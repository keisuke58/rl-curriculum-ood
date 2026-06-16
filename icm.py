"""
Intrinsic Curiosity Module (Pathak et al., 2017).

Architecture:
  - Encoder φ(s): obs → feature embedding
  - Inverse model: [φ(s), φ(s')] → predicted action
  - Forward model: [φ(s), a] → predicted φ(s')

Intrinsic reward = ||φ(s') - forward(φ(s), a)||²  (forward model error)
Inverse model is trained jointly to make embeddings action-relevant.

Usage:
  icm = ICMModule(obs_dim, n_actions)
  intrinsic_reward = icm.intrinsic_reward(obs, action, next_obs)
  icm.update(obs_batch, action_batch, next_obs_batch)
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InverseModel(nn.Module):
    """Predicts action from (φ(s), φ(s'))."""
    def __init__(self, feature_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, phi_s: torch.Tensor, phi_s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_s, phi_s_next], dim=-1))


class ForwardModel(nn.Module):
    """Predicts φ(s') from (φ(s), one-hot action)."""
    def __init__(self, feature_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, phi_s: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_s, action_onehot], dim=-1))


class ICMModule:
    """
    Full ICM: encoder + inverse model + forward model.
    Intrinsic reward = forward prediction error (scaled).
    """

    def __init__(self, obs_dim: int, n_actions: int, feature_dim: int = 128,
                 lr: float = 1e-4, beta: float = 0.2, device: str = "cpu"):
        """
        beta: weighting between forward (beta) and inverse (1-beta) loss.
        """
        self.device = device
        self.n_actions = n_actions
        self.beta = beta

        self.encoder = Encoder(obs_dim, feature_dim).to(device)
        self.inverse = InverseModel(feature_dim, n_actions).to(device)
        self.forward_model = ForwardModel(feature_dim, n_actions).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.inverse.parameters()) +
            list(self.forward_model.parameters()),
            lr=lr,
        )

        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(x).to(self.device)

    def intrinsic_reward(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> float:
        """Forward model prediction error as intrinsic reward (normalized)."""
        with torch.no_grad():
            s  = self._to_tensor(obs).unsqueeze(0)
            s_ = self._to_tensor(next_obs).unsqueeze(0)
            a_onehot = torch.zeros(1, self.n_actions, device=self.device)
            a_onehot[0, action] = 1.0

            phi_s  = self.encoder(s)
            phi_s_ = self.encoder(s_)
            pred_phi_s_ = self.forward_model(phi_s, a_onehot)
            raw_reward = float(F.mse_loss(pred_phi_s_, phi_s_))

        self._reward_count += 1
        delta = raw_reward - self._reward_running_mean
        self._reward_running_mean += delta / self._reward_count
        self._reward_running_var += delta * (raw_reward - self._reward_running_mean)
        std = max((self._reward_running_var / max(self._reward_count - 1, 1)) ** 0.5, 1e-8)
        return raw_reward / std

    def update(self, obs_batch: np.ndarray, action_batch: np.ndarray,
               next_obs_batch: np.ndarray) -> dict:
        """Update encoder + inverse + forward. Returns loss dict."""
        s  = self._to_tensor(obs_batch)
        s_ = self._to_tensor(next_obs_batch)
        a  = torch.LongTensor(action_batch).to(self.device)
        a_onehot = F.one_hot(a, self.n_actions).float()

        phi_s  = self.encoder(s)
        phi_s_ = self.encoder(s_)

        # Forward loss
        pred_phi_s_ = self.forward_model(phi_s.detach(), a_onehot)
        forward_loss = F.mse_loss(pred_phi_s_, phi_s_.detach())

        # Inverse loss
        logits = self.inverse(phi_s, phi_s_)
        inverse_loss = F.cross_entropy(logits, a)

        loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"forward": float(forward_loss), "inverse": float(inverse_loss)}


class ICMEnvWrapper(gym.Wrapper):
    """
    Gym wrapper that adds ICM intrinsic reward to the extrinsic reward.
    Buffers (obs, action, next_obs) tuples and updates ICM periodically.
    """

    def __init__(self, env: gym.Env, icm: ICMModule, coef: float = 0.1,
                 update_freq: int = 256):
        super().__init__(env)
        self.icm = icm
        self.coef = coef
        self.update_freq = update_freq
        self._buf_obs: list = []
        self._buf_act: list = []
        self._buf_next: list = []
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._last_obs is not None:
            intr = self.icm.intrinsic_reward(self._last_obs, int(action), obs)
            self._buf_obs.append(self._last_obs.copy())
            self._buf_act.append(int(action))
            self._buf_next.append(obs.copy())
            if len(self._buf_obs) >= self.update_freq:
                self.icm.update(
                    np.stack(self._buf_obs),
                    np.array(self._buf_act),
                    np.stack(self._buf_next),
                )
                self._buf_obs, self._buf_act, self._buf_next = [], [], []
            info["intrinsic_reward"] = intr
            reward = reward + self.coef * intr
        self._last_obs = obs.copy()
        return obs, reward, terminated, truncated, info
