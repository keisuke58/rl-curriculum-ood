"""
Random Network Distillation (Burda et al., 2019) for intrinsic motivation.

Usage:
  rnd = RNDModule(obs_dim)
  intrinsic_reward = rnd.intrinsic_reward(obs_tensor)
  rnd.update(obs_tensor)
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class RNDNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RNDModule:
    """
    Wraps target (fixed) and predictor (trained) networks.
    Intrinsic reward = MSE between predictor and target outputs.
    """

    def __init__(self, obs_dim: int, lr: float = 1e-4, device: str = "cpu"):
        self.device = device
        self.target = RNDNetwork(obs_dim).to(device)
        self.predictor = RNDNetwork(obs_dim).to(device)

        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        return obs.to(self.device)

    def intrinsic_reward(self, obs: np.ndarray) -> float:
        """Compute normalized intrinsic reward for a single observation."""
        with torch.no_grad():
            x = self._to_tensor(obs).unsqueeze(0)
            raw_reward = float(((self.predictor(x) - self.target(x)) ** 2).mean())

        self._reward_count += 1
        delta = raw_reward - self._reward_running_mean
        self._reward_running_mean += delta / self._reward_count
        self._reward_running_var += delta * (raw_reward - self._reward_running_mean)
        std = max((self._reward_running_var / max(self._reward_count - 1, 1)) ** 0.5, 1e-8)
        return raw_reward / std

    def update(self, obs_batch: np.ndarray) -> float:
        """Update predictor on a batch of observations. Returns mean loss."""
        x = self._to_tensor(obs_batch)
        with torch.no_grad():
            target_feat = self.target(x)
        loss = ((self.predictor(x) - target_feat) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)


class RNDEnvWrapper(gym.Wrapper):
    """
    Gym wrapper that adds RND intrinsic reward to the extrinsic reward.
    This is the correct approach: reward augmentation happens at env.step(),
    so SB3's rollout buffer sees the augmented reward automatically.
    """

    def __init__(self, env: gym.Env, rnd: RNDModule, coef: float = 0.1,
                 update_freq: int = 256):
        super().__init__(env)
        self.rnd = rnd
        self.coef = coef
        self.update_freq = update_freq
        self._obs_buffer: list[np.ndarray] = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        intr = self.rnd.intrinsic_reward(obs)
        self._obs_buffer.append(obs.copy())
        if len(self._obs_buffer) >= self.update_freq:
            self.rnd.update(np.stack(self._obs_buffer))
            self._obs_buffer = []
        info["intrinsic_reward"] = intr
        return obs, reward + self.coef * intr, terminated, truncated, info
