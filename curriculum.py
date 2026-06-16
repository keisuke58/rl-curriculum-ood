"""
Curriculum environment: switches between MiniGrid envs based on a condition.

Strategies:
  progressive  : Easy -> Medium -> Hard (switch on success threshold)
  reverse      : Hard -> Medium -> Easy (same switching logic)
  random       : staged like progressive, but samples randomly within allowed stages
                 (distinct from mixed: tracks stage progression, only unlocks harder envs
                  once success threshold met)
  hard_only    : always Hard (no curriculum)
  mixed        : uniform sample from all 3 at all times (multi-task baseline)
  self_paced   : Easy -> Medium -> Hard, advances when learning PLATEAUS
                 (delta mean_reward over last window < progress_threshold)
"""
import random
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper
from collections import deque


class TransposeCHWWrapper(gym.ObservationWrapper):
    """Transpose (H,W,C) → (C,H,W) so SB3 CnnPolicy sees channels-first images."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, obs):
        return obs.transpose(2, 0, 1)

ENVS = {
    "easy":   "MiniGrid-Empty-8x8-v0",
    "medium": "MiniGrid-FourRooms-v0",
    "hard":   "MiniGrid-KeyCorridorS3R1-v0",
}

SEQUENCES = {
    "progressive": ["easy", "medium", "hard"],
    "reverse":     ["hard", "medium", "easy"],
    "random":      ["easy", "medium", "hard"],
    "hard_only":   ["hard"],
    "mixed":       ["easy", "medium", "hard"],
    "self_paced":  ["easy", "medium", "hard"],
}


class CurriculumEnv(gym.Env):
    """
    Wraps multiple MiniGrid environments.
    On each episode reset, selects the current env based on curriculum strategy.
    """

    def __init__(
        self,
        strategy: str,
        seed: int = 0,
        success_threshold: float = 0.7,
        window: int = 20,
        max_steps: int = 500,
        progress_threshold: float = 0.02,  # for self_paced: min delta to not be "plateau"
        use_image: bool = False,           # experiment B: raw image obs for CnnPolicy
    ):
        assert strategy in SEQUENCES, f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.seed = seed
        self.success_threshold = success_threshold
        self.window = window
        self.max_steps = max_steps
        self.progress_threshold = progress_threshold
        self.use_image = use_image

        self.sequence = SEQUENCES[strategy]
        self.stage_idx = 0
        self.recent_successes = deque(maxlen=window)
        self.reward_history = deque(maxlen=window * 2)  # for self_paced plateau detection

        self._ep_count = 0

        self._envs = {}
        for key, env_id in ENVS.items():
            env = gym.make(env_id, max_steps=max_steps)
            if use_image:
                env = ImgObsWrapper(env)       # dict → image array (H,W,C)
                env = TransposeCHWWrapper(env)  # (H,W,C) → (C,H,W) for CnnPolicy
            else:
                env = FlatObsWrapper(env)
            env.reset(seed=seed)
            self._envs[key] = env

        sample_env = self._envs["easy"]
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space

        self._current_key = self.sequence[0]
        self._current_env = self._envs[self._current_key]
        self._episode_step = 0

    def _should_advance(self) -> bool:
        """Return True if stage advancement condition is met."""
        if self.stage_idx >= len(self.sequence) - 1:
            return False

        if self.strategy in ("progressive", "reverse"):
            return (
                len(self.recent_successes) == self.window
                and np.mean(self.recent_successes) >= self.success_threshold
            )

        if self.strategy == "random":
            # Advance when success threshold met — then sample randomly from unlocked stages
            return (
                len(self.recent_successes) == self.window
                and np.mean(self.recent_successes) >= self.success_threshold
            )

        if self.strategy == "self_paced":
            # Advance when learning plateaus: recent improvement < progress_threshold
            # Guard: require older half shows agent already achieved something (> 0 mean)
            # so we don't advance from a completely stuck zero-reward state.
            if len(self.reward_history) < self.window * 2:
                return False
            hist = list(self.reward_history)
            recent = np.mean(hist[-self.window:])
            older  = np.mean(hist[-self.window * 2:-self.window])
            return older > 0 and (recent - older) < self.progress_threshold

        return False

    def _select_env_key(self) -> str:
        if self.strategy in ("progressive", "reverse", "self_paced"):
            if self._should_advance():
                self.stage_idx += 1
                self.recent_successes.clear()
                self.reward_history.clear()
            return self.sequence[self.stage_idx]

        if self.strategy == "random":
            # Advance stage if threshold met, then sample uniformly from all unlocked stages
            if self._should_advance():
                self.stage_idx += 1
                self.recent_successes.clear()
            return random.choice(self.sequence[:self.stage_idx + 1])

        if self.strategy == "mixed":
            # Multi-task baseline: always sample all 3 uniformly, no stage tracking
            return random.choice(self.sequence)

        if self.strategy == "hard_only":
            return self.sequence[0]  # always "hard"

        return self.sequence[0]

    def reset(self, **kwargs):
        self._current_key = self._select_env_key()
        self._current_env = self._envs[self._current_key]
        self._episode_step = 0
        self._ep_count += 1
        obs, info = self._current_env.reset()
        info["env_key"] = self._current_key
        info["stage"] = self.stage_idx
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        self._episode_step += 1
        done = terminated or truncated
        if done:
            success = float(terminated and reward > 0)
            self.recent_successes.append(success)
            self.reward_history.append(reward)
            info["success"] = success
            info["env_key"] = self._current_key
            info["stage"] = self.stage_idx
        return obs, reward, terminated, truncated, info

    def close(self):
        for env in self._envs.values():
            env.close()

    @property
    def current_stage(self) -> str:
        return self._current_key
