"""MiniGrid observation wrappers for SB3."""
import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper


def make_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
        env.reset(seed=seed)
        return env
    return _init


class ObsNoiseWrapper(gym.ObservationWrapper):
    """Adds zero-mean Gaussian noise to flat observations during training.

    Cheap input-level regularizer: the policy can no longer rely on exact
    one-hot activations of the flattened MiniGrid encoding, which reduces
    overfitting to training-env-specific observation patterns (a suspected
    contributor to near-zero OOD transfer). Evaluation uses unwrapped envs,
    so test-time observations stay clean.
    """

    def __init__(self, env, noise_std: float = 0.05, seed: int = 0):
        super().__init__(env)
        self.noise_std = noise_std
        self._np_rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=env.observation_space.shape, dtype=np.float32,
        )

    def observation(self, obs):
        obs = obs.astype(np.float32)
        if self.noise_std > 0:
            obs = obs + self._np_rng.normal(0.0, self.noise_std, size=obs.shape).astype(np.float32)
        return obs
