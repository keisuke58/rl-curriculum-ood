"""MiniGrid observation wrapper: converts image obs to flat vector for SB3."""
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
