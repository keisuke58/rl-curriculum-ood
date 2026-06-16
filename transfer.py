"""
Transfer learning evaluation: fine-tune trained models on new environments.

Measures how quickly each curriculum strategy adapts to an unseen task,
using the number of steps to reach 70% success rate as the key metric.

Usage:
  python transfer.py --strategy progressive --seed 0
  python transfer.py --all
"""
import io
import json
import argparse
from pathlib import Path

import torch
# Must patch before importing SB3
_orig_torch_load = torch.load
def _patched_torch_load(f, *a, **kw):
    kw["weights_only"] = False
    if hasattr(f, "read"):
        f = io.BytesIO(f.read())
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_torch_load

import numpy as np
import yaml
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

RESULTS_DIR = Path(__file__).parent / "results"
TRANSFER_DIR = RESULTS_DIR / "transfer"
TRANSFER_DIR.mkdir(exist_ok=True)

FINETUNE_STEPS = CFG.get("transfer_finetune_steps", 200_000)
SUCCESS_TARGET = 0.7
EVAL_FREQ = 10_000
N_EVAL = 20


class TransferCallback(BaseCallback):
    """Logs success rate every EVAL_FREQ steps during fine-tuning."""

    def __init__(self, env_id: str, verbose=0):
        super().__init__(verbose)
        self.env_id = env_id
        self._last_log = 0
        self.rows = []           # (step, success_rate)
        self.steps_to_target = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= EVAL_FREQ:
            self._last_log = self.num_timesteps
            monitor = self.training_env.envs[0]
            recent = monitor.episode_returns[-N_EVAL:]
            success_rate = float(np.mean([r > 0 for r in recent])) if recent else 0.0
            self.rows.append((self.num_timesteps, success_rate))

            if self.steps_to_target is None and success_rate >= SUCCESS_TARGET:
                self.steps_to_target = self.num_timesteps
        return True


def make_env(env_id: str, seed: int = 42):
    env = gym.make(env_id, max_steps=CFG["training"]["max_steps"])
    env = FlatObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def zero_shot_eval(model, env_id: str, n_episodes: int = N_EVAL) -> dict:
    env = gym.make(env_id, max_steps=CFG["training"]["max_steps"])
    env = FlatObsWrapper(env)
    rewards, successes = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(float(terminated and ep_reward > 0))
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
    }


def run_transfer(strategy: str, seed: int):
    model_path = RESULTS_DIR / f"{strategy}_seed{seed}.zip"
    if not model_path.exists():
        print(f"  Model not found: {model_path.name} — skipping")
        return

    out_path = TRANSFER_DIR / f"transfer_{strategy}_seed{seed}.json"
    if out_path.exists():
        print(f"  Already done: {out_path.name}")
        return

    results = {"strategy": strategy, "seed": seed, "transfer_envs": {}}

    for env_id in CFG["transfer_envs"]:
        print(f"  Fine-tuning {strategy}/seed{seed} on {env_id}...")
        env = make_env(env_id, seed=seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(str(model_path), device=device)

        # Zero-shot baseline before any fine-tuning
        zs = zero_shot_eval(model, env_id)
        print(f"    zero-shot: success={zs['success_rate']:.3f}  reward={zs['mean_reward']:.3f}")

        model.set_env(env)
        cb = TransferCallback(env_id)
        model.learn(total_timesteps=FINETUNE_STEPS, callback=cb, reset_num_timesteps=True)

        results["transfer_envs"][env_id] = {
            "zero_shot": zs,
            "steps_to_target": cb.steps_to_target,
            "final_success_rate": cb.rows[-1][1] if cb.rows else 0.0,
            "curve": cb.rows,
        }
        env.close()
        print(f"    steps_to_{int(SUCCESS_TARGET*100)}%={cb.steps_to_target}  "
              f"final={results['transfer_envs'][env_id]['final_success_rate']:.3f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path.name}")


def main():
    all_strategies = (
        CFG["strategies"]
        + CFG.get("rnd_strategies", [])
        + CFG.get("icm_strategies", [])
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="progressive", choices=all_strategies)
    parser.add_argument("--strategies", nargs="+", choices=all_strategies,
                        help="Run multiple strategies (overrides --strategy)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        targets = all_strategies
    elif args.strategies:
        targets = args.strategies
    else:
        targets = None

    if targets:
        for strategy in targets:
            for seed in CFG["seeds"]:
                run_transfer(strategy, seed)
    else:
        run_transfer(args.strategy, args.seed)


if __name__ == "__main__":
    main()
