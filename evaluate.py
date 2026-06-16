"""
Zero-shot OOD evaluation: load trained model, evaluate on held-out test envs.

Usage:
  python evaluate.py --strategy progressive --seed 0
  python evaluate.py --all
"""
import argparse
import csv
import io
import json
from pathlib import Path

# Patch torch.load BEFORE importing SB3 — SB3 captures `th = torch` at import time,
# so the patch must exist before `from stable_baselines3 import PPO`.
import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, *a, **kw):
    kw["weights_only"] = False
    if hasattr(f, "read"):
        f = io.BytesIO(f.read())
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_torch_load

import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

import yaml

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

RESULTS_DIR = Path(__file__).parent / "results"
TEST_ENVS = CFG["test_envs"]
N_EVAL = CFG["eval"]["n_eval_episodes"]


def make_test_env(env_id: str, seed: int = 42):
    env = gym.make(env_id, max_steps=CFG["training"]["max_steps"])
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def evaluate_model(model, env_id: str, n_episodes: int = N_EVAL) -> dict:
    env = make_test_env(env_id)
    rewards, successes = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(float(terminated and ep_reward > 0))
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
    }


def run_eval(strategy: str, seed: int, force: bool = False):
    model_path = RESULTS_DIR / f"{strategy}_seed{seed}.zip"
    if not model_path.exists():
        print(f"  Model not found: {model_path.name} — skipping")
        return

    out_path = RESULTS_DIR / f"ood_{strategy}_seed{seed}.json"
    if out_path.exists() and not force:
        # Re-evaluate if number of test envs has changed
        import json as _json
        cached = _json.loads(out_path.read_text())
        if len(cached.get("test_envs", {})) == len(TEST_ENVS):
            print(f"  Already evaluated: {out_path.name}")
            return
        print(f"  Re-evaluating (env count changed): {out_path.name}")

    model = PPO.load(str(model_path))
    results = {"strategy": strategy, "seed": seed, "test_envs": {}}

    for env_id in TEST_ENVS:
        print(f"  Evaluating {strategy}/seed{seed} on {env_id}...")
        stats = evaluate_model(model, env_id)
        results["test_envs"][env_id] = stats
        print(f"    success_rate={stats['success_rate']:.3f}  mean_reward={stats['mean_reward']:.3f}")

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
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--n-eval",  type=int, default=None,
                        help="Override n_eval_episodes from config")
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    global N_EVAL
    if args.n_eval:
        N_EVAL = args.n_eval

    if args.all:
        for strategy in all_strategies:
            for seed in CFG["seeds"]:
                run_eval(strategy, seed, force=args.force)
    else:
        run_eval(args.strategy, args.seed, force=args.force)


if __name__ == "__main__":
    main()
