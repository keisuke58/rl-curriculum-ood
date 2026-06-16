"""
Ablation: effect of success threshold on curriculum switching speed.
Runs progressive strategy with threshold in {0.6, 0.7, 0.8} × 5 seeds.

Usage:
  python ablation_threshold.py
  python ablation_threshold.py --thresholds 0.6 0.8 --seeds 0 1 2
"""
import argparse
import csv
import time
from pathlib import Path

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from curriculum import CurriculumEnv

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)


class LogCallback(BaseCallback):
    def __init__(self, csv_path, eval_freq=50_000, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self._last_log = 0
        self._rows = []

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self.eval_freq:
            self._last_log = self.num_timesteps
            monitor = self.training_env.envs[0]
            curriculum_env = monitor.env
            key = getattr(curriculum_env, "_current_key", "?")
            stage = getattr(curriculum_env, "stage_idx", 0)
            recent = monitor.episode_returns[-20:]
            mean_r = float(np.mean(recent)) if recent else 0.0
            std_r = float(np.std(recent)) if recent else 0.0
            self._rows.append([self.num_timesteps, mean_r, std_r, key, stage])
        return True

    def _on_training_end(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_reward", "std_reward", "env_key", "stage"])
            writer.writerows(self._rows)


def run_ablation(threshold: float, seed: int):
    tag = f"threshold{int(threshold*100)}_seed{seed}"
    csv_path = RESULTS_DIR / f"{tag}.csv"
    if csv_path.exists():
        print(f"  Exists, skipping: {csv_path.name}")
        return

    env = CurriculumEnv(
        strategy="progressive",
        seed=seed,
        success_threshold=threshold,
        window=CFG["training"]["window"],
        max_steps=CFG["training"]["max_steps"],
    )
    env = Monitor(env)

    ppo_cfg = CFG["ppo"].copy()
    policy = ppo_cfg.pop("policy")
    model = PPO(policy, env, seed=seed, verbose=0, **ppo_cfg)

    t0 = time.time()
    model.learn(
        total_timesteps=CFG["training"]["total_timesteps"],
        callback=LogCallback(csv_path, eval_freq=CFG["eval"]["eval_freq"]),
    )
    elapsed = time.time() - t0
    print(f"  threshold={threshold}  seed={seed}  {elapsed:.0f}s → {csv_path.name}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    for threshold in args.thresholds:
        for seed in args.seeds:
            run_ablation(threshold, seed)


if __name__ == "__main__":
    main()
