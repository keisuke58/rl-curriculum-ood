"""
Catastrophic Forgetting Analysis.

Evaluates each trained model on ALL THREE training environments (Easy, Medium, Hard)
to measure how well each curriculum strategy retains previously learned skills.

Progressive curriculum is expected to show forgetting on Easy/Medium envs.
Mixed (multi-task) curriculum should retain all three.

Usage:
  python analysis/plot_forgetting.py
  python analysis/plot_forgetting.py --n-eval 30
"""
import io
import sys
import json
import argparse
from pathlib import Path

import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, *a, **kw):
    kw["weights_only"] = False
    if hasattr(f, "read"):
        f = io.BytesIO(f.read())
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_torch_load

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

TRAIN_ENVS = {
    "easy":   "MiniGrid-Empty-8x8-v0",
    "medium": "MiniGrid-FourRooms-v0",
    "hard":   "MiniGrid-KeyCorridorS3R1-v0",
}

COLORS = {
    "progressive": "#2196F3",
    "reverse":     "#F44336",
    "random":      "#9C27B0",
    "hard_only":   "#FF9800",
    "mixed":       "#4CAF50",
    "self_paced":  "#00BCD4",
}
LABELS = {
    "progressive": "Progressive",
    "reverse":     "Reverse",
    "random":      "Random",
    "hard_only":   "Hard Only",
    "mixed":       "Mixed",
    "self_paced":  "Self-Paced",
}


def eval_model_on_env(model, env_id: str, n_episodes: int = 20, max_steps: int = 500) -> float:
    env = gym.make(env_id, max_steps=max_steps)
    env = FlatObsWrapper(env)
    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, ep_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            done = terminated or truncated
        successes.append(float(terminated and ep_r > 0))
    env.close()
    return float(np.mean(successes))


def load_all_forgetting(strategies: list, seeds: list, n_eval: int) -> pd.DataFrame:
    cache_path = RESULTS_DIR / "forgetting_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        cache = {}

    rows = []
    for strategy in strategies:
        for seed in seeds:
            model_path = RESULTS_DIR / f"{strategy}_seed{seed}.zip"
            if not model_path.exists():
                continue
            cache_key = f"{strategy}_seed{seed}"
            if cache_key in cache:
                for env_key, sr in cache[cache_key].items():
                    rows.append({"strategy": strategy, "seed": seed,
                                 "env_key": env_key, "success_rate": sr})
                continue

            print(f"  Evaluating {strategy}/seed{seed} on train envs...", flush=True)
            model = PPO.load(str(model_path))
            cache[cache_key] = {}
            for env_key, env_id in TRAIN_ENVS.items():
                sr = eval_model_on_env(model, env_id, n_eval)
                cache[cache_key][env_key] = sr
                rows.append({"strategy": strategy, "seed": seed,
                             "env_key": env_key, "success_rate": sr})
                print(f"    {env_key}: {sr:.2f}", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2))
    return pd.DataFrame(rows)


def plot_forgetting(df: pd.DataFrame, strategies: list):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    env_keys = ["easy", "medium", "hard"]
    env_labels = ["Easy\n(Empty-8x8)", "Medium\n(FourRooms)", "Hard\n(KeyCorridor)"]

    for ax, env_key, env_label in zip(axes, env_keys, env_labels):
        sub = df[df["env_key"] == env_key]
        agg = sub.groupby("strategy")["success_rate"].agg(["mean", "std"])
        strats = [s for s in strategies if s in agg.index]
        means = agg.loc[strats, "mean"].values
        stds  = agg.loc[strats, "std"].fillna(0).values
        x = np.arange(len(strats))
        colors = [COLORS.get(s, "#888") for s in strats]

        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                      width=0.6, edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(s, s) for s in strats],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Success Rate" if env_key == "easy" else "")
        ax.set_title(env_label, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Catastrophic Forgetting: Success Rate on Training Environments\n"
                 "(Progressive trades Easy/Medium retention for Hard performance)",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"catastrophic_forgetting.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/catastrophic_forgetting.{pdf,png}")


def print_summary(df: pd.DataFrame):
    print("\n=== Catastrophic Forgetting Summary (mean success rate) ===")
    pivot = df.groupby(["strategy", "env_key"])["success_rate"].mean().unstack()
    pivot = pivot[["easy", "medium", "hard"]]
    pivot.columns = ["Easy", "Medium", "Hard"]
    pivot["Mean"] = pivot.mean(axis=1)
    print(pivot.round(3).to_string())
    pivot.to_csv(RESULTS_DIR / "forgetting_summary.csv")
    print("Saved → results/forgetting_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--strategies", nargs="+",
                        default=["progressive", "reverse", "random",
                                 "hard_only", "mixed", "self_paced"])
    args = parser.parse_args()

    df = load_all_forgetting(args.strategies, args.seeds, args.n_eval)
    if df.empty:
        print("No model files found in results/. Train first.")
        sys.exit(0)
    print_summary(df)
    plot_for = [s for s in args.strategies if s in df["strategy"].unique()]
    plot_forgetting(df, plot_for)
