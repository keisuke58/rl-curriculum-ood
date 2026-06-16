"""
Plot learning curves (mean reward over timesteps) for all curriculum strategies.
Aggregates over seeds with mean ± 1 std shading.

Usage:
  python analysis/plot_learning_curves.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

STRATEGIES = ["progressive", "reverse", "random", "hard_only", "mixed", "self_paced"]
RND_STRATEGIES = ["progressive_rnd", "random_rnd", "hard_only_rnd", "progressive_icm"]
COLORS = {
    "progressive":     "#2196F3",
    "reverse":         "#F44336",
    "random":          "#9C27B0",
    "hard_only":       "#FF9800",
    "mixed":           "#4CAF50",
    "self_paced":      "#00BCD4",
    "progressive_rnd": "#1565C0",
    "random_rnd":      "#6A1B9A",
    "hard_only_rnd":   "#E65100",
    "progressive_icm": "#0097A7",
}
LABELS = {
    "progressive":     "Progressive (Easy→Hard)",
    "reverse":         "Reverse (Hard→Easy)",
    "random":          "Random (staged)",
    "hard_only":       "Hard Only",
    "mixed":           "Mixed (Multi-task)",
    "self_paced":      "Self-Paced (plateau)",
    "progressive_rnd": "Progressive + RND",
    "random_rnd":      "Random + RND",
    "hard_only_rnd":   "Hard Only + RND",
    "progressive_icm": "Progressive + ICM",
}


def load_strategy(strategy: str) -> pd.DataFrame | None:
    dfs = []
    for csv in RESULTS_DIR.glob(f"{strategy}_seed*.csv"):
        df = pd.read_csv(csv)
        seed = int(csv.stem.split("seed")[1])
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def compute_auc(steps: np.ndarray, rewards: np.ndarray) -> float:
    """Normalized area under the learning curve."""
    if len(steps) < 2:
        return float(rewards[-1]) if len(rewards) > 0 else 0.0
    total = float(np.trapezoid(rewards, steps))
    return total / (steps[-1] - steps[0])


def plot_curves(strategies: list, title: str, out_stem: str):
    mpl.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    auc_data = {}

    for strat in strategies:
        df = load_strategy(strat)
        if df is None:
            continue

        pivot = df.pivot_table(index="step", values="mean_reward", aggfunc=["mean", "std"])
        steps = pivot.index.values
        mu    = pivot["mean"]["mean_reward"].values
        sigma = pivot["std"]["mean_reward"].fillna(0).values

        color = COLORS.get(strat, "#888")
        ax.plot(steps / 1e6, mu, color=color, label=LABELS.get(strat, strat), linewidth=2)
        ax.fill_between(steps / 1e6, mu - sigma, mu + sigma, color=color, alpha=0.15)
        auc_data[strat] = compute_auc(steps, mu)

    ax.set_xlabel("Timesteps (×10⁶)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → figures/{out_stem}.{{pdf,png}}")

    # Print AUC table
    if auc_data:
        print("\nSample Efficiency (AUC, normalized):")
        for s, v in sorted(auc_data.items(), key=lambda x: -x[1]):
            print(f"  {LABELS.get(s, s):<30s}: {v:.4f}")

    return auc_data


if __name__ == "__main__":
    # Plot 1: base strategies
    auc1 = plot_curves(
        strategies=STRATEGIES,
        title="Curriculum Learning Curves — Base Strategies (10 seeds)",
        out_stem="learning_curves",
    )
    # Plot 2: intrinsic motivation comparison
    avail_rnd = [s for s in RND_STRATEGIES if load_strategy(s) is not None]
    if avail_rnd:
        plot_curves(
            strategies=["progressive"] + avail_rnd,
            title="Intrinsic Motivation: RND vs ICM vs Baseline",
            out_stem="learning_curves_rnd",
        )
