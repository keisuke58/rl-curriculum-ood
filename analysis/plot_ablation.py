"""
Ablation: effect of success threshold on curriculum switching speed.
Plots learning curves for different thresholds and stage transition timing.

Usage:
  python analysis/plot_ablation.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ablation"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

PALETTE = ["#1565C0", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]


def load_ablation() -> dict[float, pd.DataFrame]:
    """Returns {threshold: concat_df_over_seeds}."""
    data: dict[float, list] = {}
    for csv in sorted(RESULTS_DIR.glob("threshold*_seed*.csv")):
        parts = csv.stem.split("_")
        thr = int(parts[0].replace("threshold", "")) / 100.0
        df = pd.read_csv(csv)
        df["seed"] = int(parts[1].replace("seed", ""))
        data.setdefault(thr, []).append(df)
    return {thr: pd.concat(dfs, ignore_index=True) for thr, dfs in sorted(data.items())}


def plot_learning_curves(by_thr: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (thr, df), color in zip(by_thr.items(), PALETTE):
        pivot = df.pivot_table(index="step", values="mean_reward", aggfunc=["mean", "std"])
        steps = pivot.index.values
        mu = pivot["mean"]["mean_reward"].values
        sigma = pivot["std"]["mean_reward"].fillna(0).values
        ax.plot(steps / 1e6, mu, label=f"threshold={thr}", color=color, linewidth=2)
        ax.fill_between(steps / 1e6, mu - sigma, mu + sigma, color=color, alpha=0.15)

    ax.set_xlabel("Timesteps (×10⁶)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Ablation: Success Threshold vs Learning Speed (Progressive)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"ablation_learning_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/ablation_learning_curves.{pdf,png}")


def plot_stage_transitions(by_thr: dict):
    """Box plot of first step where stage >= 1 (medium) and stage >= 2 (hard)."""
    rows = []
    for thr, df in by_thr.items():
        for seed, sdf in df.groupby("seed"):
            for target_stage, label in [(1, "→Medium"), (2, "→Hard")]:
                reached = sdf[sdf["stage"] >= target_stage]
                first = int(reached["step"].iloc[0]) if not reached.empty else None
                rows.append({"threshold": thr, "transition": label, "step": first, "seed": seed})

    tdf = pd.DataFrame(rows).dropna()
    if tdf.empty:
        print("No stage transition data yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, trans in zip(axes, ["→Medium", "→Hard"]):
        sub = tdf[tdf["transition"] == trans]
        thresholds = sorted(sub["threshold"].unique())
        data_per_thr = [sub[sub["threshold"] == t]["step"].values / 1e3 for t in thresholds]
        bp = ax.boxplot(data_per_thr, labels=[str(t) for t in thresholds],
                        patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Success Threshold")
        ax.set_ylabel("Step (×10³)")
        ax.set_title(f"First Transition {trans}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Curriculum Transition Timing vs Threshold", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"ablation_transitions.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/ablation_transitions.{pdf,png}")


def print_summary(by_thr: dict):
    print("\nAblation Summary (mean reward at final step, mean ± std over seeds):")
    for thr, df in by_thr.items():
        final = df.groupby("seed")["mean_reward"].last()
        print(f"  threshold={thr:.1f}:  {final.mean():.3f} ± {final.std():.3f}  (n={len(final)})")


if __name__ == "__main__":
    by_thr = load_ablation()
    if not by_thr:
        print("No ablation results found. Run ablation_threshold.py first.")
        sys.exit(0)
    print_summary(by_thr)
    plot_learning_curves(by_thr)
    plot_stage_transitions(by_thr)
