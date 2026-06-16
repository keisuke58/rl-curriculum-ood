"""
Plot curriculum stage transitions over time for progressive/reverse strategies.
Shows which env was active at each logging checkpoint.

Usage:
  python analysis/plot_stage_transitions.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

ENV_COLORS = {
    "easy":   "#A5D6A7",
    "medium": "#FFD54F",
    "hard":   "#EF9A9A",
    "?":      "#BDBDBD",
}
STRATEGIES = ["progressive", "reverse"]


def plot_transitions():
    fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(10, 3 * len(STRATEGIES)), sharex=True)
    if len(STRATEGIES) == 1:
        axes = [axes]

    for ax, strat in zip(axes, STRATEGIES):
        seed_dfs = []
        for csv in sorted(RESULTS_DIR.glob(f"{strat}_seed*.csv")):
            df = pd.read_csv(csv)
            seed = int(csv.stem.split("seed")[1])
            df["seed"] = seed
            seed_dfs.append(df)

        if not seed_dfs:
            ax.set_title(f"{strat} — no data")
            continue

        all_df = pd.concat(seed_dfs)
        seeds = sorted(all_df["seed"].unique())

        for row_idx, seed in enumerate(seeds):
            sdf = all_df[all_df["seed"] == seed].sort_values("step")
            for _, r in sdf.iterrows():
                color = ENV_COLORS.get(r["env_key"], "#BDBDBD")
                ax.barh(
                    row_idx,
                    width=50_000,
                    left=r["step"] - 25_000,
                    color=color,
                    height=0.8,
                    align="center",
                )

        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=8)
        ax.set_title(f"{strat.capitalize()} Strategy — Stage Timeline")
        ax.set_xlabel("Timesteps")

    patches = [mpatches.Patch(color=c, label=k) for k, c in ENV_COLORS.items() if k != "?"]
    fig.legend(handles=patches, loc="upper right", title="Env Difficulty")
    fig.tight_layout()
    out = FIGS_DIR / "stage_transitions.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    plot_transitions()
