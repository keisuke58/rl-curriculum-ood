"""
Stage transition heatmap for progressive, reverse, random, self_paced strategies.

For each strategy:
  X axis: training step
  Y axis: seed (0-9)
  Color:  current env (easy=blue, medium=orange, hard=red)

Shows variance in transition timing across seeds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

STAGE_COLORS = {"easy": "#4878cf", "medium": "#f28e2b", "hard": "#e15759"}
STAGED_STRATEGIES = ["progressive", "reverse", "random", "self_paced"]


def load_csv(strategy: str, seed: int) -> pd.DataFrame | None:
    p = RESULTS_DIR / f"{strategy}_seed{seed}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def plot_heatmap(strategy: str, ax: plt.Axes, seeds: list[int]):
    dfs = {}
    for seed in seeds:
        df = load_csv(strategy, seed)
        if df is not None and "env_key" in df.columns:
            dfs[seed] = df

    if not dfs:
        ax.set_title(f"{strategy} (no data)")
        return

    # Collect all steps
    all_steps = sorted(set().union(*[set(df["step"]) for df in dfs.values()]))

    img = np.zeros((len(seeds), len(all_steps), 3))
    step_idx = {s: i for i, s in enumerate(all_steps)}

    # Map env key to RGB
    color_map = {
        "easy":   np.array([72,  120, 207]) / 255,
        "medium": np.array([242, 142,  43]) / 255,
        "hard":   np.array([225,  87,  89]) / 255,
    }

    for row, seed in enumerate(seeds):
        if seed not in dfs:
            img[row] = 0.5  # grey for missing
            continue
        df = dfs[seed]
        for _, r in df.iterrows():
            col = step_idx.get(int(r["step"]))
            if col is not None:
                key = str(r.get("env_key", "easy"))
                img[row, col] = color_map.get(key, np.array([0.5, 0.5, 0.5]))

    ax.imshow(img, aspect="auto", interpolation="nearest",
              extent=[min(all_steps)/1000, max(all_steps)/1000, len(seeds)-0.5, -0.5])
    ax.set_xlabel("Training steps (×1000)")
    ax.set_ylabel("Seed")
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels(seeds, fontsize=7)
    ax.set_title(strategy.replace("_", " ").title())


def main():
    available = [s for s in STAGED_STRATEGIES
                 if any((RESULTS_DIR / f"{s}_seed{i}.csv").exists() for i in range(10))]

    if not available:
        print("No CSV files found for staged strategies.")
        sys.exit(0)

    seeds = list(range(10))
    ncols = min(len(available), 2)
    nrows = (len(available) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, strategy in enumerate(available):
        plot_heatmap(strategy, axes[i], seeds)

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=k.title())
               for k, c in STAGE_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Curriculum Stage Transitions per Seed", fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"stage_heatmap.{ext}", bbox_inches="tight", dpi=150)
    print(f"Saved → figures/stage_heatmap.{{pdf,png}}")


if __name__ == "__main__":
    main()
