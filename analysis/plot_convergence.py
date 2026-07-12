"""
Convergence detection via intrinsic signals (extension).

Research question (secondary): can the RND predictor error serve as an
intrinsic proxy for "training has converged", and does it flag the plateau
earlier / more reliably than the reward curve itself?  If so, it could drive
automatic early stopping instead of a fixed compute budget.

Two signals are compared per RND run:
  1. RND EMA loss  (intrinsic; logged as `rnd_loss` by train.py --rnd)
  2. Reward-curve slope (extrinsic baseline; derived from `mean_reward`)

For each we detect the first sustained plateau and report how much of the 1M
step budget could have been saved by stopping there.

Usage:
  python analysis/plot_convergence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style; style.apply()

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# RND runs carry the intrinsic signal. Add/remove tags as experiments grow.
RND_STRATEGIES = ["progressive_rnd", "random_rnd", "hard_only_rnd"]
LABELS = {
    "progressive_rnd": "Progressive + RND",
    "random_rnd":      "Random + RND",
    "hard_only_rnd":   "Hard Only + RND",
}
COLORS = {
    "progressive_rnd": "#1565C0",
    "random_rnd":      "#6A1B9A",
    "hard_only_rnd":   "#E65100",
}


def load_strategy(strategy: str) -> pd.DataFrame | None:
    dfs = []
    for csv in RESULTS_DIR.glob(f"{strategy}_seed*.csv"):
        df = pd.read_csv(csv)
        df["seed"] = int(csv.stem.split("seed")[1])
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _trailing_slope(y: np.ndarray, w: int) -> np.ndarray:
    """|Normalized slope| over a trailing window of length w (per-sample)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    scale = np.nanmax(y) - np.nanmin(y)
    scale = scale if scale > 1e-9 else 1.0
    out = np.full(n, np.nan)
    for i in range(w, n):
        seg = y[i - w:i]
        xs = np.arange(w)
        # least-squares slope of the trailing window, normalized by signal range
        b = np.polyfit(xs, seg, 1)[0]
        out[i] = abs(b) / scale
    return out


def first_plateau(steps: np.ndarray, values: np.ndarray,
                  window: int = 5, tol: float = 0.01, patience: int = 3) -> float | None:
    """
    First step at which the normalized trailing slope stays below `tol` for
    `patience` consecutive samples (a sustained plateau). Returns the step, or
    None if never sustained.
    """
    slope = _trailing_slope(values, window)
    run = 0
    for i in range(len(slope)):
        if np.isnan(slope[i]):
            continue
        if slope[i] < tol:
            run += 1
            if run >= patience:
                return float(steps[i - patience + 1])
        else:
            run = 0
    return None


def analyze(strategy: str):
    """Return per-seed convergence points for RND signal vs reward baseline."""
    df = load_strategy(strategy)
    if df is None or "rnd_loss" not in df.columns:
        return None

    rows = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values("step")
        steps = g["step"].values
        reward = g["mean_reward"].values
        rnd = g["rnd_loss"].values
        if np.all(np.isnan(rnd)):
            continue
        conv_rnd = first_plateau(steps, rnd, window=5, tol=0.01, patience=3)
        conv_rew = first_plateau(steps, reward, window=5, tol=0.01, patience=3)
        budget = float(steps[-1])
        rows.append({
            "seed": seed,
            "conv_rnd": conv_rnd,
            "conv_reward": conv_rew,
            "budget": budget,
        })
    return rows if rows else None


def plot_signals(strategy: str, ax):
    """Twin-axis plot: mean reward (left) + RND EMA loss (right), aggregated over seeds."""
    df = load_strategy(strategy)
    if df is None or "rnd_loss" not in df.columns:
        return False

    piv_r = df.pivot_table(index="step", values="mean_reward", aggfunc="mean")
    piv_l = df.pivot_table(index="step", values="rnd_loss", aggfunc="mean")
    steps = piv_r.index.values / 1e6
    color = COLORS.get(strategy, "#333")

    ax.plot(steps, piv_r["mean_reward"].values, color=color, lw=2, label="Mean reward")
    ax.set_ylabel("Mean Episode Reward", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.set_xlabel("Timesteps (×10⁶)")
    ax.set_title(LABELS.get(strategy, strategy))

    ax2 = ax.twinx()
    ax2.plot(piv_l.index.values / 1e6, piv_l["rnd_loss"].values,
             color="#666", lw=1.5, ls="--", label="RND loss (EMA)")
    ax2.set_ylabel("RND predictor loss (EMA)", color="#666")
    ax2.tick_params(axis="y", labelcolor="#666")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    # Mark detected convergence points (median over seeds)
    rows = analyze(strategy)
    if rows:
        cr = [r["conv_rnd"] for r in rows if r["conv_rnd"] is not None]
        cw = [r["conv_reward"] for r in rows if r["conv_reward"] is not None]
        if cr:
            ax.axvline(np.median(cr) / 1e6, color="#C62828", lw=1.2, ls=":",
                       label="RND-detected convergence")
        if cw:
            ax.axvline(np.median(cw) / 1e6, color="#2E7D32", lw=1.2, ls=":",
                       label="Reward-plateau")
    return True


def main():
    loaded = {s: load_strategy(s) for s in RND_STRATEGIES}
    avail = [s for s, df in loaded.items() if df is not None and "rnd_loss" in df.columns
             and not df["rnd_loss"].isna().all()]

    # Most likely failure mode: RND CSVs exist but predate the rnd_loss logger.
    stale = [s for s, df in loaded.items()
             if df is not None and ("rnd_loss" not in df.columns or df["rnd_loss"].isna().all())]
    if stale and not avail:
        print("Found RND result CSVs WITHOUT the `rnd_loss` convergence signal:")
        for s in stale:
            print(f"  - {s}")
        print("\nThese predate the updated logger. Re-run to regenerate them:")
        print("  bash launch_extensions.sh          # re-syncs updated code + relaunches RND runs")
        print("  # or locally, per strategy:")
        print("  python train.py --strategy progressive --seeds 0 1 2 3 4 --rnd")
        return
    if not avail:
        print("No RND results found. Run e.g.:")
        print("  bash launch_extensions.sh")
        print("  python train.py --strategy progressive --seeds 0 1 2 3 4 --rnd")
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5.2 * len(avail), 4.2), squeeze=False)
    for ax, strat in zip(axes[0], avail):
        plot_signals(strat, ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Convergence Detection: RND Intrinsic Signal vs Reward Plateau", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"convergence_detection.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/convergence_detection.{pdf,png}")

    # Summary table: how early does the RND signal fire vs the reward curve?
    print("\nConvergence step (median over seeds) and budget saved by early stop:")
    print(f"{'strategy':<20s} {'RND step':>10s} {'reward step':>12s} {'budget saved':>13s}")
    for strat in avail:
        rows = analyze(strat)
        if not rows:
            continue
        budget = np.median([r["budget"] for r in rows])
        cr = [r["conv_rnd"] for r in rows if r["conv_rnd"] is not None]
        cw = [r["conv_reward"] for r in rows if r["conv_reward"] is not None]
        rnd_step = np.median(cr) if cr else float("nan")
        rew_step = np.median(cw) if cw else float("nan")
        saved = 1.0 - (rnd_step / budget) if cr else float("nan")
        print(f"{LABELS.get(strat, strat):<20s} {rnd_step:>10.0f} {rew_step:>12.0f} {saved:>12.1%}")


if __name__ == "__main__":
    main()
