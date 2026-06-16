"""
OOD Gap Analysis — key figure for the paper.

OOD gap = (training mean_reward) - (OOD mean_reward)
         per strategy, averaged over seeds and test environments.

Produces:
  - ood_gap_bar.pdf/png : bar chart of OOD gap per strategy
  - ood_gap_by_env.pdf/png : gap broken down per test environment
  - ood_gap_summary.csv : numeric summary table
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

STRATEGY_COLORS = {
    "progressive":     "#4878cf",
    "reverse":         "#e15759",
    "random":          "#9467bd",
    "hard_only":       "#f28e2b",
    "mixed":           "#59a14f",
    "self_paced":      "#17becf",
    "progressive_rnd": "#4878cf",
    "random_rnd":      "#9467bd",
    "hard_only_rnd":   "#f28e2b",
    "progressive_icm": "#4878cf",
}

STRATEGY_LABELS = {
    "progressive":     "Progressive",
    "reverse":         "Reverse",
    "random":          "Random (staged)",
    "hard_only":       "Hard Only",
    "mixed":           "Mixed",
    "self_paced":      "Self-Paced",
    "progressive_rnd": "Progressive+RND",
    "random_rnd":      "Random+RND",
    "hard_only_rnd":   "HardOnly+RND",
    "progressive_icm": "Progressive+ICM",
}


def load_train_perf(strategy: str, seeds: list[int]) -> float:
    """Load post-training Hard success rate from forgetting cache.

    Uses the forgetting cache (extrinsic-only evaluation) rather than
    training-time rewards, which may include intrinsic bonuses for RND variants.
    Falls back to the Hard-stage CSV reward if the cache is missing.
    """
    cache_path = RESULTS_DIR / "forgetting_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        vals = []
        for seed in seeds:
            key = f"{strategy}_seed{seed}"
            if key in cache:
                entry = cache[key]
                # Use mean success rate across all training stages
                stage_vals = list(entry.values())
                vals.append(float(np.mean(stage_vals)))
        if vals:
            return float(np.mean(vals))
    # Fallback: training CSV reward
    vals = []
    for seed in seeds:
        p = RESULTS_DIR / f"{strategy}_seed{seed}.csv"
        if p.exists():
            df = pd.read_csv(p)
            vals.append(df["mean_reward"].iloc[-1])
    return float(np.mean(vals)) if vals else np.nan


def load_ood_perf(strategy: str, seeds: list[int]) -> dict:
    """Load OOD results. Returns {env_id: [success_rates_across_seeds]}."""
    env_results: dict[str, list] = {}
    for seed in seeds:
        p = RESULTS_DIR / f"ood_{strategy}_seed{seed}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for env_id, stats in data.get("test_envs", {}).items():
            env_results.setdefault(env_id, []).append(stats.get("mean_reward", 0.0))
    return env_results


def main():
    seeds = list(range(10))
    ood_json_files = list(RESULTS_DIR.glob("ood_*.json"))
    if not ood_json_files:
        print("No OOD JSON files found. Run evaluate.py --all first.")
        sys.exit(0)

    # Discover which strategies have OOD results.
    # Exclude RND/ICM variants from the gap chart: their training rewards include
    # intrinsic bonuses, making the gap non-comparable to base strategies.
    # Their OOD=0.00 result is captured in the success-rate figure.
    strategies = sorted(set(
        f.stem.replace("ood_", "").rsplit("_seed", 1)[0]
        for f in ood_json_files
        if "_rnd" not in f.stem and "_icm" not in f.stem
    ))

    rows = []
    env_rows = []

    for strategy in strategies:
        train_r = load_train_perf(strategy, seeds)
        ood_by_env = load_ood_perf(strategy, seeds)
        if not ood_by_env:
            continue

        all_ood = [v for vals in ood_by_env.values() for v in vals]
        mean_ood = float(np.mean(all_ood))
        std_ood  = float(np.std(all_ood))
        gap = train_r - mean_ood

        rows.append({
            "strategy": strategy,
            "train_reward": train_r,
            "ood_mean_reward": mean_ood,
            "ood_std_reward": std_ood,
            "ood_gap": gap,
        })

        for env_id, vals in ood_by_env.items():
            env_rows.append({
                "strategy": strategy,
                "env": env_id.split("-")[1],  # short name
                "ood_mean_reward": float(np.mean(vals)),
                "ood_std_reward":  float(np.std(vals)),
            })

    if not rows:
        print("No data to plot.")
        sys.exit(0)

    df_gap = pd.DataFrame(rows).set_index("strategy")
    df_env = pd.DataFrame(env_rows)

    # Save summary
    df_gap.to_csv(RESULTS_DIR / "ood_gap_summary.csv")
    print(df_gap[["train_reward", "ood_mean_reward", "ood_gap"]].round(3).to_string())

    # --- Figure 1: OOD Gap bar chart ---
    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 1.2), 4))
    x = np.arange(len(strategies))
    colors = [STRATEGY_COLORS.get(s, "#888") for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    bars = ax.bar(x, df_gap.loc[strategies, "ood_gap"], color=colors,
                  edgecolor="white", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("OOD Gap (train − OOD mean reward)")
    ax.set_title("Generalization Gap by Curriculum Strategy\n"
                 "(higher = worse OOD transfer)")

    for bar, gap in zip(bars, df_gap.loc[strategies, "ood_gap"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{gap:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"ood_gap_bar.{ext}", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Figure 2: OOD performance by test environment ---
    if not df_env.empty:
        fig2, ax2 = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 4))
        envs = df_env["env"].unique()
        n_envs = len(envs)
        width = 0.8 / n_envs
        x2 = np.arange(len(strategies))

        for i, env in enumerate(envs):
            sub = df_env[df_env["env"] == env].set_index("strategy")
            vals = [sub.loc[s, "ood_mean_reward"] if s in sub.index else 0.0
                    for s in strategies]
            errs = [sub.loc[s, "ood_std_reward"]  if s in sub.index else 0.0
                    for s in strategies]
            ax2.bar(x2 + i * width - 0.4 + width / 2, vals, width,
                    yerr=errs, capsize=3, label=env, alpha=0.85)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax2.set_ylabel("Mean Reward on OOD Environment")
        ax2.set_title("OOD Performance per Test Environment")
        ax2.legend(fontsize=8, ncol=min(n_envs, 3))
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(FIGURES_DIR / f"ood_gap_by_env.{ext}", bbox_inches="tight", dpi=150)
        plt.close()

    print("Saved → figures/ood_gap_bar.{pdf,png}, ood_gap_by_env.{pdf,png}")
    print(f"        results/ood_gap_summary.csv")


if __name__ == "__main__":
    main()
