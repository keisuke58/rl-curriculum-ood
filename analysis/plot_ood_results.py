"""
Plot OOD evaluation results: success rate and mean reward across strategies and test envs.
Reads results/ood_*.json and produces bar charts + summary table.

Usage:
  python analysis/plot_ood_results.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

LABELS = {
    "progressive":     "Progressive",
    "reverse":         "Reverse",
    "random":          "Random",
    "hard_only":       "Hard Only",
    "mixed":           "Mixed",
    "self_paced":      "Self-Paced",
    "progressive_rnd": "Progressive+RND",
    "random_rnd":      "Random+RND",
    "hard_only_rnd":   "HardOnly+RND",
    "progressive_icm": "Progressive+ICM",
}
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


def load_ood_data() -> pd.DataFrame:
    rows = []
    for jf in RESULTS_DIR.glob("ood_*.json"):
        with open(jf) as f:
            d = json.load(f)
        strat = d["strategy"]
        seed = d["seed"]
        for env_id, stats in d["test_envs"].items():
            rows.append({
                "strategy": strat,
                "seed": seed,
                "env_id": env_id,
                "success_rate": stats["success_rate"],
                "mean_reward": stats["mean_reward"],
                "std_reward": stats["std_reward"],
            })
    return pd.DataFrame(rows)


def plot_bar_chart(df: pd.DataFrame, metric: str, ylabel: str, title: str, fname: str):
    mpl.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    strategies = sorted(df["strategy"].unique())
    test_envs = sorted(df["env_id"].unique())
    n_envs = len(test_envs)
    n_strats = len(strategies)

    fig, axes = plt.subplots(1, n_envs, figsize=(max(4, 2 * n_strats) * n_envs, 5), sharey=False)
    if n_envs == 1:
        axes = [axes]

    for ax, env_id in zip(axes, test_envs):
        sub = df[df["env_id"] == env_id]
        agg = sub.groupby("strategy")[metric].agg(["mean", "std"]).reindex(strategies)

        x = np.arange(n_strats)
        ax.bar(
            x,
            agg["mean"].fillna(0).values,
            yerr=agg["std"].fillna(0).values,
            color=[COLORS.get(s, "#888") for s in strategies],
            capsize=4,
            width=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(s, s) for s in strategies], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(env_id.replace("MiniGrid-", "").replace("-v0", ""), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGS_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def print_summary_table(df: pd.DataFrame):
    print("\n=== OOD Success Rate (mean ± std over seeds) ===")
    tbl = df.groupby(["strategy", "env_id"])["success_rate"].agg(["mean", "std"])
    tbl = tbl.round(3)
    print(tbl.to_string())
    print()

    csv_out = RESULTS_DIR / "ood_summary.csv"
    tbl.reset_index().to_csv(csv_out, index=False)
    print(f"Summary table saved → {csv_out}")


if __name__ == "__main__":
    df = load_ood_data()
    if df.empty:
        print("No OOD result files found yet. Run evaluate.py first.")
    else:
        print_summary_table(df)
        plot_bar_chart(df, "success_rate", "Success Rate", "OOD Success Rate by Strategy", "ood_success_rate.pdf")
        plot_bar_chart(df, "mean_reward", "Mean Reward", "OOD Mean Reward by Strategy", "ood_mean_reward.pdf")
