"""
Statistical significance testing: Kruskal-Wallis + post-hoc Mann-Whitney U
on OOD success rates across curriculum strategies.

Usage:
  python analysis/stats_test.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_ood_data() -> pd.DataFrame:
    rows = []
    for jf in RESULTS_DIR.glob("ood_*.json"):
        with open(jf) as f:
            d = json.load(f)
        for env_id, s in d["test_envs"].items():
            rows.append({
                "strategy": d["strategy"],
                "seed": d["seed"],
                "env_id": env_id,
                "success_rate": s["success_rate"],
            })
    return pd.DataFrame(rows)


def run_tests(df: pd.DataFrame):
    strategies = sorted(df["strategy"].unique())
    for env_id in sorted(df["env_id"].unique()):
        sub = df[df["env_id"] == env_id]
        groups = [sub[sub["strategy"] == s]["success_rate"].values for s in strategies]
        groups = [g for g in groups if len(g) > 0]  # drop strategies with no data
        active = [s for s, g in zip(strategies, [sub[sub["strategy"]==s]["success_rate"].values for s in strategies]) if len(g) > 0]

        print(f"\n{'='*60}")
        print(f"Env: {env_id}")
        print(f"{'='*60}")

        # Kruskal-Wallis (skip if all values identical)
        all_vals = np.concatenate(groups)
        if np.all(all_vals == all_vals[0]):
            print(f"Kruskal-Wallis: SKIPPED (all values identical = {all_vals[0]:.3f})")
            continue
        stat, p = stats.kruskal(*groups)
        print(f"Kruskal-Wallis: H={stat:.3f}, p={p:.4f}  {'*SIGNIFICANT*' if p < 0.05 else ''}")

        # Pairwise Mann-Whitney U with Bonferroni correction
        pairs = list(combinations(range(len(active)), 2))
        n_comparisons = max(len(pairs), 1)
        print(f"\nPairwise Mann-Whitney U (Bonferroni α={0.05/n_comparisons:.4f}):")
        for i, j in pairs:
            si, sj = active[i], active[j]
            u, p_val = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            sig = "*" if p_val < 0.05 / n_comparisons else ""
            print(f"  {si:15s} vs {sj:15s}: U={u:.0f}, p={p_val:.4f} {sig}")


if __name__ == "__main__":
    df = load_ood_data()
    if df.empty:
        print("No OOD result files found. Run evaluate.py first.")
    else:
        run_tests(df)
