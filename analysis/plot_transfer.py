"""
Plot transfer learning results: fine-tuning adaptation speed per strategy.

Reads results/transfer/transfer_*.json and produces:
  figures/transfer_adaptation.pdf  — bar: steps-to-70% per strategy × env
  figures/transfer_final.pdf       — bar: final success rate per strategy × env

Usage:
  python analysis/plot_transfer.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
TRANSFER_DIR = RESULTS_DIR / "transfer"
FIGS_DIR = Path(__file__).parent.parent / "figures"
FIGS_DIR.mkdir(exist_ok=True)

STRATEGY_ORDER = ["progressive", "reverse", "random", "hard_only", "mixed"]
LABELS = {
    "progressive": "Progressive",
    "reverse":     "Reverse",
    "random":      "Random",
    "hard_only":   "Hard-Only",
    "mixed":       "Mixed",
}
COLORS = {
    "progressive": "#2196F3",
    "reverse":     "#F44336",
    "random":      "#9C27B0",
    "hard_only":   "#FF9800",
    "mixed":       "#4CAF50",
}
ENV_SHORT = {
    "MiniGrid-DoorKey-8x8-v0":      "DoorKey-8x8",
    "MiniGrid-MultiRoom-N4-S5-v0":  "MultiRoom-N4-S5",
}
FINETUNE_STEPS = 200_000


def load_transfer_data():
    rows = []
    for jf in sorted(TRANSFER_DIR.glob("transfer_*.json")):
        with open(jf) as f:
            d = json.load(f)
        strategy = d["strategy"]
        seed = d["seed"]
        for env_id, res in d["transfer_envs"].items():
            rows.append({
                "strategy":       strategy,
                "seed":           seed,
                "env_id":         env_id,
                "zero_shot":      res["zero_shot"]["success_rate"],
                "steps_to_70":    res["steps_to_target"],   # None = never reached
                "final_success":  res["final_success_rate"],
            })
    return rows


def aggregate(rows, metric, envs, strategies=STRATEGY_ORDER):
    result = {}
    for strat in strategies:
        result[strat] = {}
        for env in envs:
            vals = [r[metric] for r in rows
                    if r["strategy"] == strat and r["env_id"] == env]
            if metric == "steps_to_70":
                # Replace None with FINETUNE_STEPS (never adapted)
                vals_filled = [v if v is not None else FINETUNE_STEPS for v in vals]
                result[strat][env] = (np.mean(vals_filled) if vals_filled else np.nan,
                                      np.std(vals_filled) if vals_filled else np.nan,
                                      len(vals_filled))
            else:
                result[strat][env] = (np.mean(vals) if vals else np.nan,
                                      np.std(vals) if vals else np.nan,
                                      len(vals))
    return result


def plot_transfer_bars(rows):
    envs = sorted({r["env_id"] for r in rows})
    available_strats = [s for s in STRATEGY_ORDER
                        if any(r["strategy"] == s for r in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: steps-to-70%
    agg_steps = aggregate(rows, "steps_to_70", envs, available_strats)
    ax = axes[0]
    x = np.arange(len(available_strats))
    width = 0.35
    for i, env in enumerate(envs):
        means = [agg_steps[s][env][0] / 1000 for s in available_strats]
        stds  = [agg_steps[s][env][1] / 1000 for s in available_strats]
        counts = [agg_steps[s][env][2] for s in available_strats]
        bars = ax.bar(x + (i - 0.5) * width, means, width,
                      yerr=stds, capsize=3,
                      color=[COLORS[s] for s in available_strats],
                      alpha=0.6 + 0.4 * i,
                      label=ENV_SHORT.get(env, env))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in available_strats], rotation=20, ha="right")
    ax.set_ylabel("Steps to 70% success (k)")
    ax.set_title("Adaptation speed (lower = faster)")
    ax.axhline(FINETUNE_STEPS / 1000, ls="--", lw=0.8, color="gray", label="Max steps (200k)")
    ax.legend(fontsize=7)

    # Right: final success rate
    agg_final = aggregate(rows, "final_success", envs, available_strats)
    ax = axes[1]
    for i, env in enumerate(envs):
        means = [agg_final[s][env][0] for s in available_strats]
        stds  = [agg_final[s][env][1] for s in available_strats]
        ax.bar(x + (i - 0.5) * width, means, width,
               yerr=stds, capsize=3,
               color=[COLORS[s] for s in available_strats],
               alpha=0.6 + 0.4 * i,
               label=ENV_SHORT.get(env, env))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in available_strats], rotation=20, ha="right")
    ax.set_ylabel("Final success rate")
    ax.set_title("Final performance after fine-tuning")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

    n_seeds = max(agg_final[s][envs[0]][2] for s in available_strats)
    fig.suptitle(f"Transfer learning ($2\\times10^5$ steps, "
                 f"n={n_seeds} seeds each)", fontsize=10)
    fig.tight_layout()
    out = FIGS_DIR / "transfer_adaptation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def print_summary(rows):
    envs = sorted({r["env_id"] for r in rows})
    available_strats = [s for s in STRATEGY_ORDER
                        if any(r["strategy"] == s for r in rows)]
    print(f"\n{'Strategy':<20} {'Env':<25} {'Seeds':>5} {'Steps-to-70%':>14} {'Final':>7}")
    print("-" * 75)
    for strat in available_strats:
        for env in envs:
            these = [r for r in rows if r["strategy"] == strat and r["env_id"] == env]
            if not these:
                continue
            steps = [r["steps_to_70"] for r in these]
            n_reach = sum(1 for s in steps if s is not None)
            finals = [r["final_success"] for r in these]
            steps_str = f"{int(np.mean([s for s in steps if s is not None]) / 1000):>6}k" \
                        if n_reach > 0 else "   never"
            print(f"{strat:<20} {ENV_SHORT.get(env, env):<25} {len(these):>5} "
                  f"{steps_str:>14} {np.mean(finals):>7.3f}")


if __name__ == "__main__":
    rows = load_transfer_data()
    if not rows:
        print("No transfer result files found. Run transfer.py first.")
    else:
        print(f"Loaded {len(rows)} transfer records from {len({r['strategy'] for r in rows})} strategies.")
        print_summary(rows)
        plot_transfer_bars(rows)
