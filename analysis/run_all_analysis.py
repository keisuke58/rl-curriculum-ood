"""
Master script: run all analysis and produce all figures.

Usage:
  python analysis/run_all_analysis.py
"""
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ANALYSIS_DIR = Path(__file__).parent

scripts = [
    "plot_learning_curves.py",
    "plot_stage_transitions.py",
    "plot_stage_heatmap.py",
    "plot_ood_results.py",
    "plot_ood_gap.py",
    "plot_forgetting.py",
    "stats_test.py",
    "plot_ablation.py",
    "plot_transfer.py",
    "plot_convergence.py",   # extension: RND-signal convergence detection
]

for script in scripts:
    path = ANALYSIS_DIR / script
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print("="*60)
    result = subprocess.run([PYTHON, str(path)], capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR in {script} (exit {result.returncode})")
