# Curriculum Ordering, Catastrophic Forgetting, and OOD Generalization in PPO Agents

RL Course Project — SoSe2026, Leibniz Universität Hannover  
Keisuke Nishioka

## Research Questions

1. Does progressive (easy→hard) curriculum outperform alternatives in-distribution, and does it induce catastrophic forgetting?
2. Do curriculum-trained agents generalize zero-shot to structurally related but unseen environments?
3. Is there a performance–generalization trade-off across curriculum conditions?

## Key Findings

- **Progressive** achieves highest Hard success (0.97±0.02) but catastrophically forgets Easy/Medium (0.00/0.02)
- **OOD transfer is near-zero** for all strategies — Reverse's higher point estimate on DistShift1 is a high-variance *endpoint-alignment* artefact; the best *reliable* transfer comes from the multi-task baselines (Mixed/Random, 0.059±0.030), confirming **diversity beats ordering**
- The trade-off is between Hard-only (max in-distribution specialization, zero OOD) and Mixed/Random (broad competence, modest but reliable OOD) — Progressive achieves neither

> Additional exploratory experiments (RND/ICM intrinsic motivation in `rnd.py`/`icm.py`, EWC in `ewc.py`) were run but excluded from the report to keep scope focused; RND did not improve OOD transfer in our setting.

## Improvement Measures (向上施策)

Three follow-up measures targeting the findings above are implemented — see
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for design rationale and the full experiment matrix:

1. **Interleaved replay curriculum** (`--strategy progressive_replay`) — progressive staging with
   probabilistic rehearsal of earlier stages, targeting catastrophic forgetting. Replay episodes
   are excluded from the stage-advancement window.
2. **Per-tier environment pools** (`--diverse`) — each difficulty tier samples from multiple
   structurally distinct envs (held-out test/transfer families excluded), directly increasing the
   training diversity that the results identify as the driver of OOD transfer.
3. **Observation-noise regularization** (`--obs-noise`) — Gaussian noise on flat observations
   during training to reduce overfitting to training-env-specific observation patterns.

```bash
python train.py --strategy progressive_replay --seeds 0 1 2 3 4 5 6 7 8 9
python train.py --strategy mixed --diverse --seeds 0 1 2 3 4 5 6 7 8 9
python evaluate.py --strategy progressive_replay --seed 0   # variants: mixed_div, mixed_noise, ...
```

## Setup

```bash
pip install stable-baselines3 minigrid gymnasium numpy pandas matplotlib scipy pyyaml
```

## Structure

```
train.py              # Main training script (PPO + curriculum strategies)
evaluate.py           # Zero-shot OOD evaluation
transfer.py           # Fine-tuning on transfer environments
curriculum.py         # Curriculum strategy implementations
rnd.py                # RND intrinsic motivation module
icm.py                # ICM intrinsic motivation module
ablation_threshold.py # Threshold ablation for Progressive strategy
configs/
  default.yaml        # All hyperparameters (seeds, envs, thresholds)
analysis/
  plot_learning_curves.py
  plot_forgetting.py
  plot_ood_results.py
  plot_ood_gap.py
  plot_ablation.py
  stats_test.py
  run_all_analysis.py  # Run all analysis scripts at once
results/              # Saved model weights + evaluation JSONs (not tracked by git)
figures/              # Generated figures (PDF + PNG)
```

## Reproducing Results

### Training (5 strategies × 10 seeds = 50 runs)
```bash
# Single strategy
python train.py --strategy progressive --seed 0

# All strategies, all seeds (use --seeds flag for multiple)
for strategy in progressive reverse random hard_only mixed; do
  python train.py --strategy $strategy --seeds 0 1 2 3 4 5 6 7 8 9
done
```

### OOD Evaluation
```bash
python evaluate.py --all --n-eval 50
```

### Forgetting Analysis
```bash
python analysis/plot_forgetting.py --n-eval 20
```

### Transfer Learning
```bash
python transfer.py --all
```

### Threshold Ablation
```bash
python ablation_threshold.py --thresholds 0.5 0.6 0.7 0.8 0.9 --seeds 0 1 2 3 4
```

### Regenerate All Figures
```bash
python analysis/run_all_analysis.py
```

## Configuration

All hyperparameters are in `configs/default.yaml`:
- `seeds`: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- `training.max_steps`: 500
- `training.total_timesteps`: 1_000_000
- `training.advancement_threshold`: 0.7
- `n_eval_episodes`: 50
- `transfer_finetune_steps`: 200_000

## Acknowledgements

Uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO
and [MiniGrid](https://github.com/Farama-Foundation/Minigrid) for environments.
RND implementation follows [Burda et al. 2019](https://arxiv.org/abs/1810.12894).
