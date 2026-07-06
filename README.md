# Curriculum Ordering, Catastrophic Forgetting, and OOD Generalization in PPO Agents

RL Course Project — SoSe2026, Leibniz Universität Hannover  
Keisuke Nishioka

PPO agents are trained on a three-tier MiniGrid curriculum (Easy → Medium → Hard) under
different ordering strategies, then evaluated zero-shot on five held-out environments to
measure out-of-distribution (OOD) generalization and catastrophic forgetting.

**Deliverables:** [report.pdf](report.pdf) · [poster.pdf](poster.pdf) · [proposal.pdf](proposal.pdf)

## Research Questions

1. Does progressive (easy→hard) curriculum outperform alternatives in-distribution, and does it induce catastrophic forgetting?
2. Do curriculum-trained agents generalize zero-shot to structurally related but unseen environments?
3. Is there a performance–generalization trade-off across curriculum conditions?

## Experimental Setup

| Role | Environments |
|---|---|
| Training — Easy | `MiniGrid-Empty-8x8-v0` |
| Training — Medium | `MiniGrid-FourRooms-v0` |
| Training — Hard | `MiniGrid-KeyCorridorS3R1-v0` |
| OOD test (zero-shot) | `DoorKey-5x5`, `MultiRoom-N2-S4`, `LavaCrossingS9N1`, `SimpleCrossingS9N1`, `DistShift1` |
| Transfer (fine-tune) | `DoorKey-8x8`, `MultiRoom-N4-S5` |

**Curriculum strategies:** `progressive` (easy→hard on success threshold), `reverse`,
`random` (staged unlock + uniform sampling), `hard_only` (no curriculum), `mixed`
(uniform multi-task baseline), `self_paced` (advance on learning plateau), and
`progressive_replay` (see [Improvement Measures](#improvement-measures-向上施策)).

Each condition runs PPO (Stable-Baselines3, `MlpPolicy` on flattened observations) for
1M timesteps across 10 seeds.

## Key Findings

- **Progressive** achieves highest Hard success (0.97±0.02) but catastrophically forgets Easy/Medium (0.00/0.02)
- **OOD transfer is near-zero** for all strategies — Reverse's higher point estimate on DistShift1 is a high-variance *endpoint-alignment* artefact; the best *reliable* transfer comes from the multi-task baselines (Mixed/Random, 0.059±0.030), confirming **diversity beats ordering**
- The trade-off is between Hard-only (max in-distribution specialization, zero OOD) and Mixed/Random (broad competence, modest but reliable OOD) — Progressive achieves neither

> Additional exploratory experiments (RND/ICM intrinsic motivation in `rnd.py`/`icm.py`, EWC in `ewc.py`) were run but excluded from the report to keep scope focused; RND did not improve OOD transfer in our setting.

## Improvement Measures (向上施策)

Three follow-up measures targeting the findings above are implemented — see
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for design rationale and the full experiment matrix:

1. **Interleaved replay curriculum** (`--strategy progressive_replay`) — progressive staging with
   probabilistic rehearsal of earlier stages (`replay_prob: 0.2`), targeting catastrophic
   forgetting. Replay episodes are excluded from the stage-advancement window so easy successes
   cannot trigger premature advancement.
2. **Per-tier environment pools** (`--diverse`) — each difficulty tier samples from multiple
   structurally distinct envs (held-out test/transfer families excluded), directly increasing the
   training diversity that the results identify as the driver of OOD transfer.
3. **Observation-noise regularization** (`--obs-noise`) — Gaussian noise on flat observations
   during training (`obs_noise_std: 0.05`) to reduce overfitting to training-env-specific
   observation patterns. Evaluation observations stay clean.

```bash
python train.py --strategy progressive_replay --seeds 0 1 2 3 4 5 6 7 8 9
python train.py --strategy mixed --diverse --seeds 0 1 2 3 4 5 6 7 8 9
python train.py --strategy mixed --obs-noise --seeds 0 1 2 3 4 5 6 7 8 9
python evaluate.py --strategy progressive_replay --seed 0   # variants: mixed_div, mixed_noise, ...
```

Models trained with `--diverse` / `--obs-noise` are tagged with `_div` / `_noise` suffixes
(e.g. `mixed_div_seed0.zip`) and are recognized by `evaluate.py` via `variant_strategies`
in the config.

## Setup

Requires Python 3.10+.

```bash
pip install stable-baselines3 minigrid gymnasium numpy pandas matplotlib scipy pyyaml
```

## Structure

```
train.py              # Main training script (PPO + curriculum strategies + variants)
evaluate.py           # Zero-shot OOD evaluation on held-out test envs
transfer.py           # Fine-tuning on transfer environments
curriculum.py         # Curriculum strategies, ENV tiers, per-tier env pools
ablation_threshold.py # Advancement-threshold ablation for Progressive
rnd.py                # RND intrinsic motivation module (exploratory)
icm.py                # ICM intrinsic motivation module (exploratory)
ewc.py                # Elastic Weight Consolidation callback (exploratory)
envs/
  wrappers.py         # Observation wrappers (flat obs, ObsNoiseWrapper)
configs/
  default.yaml        # All hyperparameters (seeds, envs, thresholds, variants)
analysis/
  plot_learning_curves.py
  plot_forgetting.py
  plot_ood_results.py
  plot_ood_gap.py
  plot_transfer.py
  plot_ablation.py
  plot_stage_transitions.py
  plot_stage_heatmap.py
  stats_test.py        # Significance tests (Kruskal-Wallis + post-hoc Mann-Whitney U)
  run_all_analysis.py  # Run all analysis scripts at once
  style.py             # Shared matplotlib style
docs/
  IMPROVEMENTS.md      # Improvement measures: rationale + experiment matrix
results/              # Saved model weights + evaluation JSONs (not tracked by git)
figures/              # Generated figures (PDF + PNG)
```

## Reproducing Results

### Training (main study: 5 strategies × 10 seeds = 50 runs)
```bash
# Single strategy
python train.py --strategy progressive --seed 0

# All main-study strategies, all seeds
for strategy in progressive reverse random hard_only mixed; do
  python train.py --strategy $strategy --seeds 0 1 2 3 4 5 6 7 8 9
done

# Additional conditions (not part of the main report)
python train.py --strategy self_paced --seeds 0 1 2 3 4 5 6 7 8 9
python train.py --strategy progressive_replay --seeds 0 1 2 3 4 5 6 7 8 9
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

| Key | Default | Meaning |
|---|---|---|
| `seeds` | `[0..9]` | Random seeds per condition |
| `training.total_timesteps` | `1_000_000` | PPO training budget |
| `training.max_steps` | `500` | Episode step limit |
| `training.success_threshold` | `0.7` | Success rate to advance a curriculum stage |
| `training.window` | `20` | Episodes in the advancement window |
| `training.replay_prob` | `0.2` | `progressive_replay`: prob. of revisiting an earlier stage |
| `training.obs_noise_std` | `0.05` | `--obs-noise`: Gaussian std on flat observations |
| `eval.n_eval_episodes` | `50` | Episodes per test env in OOD evaluation |
| `transfer_finetune_steps` | `200_000` | Fine-tuning budget per transfer env |

## Acknowledgements

Uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO
and [MiniGrid](https://github.com/Farama-Foundation/Minigrid) for environments.
RND implementation follows [Burda et al. 2019](https://arxiv.org/abs/1810.12894).
