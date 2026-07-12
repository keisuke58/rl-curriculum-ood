"""
Train PPO on a curriculum strategy and log results to CSV.

Usage:
  python train.py --strategy progressive --seed 0
  python train.py --strategy all --seed 0
  python train.py --strategy progressive --seeds 0 1 2 3 4 --rnd     # RND
  python train.py --strategy progressive --seeds 0 1 2 3 4 --icm     # ICM
  python train.py --strategy self_paced --seed 0                      # self-paced
  python train.py --strategy progressive --ewc --seed 0               # EWC (exp A)
  python train.py --strategy progressive --image --seed 0             # image obs (exp B)
  python train.py --strategy progressive_replay --seed 0              # interleaved replay
  python train.py --strategy mixed --diverse --seed 0                 # per-tier env pools
  python train.py --strategy mixed --obs-noise --seed 0               # obs-noise augmentation
"""
import argparse
import csv
import time
from pathlib import Path

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from curriculum import CurriculumEnv, ENVS, SEQUENCES
from envs.wrappers import ObsNoiseWrapper
from rnd import RNDModule, RNDEnvWrapper
from icm import ICMModule, ICMEnvWrapper
from ewc import EWCCallback

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

ALL_STRATEGIES = list(SEQUENCES.keys())


class LogCallback(BaseCallback):
    """Logs per-episode stats to CSV every eval_freq steps."""

    def __init__(self, csv_path: Path, eval_freq: int, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self._last_log = 0
        self._rows = []

    def _find_rnd(self, env):
        """Walk the gym.Wrapper chain to find an RNDEnvWrapper, if any."""
        while env is not None:
            if hasattr(env, "rnd"):
                return env.rnd
            env = getattr(env, "env", None)
        return None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.eval_freq:
            self._last_log = self.num_timesteps

            monitor = self.training_env.envs[0]
            # .unwrapped traverses all gym.Wrapper layers to reach CurriculumEnv
            curriculum_env = monitor.unwrapped

            key   = getattr(curriculum_env, "_current_key", "?")
            stage = getattr(curriculum_env, "stage_idx", 0)

            recent = monitor.episode_returns[-20:]
            mean_r = float(np.mean(recent)) if recent else 0.0
            std_r  = float(np.std(recent))  if recent else 0.0

            # Convergence-detection signal: RND predictor loss (EMA) if RND is active,
            # else NaN. Logged as an intrinsic proxy for "has learning stopped".
            rnd = self._find_rnd(monitor)
            rnd_loss = float(rnd.ema_loss) if (rnd is not None and rnd.ema_loss is not None) else float("nan")

            self._rows.append([self.num_timesteps, mean_r, std_r, key, stage, rnd_loss])
            print(f"  step={self.num_timesteps:>8d}  mean_r={mean_r:.3f}  env={key}  stage={stage}"
                  + (f"  rnd={rnd_loss:.4f}" if rnd_loss == rnd_loss else ""))
        return True

    def _on_training_end(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_reward", "std_reward", "env_key", "stage", "rnd_loss"])
            writer.writerows(self._rows)


def make_env(strategy: str, seed: int,
             use_rnd: bool = False, use_icm: bool = False,
             use_image: bool = False, use_diverse: bool = False,
             use_obs_noise: bool = False):
    env = CurriculumEnv(
        strategy=strategy,
        seed=seed,
        success_threshold=CFG["training"]["success_threshold"],
        window=CFG["training"]["window"],
        max_steps=CFG["training"]["max_steps"],
        progress_threshold=CFG["training"].get("progress_threshold", 0.02),
        use_image=use_image,
        replay_prob=CFG["training"].get("replay_prob", 0.2),
        env_pool=use_diverse,
    )
    if use_obs_noise and not use_image:
        env = ObsNoiseWrapper(env, noise_std=CFG["training"].get("obs_noise_std", 0.05),
                              seed=seed)
    if use_rnd:
        obs_dim = env.observation_space.shape[0]
        rnd = RNDModule(obs_dim=obs_dim, lr=CFG.get("rnd", {}).get("lr", 1e-4))
        env = RNDEnvWrapper(env, rnd, coef=CFG.get("rnd", {}).get("coef", 0.1))
    elif use_icm:
        obs_dim    = env.observation_space.shape[0]
        n_actions  = env.action_space.n
        icm = ICMModule(obs_dim=obs_dim, n_actions=n_actions,
                        lr=CFG.get("icm", {}).get("lr", 1e-4),
                        beta=CFG.get("icm", {}).get("beta", 0.2))
        env = ICMEnvWrapper(env, icm, coef=CFG.get("icm", {}).get("coef", 0.1))
    env = Monitor(env)  # outermost so LogCallback reads correct episode stats
    return env


def train(strategy: str, seed: int,
          use_rnd: bool = False, use_icm: bool = False,
          use_ewc: bool = False, use_image: bool = False,
          use_diverse: bool = False, use_obs_noise: bool = False):
    suffix = "_rnd" if use_rnd else ("_icm" if use_icm else
             ("_ewc" if use_ewc else ("_image" if use_image else "")))
    if use_diverse:
        suffix += "_div"
    if use_obs_noise:
        suffix += "_noise"
    tag = f"{strategy}{suffix}"

    print(f"\n{'='*50}")
    print(f"  Strategy: {tag}  |  Seed: {seed}")
    print(f"{'='*50}")

    csv_path = RESULTS_DIR / f"{tag}_seed{seed}.csv"
    if csv_path.exists():
        print(f"  Already exists, skipping: {csv_path.name}")
        return

    env = make_env(strategy, seed, use_rnd=use_rnd, use_icm=use_icm, use_image=use_image,
                   use_diverse=use_diverse, use_obs_noise=use_obs_noise)
    ppo_cfg = CFG["ppo"].copy()
    policy  = ppo_cfg.pop("policy")
    if use_image:
        policy = "CnnPolicy"
    model   = PPO(policy, env, seed=seed, verbose=0, **ppo_cfg)

    log_cb = LogCallback(csv_path=csv_path, eval_freq=CFG["eval"]["eval_freq"])
    if use_ewc:
        ewc_lam = CFG.get("ewc", {}).get("lambda", 5_000.0)
        ewc_n   = CFG.get("ewc", {}).get("n_fisher", 400)
        callback = CallbackList([log_cb, EWCCallback(ewc_lambda=ewc_lam, n_fisher=ewc_n)])
    else:
        callback = log_cb

    t0 = time.time()
    model.learn(total_timesteps=CFG["training"]["total_timesteps"], callback=callback)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s → {csv_path.name}")

    model.save(str(RESULTS_DIR / f"{tag}_seed{seed}.zip"))
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="progressive",
                        choices=ALL_STRATEGIES + ["all"])
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--rnd",   action="store_true", help="Add RND intrinsic reward")
    parser.add_argument("--icm",   action="store_true", help="Add ICM intrinsic reward")
    parser.add_argument("--ewc",   action="store_true", help="EWC regularisation (exp A)")
    parser.add_argument("--image", action="store_true", help="Raw image obs + CnnPolicy (exp B)")
    parser.add_argument("--diverse",   action="store_true",
                        help="Sample per-tier env variants from ENV_POOLS (diversity boost)")
    parser.add_argument("--obs-noise", action="store_true",
                        help="Gaussian noise on flat observations (regularizer)")
    args = parser.parse_args()

    if args.rnd and args.icm:
        raise ValueError("Cannot use --rnd and --icm simultaneously")
    if args.ewc and args.image:
        raise ValueError("Cannot use --ewc and --image simultaneously")
    if args.obs_noise and args.image:
        raise ValueError("--obs-noise only applies to flat observations, not --image")

    strategies = ALL_STRATEGIES if args.strategy == "all" else [args.strategy]
    seeds      = args.seeds if args.seeds else [args.seed]

    for strategy in strategies:
        for seed in seeds:
            train(strategy, seed,
                  use_rnd=args.rnd, use_icm=args.icm,
                  use_ewc=args.ewc, use_image=args.image,
                  use_diverse=args.diverse, use_obs_noise=args.obs_noise)


if __name__ == "__main__":
    main()
