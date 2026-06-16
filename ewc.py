"""
Elastic Weight Consolidation (EWC) for SB3 PPO.

On each curriculum stage transition, EWC:
  1. Estimates the diagonal Fisher information F_i for all policy params.
  2. Stores the current params as anchor θ*.
  3. Registers gradient hooks that add  λ · F_i · (θ_i - θ*_i)  to each
     gradient, equivalent to penalising   λ/2 · Σ F_i (θ_i - θ*_i)².

Usage:
  python train.py --strategy progressive_ewc --seed 0
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class EWCCallback(BaseCallback):

    def __init__(self, ewc_lambda: float = 5_000.0, n_fisher: int = 400, verbose: int = 0):
        super().__init__(verbose)
        self.ewc_lambda = ewc_lambda
        self.n_fisher = n_fisher
        self._prev_stage = 0
        self._anchors: list[dict] = []   # [{params_star, fisher}]
        self._hooks: list = []

    # ── stage detection ────────────────────────────────────────────────────────

    def _current_stage(self) -> int:
        try:
            return int(self.training_env.envs[0].unwrapped.stage_idx)
        except Exception:
            return 0

    # ── Fisher estimation ──────────────────────────────────────────────────────

    def _compute_fisher(self) -> dict[str, torch.Tensor]:
        policy = self.model.policy
        dev = next(policy.parameters()).device

        buf = self.model.rollout_buffer
        obs_np = buf.observations
        obs_np = obs_np.reshape(-1, *obs_np.shape[2:])  # works for flat or image obs
        n = min(self.n_fisher, len(obs_np))
        idx = np.random.choice(len(obs_np), n, replace=False)
        sample = torch.as_tensor(obs_np[idx], dtype=torch.float32, device=dev)

        fisher = {nm: torch.zeros_like(p) for nm, p in policy.named_parameters() if p.requires_grad}

        policy.set_training_mode(True)
        with torch.enable_grad():
            for i in range(n):
                policy.zero_grad()
                ob = sample[i : i + 1]
                dist = policy.get_distribution(ob)
                act = dist.distribution.sample()
                log_p = dist.distribution.log_prob(act)
                (-log_p.sum()).backward()
                for nm, p in policy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[nm] += p.grad.detach() ** 2 / n
        policy.zero_grad()
        return fisher

    # ── anchor management ──────────────────────────────────────────────────────

    def _store_anchor(self):
        policy = self.model.policy
        params_star = {nm: p.detach().clone() for nm, p in policy.named_parameters() if p.requires_grad}
        fisher      = self._compute_fisher()
        self._anchors.append({"params_star": params_star, "fisher": fisher})
        self._register_hooks()
        print(f"  [EWC] anchor #{len(self._anchors)} stored (stage {self._prev_stage}→{self._prev_stage+1})")

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        anchors = self._anchors
        lam = self.ewc_lambda

        for nm, p in self.model.policy.named_parameters():
            if not p.requires_grad:
                continue
            # Capture per-parameter EWC contribution across all anchors
            ewc_f = sum(a["fisher"][nm] for a in anchors)
            ewc_p = sum(a["fisher"][nm] * a["params_star"][nm] for a in anchors)

            def make_hook(f, s):
                def hook(grad):
                    return grad + lam * (f * s.param - s.star)
                return hook

            class _State:
                pass
            state = _State()
            state.param = p
            state.star   = ewc_p / (ewc_f + 1e-8)  # weighted mean anchor
            state.f      = ewc_f

            h = p.register_hook(lambda grad, f=ewc_f, star=state.star, param=p:
                                 grad + lam * f * (param - star))
            self._hooks.append(h)

    # ── SB3 hooks ──────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        stage = self._current_stage()
        if stage > self._prev_stage:
            self._store_anchor()
            self._prev_stage = stage
        return True

    def _on_training_end(self):
        for h in self._hooks:
            h.remove()
