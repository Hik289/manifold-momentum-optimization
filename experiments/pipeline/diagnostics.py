"""Record training metrics and optimizer diagnostics at evaluation points.

DiagnosticRecorder collects these values when they are available:
  1  train_loss                 (passed to record)
  2  val_metric                 (passed to record)
  3  final_test_metric          (passed at the end of a run)
  4  learning_rate              (passed to record)
  5  raw_grad_norm              (optimizer.last_diag)
  6  first_moment_norm          (optimizer.last_diag)
  7  precond_update_norm        (optimizer.last_diag)
  8  norm_matching_scale s_t    (optimizer.last_diag)
  9  lambda_min(L), lambda_max(L)
  10 lambda_min(R), lambda_max(R)
  11 kappa(L), kappa(R)
  12 tr(L)/m, tr(R)/n
  13 logdet(L), logdet(R)
  14 d_BW(L_t, L_{t-1})
  15 d_BW(R_t, R_{t-1})
  16 ||L_t - L_{t-1}||_F
  17 ||R_t - R_{t-1}||_F
  18 root_cache_error_proxy
  19 clipping_activation_rate
  20 wall_clock
  21 peak_memory
  22 checkpoint_metadata        (passed to record)
  23 nan_inf / eig_fail / state_reset events

build_optimizer(name, params, **hp) provides one entry point for the five
baseline and BuresMomentum optimizers.
"""

import time
import json
import torch

from bures.core import bw_distance, symmetrize, _robust_eigvalsh
from bures.optimizer import (
    MatrixOptimizer, AdamWRef, SGDMRef, ShampooRef,
)


# ---------------------------------------------------------------------------
# Shared optimizer interface
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY = {
    "sgdm": "B1", "adamw": "B2", "euclidean_kron": "B3",
    "shampoo": "B4", "buresmomentum": "B5",
}


def build_optimizer(name, params, lr=1e-3, weight_decay=0.0, **kw):
    """Build an optimizer by its registry name.

    The Euclidean and Bures variants share settings except covariance_rule.
    """
    name = name.lower()
    if name == "sgdm":
        return SGDMRef(params, lr=lr, momentum=kw.get("momentum", 0.9),
                       weight_decay=weight_decay)
    if name == "adamw":
        return AdamWRef(params, lr=lr, betas=kw.get("betas", (0.9, 0.999)),
                        eps=kw.get("eps", 1e-8), weight_decay=weight_decay)
    if name == "shampoo":
        return ShampooRef(params, lr=lr, momentum=kw.get("momentum", 0.9),
                          weight_decay=weight_decay,
                          update_freq=kw.get("tau_root", 1),
                          block_size=kw.get("block_size", 256))
    if name in ("euclidean_kron", "buresmomentum", "scale_only"):
        rule = {"euclidean_kron": "euclidean", "buresmomentum": "bures",
                "scale_only": "scale_only"}[name]
        # These variants share settings except for the averaging rule.
        return MatrixOptimizer(
            params, lr=lr, betas=kw.get("betas", (0.9, 0.99)),
            eps_stat=kw.get("eps_stat", 1e-6),
            eps_norm=kw.get("eps_norm", 1e-12),
            weight_decay=weight_decay,
            lam_min=kw.get("lam_min", 1e-6), lam_max=kw.get("lam_max", 1e6),
            s_min=kw.get("s_min", 0.1), s_max=kw.get("s_max", 10.0),
            tau_root=kw.get("tau_root", 10),
            block_size=kw.get("block_size", 256),
            covariance_rule=rule)
    raise ValueError(f"unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Diagnostic recorder
# ---------------------------------------------------------------------------

class DiagnosticRecorder:
    """Store a diagnostic record at each evaluation point."""

    def __init__(self, optimizer, track_spd=True):
        self.opt = optimizer
        self.track_spd = track_spd
        self.trajectory = []
        self.prev_states = {}  # param_id -> {"L":{blk:mat}, "R":{blk:mat}}
        self.t0 = time.time()
        self.event_log = []  # nan/inf/eig_fail/state_reset
        # Use the optimizer's clipping bounds in condition-number diagnostics.
        self._state_lam_min = 1e-6
        self._state_lam_max = 1e6
        try:
            g0 = optimizer.param_groups[0]
            self._state_lam_min = float(g0.get("lam_min", 1e-6))
            self._state_lam_max = float(g0.get("lam_max", 1e6))
        except Exception:
            pass

    def _spd_diag(self):
        """Summarize SPD states across all parameters and matrix blocks.

        Return mean values and, where useful, maxima.
        """
        agg = {
            "lambda_min_L": [], "lambda_max_L": [],
            "lambda_min_R": [], "lambda_max_R": [],
            "kappa_L": [], "kappa_R": [],
            "trace_L_over_m": [], "trace_R_over_n": [],
            "logdet_L": [], "logdet_R": [],
            "bw_drift_L": [], "bw_drift_R": [],
            "fro_drift_L": [], "fro_drift_R": [],
        }
        if not isinstance(self.opt, MatrixOptimizer):
            return {k: float("nan") for k in _spd_keys()}

        for gi, group in enumerate(self.opt.param_groups):
            for pi, p in enumerate(group["params"]):
                st = self.opt.state.get(p, {})
                if "L" not in st:
                    continue
                pid = (gi, pi)
                prev = self.prev_states.get(pid, {"L": {}, "R": {}})
                for side, key_lm, key_lM, key_k, key_tr, key_ld, \
                        key_bw, key_fr, norm_dim in [
                        ("L", "lambda_min_L", "lambda_max_L", "kappa_L",
                         "trace_L_over_m", "logdet_L", "bw_drift_L",
                         "fro_drift_L", 0),
                        ("R", "lambda_min_R", "lambda_max_R", "kappa_R",
                         "trace_R_over_n", "logdet_R", "bw_drift_R",
                         "fro_drift_R", 1)]:
                    for blk, mat in st[side].items():
                        try:
                            ev = _robust_eigvalsh(mat)
                        except Exception as e:
                            self.event_log.append(
                                {"t": self.opt_step(), "event": "eig_fail",
                                 "detail": str(e)})
                            continue
                        # Match the state's clipping bounds. A fresh
                        # eigendecomposition can otherwise create tiny
                        # round-off eigenvalues and inflate the condition
                        # number far beyond lam_max / lam_min.
                        ev = ev.clamp(min=self._state_lam_min,
                                      max=self._state_lam_max)
                        lmin, lmax = ev.min().item(), ev.max().item()
                        dm = mat.shape[-1]
                        agg[key_lm].append(lmin)
                        agg[key_lM].append(lmax)
                        agg[key_k].append(lmax / lmin)
                        agg[key_tr].append(ev.sum().item() / dm)
                        agg[key_ld].append(torch.log(ev).sum().item())
                        pm = prev[side].get(blk)
                        if pm is not None:
                            agg[key_bw].append(
                                bw_distance(pm, mat).item())
                            agg[key_fr].append((mat - pm).norm().item())
                    prev[side] = {blk: mat.clone()
                                  for blk, mat in st[side].items()}
                self.prev_states[pid] = prev

        out = {}
        for k, vals in agg.items():
            out[k] = (sum(vals) / len(vals)) if vals else float("nan")
            if vals and ("kappa" in k or "lambda_max" in k):
                out[k + "_max"] = max(vals)
        return out

    def opt_step(self):
        for group in self.opt.param_groups:
            for p in group["params"]:
                st = self.opt.state.get(p, {})
                if "step" in st:
                    return st["step"]
        return -1

    def record(self, step, train_loss=None, val_metric=None,
               learning_rate=None, checkpoint_meta=None,
               peak_memory_gb=None):
        d = self.opt.last_diag if hasattr(self.opt, "last_diag") else {}
        entry = {
            "step": step,
            "train_loss": train_loss,
            "val_metric": val_metric,
            "learning_rate": learning_rate,
            "raw_grad_norm": d.get("raw_grad_norm"),
            "first_moment_norm": d.get("first_moment_norm"),
            "precond_update_norm": d.get("precond_update_norm"),
            "scale_mean": d.get("scale_mean"),
            "clip_events": d.get("clip_events", 0),
            "nan_inf_events": d.get("nan_inf_events", 0),
            "wall_clock": time.time() - self.t0,
            "peak_memory_gb": peak_memory_gb,
            "checkpoint_meta": checkpoint_meta,
        }
        if self.track_spd:
            entry.update(self._spd_diag())
        # Store the clipping event count under the existing diagnostic key.
        entry["clip_activation_rate"] = (
            d.get("clip_events", 0))
        self.trajectory.append(entry)
        return entry

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"trajectory": self.trajectory,
                       "event_log": self.event_log}, f, indent=2,
                      default=str)


def _spd_keys():
    return ["lambda_min_L", "lambda_max_L", "lambda_min_R", "lambda_max_R",
            "kappa_L", "kappa_R", "trace_L_over_m", "trace_R_over_n",
            "logdet_L", "logdet_R", "bw_drift_L", "bw_drift_R",
            "fro_drift_L", "fro_drift_R"]
