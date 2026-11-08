"""Matrix optimizers and comparison methods.

BuresMomentum and its paired Euclidean baseline share the same gradient,
momentum, preconditioner, norm matching, clipping, and parameter update. The
`covariance_rule` setting changes only how each covariance state is averaged
with a new gradient statistic. Non-matrix parameters use AdamW. Large
matrices are split into independent tiles, and matrix states use at least
float32.
"""

import math
import torch
from torch.optim import Optimizer

from .core import (
    symmetrize, spd_inv_sqrt, bures_barycenter_2pt, clip_spd,
    bw_distance, scale_only_average,
)


def _split_blocks(d, block):
    """Split a dimension into consecutive blocks of at most `block` entries."""
    if d <= block:
        return [(0, d)]
    bounds = []
    start = 0
    while start < d:
        end = min(start + block, d)
        bounds.append((start, end))
        start = end
    return bounds


class MatrixOptimizer(Optimizer):
    """Precondition matrix parameters using left and right covariance states.

    Set `covariance_rule` to "bures" or "euclidean" for the paired comparison.
    Only two-dimensional parameters use matrix states; other parameters use
    AdamW. Each matrix tile has its own states and cached roots. Checkpoints
    made with the older shared-state layout cannot be resumed here.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99),
                 eps_stat=1e-6, eps_norm=1e-12, weight_decay=0.0,
                 lam_min=1e-6, lam_max=1e6, s_min=0.1, s_max=10.0,
                 tau_root=10, block_size=256, covariance_rule="bures",
                 adamw_eps=1e-8, state_dtype=torch.float32, ablation="full"):
        if covariance_rule not in ("bures", "euclidean", "scale_only", "ai", "log_euclidean"):
            raise ValueError(
                "covariance_rule must be one of: bures, euclidean, "
                "scale_only, ai, log_euclidean")
        defaults = dict(
            lr=lr, betas=betas, eps_stat=eps_stat, eps_norm=eps_norm,
            weight_decay=weight_decay, lam_min=lam_min, lam_max=lam_max,
            s_min=s_min, s_max=s_max, tau_root=tau_root, block_size=block_size,
            covariance_rule=covariance_rule, adamw_eps=adamw_eps,
            state_dtype=state_dtype, ablation=ablation,
        )
        super().__init__(params, defaults)
        # Diagnostics from the latest optimizer step.
        self.last_diag = {}

    # Covariance update: the only difference in the paired comparison.
    @staticmethod
    def _update_cov(rule, prev, S, alpha, lam_min, lam_max):
        """Average one covariance state with a new statistic, then clip it.

        "bures" uses the BW average; "euclidean" uses an arithmetic moving
        average. "scale_only", "ai", and "log_euclidean" are ablation rules.
        Return the new state and whether clipping changed an eigenvalue.
        """
        if rule in ("ai", "log_euclidean"):
            from .core import spd_power
            a, q = torch.linalg.eigh(prev)
            if rule == "ai":
                ah=spd_power(prev,.5); aih=spd_power(prev,-.5)
                new=ah @ spd_power(aih @ S @ aih,alpha) @ ah
            else:
                b, u=torch.linalg.eigh(S)
                la=(q*a.clamp(min=1e-30).log().unsqueeze(0))@q.T
                lb=(u*b.clamp(min=1e-30).log().unsqueeze(0))@u.T
                c,v=torch.linalg.eigh((1-alpha)*la+alpha*lb)
                new=(v*c.exp().unsqueeze(0))@v.T
        elif rule == "bures":
            new = bures_barycenter_2pt(prev, S, alpha)
        elif rule == "scale_only":
            new = scale_only_average(prev, S, alpha)
        else:
            beta2 = 1.0 - alpha
            new = symmetrize(beta2 * prev + (1.0 - beta2) * S)
        new, trig = clip_spd(new, lam_min, lam_max)
        return new, trig

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        diag_accum = {
            "raw_grad_norm": 0.0, "first_moment_norm": 0.0,
            "precond_update_norm": 0.0, "scale_sum": 0.0, "scale_count": 0,
            "clip_events": 0, "nan_inf_events": 0,
        }

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            alpha = 1.0 - beta2
            eps_stat = group["eps_stat"]
            eps_norm = group["eps_norm"]
            wd = group["weight_decay"]
            lam_min, lam_max = group["lam_min"], group["lam_max"]
            s_min, s_max = group["s_min"], group["s_max"]
            tau_root = group["tau_root"]
            block = group["block_size"]
            rule = group["covariance_rule"]
            abl = group["ablation"]
            if abl == "no_clip":
                lam_min, lam_max = 1e-20, float("inf")
            adamw_eps = group["adamw_eps"]
            sdtype = group["state_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    diag_accum["nan_inf_events"] += 1
                    continue
                diag_accum["raw_grad_norm"] += float(g.norm().item()) ** 2

                st = self.state[p]

                # Use AdamW for parameters that are not matrices.
                if g.dim() != 2:
                    self._adamw_step(p, g, st, lr, beta1, beta2,
                                     adamw_eps, wd)
                    continue

                # Precondition matrix parameters.
                m, n = g.shape
                g32 = g.to(sdtype)
                if len(st) == 0:
                    st["step"] = 0
                    st["M"] = torch.zeros_like(p)
                    st["blocks_m"] = _split_blocks(m, block)
                    st["blocks_n"] = _split_blocks(n, block)
                    st["L"] = {}
                    st["R"] = {}
                    st["Lih_cache"] = {}
                    st["Rih_cache"] = {}
                    for (i0, i1) in st["blocks_m"]:
                        for (j0, j1) in st["blocks_n"]:
                            key = (i0, j0)
                            st["L"][key] = torch.eye(i1-i0, dtype=sdtype, device=p.device)
                            st["R"][key] = torch.eye(j1-j0, dtype=sdtype, device=p.device)
                st["step"] += 1
                t = st["step"]

                # Step 1: update and bias-correct the first moment.
                M = st["M"]
                M.mul_(beta1).add_(g, alpha=1.0 - beta1)
                M_hat = (M / (1.0 - beta1 ** t)).to(sdtype)
                diag_accum["first_moment_norm"] += float(M.norm().item()) ** 2

                D_raw = torch.zeros_like(M_hat)
                refresh = (t % tau_root == 0) or (t == 1) or (tau_root == 1)

                # Each rectangular tile has its own covariance states.
                for (i0, i1) in st["blocks_m"]:
                    gi = g32[i0:i1, :]
                    for (j0, j1) in st["blocks_n"]:
                        gij = gi[:, j0:j1]
                        key = (i0, j0)
                        bm, bn = i1 - i0, j1 - j0
                        Im = torch.eye(bm, dtype=sdtype, device=p.device)
                        In = torch.eye(bn, dtype=sdtype, device=p.device)
                        # Step 2: form positive-definite gradient statistics.
                        S_L = symmetrize(gij @ gij.transpose(-1, -2)) / bn \
                            + eps_stat * Im
                        S_R = symmetrize(gij.transpose(-1, -2) @ gij) / bm \
                            + eps_stat * In

                        # Step 3: update the covariance states.
                        L_new, tL = self._update_cov(
                            rule, st["L"][key], S_L, alpha, lam_min, lam_max)
                        R_new, tR = self._update_cov(
                            rule, st["R"][key], S_R, alpha, lam_min, lam_max)
                        st["L"][key] = L_new
                        st["R"][key] = R_new
                        if tL or tR:
                            diag_accum["clip_events"] += 1

                        if abl in ("left_only", "identity"):
                            R_new = In
                        if abl in ("right_only", "identity"):
                            L_new = Im
                        if abl == "diagonal":
                            L_new = torch.diag(torch.diag(L_new))
                            R_new = torch.diag(torch.diag(R_new))
                        if abl in ("rotate", "permute"):
                            for side, val in (("L", L_new), ("R", R_new)):
                                e, q = torch.linalg.eigh(val)
                                gen=torch.Generator(device=val.device)
                                gen.manual_seed(12345+t*131+i0*17+j0*31+(side=="R"))
                                if abl == "rotate":
                                    q,_=torch.linalg.qr(torch.randn(val.shape,device=val.device,dtype=val.dtype,generator=gen))
                                else:
                                    q=q[:,torch.randperm(len(e),device=val.device,generator=gen)]
                                v=(q*e.unsqueeze(0))@q.T
                                if side=="L": L_new=v
                                else: R_new=v
                        st["L"][key], st["R"][key] = L_new, R_new

                        # Refresh inverse square roots every tau_root steps.
                        if refresh or (key not in st["Lih_cache"]):
                            st["Lih_cache"][key] = spd_inv_sqrt(L_new)
                        if refresh or (key not in st["Rih_cache"]):
                            st["Rih_cache"][key] = spd_inv_sqrt(R_new)
                        Lih = st["Lih_cache"][key]
                        Rih = st["Rih_cache"][key]

                        # Step 4: precondition the first moment.
                        Mij = M_hat[i0:i1, j0:j1]
                        D_raw[i0:i1, j0:j1] = Lih @ Mij @ Rih

                # Match the step size across the full parameter matrix.
                num = M_hat.norm()
                den = D_raw.norm() + eps_norm
                scale = torch.clamp(num / den, min=s_min, max=s_max)
                if abl == "no_norm":
                    scale = torch.ones_like(scale)
                D = (scale * D_raw).to(p.dtype)
                diag_accum["precond_update_norm"] += float(D.norm().item()) ** 2
                diag_accum["scale_sum"] += float(scale.item())
                diag_accum["scale_count"] += 1

                # Step 5: apply weight decay and the parameter update.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(D, alpha=-lr)

        # Combine tile diagnostics into parameter-level values.
        for k in ("raw_grad_norm", "first_moment_norm", "precond_update_norm"):
            diag_accum[k] = math.sqrt(diag_accum[k])
        if diag_accum["scale_count"] > 0:
            diag_accum["scale_mean"] = \
                diag_accum["scale_sum"] / diag_accum["scale_count"]
        else:
            diag_accum["scale_mean"] = float("nan")
        self.last_diag = diag_accum
        return loss

    # AdamW update for non-matrix parameters.
    @staticmethod
    def _adamw_step(p, g, st, lr, beta1, beta2, eps, wd):
        if len(st) == 0:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
        st["step"] += 1
        t = st["step"]
        ea, eas = st["exp_avg"], st["exp_avg_sq"]
        ea.mul_(beta1).add_(g, alpha=1.0 - beta1)
        eas.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        denom = (eas.sqrt() / math.sqrt(bc2)).add_(eps)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.addcdiv_(ea, denom, value=-lr / bc1)


class AdamWRef(Optimizer):
    """AdamW baseline implemented with the same non-matrix update path."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                MatrixOptimizer._adamw_step(
                    p, p.grad, self.state[p], group["lr"],
                    group["betas"][0], group["betas"][1],
                    group["eps"], group["weight_decay"])
        return loss


class ShampooRef(Optimizer):
    """Shampoo baseline with cumulative left and right gradient statistics.

    Matrix gradients are scaled by L**(-1/4) and R**(-1/4). Non-matrix
    parameters use AdamW.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, eps=1e-6,
                 weight_decay=0.0, update_freq=1, block_size=256,
                 betas=(0.9, 0.999), adamw_eps=1e-8,
                 state_dtype=torch.float32):
        defaults = dict(lr=lr, momentum=momentum, eps=eps,
                        weight_decay=weight_decay, update_freq=update_freq,
                        block_size=block_size, betas=betas,
                        adamw_eps=adamw_eps, state_dtype=state_dtype)
        super().__init__(params, defaults)
        self.last_diag = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            eps = group["eps"]
            wd = group["weight_decay"]
            uf = group["update_freq"]
            sdtype = group["state_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() != 2:
                    MatrixOptimizer._adamw_step(
                        p, g, self.state[p], lr, group["betas"][0],
                        group["betas"][1], group["adamw_eps"], wd)
                    continue
                m, n = g.shape
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["L"] = eps * torch.eye(m, dtype=sdtype, device=p.device)
                    st["R"] = eps * torch.eye(n, dtype=sdtype, device=p.device)
                    st["buf"] = torch.zeros_like(p)
                    st["Lp"] = None
                    st["Rp"] = None
                st["step"] += 1
                t = st["step"]
                g32 = g.to(sdtype)
                st["L"] = st["L"] + g32 @ g32.transpose(-1, -2)
                st["R"] = st["R"] + g32.transpose(-1, -2) @ g32
                if (t - 1) % uf == 0 or st["Lp"] is None:
                    st["Lp"] = _mat_power(st["L"], -0.25)
                    st["Rp"] = _mat_power(st["R"], -0.25)
                D = (st["Lp"] @ g32 @ st["Rp"]).to(p.dtype)
                buf = st["buf"]
                buf.mul_(mom).add_(D)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(buf, alpha=-lr)
        return loss


def _mat_power(A, p):
    from .core import spd_power
    return spd_power(A, p)


class SGDMRef(Optimizer):
    """SGD with momentum, used as a baseline."""

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom = group["momentum"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.clone(g).detach()
                else:
                    st["buf"].mul_(mom).add_(g)
                p.add_(st["buf"], alpha=-lr)
        return loss
