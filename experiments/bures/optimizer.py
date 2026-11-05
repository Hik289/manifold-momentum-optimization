"""
BuresMomentum (B5) 与配对基线 Euclidean-Kronecker (B3) 的 optimizer 实现.

设计要点 (对应 README §3, §9, hypothesis_tree H0.anchor_2 / H0.H4):
  - B3 与 B5 **共享全部超参与实现路径**, 唯一差别 = 协方差状态更新规则:
        B5: L_t = ClipSPD(Bar_BW(L_{t-1}, S_L))   [Bures barycenter]
        B3: L_t = beta2 * L_{t-1} + (1-beta2) * S_L [欧氏 EMA]
  - 其余步骤 (一阶动量 / 瞬时统计 / 逆平方根预条件 / norm matching /
    参数更新 / SPD 谱裁剪 / root cache) 完全一致.
  - 一维参数走 AdamW (README §3.1).
  - 大矩阵按 block size 256 分块, 每块独立跑同一算法 (README §3.5, §6.4).
  - SPD 状态与矩阵分解至少 FP32 (README §14.18).

covariance_rule in {"bures", "euclidean"} 是 B5/B3 的唯一开关.
"""

import math
import torch
from torch.optim import Optimizer

from .core import (
    symmetrize, spd_inv_sqrt, bures_barycenter_2pt, clip_spd,
    bw_distance, scale_only_average,
)


def _split_blocks(d, block):
    """把维度 d 切成 <=block 的连续块边界列表."""
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
    """统一矩阵预条件优化器. covariance_rule 切换 B5(bures)/B3(euclidean).

    仅对 2D 参数构造 SPD 状态; 其余参数走内置 AdamW 分支.
    Each rectangular tile owns independent L/R/root states (schema v2).
    Historical schema-v1 checkpoints must not be resumed with this version.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99),
                 eps_stat=1e-6, eps_norm=1e-12, weight_decay=0.0,
                 lam_min=1e-6, lam_max=1e6, s_min=0.1, s_max=10.0,
                 tau_root=10, block_size=256, covariance_rule="bures",
                 adamw_eps=1e-8, state_dtype=torch.float32, ablation="full"):
        if covariance_rule not in ("bures", "euclidean", "scale_only", "ai", "log_euclidean"):
            raise ValueError(
                "covariance_rule must be 'bures'/'euclidean'/'scale_only'")
        defaults = dict(
            lr=lr, betas=betas, eps_stat=eps_stat, eps_norm=eps_norm,
            weight_decay=weight_decay, lam_min=lam_min, lam_max=lam_max,
            s_min=s_min, s_max=s_max, tau_root=tau_root, block_size=block_size,
            covariance_rule=covariance_rule, adamw_eps=adamw_eps,
            state_dtype=state_dtype, ablation=ablation,
        )
        super().__init__(params, defaults)
        # 每步聚合的诊断量 (供 pipeline 记录, README §7.3)
        self.last_diag = {}

    # ---- SPD 状态更新: 唯一区分 B5/B3 的地方 ----
    @staticmethod
    def _update_cov(rule, prev, S, alpha, lam_min, lam_max):
        """返回 (new_state, clip_triggered).

        rule='bures':      Bures barycenter, alpha = 1 - beta2
        rule='euclidean':  欧氏 EMA, new = beta2*prev + (1-beta2)*S
        rule='scale_only': 尺度-only 平均 (固定旧基, √λ 平均, 去旋转) [Stage2 消融]
        三者更新后都做同样的 ClipSPD.
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

                # ---- 1D / 非2D 参数: AdamW 分支 ----
                if g.dim() != 2:
                    self._adamw_step(p, g, st, lr, beta1, beta2,
                                     adamw_eps, wd)
                    continue

                # ---- 2D 参数: 矩阵预条件 ----
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

                # Step 1: 一阶动量 + bias correction
                M = st["M"]
                M.mul_(beta1).add_(g, alpha=1.0 - beta1)
                M_hat = (M / (1.0 - beta1 ** t)).to(sdtype)
                diag_accum["first_moment_norm"] += float(M.norm().item()) ** 2

                D_raw = torch.zeros_like(M_hat)
                refresh = (t % tau_root == 0) or (t == 1) or (tau_root == 1)

                # 分块处理 (每个 (行块, 列块) 组合独立)
                for (i0, i1) in st["blocks_m"]:
                    gi = g32[i0:i1, :]
                    for (j0, j1) in st["blocks_n"]:
                        gij = gi[:, j0:j1]
                        key = (i0, j0)
                        bm, bn = i1 - i0, j1 - j0
                        Im = torch.eye(bm, dtype=sdtype, device=p.device)
                        In = torch.eye(bn, dtype=sdtype, device=p.device)
                        # Step 2: 瞬时 SPD 统计
                        S_L = symmetrize(gij @ gij.transpose(-1, -2)) / bn \
                            + eps_stat * Im
                        S_R = symmetrize(gij.transpose(-1, -2) @ gij) / bm \
                            + eps_stat * In

                        # Step 3: 协方差状态更新 (B5/B3 唯一分歧)
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

                        # root cache: 每 tau_root 步刷新逆平方根
                        if refresh or (key not in st["Lih_cache"]):
                            st["Lih_cache"][key] = spd_inv_sqrt(L_new)
                        if refresh or (key not in st["Rih_cache"]):
                            st["Rih_cache"][key] = spd_inv_sqrt(R_new)
                        Lih = st["Lih_cache"][key]
                        Rih = st["Rih_cache"][key]

                        # Step 4: 预条件方向 (raw)
                        Mij = M_hat[i0:i1, j0:j1]
                        D_raw[i0:i1, j0:j1] = Lih @ Mij @ Rih

                # Step 4b: Frobenius norm matching (全参数级别)
                num = M_hat.norm()
                den = D_raw.norm() + eps_norm
                scale = torch.clamp(num / den, min=s_min, max=s_max)
                if abl == "no_norm":
                    scale = torch.ones_like(scale)
                D = (scale * D_raw).to(p.dtype)
                diag_accum["precond_update_norm"] += float(D.norm().item()) ** 2
                diag_accum["scale_sum"] += float(scale.item())
                diag_accum["scale_count"] += 1

                # Step 5: 参数更新 (decoupled weight decay)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(D, alpha=-lr)

        # 汇总诊断
        for k in ("raw_grad_norm", "first_moment_norm", "precond_update_norm"):
            diag_accum[k] = math.sqrt(diag_accum[k])
        if diag_accum["scale_count"] > 0:
            diag_accum["scale_mean"] = \
                diag_accum["scale_sum"] / diag_accum["scale_count"]
        else:
            diag_accum["scale_mean"] = float("nan")
        self.last_diag = diag_accum
        return loss

    # ---- 内置 AdamW (1D / 非矩阵参数) ----
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
    """标准 AdamW 对照基线 (B2), 用于 anchor_2 复现合理曲线."""

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
    """Shampoo 对照基线 (B4, README §9).

    标准 Shampoo: 维护 L,R 累积 Gram 统计, 预条件 = L^{-1/4} G R^{-1/4}.
    与 B3/B5 不同 (那两者是 EMA / Bures 的 -1/2 幂), Shampoo 用 -1/4 幂 +
    累积统计, 作为独立对照. 一维参数走 AdamW.
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
    """SGD-with-momentum 对照基线 (B1)."""

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
