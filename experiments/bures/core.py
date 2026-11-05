"""
Bures-Wasserstein SPD 几何核心模块 (BuresMomentum)

实现 README §1.3 / §3 / §4 定义的全部几何原语:
  - SPD 平方根 / 逆平方根 (对称特征分解)
  - Bures-Wasserstein 距离 d_BW
  - 最优传输映射 T(A,B)
  - BW 测地线 Γ_s(A,B)
  - 两点 Bures barycenter B_α(A,B)
  - SPD 谱裁剪 ClipSPD

数值约定:
  - 所有 SPD 状态与矩阵分解至少 FP32 (README §3.5 / §14.18)
  - 测试默认用 FP64 以稳妥通过 1e-6 阈值
  - 特征分解走 torch.linalg.eigh (对称路径)
"""

import warnings
import torch


# 全局计数: eigh 回退事件 (供诊断/日志, hypothesis_tree 意外现象追踪)
EIGH_FALLBACK_STATS = {"cpu_retry": 0, "jitter_retry": 0, "svd_fallback": 0,
                       "total_calls": 0}


# ---------------------------------------------------------------------------
# 基础: 对称化 + 鲁棒特征分解
# ---------------------------------------------------------------------------

def symmetrize(A):
    """强制对称, 消除有限精度非对称漂移."""
    return 0.5 * (A + A.transpose(-1, -2))


def _robust_eigh(A):
    """鲁棒对称特征分解, 抵御 GPU cuSOLVER 在病态/重根矩阵上的
    error-65 (failed to converge).

    回退阶梯 (逐级更稳, 只在失败时才升级):
      1) 原设备 torch.linalg.eigh (GPU 上走 cuSOLVER syevd)
      2) CPU LAPACK eigh (divide-and-conquer, 对病态/重根远更稳)
      3) CPU 上加对角 jitter (相对 trace 尺度) 后再 eigh
      4) CPU SVD 兜底 (对称阵 U≈V, 用 U 与 signed singular values 重构谱)
    结果搬回原 dtype/device. 返回 (evals, evecs).
    """
    EIGH_FALLBACK_STATS["total_calls"] += 1
    A = symmetrize(A)
    dev, dt = A.device, A.dtype

    # 1) 原设备直接尝试
    try:
        return torch.linalg.eigh(A)
    except Exception:
        pass

    # 2) CPU LAPACK 重试 (FP64 提升数值稳定)
    A_cpu = A.detach().to("cpu", torch.float64)
    A_cpu = symmetrize(A_cpu)
    try:
        evals, evecs = torch.linalg.eigh(A_cpu)
        EIGH_FALLBACK_STATS["cpu_retry"] += 1
        return evals.to(dt).to(dev), evecs.to(dt).to(dev)
    except Exception:
        pass

    # 3) CPU + 相对 jitter
    d = A_cpu.shape[-1]
    scale = torch.diagonal(A_cpu, dim1=-2, dim2=-1).abs().mean().clamp(min=1e-12)
    for j in (1e-10, 1e-8, 1e-6, 1e-4):
        jit = A_cpu + (j * scale) * torch.eye(d, dtype=torch.float64)
        try:
            evals, evecs = torch.linalg.eigh(symmetrize(jit))
            EIGH_FALLBACK_STATS["jitter_retry"] += 1
            return evals.to(dt).to(dev), evecs.to(dt).to(dev)
        except Exception:
            continue

    # 4) SVD 兜底 (对称阵: A = U S V^T, 特征值 = sign 对齐的奇异值)
    U, S, Vh = torch.linalg.svd(A_cpu)
    signs = torch.sign((U * Vh.transpose(-1, -2)).sum(-2))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    evals = (S * signs)
    order = torch.argsort(evals, dim=-1)
    evals = torch.gather(evals, -1, order)
    evecs = torch.gather(U, -1, order.unsqueeze(-2).expand_as(U))
    EIGH_FALLBACK_STATS["svd_fallback"] += 1
    warnings.warn("_robust_eigh fell back to SVD (eigh non-convergent)")
    return evals.to(dt).to(dev), evecs.to(dt).to(dev)


def _robust_eigvalsh(A):
    """仅特征值版本, 同鲁棒回退."""
    try:
        return torch.linalg.eigvalsh(symmetrize(A))
    except Exception:
        evals, _ = _robust_eigh(A)
        return evals


def _eigh_psd(A, floor=0.0):
    """对称特征分解, 特征值下限截断 (保证 PSD/SPD). 走鲁棒 eigh."""
    A = symmetrize(A)
    evals, evecs = _robust_eigh(A)
    if floor > 0.0:
        evals = torch.clamp(evals, min=floor)
    return evals, evecs


def spd_power(A, p, floor=None, rel_floor=1e-12):
    """SPD 矩阵幂 A^p, 经对称特征分解.

    p=0.5 -> sqrt, p=-0.5 -> inverse sqrt, p=-1 -> inverse.

    数值鲁棒性 (修复 Stage1 exploded_gram → inv-sqrt 爆炸 → NaN):
      - 负幂 (inverse 类) 对近零/负特征值极敏感。FP32 eigh 在 cond>1e7 的
        矩阵上会分解出微小负特征值, clamp(1e-30)**(-0.5) 会爆到 1e15。
      - 因此特征值先按 **相对 floor = rel_floor * lambda_max** 截断
        (谱尺度自适应), 再按绝对 floor 兜底。正幂基本不受影响。
    """
    evals, evecs = _eigh_psd(A, floor=0.0)
    lam_max = float(evals.max().clamp(min=0.0).item())
    eff_floor = rel_floor * lam_max            # 谱尺度自适应
    if floor is not None:
        eff_floor = max(eff_floor, float(floor))
    abs_floor = 1e-30 if p >= 0 else 1e-20     # 负幂给更大绝对兜底
    eff_floor = max(eff_floor, abs_floor)
    powered = evals.clamp(min=eff_floor) ** p
    return symmetrize((evecs * powered.unsqueeze(-2)) @ evecs.transpose(-1, -2))


def spd_sqrt(A, floor=None):
    return spd_power(A, 0.5, floor=floor)


def spd_inv_sqrt(A, floor=None):
    return spd_power(A, -0.5, floor=floor)


def spd_inv(A, floor=None):
    return spd_power(A, -1.0, floor=floor)


# ---------------------------------------------------------------------------
# Bures-Wasserstein 距离 (README §1.3, §15.3-15.5)
# ---------------------------------------------------------------------------

def bw_distance_sq(A, B, floor=1e-30, clamp_neg=True):
    """d_BW^2(A,B) = tr(A) + tr(B) - 2 tr[(A^{1/2} B A^{1/2})^{1/2}].

    有限精度下允许的极小负值截断到 0 (README §15.3), 返回 (value, was_clamped).
    """
    A = symmetrize(A)
    B = symmetrize(B)
    Ah = spd_sqrt(A, floor=floor)
    M = symmetrize(Ah @ B @ Ah)
    # tr[(A^1/2 B A^1/2)^1/2] = sum sqrt(eig(M))
    evals = _robust_eigvalsh(M).clamp(min=0.0)
    cross = torch.sqrt(evals).sum(-1)
    trA = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    trB = torch.diagonal(B, dim1=-2, dim2=-1).sum(-1)
    d2 = trA + trB - 2.0 * cross
    was_clamped = bool((d2 < 0).any().item()) if clamp_neg else False
    if clamp_neg:
        d2 = torch.clamp(d2, min=0.0)
    return d2, was_clamped


def bw_distance(A, B, floor=1e-30):
    d2, _ = bw_distance_sq(A, B, floor=floor)
    return torch.sqrt(d2)


# ---------------------------------------------------------------------------
# 最优传输映射 + 测地线 + barycenter (README §1.3, §3.2)
# ---------------------------------------------------------------------------

def ot_map(A, B, floor=1e-30):
    """T(A,B) = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}.

    满足 T A T = B (README §15.10).
    """
    A = symmetrize(A)
    B = symmetrize(B)
    Ah = spd_sqrt(A, floor=floor)
    Aih = spd_inv_sqrt(A, floor=floor)
    M = symmetrize(Ah @ B @ Ah)
    Msqrt = spd_sqrt(M, floor=floor)
    return symmetrize(Aih @ Msqrt @ Aih)


def bw_geodesic(A, B, s, floor=1e-30):
    """Γ_s(A,B) = [(1-s)I + s T(A,B)] A [(1-s)I + s T(A,B)]."""
    A = symmetrize(A)
    T = ot_map(A, B, floor=floor)
    d = A.shape[-1]
    I = torch.eye(d, dtype=A.dtype, device=A.device)
    H = (1.0 - s) * I + s * T
    return symmetrize(H @ A @ H)


def bures_barycenter_2pt(A, B, alpha, floor=1e-30):
    """两点 Bures barycenter B_α(A,B) = Γ_α(A,B) (README §3.2 Step 3).

    在 BuresMomentum 更新中 alpha = 1 - beta2, A=旧状态, B=瞬时统计.
    Fréchet 权重: Bar_{BW,β}(A,B)=Γ_{1-β}, 这里 β=beta2 => s=1-beta2=alpha.
    """
    return bw_geodesic(A, B, alpha, floor=floor)


def scale_only_average(A, B, alpha, floor=1e-30):
    """尺度-only 平均 (Stage2 INTERVENE 消融臂).

    隔离"√λ 尺度平均"与"Bures 旋转/非对易处理"。
    做法: 固定旧状态 A 的特征基 Q_A (不旋转), 把瞬时统计 B 投影到该基,
    取对角 (沿 A 的主轴的方差), 然后按 §4.3 交换情形公式对 **标准差** 做
    线性平均再平方:
        λ_i^new = [(1-α) sqrt(a_i) + α sqrt(b_i^proj)]^2
    其中 a_i = A 在 Q_A 下的特征值, b_i^proj = diag(Q_A^T B Q_A)_i。

    与完整 Bures 的唯一区别: 特征向量**不随 B 旋转** (Q_A 固定), 因此去掉了
    Bures 对特征子空间旋转的完整处理, 只保留尺度 (√λ vs λ) 效应。
    A=B 或 A,B 共基时, 该式与完整 Bures 完全一致 (§4.3), 差异只在非对易分量。
    """
    A = symmetrize(A)
    B = symmetrize(B)
    a, QA = _robust_eigh(A)          # A = QA diag(a) QA^T
    a = a.clamp(min=floor)
    # B 投影到 A 的特征基, 取对角 (沿 A 主轴的方差)
    Bproj = QA.transpose(-1, -2) @ B @ QA
    b = torch.diagonal(Bproj, dim1=-2, dim2=-1).clamp(min=floor)
    # √标准差线性平均再平方 (§4.3 交换情形核心, 无旋转)
    lam_new = ((1.0 - alpha) * torch.sqrt(a) + alpha * torch.sqrt(b)) ** 2
    return symmetrize((QA * lam_new.unsqueeze(-2)) @ QA.transpose(-1, -2))


# ---------------------------------------------------------------------------
# SPD 谱裁剪 (README §3.2 Step 3, §14 lambda_min/max)
# ---------------------------------------------------------------------------

def clip_spd(A, lam_min=1e-6, lam_max=1e6):
    """ClipSPD: 把特征值裁到 [lam_min, lam_max], 返回 (裁剪后矩阵, 是否触发)."""
    evals, evecs = _eigh_psd(A, floor=0.0)
    clipped = torch.clamp(evals, min=lam_min, max=lam_max)
    triggered = bool(((evals < lam_min) | (evals > lam_max)).any().item())
    out = symmetrize((evecs * clipped.unsqueeze(-2)) @ evecs.transpose(-1, -2))
    return out, triggered


# ---------------------------------------------------------------------------
# 诊断量 (README §7.3)
# ---------------------------------------------------------------------------

def spd_stats(A):
    """返回 (lambda_min, lambda_max, cond, trace/d, logdet)."""
    evals = _robust_eigvalsh(A).clamp(min=1e-30)
    lmin = evals.min().item()
    lmax = evals.max().item()
    d = A.shape[-1]
    return {
        "lambda_min": lmin,
        "lambda_max": lmax,
        "cond": lmax / lmin,
        "trace_over_d": (evals.sum().item() / d),
        "logdet": torch.log(evals).sum().item(),
    }
