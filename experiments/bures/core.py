"""Bures-Wasserstein operations for symmetric positive-definite matrices.

This module computes matrix roots, the BW distance and transport map,
two-point BW averages, and eigenvalue clipping. Optimizer states use at least
float32; the numerical tests use float64 for tighter accuracy checks.
"""

import warnings
import torch


# Count how often a failed eigendecomposition needs a fallback.
EIGH_FALLBACK_STATS = {"cpu_retry": 0, "jitter_retry": 0, "svd_fallback": 0,
                       "total_calls": 0}


# ---------------------------------------------------------------------------
# Symmetry and eigendecomposition
# ---------------------------------------------------------------------------

def symmetrize(A):
    """Remove small asymmetries caused by floating-point arithmetic."""
    return 0.5 * (A + A.transpose(-1, -2))


def _robust_eigh(A):
    """Compute eigenvalues and eigenvectors, retrying if the solver fails.

    Try the original device first. If needed, retry on CPU in float64, add a
    small diagonal offset, and finally use SVD. Return the result on the
    original device and with the original dtype.
    """
    EIGH_FALLBACK_STATS["total_calls"] += 1
    A = symmetrize(A)
    dev, dt = A.device, A.dtype

    # First try the original device.
    try:
        return torch.linalg.eigh(A)
    except Exception:
        pass

    # Retry on CPU in float64.
    A_cpu = A.detach().to("cpu", torch.float64)
    A_cpu = symmetrize(A_cpu)
    try:
        evals, evecs = torch.linalg.eigh(A_cpu)
        EIGH_FALLBACK_STATS["cpu_retry"] += 1
        return evals.to(dt).to(dev), evecs.to(dt).to(dev)
    except Exception:
        pass

    # Add progressively larger diagonal offsets if CPU eigh also fails.
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

    # Last resort: recover signed eigenvalues from an SVD of the symmetric matrix.
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
    """Return eigenvalues, with the same fallback if the solver fails."""
    try:
        return torch.linalg.eigvalsh(symmetrize(A))
    except Exception:
        evals, _ = _robust_eigh(A)
        return evals


def _eigh_psd(A, floor=0.0):
    """Decompose a symmetric matrix and apply an optional eigenvalue floor."""
    A = symmetrize(A)
    evals, evecs = _robust_eigh(A)
    if floor > 0.0:
        evals = torch.clamp(evals, min=floor)
    return evals, evecs


def spd_power(A, p, floor=None, rel_floor=1e-12):
    """Compute A**p from its eigendecomposition.

    Values 0.5, -0.5, and -1 give the square root, inverse square root, and
    inverse. Before taking a negative power, floor tiny eigenvalues relative
    to the largest one. This prevents unstable inverse roots when float32
    rounding makes a nearly singular matrix appear slightly indefinite.
    """
    evals, evecs = _eigh_psd(A, floor=0.0)
    lam_max = float(evals.max().clamp(min=0.0).item())
    eff_floor = rel_floor * lam_max            # Scale the floor to the spectrum.
    if floor is not None:
        eff_floor = max(eff_floor, float(floor))
    abs_floor = 1e-30 if p >= 0 else 1e-20     # Negative powers need a larger floor.
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
# Bures-Wasserstein distance
# ---------------------------------------------------------------------------

def bw_distance_sq(A, B, floor=1e-30, clamp_neg=True):
    """d_BW^2(A,B) = tr(A) + tr(B) - 2 tr[(A^{1/2} B A^{1/2})^{1/2}].

    Round-off can make a zero distance slightly negative. Clamp it to zero
    when requested, and return both the value and whether clamping occurred.
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
# Transport map, geodesic, and two-point average
# ---------------------------------------------------------------------------

def ot_map(A, B, floor=1e-30):
    """T(A,B) = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}.

    The map satisfies T A T = B up to numerical error.
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
    """Return the BW average of A and B at position alpha on the geodesic.

    The optimizer uses A as the previous state, B as the new gradient
    statistic, and alpha = 1 - beta2.
    """
    return bw_geodesic(A, B, alpha, floor=floor)


def scale_only_average(A, B, alpha, floor=1e-30):
    """Average eigenvalue scales while keeping A's eigenvectors fixed.

    This ablation removes the rotation part of the full BW update. Project B
    onto A's eigenvectors, average the square roots of the resulting diagonal
    values, and square the result:
        λ_i^new = [(1-α) sqrt(a_i) + α sqrt(b_i^proj)]^2
    The result matches full BW averaging when A and B share eigenvectors.
    """
    A = symmetrize(A)
    B = symmetrize(B)
    a, QA = _robust_eigh(A)          # A = QA diag(a) QA^T.
    a = a.clamp(min=floor)
    # Measure B along A's eigenvectors.
    Bproj = QA.transpose(-1, -2) @ B @ QA
    b = torch.diagonal(Bproj, dim1=-2, dim2=-1).clamp(min=floor)
    # Average square roots, then square; do not rotate the eigenvectors.
    lam_new = ((1.0 - alpha) * torch.sqrt(a) + alpha * torch.sqrt(b)) ** 2
    return symmetrize((QA * lam_new.unsqueeze(-2)) @ QA.transpose(-1, -2))


# ---------------------------------------------------------------------------
# Eigenvalue clipping
# ---------------------------------------------------------------------------

def clip_spd(A, lam_min=1e-6, lam_max=1e6):
    """Limit eigenvalues to [lam_min, lam_max]; return the matrix and a flag."""
    evals, evecs = _eigh_psd(A, floor=0.0)
    clipped = torch.clamp(evals, min=lam_min, max=lam_max)
    triggered = bool(((evals < lam_min) | (evals > lam_max)).any().item())
    out = symmetrize((evecs * clipped.unsqueeze(-2)) @ evecs.transpose(-1, -2))
    return out, triggered


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def spd_stats(A):
    """Return eigenvalue bounds, condition number, mean trace, and logdet."""
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
