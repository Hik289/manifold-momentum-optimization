"""Fifteen numerical checks for BW geometry and the matrix optimizer.

The checks cover symmetry, positive definiteness, BW distance, endpoints,
stationary inputs, commuting matrices, orthogonal changes of basis, transport,
geodesics, two-point optimality, two-sided preconditioning, identity and scalar
cases, repeatability with root refresh, and mixed precision. Most checks use
float64; the mixed-precision check uses float32 states. Seed: 42.
"""

import json
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bures.core import (
    symmetrize, spd_sqrt, spd_inv_sqrt, bw_distance, bw_distance_sq,
    ot_map, bw_geodesic, bures_barycenter_2pt, clip_spd,
)
from bures.optimizer import MatrixOptimizer

DTYPE = torch.float64
torch.manual_seed(42)


def rand_spd(d, dtype=DTYPE, cond_scale=1.0, device="cpu"):
    """Generate a random positive-definite matrix."""
    A = torch.randn(d, d, dtype=dtype, device=device)
    S = symmetrize(A @ A.transpose(-1, -2)) / d
    S = S + (0.1 + cond_scale) * torch.eye(d, dtype=dtype, device=device)
    return symmetrize(S)


def rand_orthogonal(d, dtype=DTYPE, device="cpu"):
    A = torch.randn(d, d, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


def relerr(x, y):
    return (x - y).norm().item() / (y.norm().item() + 1e-30)


# ---------------------------------------------------------------------------
# Each check returns a pass flag and a detail dictionary.
# ---------------------------------------------------------------------------

def T01_symmetry():
    """Check that BW averages remain symmetric."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        for alpha in (0.01, 0.3, 0.99):
            C = bures_barycenter_2pt(A, B, alpha)
            r = (C - C.transpose(-1, -2)).norm().item() / (C.norm().item())
            worst = max(worst, r)
    return worst < 1e-6, {"max_asym_rel": worst, "threshold": 1e-6}


def T02_positive_definite():
    """Check that BW averages remain positive definite."""
    worst_min = float("inf")
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        for alpha in (0.0, 0.01, 0.5, 0.99, 1.0):
            C = bures_barycenter_2pt(A, B, alpha)
            lmin = torch.linalg.eigvalsh(C).min().item()
            worst_min = min(worst_min, lmin)
    return worst_min > 0.0, {"min_eigenvalue": worst_min, "threshold": 0.0}


def T03_bw_distance_nonneg_symmetric():
    """d_BW^2 >= -1e-8; d_BW(A,B)=d_BW(B,A); d_BW(A,A)=0."""
    max_neg = 0.0
    max_asym = 0.0
    max_self = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        d2ab, _ = bw_distance_sq(A, B, clamp_neg=False)
        max_neg = min(max_neg, d2ab.item())  # Keep the lowest observed value.
        dab = bw_distance(A, B).item()
        dba = bw_distance(B, A).item()
        max_asym = max(max_asym, abs(dab - dba))
        max_self = max(max_self, bw_distance(A, A).item())
    ok = (max_neg >= -1e-8) and (max_asym < 1e-6) and (max_self < 1e-6)
    return ok, {"min_d2": max_neg, "max_asym": max_asym,
                "max_self_dist": max_self,
                "thresholds": {"nonneg": -1e-8, "asym": 1e-6, "self": 1e-6}}


def T04_barycenter_endpoints():
    """B_0(A,B)=A, B_1(A,B)=B."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        e0 = relerr(bures_barycenter_2pt(A, B, 0.0), A)
        e1 = relerr(bures_barycenter_2pt(A, B, 1.0), B)
        worst = max(worst, e0, e1)
    return worst < 1e-6, {"max_endpoint_relerr": worst, "threshold": 1e-6}


def T05_stationary():
    """B_alpha(A,A)=A."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        for alpha in (0.01, 0.3, 0.99):
            worst = max(worst, relerr(bures_barycenter_2pt(A, A, alpha), A))
    return worst < 1e-6, {"max_stationary_relerr": worst, "threshold": 1e-6}


def T06_commuting_closed_form():
    """Check the closed form when A and B share eigenvectors."""
    worst = 0.0
    for d in (4, 16, 64):
        Q = rand_orthogonal(d)
        a = torch.rand(d, dtype=DTYPE) + 0.1
        b = torch.rand(d, dtype=DTYPE) + 0.1
        A = symmetrize(Q @ torch.diag(a) @ Q.transpose(-1, -2))
        B = symmetrize(Q @ torch.diag(b) @ Q.transpose(-1, -2))
        for alpha in (0.2, 0.5, 0.8):
            got = bures_barycenter_2pt(A, B, alpha)
            H = (1 - alpha) * spd_sqrt(A) + alpha * spd_sqrt(B)
            expect = symmetrize(H @ H)
            worst = max(worst, relerr(got, expect))
    return worst < 1e-5, {"max_relerr": worst, "threshold": 1e-5}


def T07_orthogonal_equivariance():
    """B_alpha(QAQ^T, QBQ^T) = Q B_alpha(A,B) Q^T."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        Q = rand_orthogonal(d)
        for alpha in (0.2, 0.7):
            lhs = bures_barycenter_2pt(
                symmetrize(Q @ A @ Q.transpose(-1, -2)),
                symmetrize(Q @ B @ Q.transpose(-1, -2)), alpha)
            rhs = symmetrize(Q @ bures_barycenter_2pt(A, B, alpha)
                             @ Q.transpose(-1, -2))
            worst = max(worst, (lhs - rhs).norm().item())
    return worst < 1e-5, {"max_abs_diff": worst, "threshold": 1e-5}


def T08_ot_map_TAT():
    """T(A,B) A T(A,B) = B."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        T = ot_map(A, B)
        got = symmetrize(T @ A @ T)
        worst = max(worst, relerr(got, B))
    return worst < 1e-5, {"max_TAT_relerr": worst, "threshold": 1e-5}


def T09_geodesic_const_speed():
    """d_BW(Γ_s, Γ_t) = |s-t| d_BW(A,B)."""
    worst = 0.0
    for d in (4, 16, 64):
        A = rand_spd(d)
        B = rand_spd(d)
        dAB = bw_distance(A, B).item()
        pts = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                s, tt = pts[i], pts[j]
                gs = bw_geodesic(A, B, s)
                gt = bw_geodesic(A, B, tt)
                got = bw_distance(gs, gt).item()
                expect = abs(s - tt) * dAB
                worst = max(worst, abs(got - expect) / (expect + 1e-12))
    return worst < 1e-4, {"max_rel_speed_err": worst, "threshold": 1e-4}


def T10_two_point_optimality():
    """Check that the BW average beats sampled nearby alternatives.

    Compare the two-point objective at the average with random SPD
    perturbations. This is a numerical check, not a global proof.
    """
    def F(C, A, B, beta):
        d2a, _ = bw_distance_sq(C, A)
        d2b, _ = bw_distance_sq(C, B)
        return (beta * d2a + (1 - beta) * d2b).item()

    worst_gap = -float("inf")
    for d in (8, 16):
        A = rand_spd(d)
        B = rand_spd(d)
        for alpha in (0.3, 0.6):
            beta = 1.0 - alpha
            Cstar = bures_barycenter_2pt(A, B, alpha)
            Fstar = F(Cstar, A, B, beta)
            for _ in range(200):
                E = torch.randn(d, d, dtype=DTYPE)
                E = symmetrize(E) * 0.05
                Cpert = symmetrize(Cstar + E)
                # Project the perturbed matrix back to the SPD cone.
                ev, evec = torch.linalg.eigh(Cpert)
                ev = ev.clamp(min=1e-6)
                Cpert = symmetrize((evec * ev.unsqueeze(-2))
                                   @ evec.transpose(-1, -2))
                gap = Fstar - F(Cpert, A, B, beta)
                worst_gap = max(worst_gap, gap)
    return worst_gap <= 1e-6, {"max_gap": worst_gap, "threshold": 1e-6}


def T11_kronecker_direction():
    """vec(L^-1/2 M R^-1/2) = (R^-1/2 ⊗ L^-1/2) vec(M).
    Use column-major vectorization in the numerical comparison.
    """
    worst = 0.0
    for (m, n) in ((4, 6), (8, 8), (16, 12)):
        L = rand_spd(m)
        R = rand_spd(n)
        M = torch.randn(m, n, dtype=DTYPE)
        Lih = spd_inv_sqrt(L)
        Rih = spd_inv_sqrt(R)
        direct = Lih @ M @ Rih
        # For column-major vec, vec(AXB) = (B^T ⊗ A) vec(X).
        # Rih is symmetric, so vec(D) = (Rih ⊗ Lih) vec(M).
        vecM = M.transpose(-1, -2).reshape(-1)  # column-major
        kron = torch.kron(Rih, Lih)
        vecD = kron @ vecM
        direct_vec = direct.transpose(-1, -2).reshape(-1)
        worst = max(worst, relerr(vecD, direct_vec))
    return worst < 1e-6, {"max_relerr": worst, "threshold": 1e-6}


def T12_identity_degeneration():
    """Check that identity inverse roots leave a matrix unchanged.

    The identity is checked directly. A separate optimizer step checks that
    the full path runs; that step updates its covariance states.
    """
    worst = 0.0
    for (m, n) in ((4, 6), (16, 16), (32, 8)):
        Im = torch.eye(m, dtype=DTYPE)
        In = torch.eye(n, dtype=DTYPE)
        M = torch.randn(m, n, dtype=DTYPE)
        D = spd_inv_sqrt(Im) @ M @ spd_inv_sqrt(In)
        worst = max(worst, relerr(D, M))
    # Run one complete optimizer step as an additional smoke check.
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.randn(8, 10, dtype=DTYPE))
    opt = MatrixOptimizer([w], lr=0.0, betas=(0.9, 0.99), tau_root=1,
                          state_dtype=DTYPE)  # lr=0 leaves the parameter fixed.
    g = torch.randn(8, 10, dtype=DTYPE)
    w.grad = g
    opt.step()
    # The updated covariance states are not identity after this step.
    return worst < 1e-6, {"max_identity_relerr": worst, "threshold": 1e-6,
                          "note": "core identity I^-1/2 M I^-1/2 = M"}


def T13_scalar_degeneration():
    """1D: c_t = [beta2 sqrt(c_{t-1}) + (1-beta2) sqrt(s_t)]^2.
    Compare a 1x1 BW average with the scalar recurrence.
    """
    beta2 = 0.99
    alpha = 1 - beta2
    worst = 0.0
    torch.manual_seed(42)
    c = torch.tensor([[2.0]], dtype=DTYPE)
    for _ in range(50):
        s = (torch.rand(1, 1, dtype=DTYPE) * 3 + 0.5)
        c_new = bures_barycenter_2pt(c, s, alpha)  # 1x1 barycenter
        scalar = (beta2 * torch.sqrt(c[0, 0]) +
                  (1 - beta2) * torch.sqrt(s[0, 0])) ** 2
        worst = max(worst, abs(c_new[0, 0].item() - scalar.item())
                    / scalar.item())
        c = c_new
    return worst < 1e-6, {"max_scalar_relerr": worst, "threshold": 1e-6}


def T14_cache_consistency():
    """Check repeatability when inverse roots refresh at every step.

    Two identical optimizer instances receive the same gradients and should
    produce the same parameters.
    """
    torch.manual_seed(42)
    m, n = 8, 10
    w1 = torch.nn.Parameter(torch.randn(m, n, dtype=DTYPE))
    w2 = torch.nn.Parameter(w1.detach().clone())
    opt_cache = MatrixOptimizer([w1], lr=1e-2, tau_root=1, state_dtype=DTYPE,
                                covariance_rule="bures")
    opt_direct = MatrixOptimizer([w2], lr=1e-2, tau_root=1, state_dtype=DTYPE,
                                 covariance_rule="bures")
    worst = 0.0
    torch.manual_seed(123)
    for _ in range(10):
        g = torch.randn(m, n, dtype=DTYPE)
        w1.grad = g.clone()
        w2.grad = g.clone()
        opt_cache.step()
        opt_direct.step()
        worst = max(worst, (w1 - w2).norm().item())
    # Both instances refresh roots every step and should match.
    return worst < 1e-10, {"max_path_diff": worst, "threshold": 1e-10}


def T15_mixed_precision():
    """Check that the optimizer keeps finite float32 covariance states.

    Feed float16-generated gradients after converting them to float32.
    """
    torch.manual_seed(42)
    m, n = 8, 10
    w = torch.nn.Parameter(torch.randn(m, n, dtype=torch.float32))
    opt = MatrixOptimizer([w], lr=1e-3, state_dtype=torch.float32,
                          covariance_rule="bures")
    finite_ok = True
    max_asym = 0.0
    for _ in range(20):
        g16 = torch.randn(m, n, dtype=torch.float16)
        w.grad = g16.to(torch.float32)  # FP16 grad -> FP32
        opt.step()
        st = opt.state[w]
        for key in ("L", "R"):
            for blk in st[key].values():
                if blk.dtype != torch.float32 and blk.dtype != torch.float64:
                    finite_ok = False
                if not torch.isfinite(blk).all():
                    finite_ok = False
                asym = (blk - blk.transpose(-1, -2)).norm().item() \
                    / (blk.norm().item() + 1e-30)
                max_asym = max(max_asym, asym)
    ok = finite_ok and torch.isfinite(w).all().item() and max_asym < 1e-4
    return ok, {"state_dtype_ok": finite_ok, "max_state_asym": max_asym,
                "threshold": 1e-4}


TESTS = [
    ("T01_symmetry", T01_symmetry),
    ("T02_positive_definite", T02_positive_definite),
    ("T03_bw_distance_nonneg_symmetric", T03_bw_distance_nonneg_symmetric),
    ("T04_barycenter_endpoints", T04_barycenter_endpoints),
    ("T05_stationary", T05_stationary),
    ("T06_commuting_closed_form", T06_commuting_closed_form),
    ("T07_orthogonal_equivariance", T07_orthogonal_equivariance),
    ("T08_ot_map_TAT", T08_ot_map_TAT),
    ("T09_geodesic_const_speed", T09_geodesic_const_speed),
    ("T10_two_point_optimality", T10_two_point_optimality),
    ("T11_kronecker_direction", T11_kronecker_direction),
    ("T12_identity_degeneration", T12_identity_degeneration),
    ("T13_scalar_degeneration", T13_scalar_degeneration),
    ("T14_cache_consistency", T14_cache_consistency),
    ("T15_mixed_precision", T15_mixed_precision),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/numerical_results.json")
    args = ap.parse_args()

    results = []
    n_pass = 0
    for name, fn in TESTS:
        try:
            passed, detail = fn()
        except Exception as e:
            import traceback
            passed, detail = False, {"error": str(e),
                                     "trace": traceback.format_exc()}
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        print(f"[{status}] {name}: {json.dumps(detail, default=str)}")
        results.append({"name": name, "passed": passed, "detail": detail})

    summary = {
        "total": len(TESTS), "passed": n_pass, "failed": len(TESTS) - n_pass,
        "all_pass": n_pass == len(TESTS),
        "seed": 42, "geometry_dtype": "float64",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== {n_pass}/{len(TESTS)} passed ===")
    print(f"written to {args.out}")
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
