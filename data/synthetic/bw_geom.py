"""
Independent Bures-Wasserstein geometry (NumPy / FP64).

⚠️ DESIGN PRINCIPLE (Data Scientist as independent judge, SOUL.md):
This module is a DELIBERATELY INDEPENDENT re-implementation of the BW geometry,
written in NumPy + FP64, with NO dependency on the ML engineer's torch-based
`experiments/bures/core.py`. It is the *referee* that produces high-precision
offline ground truth for G1/G2. Cross-checking two independent backends
(numpy-FP64 here vs torch-FP32/64 there) is itself a correctness guarantee.

All matrices are real symmetric positive definite (SPD), dtype float64.
References:
  Bhatia-Jain-Lim 2019 (BW distance); Alvarez-Esteban et al. 2016 (barycenter
  fixed point); Agueh-Carlier 2011 (barycenter variational form).
"""
import numpy as np

FLOOR = 1e-300  # eigenvalue floor to avoid div-by-zero; GT inputs are well-conditioned


# --------------------------------------------------------------------------
# symmetric eigen-powers
# --------------------------------------------------------------------------
def sym(A):
    return 0.5 * (A + A.T)


def spd_power(A, p, floor=FLOOR):
    A = sym(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, floor, None)
    return sym((V * (w ** p)) @ V.T)


def sqrtm(A):
    return spd_power(A, 0.5)


def inv_sqrtm(A):
    return spd_power(A, -0.5)


def invm(A):
    return spd_power(A, -1.0)


# --------------------------------------------------------------------------
# BW distance, OT map, geodesic
# --------------------------------------------------------------------------
def bw_distance_sq(A, B):
    """d_BW^2(A,B) = tr A + tr B - 2 tr[(A^1/2 B A^1/2)^1/2]. Neg clamp -> 0."""
    A = sym(A); B = sym(B)
    Ah = sqrtm(A)
    M = sym(Ah @ B @ Ah)
    ev = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    cross = np.sqrt(ev).sum()
    d2 = np.trace(A) + np.trace(B) - 2.0 * cross
    return max(float(d2), 0.0)


def bw_distance(A, B):
    return np.sqrt(bw_distance_sq(A, B))


def ot_map(A, B):
    """T(A,B) = A^-1/2 (A^1/2 B A^1/2)^1/2 A^-1/2 ; satisfies T A T = B."""
    A = sym(A); B = sym(B)
    Ah = sqrtm(A); Aih = inv_sqrtm(A)
    M = sym(Ah @ B @ Ah)
    return sym(Aih @ sqrtm(M) @ Aih)


def bw_geodesic(A, B, s):
    """Gamma_s(A,B) = [(1-s)I + s T(A,B)] A [(1-s)I + s T(A,B)]."""
    A = sym(A)
    T = ot_map(A, B)
    d = A.shape[0]
    H = (1.0 - s) * np.eye(d) + s * T
    return sym(H @ A @ H)


# --------------------------------------------------------------------------
# high-precision population barycenter (Alvarez-Esteban fixed point)
# --------------------------------------------------------------------------
def bures_barycenter(mats, weights=None, tol=1e-14, max_iter=2000, init=None):
    """Population Bures-Wasserstein barycenter of SPD matrices {mats} with
    weights. Fixed-point map G(B)=B^-1/2 [ sum_i w_i (B^1/2 S_i B^1/2)^1/2 ]^2 B^-1/2,
    provably convergent (Alvarez-Esteban 2016). Returns (Bstar, info).
    """
    mats = [sym(np.asarray(S, dtype=np.float64)) for S in mats]
    n = len(mats)
    if weights is None:
        weights = np.full(n, 1.0 / n)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    B = sym(np.mean(mats, axis=0)) if init is None else sym(np.asarray(init, float))
    hist = []
    for it in range(max_iter):
        Bh = sqrtm(B); Bih = inv_sqrtm(B)
        acc = np.zeros_like(B)
        for w, S in zip(weights, mats):
            acc += w * sqrtm(sym(Bh @ S @ Bh))
        Bnew = sym(Bih @ (acc @ acc) @ Bih)
        step = np.linalg.norm(Bnew - B) / (np.linalg.norm(B) + 1e-30)
        B = Bnew
        hist.append(step)
        if step < tol:
            break
    resid = frechet_residual(B, mats, weights)
    return B, {"iters": it + 1, "last_step": step, "frechet_residual": resid}


def frechet_residual(B, mats, weights=None):
    """First-order optimality residual: || sum_i w_i T_{B->S_i} - I ||_F.
    Barycenter satisfies sum_i w_i T(B,S_i) = I."""
    mats = [sym(np.asarray(S, float)) for S in mats]
    n = len(mats)
    if weights is None:
        weights = np.full(n, 1.0 / n)
    weights = np.asarray(weights, float); weights = weights / weights.sum()
    d = B.shape[0]
    acc = np.zeros((d, d))
    for w, S in zip(weights, mats):
        acc += w * ot_map(B, S)
    return float(np.linalg.norm(acc - np.eye(d)))


# --------------------------------------------------------------------------
# off-geodesic error (independent judge for G1)
# --------------------------------------------------------------------------
def off_geodesic_error(C, A, B, grid=4001, refine=True):
    """inf_{tau in [0,1]} d_BW(C, Gamma_tau(A,B)).
    High-precision: dense grid + local golden refine. Returns (err, tau*)."""
    taus = np.linspace(0.0, 1.0, grid)
    d = np.array([bw_distance(C, bw_geodesic(A, B, t)) for t in taus])
    k = int(np.argmin(d))
    best_t, best_d = taus[k], d[k]
    if refine:
        lo = taus[max(k - 1, 0)]
        hi = taus[min(k + 1, grid - 1)]
        gr = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1 = b - gr * (b - a); c2 = a + gr * (b - a)
        f1 = bw_distance(C, bw_geodesic(A, B, c1))
        f2 = bw_distance(C, bw_geodesic(A, B, c2))
        for _ in range(80):
            if f1 < f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - gr * (b - a)
                f1 = bw_distance(C, bw_geodesic(A, B, c1))
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + gr * (b - a)
                f2 = bw_distance(C, bw_geodesic(A, B, c2))
        tm = 0.5 * (a + b)
        dm = bw_distance(C, bw_geodesic(A, B, tm))
        if dm < best_d:
            best_t, best_d = tm, dm
    return float(best_d), float(best_t)


# --------------------------------------------------------------------------
# subspace angle (independent judge for rotation setting)
# --------------------------------------------------------------------------
def principal_subspace(C, r):
    """Top-r eigenvector matrix (columns) of SPD C, descending eigenvalue."""
    w, V = np.linalg.eigh(sym(C))
    idx = np.argsort(w)[::-1]
    return V[:, idx[:r]]


def sin_theta_fro(U, Uref):
    """||sin Theta(U, Uref)||_F between two orthonormal bases (same #cols)."""
    # principal angles via SVD of U^T Uref
    s = np.linalg.svd(U.T @ Uref, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    sin2 = np.clip(1.0 - s ** 2, 0.0, 1.0)
    return float(np.sqrt(sin2.sum()))


# --------------------------------------------------------------------------
# Nearest Kronecker product for a covariance with imperfect factorization.
# --------------------------------------------------------------------------
def nearest_kron(Sigma, m, n):
    """Van Loan-Pitsianis: best Frobenius rank-1 Kronecker approx
    Sigma ~ R (x) L, R:n×n, L:m×m, with Sigma of size mn×mn.
    Convention here: kron(R, L) with vec index = i_row_block=n-block.
    Returns (L, R, residual_fro). L,R symmetrized; not guaranteed SPD but
    projected to SPD via eigen-floor for downstream use."""
    Sigma = np.asarray(Sigma, float)
    # Rearrangement R(Sigma): reshape so that ||Sigma - kron(R,L)||_F =
    # ||Rearr - vec(R) vec(L)^T||_F. Use pitsianis rearrangement.
    Rearr = np.zeros((n * n, m * m))
    for i in range(n):
        for j in range(n):
            block = Sigma[i * m:(i + 1) * m, j * m:(j + 1) * m]  # m×m
            Rearr[i * n + j, :] = block.reshape(-1)
    U, s, Vt = np.linalg.svd(Rearr, full_matrices=False)
    sigma1 = s[0]
    r_vec = np.sqrt(sigma1) * U[:, 0]
    l_vec = np.sqrt(sigma1) * Vt[0, :]
    R = r_vec.reshape(n, n)
    L = l_vec.reshape(m, m)
    # fix sign so both have positive trace
    if np.trace(L) < 0:
        L, R = -L, -R
    L, R = sym(L), sym(R)
    resid = float(np.sqrt(max(np.sum(s[1:] ** 2), 0.0)))
    return L, R, resid


def project_spd(A, floor=1e-10):
    w, V = np.linalg.eigh(sym(A))
    w = np.clip(w, floor, None)
    return sym((V * w) @ V.T)
