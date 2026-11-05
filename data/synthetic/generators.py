"""
Synthetic SPD-dynamics generators for BuresMomentum README §10.

Each generator produces a *sequence* of instantaneous SPD statistics S_t
(the observations fed to an optimizer's covariance update) AND a companion
population ground-truth path C*_t computed by an offline high-precision routine.

Ground truth is the referee for G1/G2. Generators are seed-parameterized so
synthetic experiments may use multiple independent seeds (validate scaling
laws), while formal DL tasks stay fixed at seed=42.

All arrays float64. A generator returns a dict:
  {
    "name", "d", "T", "params", "seed",
    "S":      (T, d, d)   instantaneous SPD observations,
    "Cstar":  (T, d, d)   population ground-truth SPD path (None if not defined),
    ... setting-specific extra keys (e.g. tau for geodesic path) ...
  }
"""
import numpy as np
from bw_geom import bw_geodesic, sym, project_spd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def random_spd(d, rng, lo=0.5, hi=3.0):
    """Random SPD with eigenvalues in [lo,hi] and Haar-random eigenvectors."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = rng.uniform(lo, hi, size=d)
    return sym((Q * lam) @ Q.T)


def random_skew(d, rng, scale=1.0):
    """Random skew-symmetric generator Omega (Omega^T = -Omega)."""
    X = rng.standard_normal((d, d))
    W = X - X.T
    W = W / (np.linalg.norm(W) + 1e-30)
    return scale * W


def expm_skew(Omega, t):
    """exp(t*Omega) via eigen of skew (returns orthogonal). Uses scipy-free
    Rodrigues-style through complex eigendecomposition."""
    from scipy.linalg import expm
    return expm(t * Omega)


def add_wishart_noise(C, rng, dof_scale, eps_stat=1e-6):
    """Draw a noisy instantaneous SPD sample around population C:
    S = (1/dof) sum g g^T  with g ~ N(0,C). Larger dof_scale -> less noise.
    dof = dof_scale * d. Adds eps_stat*I damping (README §2.2)."""
    d = C.shape[0]
    dof = max(int(round(dof_scale * d)), d + 1)
    L = np.linalg.cholesky(sym(C) + 1e-12 * np.eye(d))
    G = L @ rng.standard_normal((d, dof))
    S = (G @ G.T) / dof + eps_stat * np.eye(d)
    return sym(S)


# --------------------------------------------------------------------------
# §10.1 static SPD barycenter
# --------------------------------------------------------------------------
def gen_static(d=16, T=2000, seed=42, dof_scale=8.0, eps_stat=1e-6):
    """Static population covariance C*; observations are noisy Wishart samples.
    Ground truth Cstar is constant. Population barycenter of the sample stream
    equals C* (up to Wishart bias, documented). Offline GT = C* exactly."""
    rng = np.random.default_rng(seed)
    Cstar0 = random_spd(d, rng, 0.5, 3.0)
    S = np.stack([add_wishart_noise(Cstar0, rng, dof_scale, eps_stat) for _ in range(T)])
    Cstar = np.repeat(Cstar0[None], T, axis=0)
    return {"name": "static", "d": d, "T": T, "seed": seed,
            "params": {"dof_scale": dof_scale, "eps_stat": eps_stat},
            "S": S, "Cstar": Cstar}


# --------------------------------------------------------------------------
# §10.2 pure scale change (commuting; fixed eigenbasis)
# --------------------------------------------------------------------------
def gen_scale(d=16, T=2000, seed=42, omega=3e-3, dof_scale=16.0, eps_stat=1e-6,
              noiseless=False):
    """C*_t = Q Lambda_t Q^T, Q fixed, lambda_{i,t}=exp(a_i + b_i sin(omega t + phi_i)).
    Commuting case: validates closed-form Eq §4.3 / §10.2. If noiseless, S=C*."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a = rng.uniform(-0.5, 0.5, d)
    b = rng.uniform(0.2, 0.8, d)
    phi = rng.uniform(0, 2 * np.pi, d)
    Cstar = np.empty((T, d, d))
    S = np.empty((T, d, d))
    for t in range(T):
        lam = np.exp(a + b * np.sin(omega * t + phi))
        C = sym((Q * lam) @ Q.T)
        Cstar[t] = C
        S[t] = C if noiseless else add_wishart_noise(C, rng, dof_scale, eps_stat)
    return {"name": "scale", "d": d, "T": T, "seed": seed,
            "params": {"omega": omega, "dof_scale": dof_scale, "eps_stat": eps_stat,
                       "noiseless": noiseless},
            "Q": Q, "a": a, "b": b, "phi": phi, "S": S, "Cstar": Cstar}


# --------------------------------------------------------------------------
# §10.3 pure rotation (fixed spectrum, rotating eigenvectors)
# --------------------------------------------------------------------------
def gen_rotation(d=16, T=2000, seed=42, omega=1e-3, r=4,
                 dof_scale=16.0, eps_stat=1e-6, noiseless=False,
                 gap_boost=True):
    """C*_t = Q_t Lambda Q_t^T, Q_t = exp(t omega Omega). Fixed spectrum Lambda.
    r = size of principal subspace tracked. gap_boost makes a clear eigengap at r."""
    rng = np.random.default_rng(seed)
    Omega = random_skew(d, rng, scale=1.0)
    if gap_boost:
        top = np.linspace(3.0, 2.0, r)
        bot = np.linspace(0.9, 0.4, d - r)
        lam = np.concatenate([top, bot])
    else:
        lam = np.sort(rng.uniform(0.4, 3.0, d))[::-1].copy()
    Lambda = np.diag(lam)
    Cstar = np.empty((T, d, d))
    S = np.empty((T, d, d))
    Qs = np.empty((T, d, d))
    for t in range(T):
        Qt = expm_skew(Omega, t * omega)
        C = sym(Qt @ Lambda @ Qt.T)
        Qs[t] = Qt
        Cstar[t] = C
        S[t] = C if noiseless else add_wishart_noise(C, rng, dof_scale, eps_stat)
    return {"name": "rotation", "d": d, "T": T, "seed": seed, "r": r,
            "params": {"omega": omega, "r": r, "dof_scale": dof_scale,
                       "eps_stat": eps_stat, "noiseless": noiseless,
                       "gap_boost": gap_boost},
            "Omega": Omega, "lam": lam, "S": S, "Cstar": Cstar, "Q": Qs}


# --------------------------------------------------------------------------
# §10.4 Bures geodesic path
# --------------------------------------------------------------------------
def gen_geodesic(d=16, T=2000, seed=42, dof_scale=32.0, eps_stat=1e-6,
                 noiseless=True, tau_mode="ramp"):
    """C*_t = Gamma_{tau_t}(A,B) on a Bures geodesic between random endpoints A,B.
    tau_mode: 'ramp' (linear 0->1) or 'sin' (oscillating). Records tau_t so the
    Theorem-4 recursion tau_hat_t = beta2 tau_hat_{t-1} + (1-beta2) tau_t can be
    checked. Default noiseless (clean geodesic to test off-geodesic error)."""
    rng = np.random.default_rng(seed)
    A = random_spd(d, rng, 0.5, 2.0)
    B = random_spd(d, rng, 0.5, 2.0)
    if tau_mode == "ramp":
        tau = np.linspace(0.0, 1.0, T)
    elif tau_mode == "sin":
        tau = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(T) / T)
    else:
        raise ValueError(tau_mode)
    Cstar = np.empty((T, d, d))
    S = np.empty((T, d, d))
    for t in range(T):
        C = bw_geodesic(A, B, float(tau[t]))
        Cstar[t] = C
        S[t] = C if noiseless else add_wishart_noise(C, rng, dof_scale, eps_stat)
    return {"name": "geodesic", "d": d, "T": T, "seed": seed,
            "params": {"tau_mode": tau_mode, "dof_scale": dof_scale,
                       "eps_stat": eps_stat, "noiseless": noiseless},
            "A": A, "B": B, "tau": tau, "S": S, "Cstar": Cstar}


# --------------------------------------------------------------------------
# §10.5 abrupt change and recovery
# --------------------------------------------------------------------------
def gen_jump(d=16, T=2000, seed=42, t0_frac=0.5, dof_scale=16.0, eps_stat=1e-6,
             noiseless=False):
    """C*_t = A for t<t0, B for t>=t0. For recovery-step / overshoot analysis."""
    rng = np.random.default_rng(seed)
    A = random_spd(d, rng, 0.5, 2.0)
    B = random_spd(d, rng, 0.5, 2.0)
    t0 = int(t0_frac * T)
    Cstar = np.empty((T, d, d))
    S = np.empty((T, d, d))
    for t in range(T):
        C = A if t < t0 else B
        Cstar[t] = C
        S[t] = C if noiseless else add_wishart_noise(C, rng, dof_scale, eps_stat)
    return {"name": "jump", "d": d, "T": T, "seed": seed,
            "params": {"t0": t0, "t0_frac": t0_frac, "dof_scale": dof_scale,
                       "eps_stat": eps_stat, "noiseless": noiseless},
            "A": A, "B": B, "t0": t0, "S": S, "Cstar": Cstar}


# --------------------------------------------------------------------------
# §10.6 Kronecker misspecification  Sigma = R* (x) L* + gamma E
# --------------------------------------------------------------------------
def gen_kron_misspec(m=8, n=8, seed=42, gamma=0.0):
    """Full mn×mn covariance Sigma = R*(x)L* + gamma*E, E a non-Kronecker SPD
    perturbation (normalized ||E||_F=||R*(x)L*||_F). Static (single instance).
    Returns Sigma plus the true factors L*,R*. Ground-truth nearest-Kron and
    residual eps_kron are computed offline in the runner."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    Lstar = sym((Q * rng.uniform(0.5, 2.5, m)) @ Q.T)
    Qn, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Rstar = sym((Qn * rng.uniform(0.5, 2.5, n)) @ Qn.T)
    base = np.kron(Rstar, Lstar)  # (mn, mn); ordering: R outer, L inner
    # non-Kronecker SPD perturbation
    X = rng.standard_normal((m * n, m * n))
    E = sym(X @ X.T)
    E = E / (np.linalg.norm(E) + 1e-30) * np.linalg.norm(base)
    Sigma = sym(base + gamma * E)
    Sigma = project_spd(Sigma, floor=1e-8)
    return {"name": "kron_misspec", "m": m, "n": n, "seed": seed,
            "params": {"gamma": gamma},
            "Lstar": Lstar, "Rstar": Rstar, "Sigma": Sigma}
