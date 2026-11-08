"""
Build all synthetic SPD-dynamics settings + high-precision offline ground truth,
and evaluate the G1/G2 quantitative verdict protocol.

Outputs (to --outdir, default ./out):
  <setting>_seed<k>.npz         S, Cstar, and setting metadata arrays
  ground_truth_summary.json     self-consistency (anchor_4) + G1/G2 referee numbers

This is the JUDGE. The reference covariance updaters below (Euclidean EMA and
Bures barycenter) are minimal NumPy re-implementations used ONLY to produce the
referee numbers for G1/G2; the ML engineer's optimizer is validated separately
against these same offline ground truths.

G1/G2 quantitative protocol (matches checklist 1.4):
  G1 (geodesic): time-averaged off-geodesic BW error of the Bures state on a
      clean Bures-geodesic path must be < 1e-4; AND Theorem-4 recursion residual
      max_t |tau_hat_t - (b2 tau_hat_{t-1} + (1-b2) tau_t)| < 1e-4.
  G2 (rotation): time-averaged BW tracking error (1/T) sum d_BW(C_t, C*_t) of the
      Bures updater must be strictly < that of the Euclidean-EMA updater, in at
      least one omega in the sweep. We report both + relative improvement.
"""
import argparse, json, os
import numpy as np

import bw_geom as G
import generators as GEN


# --------------------------------------------------------------------------
# reference covariance updaters (independent numpy referees)
# --------------------------------------------------------------------------
def run_euclidean_ema(S, beta2, C0=None):
    """C_t = beta2 C_{t-1} + (1-beta2) S_t."""
    T = S.shape[0]
    C = S[0].copy() if C0 is None else C0.copy()
    out = np.empty_like(S)
    for t in range(T):
        C = beta2 * C + (1.0 - beta2) * S[t]
        out[t] = G.sym(C)
    return out


def run_bures(S, beta2, C0=None):
    """C_t = Bar_alpha(C_{t-1}, S_t), alpha = 1 - beta2 (two-point barycenter)."""
    alpha = 1.0 - beta2
    T = S.shape[0]
    C = S[0].copy() if C0 is None else C0.copy()
    out = np.empty_like(S)
    for t in range(T):
        C = G.bw_geodesic(C, S[t], alpha)
        out[t] = G.sym(C)
    return out


def time_avg_bw_tracking(states, Cstar):
    d = np.array([G.bw_distance(states[t], Cstar[t]) for t in range(len(states))])
    return float(d.mean()), d


# --------------------------------------------------------------------------
# anchor_4 self-consistency checks on the ground truth
# --------------------------------------------------------------------------
def check_geodesic_constant_speed(A, B, n=50):
    """d_BW(Gamma_s, Gamma_t) = |s-t| d_BW(A,B). Return max abs residual."""
    dab = G.bw_distance(A, B)
    ss = np.linspace(0, 1, n)
    res = 0.0
    for i in range(n):
        for j in range(i, n):
            lhs = G.bw_distance(G.bw_geodesic(A, B, ss[i]), G.bw_geodesic(A, B, ss[j]))
            rhs = abs(ss[i] - ss[j]) * dab
            res = max(res, abs(lhs - rhs))
    return res


def check_barycenter_frechet(mats, weights=None):
    B, info = G.bures_barycenter(mats, weights)
    return B, info["frechet_residual"], info["iters"]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--save_npz", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    b2 = args.beta2
    summary = {"config": {"d": args.d, "T": args.T, "beta2": b2,
                          "seeds": args.seeds}, "settings": {}, "anchor4": {},
               "G1": {}, "G2": {}}

    # ---- anchor_4: offline GT self-consistency (seed 42) --------------------
    rng = np.random.default_rng(0)
    A = GEN.random_spd(args.d, rng); B = GEN.random_spd(args.d, rng)
    cs_res = check_geodesic_constant_speed(A, B, n=40)
    mats = [GEN.random_spd(args.d, rng) for _ in range(5)]
    Bbar, fr_res, fr_it = check_barycenter_frechet(mats)
    # barycenter vs 2-pt geodesic midpoint sanity (equal weights, 2 mats)
    Bpair, info2 = G.bures_barycenter([A, B])
    mid = G.bw_geodesic(A, B, 0.5)
    pair_vs_geo = float(np.linalg.norm(Bpair - mid) / (np.linalg.norm(mid) + 1e-30))
    summary["anchor4"] = {
        "geodesic_constant_speed_max_resid": cs_res,
        "barycenter_frechet_residual": fr_res,
        "barycenter_iters": fr_it,
        "twopt_barycenter_vs_geodesic_mid_relerr": pair_vs_geo,
        "pass": bool(cs_res < 1e-6 and fr_res < 1e-10 and pair_vs_geo < 1e-10),
    }

    # ---- per-setting generation + GT ---------------------------------------
    for seed in args.seeds:
        # static
        st = GEN.gen_static(d=args.d, T=args.T, seed=seed)
        _save(args, st, seed)
        summary["settings"].setdefault("static", {})[str(seed)] = _meta(st)

        # scale (commuting closed-form check)
        sc = GEN.gen_scale(d=args.d, T=args.T, seed=seed, noiseless=True)
        _save(args, sc, seed)
        cf = _scale_closedform_residual(sc, b2)
        summary["settings"].setdefault("scale", {})[str(seed)] = {**_meta(sc),
            "closed_form_bures_max_relerr": cf}

        # geodesic (G1) - expensive off-geodesic eval only on formal seed 42
        ge = GEN.gen_geodesic(d=args.d, T=args.T, seed=seed, noiseless=True,
                              tau_mode="ramp")
        _save(args, ge, seed)
        summary["settings"].setdefault("geodesic", {})[str(seed)] = _meta(ge)
        if seed == 42:
            summary["G1"][str(seed)] = _eval_G1(ge, b2)

        # jump
        jp = GEN.gen_jump(d=args.d, T=args.T, seed=seed, noiseless=False)
        _save(args, jp, seed)
        summary["settings"].setdefault("jump", {})[str(seed)] = _meta(jp)

    # ---- G2: rotation sweep (seed 42) --------------------------------------
    omegas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    g2 = {}
    for w in omegas:
      for noiseless in (True, False):
        rot = GEN.gen_rotation(d=args.d, T=args.T, seed=42, omega=w,
                               noiseless=noiseless)
        if not noiseless:
            _save(args, rot, 42, tag=f"omega{w:.0e}")
        euc = run_euclidean_ema(rot["S"], b2)
        bur = run_bures(rot["S"], b2)
        te_e, _ = time_avg_bw_tracking(euc, rot["Cstar"])
        te_b, _ = time_avg_bw_tracking(bur, rot["Cstar"])
        # subspace angle time-avg (r principal dirs)
        r = rot["r"]
        sa_e = np.mean([G.sin_theta_fro(G.principal_subspace(euc[t], r),
                        G.principal_subspace(rot["Cstar"][t], r)) for t in range(args.T)])
        sa_b = np.mean([G.sin_theta_fro(G.principal_subspace(bur[t], r),
                        G.principal_subspace(rot["Cstar"][t], r)) for t in range(args.T)])
        key = f"{w:.0e}" + ("_clean" if noiseless else "_noisy")
        g2[key] = {
            "omega": w, "noiseless": noiseless,
            "bw_track_euclidean": te_e, "bw_track_bures": te_b,
            "rel_improvement": float((te_e - te_b) / (te_e + 1e-30)),
            "bures_better": bool(te_b < te_e),
            "subspace_angle_euclidean": float(sa_e),
            "subspace_angle_bures": float(sa_b),
        }
    clean = {k: v for k, v in g2.items() if v["noiseless"]}
    noisy = {k: v for k, v in g2.items() if not v["noiseless"]}
    summary["G2"] = {
        "sweep": g2,
        "bures_better_in_any_omega_clean": bool(any(v["bures_better"] for v in clean.values())),
        "bures_better_in_any_omega_noisy": bool(any(v["bures_better"] for v in noisy.values())),
        "bures_better_count_clean": int(sum(v["bures_better"] for v in clean.values())),
        "bures_better_count_noisy": int(sum(v["bures_better"] for v in noisy.values())),
    }

    # ---- Kronecker misspecification sweep (offline eps_kron) ----------------
    km = {}
    for gamma in [0.0, 0.01, 0.03, 0.1, 0.3]:
        d = GEN.gen_kron_misspec(m=8, n=8, seed=42, gamma=gamma)
        L, R, resid = G.nearest_kron(d["Sigma"], 8, 8)
        eps_kron = float(np.linalg.norm(d["Sigma"] - np.kron(G.project_spd(R),
                                                             G.project_spd(L))))
        km[f"{gamma}"] = {"gamma": gamma, "nearest_kron_resid_fro": resid,
                          "eps_kron_fro": eps_kron,
                          "sigma_fro": float(np.linalg.norm(d["Sigma"]))}
    summary["kron_misspec"] = km

    with open(os.path.join(args.outdir, "ground_truth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:4000])
    print("\nWROTE", os.path.join(args.outdir, "ground_truth_summary.json"))


# --------------------------------------------------------------------------
def _meta(s):
    return {k: (v if np.isscalar(v) else None)
            for k, v in s.get("params", {}).items()} | {"d": s.get("d"),
            "m": s.get("m"), "n": s.get("n"), "T": s.get("T")}


def _save(args, s, seed, tag=None):
    if not args.save_npz:
        return
    name = s["name"] + (f"_{tag}" if tag else "") + f"_seed{seed}.npz"
    arrs = {k: v for k, v in s.items()
            if isinstance(v, np.ndarray)}
    np.savez_compressed(os.path.join(args.outdir, name), **arrs)


def _scale_closedform_residual(sc, b2):
    """Check the closed-form update for a noiseless path with shared eigenvectors:
    lambda_i,t^BW = [b2 sqrt(lam_{i,t-1}^BW) + (1-b2) sqrt(s_{i,t})]^2 in shared basis Q."""
    Q = sc["Q"]; S = sc["S"]; T = sc["T"]
    # project S into Q basis -> diagonal entries are eigenvalues s_i,t
    s_diag = np.einsum("ki,tkl,lj->tij", Q, S, Q)
    s_eig = np.diagonal(s_diag, axis1=1, axis2=2)  # (T,d)
    # run bures updater, project into Q basis, compare to scalar recursion
    bur = run_bures(S, b2)
    b_diag = np.einsum("ki,tkl,lj->tij", Q, bur, Q)
    b_eig = np.diagonal(b_diag, axis1=1, axis2=2)  # (T,d)
    # scalar closed form
    c = s_eig[0].copy()
    maxrel = 0.0
    for t in range(T):
        c = (b2 * np.sqrt(np.clip(c, 0, None)) + (1 - b2) * np.sqrt(np.clip(s_eig[t], 0, None))) ** 2
        rel = np.max(np.abs(c - b_eig[t]) / (np.abs(b_eig[t]) + 1e-12))
        maxrel = max(maxrel, rel)
    return float(maxrel)


def _eval_G1(ge, b2):
    """off-geodesic error of Bures state vs Euclidean state on clean geodesic +
    Theorem-4 tau recursion residual."""
    A, B = ge["A"], ge["B"]; S = ge["S"]; tau = ge["tau"]; T = ge["T"]
    bur = run_bures(S, b2)
    euc = run_euclidean_ema(S, b2)
    # single off-geodesic call per step returns (err, tau*); coarse grid=301 +
    # golden refine gives full precision at ~4x less eigendecomp cost.
    ob = [G.off_geodesic_error(bur[t], A, B, grid=301) for t in range(T)]
    og_b = np.array([e for e, _ in ob])
    tau_hat = np.array([tt for _, tt in ob])
    og_e = np.array([G.off_geodesic_error(euc[t], A, B, grid=301)[0] for t in range(T)])
    # Theorem-4: tau_hat_t = b2 tau_hat_{t-1} + (1-b2) tau_t
    th = tau[0]
    thm4_res = 0.0
    for t in range(T):
        th = b2 * th + (1 - b2) * tau[t]
        thm4_res = max(thm4_res, abs(th - tau_hat[t]))
    return {
        "offgeo_bures_timeavg": float(og_b.mean()),
        "offgeo_bures_max": float(og_b.max()),
        "offgeo_euclidean_timeavg": float(og_e.mean()),
        "offgeo_euclidean_max": float(og_e.max()),
        "theorem4_tau_recursion_max_resid": float(thm4_res),
        "G1_pass": bool(og_b.mean() < 1e-4 and thm4_res < 1e-4),
    }


if __name__ == "__main__":
    main()
