"""Mechanistic decomposition of the rotation-setting tracking error.

Splits time-avg d_BW(C_t, C*_t) into eigenvalue-tracking vs eigenvector(subspace)
tracking for Euclidean-EMA vs Bures-barycenter updaters, to explain WHY Bures
does not beat Euclidean on pure rotation. Clean (noiseless) to isolate geometry.
"""
import json, numpy as np
import bw_geom as G, generators as GEN
from build_ground_truth import run_bures, run_euclidean_ema

b2 = 0.99
out = {}
for w in [1e-4, 1e-3, 1e-2]:
    rot = GEN.gen_rotation(d=16, T=1500, seed=42, omega=w, noiseless=True)
    S, Cstar, r = rot["S"], rot["Cstar"], rot["r"]
    lam_true = rot["lam"]  # fixed spectrum
    bur = run_bures(S, b2); euc = run_euclidean_ema(S, b2)
    def decomp(states):
        bw = []; eigerr = []; sub = []
        spread = []
        for t in range(len(states)):
            bw.append(G.bw_distance(states[t], Cstar[t]))
            ev = np.sort(np.linalg.eigvalsh(G.sym(states[t])))[::-1]
            eigerr.append(np.linalg.norm(ev - np.sort(lam_true)[::-1]))
            spread.append(ev.max() - ev.min())
            sub.append(G.sin_theta_fro(G.principal_subspace(states[t], r),
                                       G.principal_subspace(Cstar[t], r)))
        return (float(np.mean(bw)), float(np.mean(eigerr)),
                float(np.mean(sub)), float(np.mean(spread)))
    bw_b, eig_b, sub_b, spr_b = decomp(bur)
    bw_e, eig_e, sub_e, spr_e = decomp(euc)
    out[f"{w:.0e}"] = {
        "omega": w, "true_spread": float(lam_true.max() - lam_true.min()),
        "bures":     {"bw": bw_b, "eig_err": eig_b, "subspace": sub_b, "spread": spr_b},
        "euclidean": {"bw": bw_e, "eig_err": eig_e, "subspace": sub_e, "spread": spr_e},
    }
    print(f"w={w:.0e} | BW  bur={bw_b:.4f} euc={bw_e:.4f} | "
          f"eig_err bur={eig_b:.4f} euc={eig_e:.4f} | "
          f"subspace bur={sub_b:.4f} euc={sub_e:.4f} | "
          f"spread(true={lam_true.max()-lam_true.min():.2f}) bur={spr_b:.2f} euc={spr_e:.2f}")

json.dump(out, open("out/rotation_decomp.json", "w"), indent=2)
print("WROTE out/rotation_decomp.json")
