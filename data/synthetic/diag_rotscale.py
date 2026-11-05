"""Probe: does Bures beat Euclidean EMA when rotation is COUPLED with scale
oscillation? Pure rotation (fixed spectrum) gives Bures no edge; the hypothesized
advantage (std-vs-var interpolation) needs changing eigenvalues. This builds a
combined generator and reports time-avg BW tracking error for both updaters."""
import json, numpy as np
import bw_geom as G, generators as GEN
from build_ground_truth import run_bures, run_euclidean_ema, time_avg_bw_tracking
from scipy.linalg import expm

b2 = 0.99
def gen_rot_scale(d=16, T=1500, seed=42, omega=3e-3, scale_amp=0.8,
                  scale_omega=None, dof_scale=16.0, eps_stat=1e-6, noiseless=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, d)); Om = X - X.T; Om /= np.linalg.norm(Om)
    a = rng.uniform(-0.3, 0.3, d); phi = rng.uniform(0, 2*np.pi, d)
    sw = omega * 3 if scale_omega is None else scale_omega
    S = np.empty((T, d, d)); Cstar = np.empty((T, d, d))
    for t in range(T):
        Qt = expm(t * omega * Om)
        lam = np.exp(a + scale_amp * np.sin(sw * t + phi))
        C = G.sym(Qt @ np.diag(lam) @ Qt.T)
        Cstar[t] = C
        S[t] = C if noiseless else GEN.add_wishart_noise(C, rng, dof_scale, eps_stat)
    return S, Cstar

out = {}
for w in [1e-3, 3e-3, 1e-2]:
  for noiseless in (True, False):
    S, Cstar = gen_rot_scale(omega=w, noiseless=noiseless)
    bur = run_bures(S, b2); euc = run_euclidean_ema(S, b2)
    te_b,_ = time_avg_bw_tracking(bur, Cstar)
    te_e,_ = time_avg_bw_tracking(euc, Cstar)
    tag = "clean" if noiseless else "noisy"
    rel = (te_e-te_b)/(te_e+1e-30)
    out[f"{w:.0e}_{tag}"] = {"omega": w, "noiseless": noiseless,
        "bw_euclidean": te_b*0+te_e, "bw_bures": te_b, "rel_improvement": rel,
        "bures_better": bool(te_b<te_e)}
    print(f"w={w:.0e} {tag:5s} euc={te_e:.5f} bur={te_b:.5f} rel={rel:+.4f} better={te_b<te_e}")

json.dump(out, open("out/rotscale_probe.json","w"), indent=2)
print("any_better", any(v["bures_better"] for v in out.values()))
print("WROTE out/rotscale_probe.json")
