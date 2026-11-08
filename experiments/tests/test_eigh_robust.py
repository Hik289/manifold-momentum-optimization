"""Check eigendecomposition fallbacks on difficult symmetric matrices.

Some GPU eigensolvers fail on matrices with repeated or widely spaced
eigenvalues. The core implementation retries on CPU, adds a small diagonal
offset if needed, and finally uses SVD. This script checks that the matrix
operations remain finite and records which fallbacks were used.
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bures.core import (
    symmetrize, spd_power, spd_sqrt, spd_inv_sqrt, clip_spd, ot_map,
    bw_distance, bures_barycenter_2pt, _robust_eigh, EIGH_FALLBACK_STATS,
)

torch.manual_seed(42)


def killers(d, device, dtype):
    """Build symmetric matrices that can challenge an eigensolver."""
    mats = {}
    I = torch.eye(d, device=device, dtype=dtype)
    # All eigenvalues equal.
    mats["repeated_identity"] = 3.7 * I
    # Almost repeated eigenvalues.
    ev = torch.ones(d, device=device, dtype=dtype)
    ev[-1] = 1.0 + 1e-12
    Q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype))
    mats["near_repeated"] = symmetrize(Q @ torch.diag(ev) @ Q.t())
    # Eigenvalues span roughly 14 orders of magnitude.
    ev2 = torch.logspace(-7, 7, d, device=device, dtype=dtype)
    mats["ill_conditioned"] = symmetrize(Q @ torch.diag(ev2) @ Q.t())
    # Rank-deficient matrix with a small diagonal offset.
    ev3 = torch.zeros(d, device=device, dtype=dtype)
    ev3[: d // 2] = torch.linspace(0.1, 5.0, d // 2, device=device, dtype=dtype)
    mats["rank_deficient"] = symmetrize(Q @ torch.diag(ev3) @ Q.t()) \
        + 1e-6 * I
    # Gram matrix from a gradient with one unusually large column.
    G = torch.randn(d, d, device=device, dtype=dtype)
    G[:, 0] *= 1e6
    mats["exploded_gram"] = symmetrize(G @ G.t()) / d + 1e-6 * I
    return mats


def check_on(device):
    dtype = torch.float32
    results = {}
    for d in (8, 32, 64):
        for name, A in killers(d, device, dtype).items():
            key = f"{name}_d{d}"
            entry = {"eigh_ok": False, "recon_relerr": None,
                     "clip_ok": False, "sqrt_ok": False, "bw_ok": False,
                     "ot_ok": False}
            try:
                evals, evecs = _robust_eigh(A)
                recon = symmetrize((evecs * evals.unsqueeze(-2))
                                   @ evecs.transpose(-1, -2))
                entry["recon_relerr"] = (recon - A).norm().item() \
                    / (A.norm().item() + 1e-30)
                entry["eigh_ok"] = True
            except Exception as e:
                entry["eigh_err"] = str(e)
            try:
                Ac, _ = clip_spd(A, 1e-6, 1e6)
                entry["clip_ok"] = bool(torch.isfinite(Ac).all())
            except Exception as e:
                entry["clip_err"] = str(e)
            try:
                # Make the inputs SPD before checking roots, transport, and BW distance.
                Aspd, _ = clip_spd(A, 1e-6, 1e6)
                Bspd, _ = clip_spd(symmetrize(A + 0.3 * torch.eye(
                    d, device=device, dtype=dtype)), 1e-6, 1e6)
                sq = spd_sqrt(Aspd)
                entry["sqrt_ok"] = bool(torch.isfinite(sq).all())
                T = ot_map(Aspd, Bspd)
                tat = symmetrize(T @ Aspd @ T)
                entry["ot_relerr"] = (tat - Bspd).norm().item() \
                    / (Bspd.norm().item() + 1e-30)
                entry["ot_ok"] = entry["ot_relerr"] < 1e-2
                dbw = bw_distance(Aspd, Bspd)
                entry["bw_ok"] = bool(torch.isfinite(dbw).all()
                                      and dbw.item() >= 0)
            except Exception as e:
                entry["math_err"] = str(e)
            results[key] = entry
    return results


def main():
    out = {"device_results": {}, "fallback_stats": None,
           "cuda_available": torch.cuda.is_available()}
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for dev in devices:
        print(f"\n=== device: {dev} ===")
        res = check_on(dev)
        n_ok = 0
        for k, v in res.items():
            ok = (v["eigh_ok"] and v["clip_ok"] and v["sqrt_ok"]
                  and v["bw_ok"] and v["ot_ok"])
            n_ok += int(ok)
            flag = "OK " if ok else "FAIL"
            print(f"  [{flag}] {k:26s} recon={v.get('recon_relerr')} "
                  f"ot={v.get('ot_relerr')} "
                  f"errs={ {kk: vv for kk, vv in v.items() if kk.endswith('err') and vv} }")
        out["device_results"][dev] = {"n_cases": len(res),
                                      "n_all_ok": n_ok, "detail": res}
        print(f"  {dev}: {n_ok}/{len(res)} cases fully OK")

    out["fallback_stats"] = dict(EIGH_FALLBACK_STATS)
    print(f"\n=== eigh fallback stats: {EIGH_FALLBACK_STATS} ===")

    all_ok = all(d["n_all_ok"] == d["n_cases"]
                 for d in out["device_results"].values())
    out["all_pass"] = all_ok
    with open("tests/eigh_robust_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n=== eigh robust test {'ALL PASS' if all_ok else 'HAS FAILURES'} ===")
    print("written to tests/eigh_robust_results.json")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
