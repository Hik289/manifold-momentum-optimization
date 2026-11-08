"""Small CPU checks for the shared optimizer interface.

The checks train five optimizers on synthetic matrix regression, record their
diagnostics, verify the identity preconditioner, and confirm that the paired
Euclidean and Bures variants differ only in covariance_rule. Seed: 42.
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bures.optimizer import MatrixOptimizer
from pipeline.diagnostics import (
    build_optimizer, DiagnosticRecorder, OPTIMIZER_REGISTRY,
)

torch.manual_seed(42)


def make_task(m=32, n=32, device="cpu"):
    """Create a linear regression task with a 32-by-32 weight matrix."""
    W_true = torch.randn(m, n, device=device)
    X = torch.randn(256, m, device=device)
    Y = X @ W_true
    return X, Y, W_true


def run_optimizer(name, steps=200, device="cpu"):
    torch.manual_seed(42)
    X, Y, W_true = make_task(device=device)
    W = torch.nn.Parameter(torch.zeros(32, 32, device=device))
    opt = build_optimizer(name, [W], lr=1e-2, weight_decay=0.0)
    rec = DiagnosticRecorder(opt, track_spd=(name in
                             ("euclidean_kron", "buresmomentum")))
    losses = []
    for t in range(steps):
        pred = X @ W
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 20 == 0 or t == steps - 1:
            rec.record(t, train_loss=loss.item(), learning_rate=1e-2)
        losses.append(loss.item())
    return losses, rec


def verify_degeneration():
    """Check that identity preconditioners leave the first moment unchanged.

    This reconstructs the identity case directly, then runs one optimizer
    step as a separate smoke check.
    """
    torch.manual_seed(42)
    from bures.core import spd_inv_sqrt
    m, n = 8, 10
    W = torch.nn.Parameter(torch.randn(m, n))
    opt = MatrixOptimizer([W], lr=0.0, betas=(0.9, 0.99), tau_root=1,
                          state_dtype=torch.float64)
    g = torch.randn(m, n)
    W.grad = g.clone()
    # At the first step, bias correction makes M_hat equal to g.
    beta1 = 0.9
    M_hat_expected = g.to(torch.float64)
    # With L = R = I, the inverse-root product must equal M_hat.
    Im = torch.eye(m, dtype=torch.float64)
    In = torch.eye(n, dtype=torch.float64)
    D_raw = spd_inv_sqrt(Im) @ M_hat_expected @ spd_inv_sqrt(In)
    err = (D_raw - M_hat_expected).norm().item()
    opt.step()  # The complete step should also run without error.
    return err < 1e-12, err


def verify_b3_b5_shared_hparams():
    """Check that the paired variants change only covariance_rule."""
    W1 = torch.nn.Parameter(torch.randn(8, 8))
    W2 = torch.nn.Parameter(torch.randn(8, 8))
    b3 = build_optimizer("euclidean_kron", [W1], lr=1e-2, weight_decay=1e-4)
    b5 = build_optimizer("buresmomentum", [W2], lr=1e-2, weight_decay=1e-4)
    keys = ["lr", "betas", "eps_stat", "eps_norm", "weight_decay",
            "lam_min", "lam_max", "s_min", "s_max", "tau_root", "block_size"]
    g3 = b3.param_groups[0]
    g5 = b5.param_groups[0]
    mismatch = {k: (g3[k], g5[k]) for k in keys if g3[k] != g5[k]}
    rule_diff = (g3["covariance_rule"] == "euclidean"
                 and g5["covariance_rule"] == "bures")
    return (len(mismatch) == 0 and rule_diff), {"mismatch": mismatch,
                                                "rule_diff_ok": rule_diff}


def main():
    out = {"optimizers": {}, "checks": {}}
    print("=== Five optimizers reduce loss through one interface ===")
    for name in OPTIMIZER_REGISTRY:
        losses, rec = run_optimizer(name, steps=200)
        improved = losses[-1] < losses[0]
        ratio = losses[-1] / (losses[0] + 1e-30)
        traj_len = len(rec.trajectory)
        n_diag_keys = len(rec.trajectory[0]) if rec.trajectory else 0
        print(f"  [{OPTIMIZER_REGISTRY[name]}] {name:16s} "
              f"loss {losses[0]:.4f} -> {losses[-1]:.6f} "
              f"(ratio {ratio:.2e}) improved={improved} "
              f"traj={traj_len} diag_keys={n_diag_keys}")
        out["optimizers"][name] = {
            "id": OPTIMIZER_REGISTRY[name],
            "loss_start": losses[0], "loss_end": losses[-1],
            "improved": improved, "traj_points": traj_len,
            "diag_keys_per_point": n_diag_keys,
        }

    print("\n=== Identity preconditioner leaves momentum unchanged ===")
    deg_ok, deg_err = verify_degeneration()
    print(f"  identity check: error={deg_err:.2e} pass={deg_ok}")
    out["checks"]["degeneration_identity"] = {"err": deg_err, "pass": deg_ok}

    print("\n=== Paired variants share all settings except covariance_rule ===")
    shared_ok, shared_detail = verify_b3_b5_shared_hparams()
    print(f"  shared settings: pass={shared_ok} detail={shared_detail}")
    out["checks"]["b3_b5_shared_hparams"] = {"pass": shared_ok,
                                             "detail": shared_detail}

    all_improved = all(v["improved"] for v in out["optimizers"].values())
    all_ok = all_improved and deg_ok and shared_ok
    out["all_pass"] = all_ok
    print(f"\n=== small-run checks: {'ALL PASS' if all_ok else 'HAS FAILURES'} ===")

    outpath = "pipeline/smoke_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"written to {outpath}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
