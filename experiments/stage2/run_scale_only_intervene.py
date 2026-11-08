#!/usr/bin/env python
"""Compare full BW, scale-only BW, and Euclidean averaging on Cora.

The three runs use the same seed, model, data, and optimizer settings except
for covariance_rule. The scale-only rule keeps the old eigenvectors and
averages their scales. The script records validation-selected accuracy,
early accuracy targets, and average early validation accuracy. Its threshold
labels summarize this single-seed comparison; they are not statistical tests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
for p in (str(SCRIPT_DIR), str(ROOT), str(ROOT / "stage1")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import SmallGCN  # noqa: E402
from pipeline.diagnostics import DiagnosticRecorder, build_optimizer  # noqa

TASK_NAMES = {"Cora": "Cora", "Citeseer": "CiteSeer", "PubMed": "PubMed"}
HP = {"lr": 0.01, "beta1": 0.9, "beta2": 0.99, "weight_decay": 1e-4,
      "eps_stat": 1e-6, "eps_norm": 1e-12, "lam_min": 1e-6, "lam_max": 1e6,
      "s_min": 0.1, "s_max": 10.0, "tau_root": 10, "block_size": 256}
ARMS = [("bures", "buresmomentum"), ("scale_only", "scale_only"),
        ("euclidean", "euclidean_kron")]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_data(task, data_dir):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    name = TASK_NAMES[task]
    ds = Planetoid(root=str(Path(data_dir) / "graph" / name), name=name,
                   transform=NormalizeFeatures())
    return ds, ds[0]


def mk_opt(opt_name, params):
    return build_optimizer(
        opt_name, params, lr=HP["lr"], weight_decay=HP["weight_decay"],
        betas=(HP["beta1"], HP["beta2"]), eps_stat=HP["eps_stat"],
        eps_norm=HP["eps_norm"], lam_min=HP["lam_min"], lam_max=HP["lam_max"],
        s_min=HP["s_min"], s_max=HP["s_max"], tau_root=HP["tau_root"],
        block_size=HP["block_size"])


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    out = {}
    for split, mask in [("train_acc", data.train_mask),
                        ("val_acc", data.val_mask),
                        ("test_acc", data.test_mask)]:
        out[split] = float((pred[mask] == data.y[mask]).float().mean().item())
    return out


def run_arm(arm, opt_name, dataset, data, args, device):
    set_seed(args.seed)  # Pair initialization across averaging rules.
    model = SmallGCN(dataset.num_features, 64, dataset.num_classes).to(device)
    opt = mk_opt(opt_name, model.parameters())
    rec = DiagnosticRecorder(opt, track_spd=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        m = evaluate(model, data)
        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                        "val_acc": m["val_acc"], "test_acc": m["test_acc"]})
        if epoch % args.diag_interval == 0 or epoch == args.epochs:
            rec.record(epoch, train_loss=float(loss.item()),
                       val_metric=m["val_acc"], learning_rate=HP["lr"])
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"[INT] {args.task} {arm}({opt_name}) ep={epoch} "
                  f"loss={loss.item():.4f} val={m['val_acc']:.4f} "
                  f"test={m['test_acc']:.4f}", flush=True)
    return history, rec.trajectory, opt.param_groups[0].get("covariance_rule")


def epoch_to_target(history, target, key="test_acc"):
    for r in history:
        if r[key] >= target:
            return r["epoch"]
    return None


def val_auc(history, upto):
    v = [r["val_acc"] for r in history if r["epoch"] <= upto]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Cora")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--diag-interval", type=int, default=25)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--early-window", type=int, default=100)
    ap.add_argument("--targets", type=str, default="0.70,0.75,0.78,0.80")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--output",
                    default=str(SCRIPT_DIR / "results_scale_only_intervene.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    dataset, data = load_data(args.task, args.data_dir)
    data = data.to(device)
    targets = [float(x) for x in args.targets.split(",")]

    arms = {}
    rules = {}
    for arm, opt_name in ARMS:
        hist, diag, rule = run_arm(arm, opt_name, dataset, data, args, device)
        best = max(hist, key=lambda r: r["val_acc"])
        arms[arm] = {
            "opt_name": opt_name, "covariance_rule": rule,
            "best": best, "final": hist[-1],
            "best_test_at_best_val": best["test_acc"],
            "early_epoch_to_target": {
                f"{t}": epoch_to_target(hist, t) for t in targets},
            "val_auc_early": val_auc(hist, args.early_window),
            "history": hist, "diagnostics": diag,
        }
        rules[arm] = rule

    # Compare how much of the BW-versus-Euclidean gap scale-only recovers.
    bt = arms["bures"]["best_test_at_best_val"]
    st = arms["scale_only"]["best_test_at_best_val"]
    et = arms["euclidean"]["best_test_at_best_val"]
    bures_gain = bt - et            # Full BW minus Euclidean.
    scale_gain = st - et            # Scale-only minus Euclidean.
    frac = (scale_gain / bures_gain) if abs(bures_gain) > 1e-6 else None

    # Record the first epoch that reaches 0.78 accuracy.
    def e2t(a):
        return arms[a]["early_epoch_to_target"].get("0.78")
    verdict = "INCONCLUSIVE"
    if frac is not None:
        if frac >= 0.6:
            verdict = "SCALE_MECHANISM (scale-only recovers most of the gap)"
        elif frac <= 0.3:
            verdict = "ROTATION_MECHANISM (scale-only recovers little of the gap)"
        else:
            verdict = "MIXED (scale-only recovers part of the gap)"

    summary = {
        "task": args.task, "seed": args.seed, "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "arm_rules": rules,
        "best_test": {"bures": bt, "scale_only": st, "euclidean": et},
        "bures_gain_vs_euclidean_pp": bures_gain * 100,
        "scale_only_gain_vs_euclidean_pp": scale_gain * 100,
        "scale_reproduces_fraction": frac,
        "epoch_to_0.78": {a: e2t(a) for a in ("bures", "scale_only",
                                              "euclidean")},
        "mechanism_verdict": verdict,
        "arms": arms,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INTERVENE DONE] bures={bt:.4f} scale_only={st:.4f} "
          f"euclid={et:.4f}", flush=True)
    print(f"  bures_gain={bures_gain*100:.2f}pp "
          f"scale_gain={scale_gain*100:.2f}pp "
          f"frac={frac} => {verdict}", flush=True)
    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
