#!/usr/bin/env python
"""
Stage2-P0 任务1: A1 消融 — 坐实 G4 (debug_move=INTERVENE)

科学问题: Cora 上 B5>B3 的 +13pp 优势, 是否真来自 Bures 几何本身?
干预: 在完全相同的 M/β1/β2/ε/matrix-root实现/更新频率/block/clip/norm-match 下,
      **只把协方差平均规则从 Bures barycenter (B5) 换成欧氏 EMA (=A1)**。
      A1 就是 B3, 但作为受控消融跑, 显式确认唯一变量 = covariance_rule。

期望:
  A1(欧氏) 使 +13pp 消失 (A1 test_acc ≈ B3 << B5) → 优势归因几何本身, G4 CONFIRMED。
  若 A1 test_acc 仍 ≈ B5 → 优势另有来源。

设计: 同 seed=42 同数据同初始化, 只跑三个受控臂:
  - B5_bures:      MatrixOptimizer(covariance_rule="bures")
  - A1_euclidean:  MatrixOptimizer(covariance_rule="euclidean")  [= B3]
  - (可选) 打印二者超参 diff 确认唯一变量

复用 train_graph 的 SmallGCN + 数据管线, 但用独立 driver 保证配对干净 + 记录
"唯一变量"审计。输出 stage2/results_A1_ablation.json。
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


def mk_opt(rule, params):
    return build_optimizer(
        "buresmomentum" if rule == "bures" else "euclidean_kron", params,
        lr=HP["lr"], weight_decay=HP["weight_decay"],
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


def run_arm(arm, rule, dataset, data, args, device):
    set_seed(args.seed)  # 同 seed 同初始化 => 唯一变量 = covariance_rule
    model = SmallGCN(dataset.num_features, 64, dataset.num_classes).to(device)
    opt = mk_opt(rule, model.parameters())
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
            print(f"[A1] {args.task} {arm}({rule}) epoch={epoch} "
                  f"loss={loss.item():.4f} val={m['val_acc']:.4f} "
                  f"test={m['test_acc']:.4f}", flush=True)
    best = max(history, key=lambda r: r["val_acc"])
    final = history[-1]
    return {"arm": arm, "rule": rule, "best": best, "final": final,
            "history": history, "diagnostics": rec.trajectory,
            "opt_hparams": {k: opt.param_groups[0].get(k) for k in
                            ("lr", "betas", "eps_stat", "eps_norm",
                             "weight_decay", "lam_min", "lam_max", "s_min",
                             "s_max", "tau_root", "block_size",
                             "covariance_rule")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Cora")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--diag-interval", type=int, default=25)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--output",
                    default=str(SCRIPT_DIR / "results_A1_ablation.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    dataset, data = load_data(args.task, args.data_dir)
    data = data.to(device)

    arms = {}
    for arm, rule in [("B5_bures", "bures"), ("A1_euclidean", "euclidean")]:
        arms[arm] = run_arm(arm, rule, dataset, data, args, device)

    # 唯一变量审计: 两臂超参 diff (应只有 covariance_rule)
    h5 = arms["B5_bures"]["opt_hparams"]
    h1 = arms["A1_euclidean"]["opt_hparams"]
    diff = {k: (h5[k], h1[k]) for k in h5 if h5[k] != h1[k]}

    b5_test = arms["B5_bures"]["best"]["test_acc"]
    a1_test = arms["A1_euclidean"]["best"]["test_acc"]
    gap = b5_test - a1_test
    summary = {
        "task": args.task, "seed": args.seed, "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "B5_bures_best_test": b5_test,
        "A1_euclidean_best_test": a1_test,
        "B5_minus_A1_test_pp": gap * 100,
        "single_variable_audit": {
            "hparam_diff": diff,
            "only_covariance_rule": (set(diff.keys()) == {"covariance_rule"}),
        },
        "G4_verdict": (
            "CONFIRMED (优势来自几何)" if gap > 0.03 and
            set(diff.keys()) == {"covariance_rule"}
            else "NOT_CONFIRMED (优势未消失或变量不唯一)"),
        "arms": arms,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[A1 DONE] B5={b5_test:.4f} A1(euclid)={a1_test:.4f} "
          f"gap={gap*100:.2f}pp only_cov_rule="
          f"{summary['single_variable_audit']['only_covariance_rule']} "
          f"=> {summary['G4_verdict']}", flush=True)
    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
