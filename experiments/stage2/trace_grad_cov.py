#!/usr/bin/env python
"""
Stage2-P0 任务2: Cora population 梯度协方差谱漂移 TRACE (debug_move=TRACE)

科学问题 (对应 insight_scale 升到数据层):
  Stage1 的 TRACE 在优化器状态层 (κ_L≈1 + 尺度收缩)。这里测 **population 梯度
  协方差本身** 的谱轨迹, 确认真实梯度协方差在 Cora 训练中是否经历尺度漂移
  (特征值量级变化) 而非子空间旋转主导。

方法:
  - 用 B5(BuresMomentum) 正常训练 Cora (与主结果同配置, seed=42)
  - 沿训练在若干 checkpoint epoch, 对某个 2D 权重层 W (m×n) 计算 **per-sample
    梯度** 的行空间协方差 population 估计:
        对该层, 取 train 节点逐样本 loss 的梯度 G_i (m×n),
        Σ_L = (1/N) Σ_i (G_i G_i^T)/n   ∈ S^m  (行空间, 与 L 状态对应)
    然后对 Σ_L 做特征分解, 记录:
        - 特征值谱 (排序, 完整) → 看尺度漂移
        - λ_max/λ_min, trace, 主子空间 (top-k 特征向量)
        - 相邻 checkpoint 的主子空间 principal angle (sinΘ) → 看旋转
        - 相邻 checkpoint 的特征值比 (尺度变化幅度)
  - 判据:
        尺度漂移显著 (特征值量级随训练明显变化) + 子空间角度小 (旋转弱)
        → insight_scale 数据层坐实 (尺度主导, 非旋转)

per-sample 梯度用 functorch/torch.func.vmap+grad (torch 2.1 有 torch.func)。
若不可用则退化为 mini-batch 分组近似。
seed=42. 输出 stage2/results_trace_gradcov.json。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]  # experiments/
for p in (str(SCRIPT_DIR), str(ROOT), str(ROOT / "stage1")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import SmallGCN  # noqa: E402
from bures.core import symmetrize, _robust_eigvalsh, _robust_eigh  # noqa: E402
from pipeline.diagnostics import build_optimizer  # noqa: E402

TASK_NAMES = {"Cora": "Cora", "Citeseer": "CiteSeer", "PubMed": "PubMed"}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(task, data_dir):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    name = TASK_NAMES[task]
    ds = Planetoid(root=str(Path(data_dir) / "graph" / name), name=name,
                   transform=NormalizeFeatures())
    return ds, ds[0]


def principal_angles(U1, U2):
    """两组正交基 (m×k) 的 principal angles sinΘ 的 Frobenius 范数."""
    # U1,U2 列正交
    M = U1.transpose(-1, -2) @ U2
    s = torch.linalg.svdvals(M).clamp(max=1.0)
    sin2 = (1.0 - s ** 2).clamp(min=0.0)
    return float(torch.sqrt(sin2.sum()).item())


def grad_cov_spectrum(model, data, layer_name, device, max_samples=140):
    """对指定 2D 权重层, 计算 population 行空间梯度协方差 Σ_L 的谱.

    per-sample 梯度: 对每个 train 节点 i 的 CE loss 求该层权重梯度 G_i(m×n),
    Σ_L = mean_i (G_i G_i^T)/n。用 torch.func.grad+vmap 若可用。
    """
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    if train_idx.numel() > max_samples:
        train_idx = train_idx[:max_samples]

    # 找到目标层参数
    target = dict(model.named_parameters())[layer_name]
    m, n = target.shape

    # 逐样本梯度 (显式循环, Cora train 节点少 ~140, 可接受)
    G_list = []
    for i in train_idx.tolist():
        model.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss_i = F.cross_entropy(logits[i:i + 1], data.y[i:i + 1])
        g = torch.autograd.grad(loss_i, target, retain_graph=False)[0]
        G_list.append(g.detach())
    G = torch.stack(G_list, 0)  # (N, m, n)
    N = G.shape[0]
    # 行空间协方差 Σ_L = mean_i (G_i G_i^T)/n
    SigmaL = torch.einsum("bmn,bkn->mk", G, G) / (N * n)
    SigmaL = symmetrize(SigmaL)
    evals, evecs = _robust_eigh(SigmaL)
    evals = evals.clamp(min=0.0)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals.cpu(), evecs.cpu(), (m, n, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Cora")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--checkpoints", type=str,
                    default="1,10,25,50,100,150,200")
    ap.add_argument("--layer", type=str, default=None,
                    help="2D 权重层名; 默认自动选第一个 2D 权重")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--output",
                    default=str(SCRIPT_DIR / "results_trace_gradcov.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    dataset, data = load_data(args.task, args.data_dir)
    data = data.to(device)
    model = SmallGCN(dataset.num_features, 64, dataset.num_classes).to(device)

    # 选目标 2D 层
    layer = args.layer
    if layer is None:
        for nm, p in model.named_parameters():
            if p.dim() == 2:
                layer = nm
                break
    print(f"[TRACE] task={args.task} layer={layer} "
          f"shape={dict(model.named_parameters())[layer].shape}", flush=True)

    ckpts = sorted(set(int(x) for x in args.checkpoints.split(",")))
    opt = build_optimizer("buresmomentum", model.parameters(),
                          lr=0.01, weight_decay=1e-4)

    trace = []
    prev_top = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch in ckpts:
            evals, evecs, (m, n, N) = grad_cov_spectrum(
                model, data, layer, device)
            topk = min(args.top_k, evecs.shape[1])
            Utop = evecs[:, :topk]
            lam = evals.numpy()
            lam_pos = lam[lam > 0]
            entry = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "n_samples": N,
                "lambda_max": float(lam[0]),
                "lambda_min_pos": float(lam_pos.min()) if lam_pos.size else 0.0,
                "trace": float(lam.sum()),
                "cond_posdef": (float(lam[0] / lam_pos.min())
                                if lam_pos.size else None),
                "top_eigenvalues": [float(x) for x in lam[:8]],
                "spectrum_full": [float(x) for x in lam],
            }
            if prev_top is not None:
                entry["subspace_sinTheta_vs_prev"] = principal_angles(
                    prev_top, Utop)
                # 尺度变化: top-k 特征值相对上个 checkpoint 的比
                pe = np.array(prev_entry["top_eigenvalues"][:topk])
                ce = np.array(entry["top_eigenvalues"][:topk])
                ratio = ce / (pe + 1e-30)
                entry["topk_eigval_ratio_vs_prev"] = [float(x) for x in ratio]
                entry["topk_scale_change_maxfold"] = float(
                    np.max(np.maximum(ratio, 1.0 / (ratio + 1e-30))))
            prev_top = Utop
            prev_entry = entry
            trace.append(entry)
            print(f"[TRACE] epoch={epoch} loss={loss.item():.4f} "
                  f"lam_max={entry['lambda_max']:.3e} "
                  f"trace={entry['trace']:.3e} "
                  f"cond={entry['cond_posdef']} "
                  f"sinTheta_prev={entry.get('subspace_sinTheta_vs_prev')}",
                  flush=True)

    # 汇总判据
    lam_max_series = [e["lambda_max"] for e in trace]
    tr_series = [e["trace"] for e in trace]
    sin_series = [e.get("subspace_sinTheta_vs_prev") for e in trace
                  if e.get("subspace_sinTheta_vs_prev") is not None]
    scale_fold = (max(lam_max_series) / (min(lam_max_series) + 1e-30)
                  if lam_max_series else None)
    summary = {
        "task": args.task, "layer": layer, "seed": args.seed,
        "epochs": args.epochs, "checkpoints": ckpts,
        "optimizer": "buresmomentum",
        "lambda_max_fold_change": scale_fold,
        "trace_fold_change": (max(tr_series) / (min(tr_series) + 1e-30)
                              if tr_series else None),
        "mean_subspace_sinTheta": (float(np.mean(sin_series))
                                   if sin_series else None),
        "max_subspace_sinTheta": (float(np.max(sin_series))
                                  if sin_series else None),
        "interpretation": (
            "尺度漂移显著且子空间角度小 => insight_scale 数据层坐实 (尺度主导)"),
        "trace": trace,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TRACE DONE] lambda_max fold={scale_fold:.2e} "
          f"mean_sinTheta={summary['mean_subspace_sinTheta']}", flush=True)
    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
