#!/usr/bin/env python
"""Track gradient-covariance eigenvalues during a Cora run.

At selected epochs, the script computes one gradient per training node for a
matrix layer. It averages their row-side Gram matrices and records the full
eigenvalue spectrum, changes in eigenvalue scale, and angles between leading
eigenspaces. This helps compare scale changes with subspace rotation. The
per-node gradients are computed in a loop. Default seed: 42.
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
    """Measure the difference between two orthonormal subspace bases."""
    # U1 and U2 have orthonormal columns.
    M = U1.transpose(-1, -2) @ U2
    s = torch.linalg.svdvals(M).clamp(max=1.0)
    sin2 = (1.0 - s ** 2).clamp(min=0.0)
    return float(torch.sqrt(sin2.sum()).item())


def grad_cov_spectrum(model, data, layer_name, device, max_samples=140):
    """Return the row-side gradient-covariance spectrum for one matrix layer.

    Each training node gives a gradient G_i. The estimate is the mean of
    G_i G_i^T / n over the sampled nodes.
    """
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    if train_idx.numel() > max_samples:
        train_idx = train_idx[:max_samples]

    # Select the requested matrix parameter.
    target = dict(model.named_parameters())[layer_name]
    m, n = target.shape

    # Compute one gradient per node; Cora's training split is small.
    G_list = []
    for i in train_idx.tolist():
        model.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss_i = F.cross_entropy(logits[i:i + 1], data.y[i:i + 1])
        g = torch.autograd.grad(loss_i, target, retain_graph=False)[0]
        G_list.append(g.detach())
    G = torch.stack(G_list, 0)  # (N, m, n)
    N = G.shape[0]
    # Average row-side Gram matrices.
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
                    help="matrix layer name (default: first matrix parameter)")
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

    # Use the first matrix parameter unless a layer is specified.
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
                # Compare leading eigenvalues with the previous checkpoint.
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

    # Summarize the recorded changes without assigning a fixed conclusion.
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
            "Compare eigenvalue-scale changes with subspace-angle changes."),
        "trace": trace,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TRACE DONE] lambda_max fold={scale_fold:.2e} "
          f"mean_sinTheta={summary['mean_subspace_sinTheta']}", flush=True)
    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
