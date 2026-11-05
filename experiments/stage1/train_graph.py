#!/usr/bin/env python
import argparse
import fcntl
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from models import SmallGCN
from pipeline.diagnostics import DiagnosticRecorder, build_optimizer


BASELINES = [
    ("B1", "sgdm"),
    ("B2", "adamw"),
    ("B3", "euclidean_kron"),
    ("B4", "shampoo"),
    ("B5", "buresmomentum"),
]

HP = {
    "lr": 0.01,
    "beta1": 0.9,
    "beta2": 0.99,
    "weight_decay": 1e-4,
    "eps_stat": 1e-6,
    "eps_norm": 1e-12,
    "lam_min": 1e-6,
    "lam_max": 1e6,
    "s_min": 0.1,
    "s_max": 10.0,
    "tau_root": 10,
    "block_size": 256,
}

TASK_NAMES = {"Cora": "Cora", "Citeseer": "CiteSeer", "PubMed": "PubMed"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_stage_optimizer(name, params):
    return build_optimizer(
        name,
        params,
        lr=HP["lr"],
        weight_decay=HP["weight_decay"],
        momentum=HP["beta1"],
        betas=(HP["beta1"], HP["beta2"]),
        eps_stat=HP["eps_stat"],
        eps_norm=HP["eps_norm"],
        lam_min=HP["lam_min"],
        lam_max=HP["lam_max"],
        s_min=HP["s_min"],
        s_max=HP["s_max"],
        tau_root=HP["tau_root"],
        block_size=HP["block_size"],
        state_dtype=torch.float32,
    )


def load_data(task, data_dir):
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except Exception as exc:
        raise ImportError(
            "torch_geometric is required for graph screening. Install it with "
            "`pip install torch_geometric` in manifoldmoment_env."
        ) from exc

    name = TASK_NAMES[task]
    dataset = Planetoid(
        root=str(Path(data_dir) / "graph" / name),
        name=name,
        transform=NormalizeFeatures(),
    )
    return dataset, dataset[0]


def peak_memory_gb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    out = {}
    for split, mask in [
        ("train_acc", data.train_mask),
        ("val_acc", data.val_mask),
        ("test_acc", data.test_mask),
    ]:
        out[split] = float((pred[mask] == data.y[mask]).float().mean().item())
    return out


def mean_loss_auc(history):
    if not history:
        return None
    return float(sum(item["train_loss"] for item in history) / len(history))


def diagnostic_highlights(diagnostics):
    keys = ("bw_drift_L", "bw_drift_R")
    vals = []
    for row in diagnostics:
        for key in keys:
            val = row.get(key)
            if isinstance(val, (int, float)) and np.isfinite(val):
                vals.append(float(val))
    return {
        "mean_bw_drift": float(np.mean(vals)) if vals else None,
        "off_geodesic_error": None,
    }


def update_results_file(path, task, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = {}
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        data[task] = payload
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def run_baseline(label, opt_name, dataset, data, args, device):
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = SmallGCN(dataset.num_features, 64, dataset.num_classes).to(device)
    data = data.to(device)
    optimizer = build_stage_optimizer(opt_name, model.parameters())
    recorder = DiagnosticRecorder(optimizer, track_spd=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        metrics = evaluate(model, data)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_acc": metrics["val_acc"],
            "test_acc": metrics["test_acc"],
        }
        history.append(row)
        if epoch % args.diag_interval == 0 or epoch == args.epochs:
            recorder.record(
                epoch,
                train_loss=float(loss.item()),
                val_metric=metrics["val_acc"],
                learning_rate=HP["lr"],
                peak_memory_gb=peak_memory_gb(device),
                checkpoint_meta={"epoch": epoch, "baseline": label},
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"{args.task} {label} epoch={epoch} "
                f"train_loss={loss.item():.6f} val_acc={metrics['val_acc']:.4f} "
                f"test_acc={metrics['test_acc']:.4f}",
                flush=True,
            )

    best = max(history, key=lambda row: row["val_acc"])
    final = history[-1]
    return {
        "label": label,
        "optimizer": opt_name,
        "history": history,
        "final": final,
        "best": best,
        "loss_auc": mean_loss_auc(history),
        "diagnostics": recorder.trajectory,
        "diagnostic_highlights": diagnostic_highlights(recorder.trajectory),
        "event_log": recorder.event_log,
        "peak_memory_gb": peak_memory_gb(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["Cora", "Citeseer", "PubMed"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--diag-interval", type=int, default=50)
    parser.add_argument(
        "--data-dir",
        default=str(ROOT.parent / "data"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "stage1" / "results_graph.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.gpu)
    dataset, data = load_data(args.task, args.data_dir)

    task_result = {
        "task": args.task,
        "domain": "graph",
        "model": "SmallGCN",
        "seed": args.seed,
        "epochs": args.epochs,
        "metric_name": "test_acc",
        "metric_mode": "max",
        "hyperparams": HP,
        "baselines": {},
        "status": "running",
    }
    update_results_file(args.output, args.task, task_result)

    for label, opt_name in BASELINES:
        result = run_baseline(label, opt_name, dataset, data, args, device)
        task_result["baselines"][label] = result
        update_results_file(args.output, args.task, task_result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    task_result["status"] = "complete"
    update_results_file(args.output, args.task, task_result)


if __name__ == "__main__":
    main()
