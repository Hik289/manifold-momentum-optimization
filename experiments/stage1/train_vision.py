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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from models import ResNet18Small
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


def dataset_spec(task):
    normalized = task.upper().replace("_", "-")
    if normalized == "CIFAR-10":
        return datasets.CIFAR10, 10
    if normalized == "CIFAR-100":
        return datasets.CIFAR100, 100
    raise ValueError("task must be CIFAR-10 or CIFAR-100")


def load_datasets(task, data_dir):
    ds_cls, num_classes = dataset_spec(task)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = Path(data_dir) / "vision"
    train_set = ds_cls(root=root, train=True, transform=tfm, download=True)
    val_set = ds_cls(root=root, train=False, transform=tfm, download=True)
    return train_set, val_set, num_classes


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(train_set, batch_size, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_val_loader(val_set, batch_size, num_workers):
    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = 0.0
    top5 = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        maxk = min(5, logits.shape[1])
        _, pred = logits.topk(maxk, dim=1)
        correct = pred.eq(y.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :maxk].sum().item()
        total += y.numel()
    return top1 / total, top5 / total


def peak_memory_gb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


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


def run_baseline(label, opt_name, train_set, val_loader, num_classes, args, device):
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = make_train_loader(
        train_set, args.batch_size, args.seed, args.num_workers
    )
    model = ResNet18Small(num_classes).to(device)
    optimizer = build_stage_optimizer(opt_name, model.parameters())
    recorder = DiagnosticRecorder(optimizer, track_spd=True)

    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            batch = y.numel()
            running_loss += loss.item() * batch
            seen += batch
            global_step += 1
            if global_step % args.diag_interval == 0:
                recorder.record(
                    global_step,
                    train_loss=float(loss.item()),
                    learning_rate=HP["lr"],
                    peak_memory_gb=peak_memory_gb(device),
                    checkpoint_meta={"epoch": epoch, "baseline": label},
                )
        train_loss = running_loss / max(seen, 1)
        val_acc, top5_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
                "top5_acc": float(top5_acc),
            }
        )
        print(
            f"{args.task} {label} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_acc={val_acc:.4f} "
            f"top5_acc={top5_acc:.4f}",
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
    parser.add_argument("--task", required=True, choices=["CIFAR-10", "CIFAR-100"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diag-interval", type=int, default=50)
    parser.add_argument(
        "--data-dir",
        default=str(ROOT.parent / "data"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "stage1" / "results_vision.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.gpu)
    train_set, val_set, num_classes = load_datasets(args.task, args.data_dir)
    val_loader = make_val_loader(val_set, args.batch_size, args.num_workers)

    task_result = {
        "task": args.task,
        "domain": "vision",
        "model": "ResNet18Small",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "metric_name": "val_acc",
        "metric_mode": "max",
        "hyperparams": HP,
        "baselines": {},
        "status": "running",
    }
    update_results_file(args.output, args.task, task_result)

    for label, opt_name in BASELINES:
        result = run_baseline(
            label, opt_name, train_set, val_loader, num_classes, args, device
        )
        task_result["baselines"][label] = result
        update_results_file(args.output, args.task, task_result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    task_result["status"] = "complete"
    update_results_file(args.output, args.task, task_result)


if __name__ == "__main__":
    main()
