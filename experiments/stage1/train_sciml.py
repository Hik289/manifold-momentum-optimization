#!/usr/bin/env python
import argparse
import fcntl
import json
import math
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

from models import PINNNet
from pipeline.diagnostics import DiagnosticRecorder, build_optimizer


BASELINES = [
    ("B1", "sgdm"),
    ("B2", "adamw"),
    ("B3", "euclidean_kron"),
    ("B4", "shampoo"),
    ("B5", "buresmomentum"),
]

HP = {
    "lr": 1e-3,
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


def rand_uniform(shape, low, high, generator, device):
    x = torch.rand(shape, generator=generator)
    x = low + (high - low) * x
    return x.to(device)


def burgers_reference(device, nx=128, dt=5e-4, store_every=20):
    nu = 1.0 / (100.0 * math.pi)
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    dx = x[1] - x[0]
    u = -np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0
    nt = int(round(1.0 / dt))
    snapshots = []
    times = []
    for step in range(nt + 1):
        if step % store_every == 0 or step == nt:
            snapshots.append(u.copy())
            times.append(step * dt)
        if step == nt:
            break
        center = u[1:-1]
        ux_back = (u[1:-1] - u[:-2]) / dx
        ux_forward = (u[2:] - u[1:-1]) / dx
        ux = np.where(center >= 0.0, ux_back, ux_forward)
        uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        new_u = u.copy()
        new_u[1:-1] = center + dt * (-center * ux + nu * uxx)
        new_u[0] = 0.0
        new_u[-1] = 0.0
        u = new_u

    tt, xx = np.meshgrid(np.asarray(times), x, indexing="ij")
    inputs = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)
    values = np.asarray(snapshots).reshape(-1, 1)
    return (
        torch.tensor(inputs, dtype=torch.float32, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
    )


def poisson_reference(device, n=512):
    x = torch.linspace(0.0, 1.0, n, device=device).unsqueeze(1)
    y = torch.sin(math.pi * x)
    return x, y


def burgers_loss(model, batch_size, generator, device):
    nu = 1.0 / (100.0 * math.pi)
    x = rand_uniform((batch_size, 1), -1.0, 1.0, generator, device)
    t = rand_uniform((batch_size, 1), 0.0, 1.0, generator, device)
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    grad_u = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    grad_u_x = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_xx = grad_u_x[:, 0:1]
    residual = u_t + u * u_x - nu * u_xx
    residual_loss = residual.pow(2).mean()

    x_ic = rand_uniform((batch_size, 1), -1.0, 1.0, generator, device)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(math.pi * x_ic)
    pred_ic = model(torch.cat([x_ic, t_ic], dim=1))

    t_bc = rand_uniform((batch_size, 1), 0.0, 1.0, generator, device)
    x_left = -torch.ones_like(t_bc)
    x_right = torch.ones_like(t_bc)
    pred_left = model(torch.cat([x_left, t_bc], dim=1))
    pred_right = model(torch.cat([x_right, t_bc], dim=1))
    bc_loss = (
        F.mse_loss(pred_ic, u_ic)
        + F.mse_loss(pred_left, torch.zeros_like(pred_left))
        + F.mse_loss(pred_right, torch.zeros_like(pred_right))
    )
    return residual_loss + bc_loss, residual_loss, bc_loss


def poisson_loss(model, batch_size, generator, device):
    x = rand_uniform((batch_size, 1), 0.0, 1.0, generator, device)
    x = x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    forcing = -(math.pi**2) * torch.sin(math.pi * x)
    residual = u_xx - forcing
    residual_loss = residual.pow(2).mean()
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    pred_bc = model(x_bc)
    bc_loss = pred_bc.pow(2).mean()
    return residual_loss + bc_loss, residual_loss, bc_loss


@torch.no_grad()
def relative_l2(model, ref_inputs, ref_values, chunk_size=8192):
    model.eval()
    preds = []
    for start in range(0, ref_inputs.shape[0], chunk_size):
        preds.append(model(ref_inputs[start : start + chunk_size]))
    pred = torch.cat(preds, dim=0)
    return float((pred - ref_values).norm().div(ref_values.norm() + 1e-12).item())


def peak_memory_gb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


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


def mean_loss_auc(history):
    if not history:
        return None
    return float(sum(item["total_loss"] for item in history) / len(history))


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


def run_baseline(label, opt_name, args, device, ref_inputs, ref_values):
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    in_dim = 2 if args.task == "Burgers-PINN" else 1
    model = PINNNet(in_dim=in_dim, hidden=64, layers=4).to(device)
    optimizer = build_stage_optimizer(opt_name, model.parameters())
    recorder = DiagnosticRecorder(optimizer, track_spd=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    history = []

    loss_fn = burgers_loss if args.task == "Burgers-PINN" else poisson_loss
    last_losses = None
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, residual_loss, bc_loss = loss_fn(
            model, args.batch_size, generator, device
        )
        total_loss.backward()
        # gradient clipping (标准训练卫生; 对所有优化器同阈值施加, 保 B3/B5 配对)
        if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_losses = (
            float(total_loss.item()),
            float(residual_loss.item()),
            float(bc_loss.item()),
        )
        if step % args.log_interval == 0 or step == args.steps:
            rel = relative_l2(model, ref_inputs, ref_values)
            row = {
                "step": step,
                "total_loss": last_losses[0],
                "residual_loss": last_losses[1],
                "bc_loss": last_losses[2],
                "relative_L2": rel,
            }
            history.append(row)
            recorder.record(
                step,
                train_loss=last_losses[0],
                val_metric=rel,
                learning_rate=HP["lr"],
                peak_memory_gb=peak_memory_gb(device),
                checkpoint_meta={
                    "step": step,
                    "baseline": label,
                    "residual_loss": last_losses[1],
                    "bc_loss": last_losses[2],
                },
            )
            print(
                f"{args.task} {label} step={step} total_loss={last_losses[0]:.6e} "
                f"residual_loss={last_losses[1]:.6e} bc_loss={last_losses[2]:.6e} "
                f"relative_L2={rel:.6e}",
                flush=True,
            )

    final = history[-1]
    best = min(history, key=lambda row: row["relative_L2"])
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
    parser.add_argument(
        "--task", required=True, choices=["Burgers-PINN", "Poisson-PINN"]
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="max grad norm (0=off); 同阈值施加于所有优化器")
    parser.add_argument(
        "--output",
        default=str(ROOT / "stage1" / "results_sciml.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.gpu)
    if args.task == "Burgers-PINN":
        ref_inputs, ref_values = burgers_reference(device)
    else:
        ref_inputs, ref_values = poisson_reference(device)

    task_result = {
        "task": args.task,
        "domain": "sciml",
        "model": "PINNNet",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "metric_name": "relative_L2",
        "metric_mode": "min",
        "hyperparams": HP,
        "grad_clip": args.grad_clip,
        "baselines": {},
        "status": "running",
    }
    update_results_file(args.output, args.task, task_result)

    for label, opt_name in BASELINES:
        result = run_baseline(label, opt_name, args, device, ref_inputs, ref_values)
        task_result["baselines"][label] = result
        update_results_file(args.output, args.task, task_result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    task_result["status"] = "complete"
    update_results_file(args.output, args.task, task_result)


if __name__ == "__main__":
    main()
