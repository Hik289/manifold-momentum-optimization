#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path


BASELINE_ORDER = ["B1", "B2", "B3", "B4", "B5"]


def load_results(stage1_dir):
    tasks = {}
    for name in ["results_vision.json", "results_graph.json", "results_sciml.json"]:
        path = Path(stage1_dir) / name
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for task, payload in data.items():
            payload["_source"] = name
            tasks[task] = payload
    return tasks


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def fmt(value):
    if value is None:
        return "n/a"
    if not finite(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.4e}"


def metric_value(run, metric_name):
    if not run:
        return None
    final = run.get("final", {})
    return final.get(metric_name)


def best_value(run, metric_name, mode):
    if not run:
        return None
    best = run.get("best", {})
    value = best.get(metric_name)
    if value is not None:
        return value
    history = run.get("history", [])
    vals = [row.get(metric_name) for row in history if row.get(metric_name) is not None]
    if not vals:
        return None
    return max(vals) if mode == "max" else min(vals)


def oriented_gap(b5, b3, mode):
    if b5 is None or b3 is None:
        return None
    return float(b5 - b3) if mode == "max" else float(b3 - b5)


def mean_bw_drift(run):
    if not run:
        return None
    highlight = run.get("diagnostic_highlights", {})
    if highlight.get("mean_bw_drift") is not None:
        return highlight.get("mean_bw_drift")
    vals = []
    for row in run.get("diagnostics", []):
        for key in ["bw_drift_L", "bw_drift_R"]:
            value = row.get(key)
            if finite(value):
                vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def off_geodesic_error(run):
    if not run:
        return None
    return run.get("diagnostic_highlights", {}).get("off_geodesic_error")


def retention_decision(task):
    baselines = task.get("baselines", {})
    mode = task.get("metric_mode", "max")
    metric = task.get("metric_name")
    b3 = baselines.get("B3")
    b5 = baselines.get("B5")
    if not b3 or not b5:
        return "NO", "missing B3/B5"

    final_gap = oriented_gap(metric_value(b5, metric), metric_value(b3, metric), mode)
    best_gap = oriented_gap(best_value(b5, metric, mode), best_value(b3, metric, mode), mode)
    loss_auc_gap = oriented_gap(b3.get("loss_auc"), b5.get("loss_auc"), "max")
    b3_drift = mean_bw_drift(b3)
    b5_drift = mean_bw_drift(b5)

    if final_gap is not None and final_gap > 0:
        return "YES", "better final metric"
    if best_gap is not None and best_gap > 0:
        return "YES", "better best metric"
    if loss_auc_gap is not None and loss_auc_gap > 0:
        return "YES", "lower loss AUC"
    if b3_drift is not None and b5_drift is not None and b5_drift < 0.95 * b3_drift:
        return "YES", "lower BW drift"
    return "NO", "no Stage 1 criterion met"


def make_table(tasks):
    headers = [
        "Task",
        "Metric",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B5 vs B3",
        "Off-geodesic",
        "BW drift B5",
        "Stage 2",
    ]
    rows = []
    for task_name in sorted(tasks):
        task = tasks[task_name]
        metric = task.get("metric_name", "metric")
        mode = task.get("metric_mode", "max")
        baselines = task.get("baselines", {})
        values = {
            label: metric_value(baselines.get(label), metric)
            for label in BASELINE_ORDER
        }
        gap = oriented_gap(values.get("B5"), values.get("B3"), mode)
        decision, reason = retention_decision(task)
        rows.append(
            [
                task_name,
                f"{metric} ({'higher' if mode == 'max' else 'lower'})",
                fmt(values.get("B1")),
                fmt(values.get("B2")),
                fmt(values.get("B3")),
                fmt(values.get("B4")),
                fmt(values.get("B5")),
                fmt(gap),
                fmt(off_geodesic_error(baselines.get("B5"))),
                fmt(mean_bw_drift(baselines.get("B5"))),
                f"{decision} - {reason}",
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-dir",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()
    tasks = load_results(args.stage1_dir)
    if not tasks:
        print("No Stage 1 result JSON files found.")
        return
    print(make_table(tasks))


if __name__ == "__main__":
    main()
