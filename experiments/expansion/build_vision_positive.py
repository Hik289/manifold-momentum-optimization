"""Publish completed vision follow-up without hiding the declared cohort."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/extension/results"
SECTIONS = ROOT / "paper/sections"
FIGURES = ROOT / "paper/figures"
METHODS = ("adamw", "euclidean", "bures")
NAMES = {"adamw": "AdamW", "euclidean": "Euclidean-Kron.", "bures": "BuresMomentum", "sgdm": "SGD momentum"}
COLORS = {"adamw": "#747983", "euclidean": "#3166a2", "bures": "#15803d"}
TASKS = {"fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
MODELS = {"mlp": "MLP-128", "resnet18_head": "ResNet-18 head", "resnet50_head": "ResNet-50 head"}


def record(job):
    path = RESULTS / (job["id"] + ".json")
    if not path.exists():
        return {}
    result = json.loads(path.read_text())
    assert result["id"] == job["id"]
    return result


def complete(job, result):
    return (
        result.get("status") == "complete"
        and result.get("stage") == "full"
        and result.get("epochs") == job["epochs"]
        and bool(result.get("history"))
        and result["history"][-1]["epoch"] == job["epochs"]
        and result.get("validation_selected", {}).get("test_metric") is not None
        and result.get("final", {}).get("test_metric") is not None
    )


def figure(comparisons):
    chosen = comparisons[:3]  # Declared task/model/rate order, never largest observed gain.
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.3 * len(chosen), 4.1), squeeze=False)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    used = []
    for col, item in enumerate(chosen):
        task, model, _, rate = item["config"]
        for method in METHODS:
            result = item["arms"][method]
            used.append(result["id"])
            for row, field in enumerate(("train_loss", "val_loss")):
                points = [p for p in result["history"] if isinstance(p.get(field), (int, float)) and math.isfinite(p[field])]
                axes[row, col].plot([p["epoch"] for p in points], [p[field] for p in points],
                                    color=COLORS[method], lw=1.1, label=NAMES[method])
        for row in range(2):
            ax = axes[row, col]
            ax.set(title=f"{TASKS[task]} / {MODELS[model]} / LR {rate:.1g}",
                   xlabel="Epoch", ylabel="Training loss" if row == 0 else "Validation loss")
            ax.grid(alpha=.15)
        axes[0, col].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "vision_positive_losses.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "vision_positive_losses.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return used


def write_ledger(jobs, records, terminal):
    rows = []
    for job in sorted(jobs, key=lambda j: (j["task"], j["model"], j["lr"], j["method"])):
        result = records[job["id"]]
        status = result.get("status", "queued")
        selected = result.get("validation_selected", {}).get("test_metric") if status == "complete" else None
        final = result.get("final", {}).get("test_metric") if status == "complete" else None
        rows.append(" & ".join((
            TASKS[job["task"]], MODELS[job["model"]], NAMES[job["method"]],
            f"{job['lr']:.1g}", status,
            f"{100*selected:.2f}" if selected is not None else "---",
            f"{100*final:.2f}" if final is not None else "---",
        )) + r"\\")
    scope = "All arms have terminated." if terminal else "Running and queued arms remain explicit."
    (SECTIONS / "vision_followup_all.tex").write_text("\n".join([
        r"\subsection{Declared vision follow-up cohort}",
        r"\begin{longtable}{lllrlrr}",
        r"\caption{All 26 declared vision arms, including two SGD-momentum controls. Accuracy is in percent; test values appear only after the full 1000-epoch budget completes. " + scope
        + r" The main text summarizes every completed three-method configuration.}\label{tab:vision_all_arms}\\",
        r"\toprule Dataset & Model & Method & LR & Status & Selected test & Final test \\ \midrule",
        r"\endfirsthead",
        r"\toprule Dataset & Model & Method & LR & Status & Selected test & Final test \\ \midrule",
        r"\endhead",
        *rows,
        r"\bottomrule\end{longtable}",
    ]) + "\n")


def build():
    jobs = [j for j in json.loads((ROOT / "experiments/expansion/jobs.json").read_text()) if j["family"] == "vision"]
    records = {j["id"]: record(j) for j in jobs}
    terminal = all(records[j["id"]].get("status") in ("complete", "failed") for j in jobs)
    audit = {
        "declared_vision_arms": len(jobs),
        "terminal": terminal,
        "status": {s: sum(records[j["id"]].get("status", "queued") == s for j in jobs)
                   for s in ("complete", "failed", "running", "queued")},
        "complete_paired_configurations": [],
        "favorable_configurations": [],
    }
    main = SECTIONS / "vision_positive_followup.tex"
    write_ledger(jobs, records, terminal)

    groups = {}
    for job in jobs:
        key = (job["task"], job["model"], job["hidden"], job["lr"])
        groups.setdefault(key, {})[job["method"]] = (job, records[job["id"]])
    complete_groups = []
    for config, arms in sorted(groups.items()):
        if not all(m in arms and complete(*arms[m]) for m in METHODS):
            continue
        paired = {m: arms[m][1] for m in METHODS}
        assert len({paired[m]["initialization_sha256"] for m in METHODS}) == 1, config
        assert len({paired[m]["source_sha256"] for m in METHODS}) == 1, config
        assert len({json.dumps(paired[m]["data_meta"], sort_keys=True) for m in METHODS}) == 1, config
        for field in ("epochs", "batch", "block", "eval_every", "diag_every", "lr", "seed"):
            assert len({paired[m][field] for m in METHODS}) == 1, (config, field)
        values = {m: paired[m]["validation_selected"]["test_metric"] for m in METHODS}
        final_values = {m: paired[m]["final"]["test_metric"] for m in METHODS}
        final_losses = {}
        for method in METHODS:
            finite = [h["val_loss"] for h in paired[method]["history"]
                      if isinstance(h.get("val_loss"), (int, float)) and math.isfinite(h["val_loss"])]
            final_losses[method] = finite[-1] if finite else None
        selected_win = values["bures"] > max(values["euclidean"], values["adamw"])
        final_win = final_values["bures"] > max(final_values["euclidean"], final_values["adamw"])
        loss_win = (all(v is not None for v in final_losses.values())
                    and final_losses["bures"] < min(final_losses["euclidean"], final_losses["adamw"]))
        item = {"config": config, "arms": paired, "selected_test": values,
                "final_test": final_values, "positive": selected_win and final_win,
                "loss_positive": loss_win}
        complete_groups.append(item)
        audit["complete_paired_configurations"].append({
            "config": config, "selected_test": values, "final_test": final_values,
            "final_validation_loss": final_losses, "selected_win": selected_win,
            "final_win": final_win, "loss_positive": loss_win, "positive": item["positive"],
            "run_sha256": {m: hashlib.sha256((RESULTS / (paired[m]["id"] + ".json")).read_bytes()).hexdigest()
                           for m in METHODS},
        })

    positives = [item for item in complete_groups if item["positive"]]
    selected_wins = [item for item in complete_groups
                     if item["selected_test"]["bures"] > max(
                         item["selected_test"]["euclidean"], item["selected_test"]["adamw"])]
    loss_positives = [item for item in positives if item["loss_positive"]]
    audit["favorable_configurations"] = [item["config"] for item in positives]
    audit["selected_checkpoint_wins"] = [item["config"] for item in selected_wins]
    audit["figure_runs"] = figure(loss_positives) if loss_positives else []
    if complete_groups:
        figure_width = r".48\textwidth" if len(loss_positives) == 1 else r".99\textwidth"
        scope = (
            "The declared 26-arm vision cohort has terminated."
            if terminal else
            f"At this interim snapshot, {audit['status']['complete']}/26 declared vision arms "
            "have completed their full budget."
        )
        rows = []
        for item in complete_groups:
            task, model, _, rate = item["config"]
            v, f = item["selected_test"], item["final_test"]
            selected = f"{100*v['bures']:.2f}"
            final = f"{100*f['bures']:.2f}"
            if item in selected_wins:
                selected = r"\textbf{" + selected + "}"
            if item["positive"]:
                final = r"\textbf{" + final + "}"
            rows.append(
                f"{TASKS[task]} & {MODELS[model]} & {rate:.1g} & "
                f"{selected} & "
                f"{100*(v['bures']-v['euclidean']):+.2f} & "
                f"{100*(v['bures']-v['adamw']):+.2f} & "
                f"{final} & "
                f"{100*(f['bures']-f['euclidean']):+.2f} & "
                f"{100*(f['bures']-f['adamw']):+.2f} " + r"\\"
            )
        text = [
            r"\subsection{Vision follow-up on completed paired configurations}",
            f"{scope} Among {len(complete_groups)} complete "
            f"three-method fixed-rate configurations, BuresMomentum exceeds both references at the "
            f"validation-selected checkpoint in {len(selected_wins)} and at both the selected and final "
            f"checkpoints in {len(positives)}. The complete-configuration results appear in "
            f"Table~\\ref{{tab:vision_complete}}; Appendix~\\ref{{tab:vision_all_arms}} retains all 26 declared arms, including "
            f"unfavorable, running, queued, and failed configurations. This is a descriptive, single-seed "
            f"comparison and does not establish a vision-family significance claim.",
        ]
        for item in positives:
            task, model, _, rate = item["config"]
            v, f = item["selected_test"], item["final_test"]
            losses = next(entry["final_validation_loss"] for entry in audit["complete_paired_configurations"]
                          if tuple(entry["config"]) == item["config"])
            loss_detail = (
                f" Final validation losses are {losses['bures']:.3f}, "
                f"{losses['euclidean']:.3f}, and {losses['adamw']:.3f}, respectively."
                if all(value is not None for value in losses.values()) else ""
            )
            head_scope = (" The matrix preconditioner acts on the classifier head; "
                          "the backbone uses the same AdamW optimizer in all three arms.") \
                         if model.startswith("resnet") else ""
            text.append(
                f"For {TASKS[task]} with {MODELS[model]} at learning rate {rate:.1g}, "
                f"BuresMomentum achieves {100*v['bures']:.2f}\\% validation-selected test "
                f"accuracy versus {100*v['euclidean']:.2f}\\% for Euclidean-Kron. "
                f"and {100*v['adamw']:.2f}\\% for AdamW. At epoch 1000, the corresponding "
                f"accuracies are {100*f['bures']:.2f}\\%, {100*f['euclidean']:.2f}\\%, "
                f"and {100*f['adamw']:.2f}\\%.{loss_detail}{head_scope} This configuration "
                f"was identified after inspecting the completed runs and remains a "
                f"single-seed descriptive result (Table~\\ref{{tab:vision_complete}})."
            )
        text.extend([
            r"\begin{table*}[t]\centering\small",
            r"\begin{tabular}{lllrrrrrr}\toprule",
            r"Dataset & Model & LR & Selected Bures & $\Delta$ vs E & $\Delta$ vs A & Final Bures & $\Delta$ vs E & $\Delta$ vs A \\ \midrule",
            *rows,
            r"\bottomrule\end{tabular}",
            r"\caption{Every complete three-method fixed-rate vision configuration under both checkpoint rules. Bures accuracies are in percent; deltas against Euclidean and AdamW are percentage points, with positive values favoring Bures. Bold marks Bures when it exceeds both references at that checkpoint. All three methods share the split, seed, initialization, and 1000-epoch budget; GPU lanes can differ. Unfinished arms and SGD controls remain in Appendix~\ref{tab:vision_all_arms}.}",
            r"\label{tab:vision_complete}\end{table*}",
        ])
        if loss_positives:
            text.extend([
                r"\begin{figure*}[t]\centering\includegraphics[width=" + figure_width + r"]{figures/vision_positive_losses.pdf}",
                r"\caption{Training and validation losses for up to three completed vision configurations whose Bures validation loss also ends below both references, in declared task/model/rate order. Every saved observation is shown without smoothing. These single-seed trajectories are descriptive; all declared outcomes remain in Appendix~\ref{tab:vision_all_arms}.}",
                r"\label{fig:vision_positive_losses}\end{figure*}",
            ])
        main.write_text("\n".join(text) + "\n")
    else:
        main.write_text("% No complete three-method vision configuration yet.\n")

    (ROOT / "analysis/VISION_FOLLOWUP.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
