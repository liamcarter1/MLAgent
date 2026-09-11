"""Score a saved checkpoint on one split and draw its evaluation figures.

Run it from the project folder:

    python evaluate.py                                  the validation split, latest checkpoint
    python evaluate.py --split test                     the test split, best run's checkpoint
    python evaluate.py --split test --checkpoint P      a specific checkpoint

Writes `eval_{split}.json` (the numbers plus the raw predictions, so the figures can be
redrawn without retraining) and PNGs into `plots/`. `train.py` imports `evaluate_split`
from here so the two scripts can never disagree about what a metric means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import load_data, make_loader, pick_device
from matplotlib.colors import LinearSegmentedColormap
from model import build_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

# --- settings ---
SCRIPT_NAME = "evaluate.py"
PROJECT_DIR = Path(".")
CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
RUNS_FILE = "runs.jsonl"
METRICS_FILE = "metrics.json"
META_FILE = "data_meta.json"
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
SPLITS = ("val", "test")
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"image_classification": "accuracy"}
EVAL_BATCH_SIZE = 64
MAX_MISCLASSIFIED = 12

# --- captions ---
# Byte-identical to the matching entries in mlagent/captions.py
# (tests/test_template_evaluate_images.py checks this); kept here too since this script
# never depends on the mlagent package.
CAPTIONS = {
    "confusion": (
        "Rows are the true label, columns are what the model predicted, so the diagonal is "
        "correct. A bright off-diagonal cell names the two classes the model keeps confusing."
    ),
    "per_class": (
        "Precision and recall for each class. Precision is how often a prediction of that "
        "class is right; recall is how much of that class the model finds. Small classes with "
        "low bars are the ones to fix."
    ),
    "misclassified": (
        "The validation images the model got most confidently wrong, each labelled "
        "true -> predicted. The same confusion repeated is a fixable labelling or "
        "[[class imbalance]] problem; a scatter of unrelated one-offs is just noise."
    ),
}

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)


# --- small helpers ---
def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            runs.append(entry)
    return runs


def metric_for(project_dir: Path, task_type: str) -> str:
    spec = read_json(project_dir / SPEC_FILE, default={}) or {}
    return str(spec.get("metric") or DEFAULT_METRIC.get(task_type, "accuracy"))


# --- scoring ---
def compute_metric(name: str, y_true, y_pred) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"unknown metric {name!r} for image classification")


def evaluate_split(model, pair, device: str, metric: str, n_classes: int,
                   batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Loss, metric and raw predictions for one (x, y) pair, in eval mode without grads."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loader = make_loader(pair, batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    n = 0
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            total_loss += float(loss_fn(logits, batch_y).item())
            n += int(batch_y.numel())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            truths.append(batch_y.cpu().numpy())
    proba = np.concatenate(probabilities) if probabilities else np.zeros((0, n_classes))
    y_true = np.concatenate(truths) if truths else np.zeros(0, dtype=np.int64)
    y_pred = proba.argmax(axis=1) if proba.size else np.zeros(0, dtype=np.int64)
    return {
        "loss": total_loss / max(1, n),
        "value": compute_metric(metric, y_true, y_pred) if n else 0.0,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": proba.tolist(),
    }


def eval_record(split: str, task_type: str, metric: str, classes, ev: dict,
                started_at: str | None) -> dict:
    return {
        "split": split,
        "task_type": task_type,
        "metric": metric,
        "value": ev["value"],
        "loss": ev["loss"],
        "classes": classes,
        "y_true": ev["y_true"],
        "y_pred": ev["y_pred"],
        "y_proba": ev["y_proba"],
        "started_at": started_at,
    }


# --- choosing a checkpoint ---
def best_checkpoint(project_dir: Path) -> tuple[str | None, int | None]:
    """The checkpoint of the best finished run, and its run id."""
    task_type = (read_json(project_dir / META_FILE, default={}) or {}).get(
        "task_type", "image_classification"
    )
    metric = metric_for(project_dir, task_type)
    higher = HIGHER_IS_BETTER.get(metric, True)
    best = None
    for run in read_runs(project_dir / RUNS_FILE):
        if run.get("status") != "done" or not run.get("checkpoint"):
            continue
        value = run.get("best_val_metric")
        if best is None:
            best = run
            continue
        incumbent = best.get("best_val_metric")
        if value is None:
            continue
        if incumbent is None or (value > incumbent if higher else value < incumbent):
            best = run
    if best is None:
        return None, None
    return str(best["checkpoint"]), best.get("run_id")


def load_checkpoint(project_dir: Path, checkpoint_path: Path, config: dict, data: dict):
    """Rebuild the network the checkpoint was saved from and load its weights."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = payload.get("config") or config
    n_classes = len(payload.get("classes") or data["classes"])
    image_size = int(payload.get("image_size") or data["image_size"])
    model = build_model(saved_config, n_classes, image_size)
    model.load_state_dict(payload["state_dict"])
    return model


# --- figures ---
def frame(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def show(fig, kind: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            display(fig)
    caption = CAPTIONS.get(kind, "")
    if caption:
        print("How to read this: " + caption.replace("[[", "").replace("]]", ""))


def save(fig, plots_dir: Path, name: str, kind: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig, kind)
    plt.close(fig)
    return f"{name}.png"


def confusion_figure(y_true, y_pred, labels: list[str]):
    n = len(labels)
    m = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    size = min(2.5 + 0.35 * n, 9)
    fig, ax = plt.subplots(figsize=(size, size), facecolor=SURFACE)
    ax.imshow(m, cmap=SEQ_CMAP)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK_2, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=INK_2, fontsize=8)
    ax.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax.set_ylabel("actual", color=INK_2, fontsize=8)
    if n <= 20 and m.size:
        threshold = m.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color=SURFACE if m[i, j] > threshold else INK)
    ax.set_title("Confusion matrix", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def per_class_figure(y_true, y_pred, labels: list[str]):
    n = len(labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * n + 2), 3.2), facecolor=SURFACE)
    x = np.arange(n)
    width = 0.38
    ax.bar(x - width / 2, precision, width=width, color=SERIES[0], label="precision")
    ax.bar(x + width / 2, recall, width=width, color=SERIES[1], label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 6 else 0,
                       ha="right" if n > 6 else "center", color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.05)
    frame(ax, "Precision and recall per class")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def misclassified_figure(images, y_true, y_pred, y_proba, labels: list[str]):
    """The wrong predictions the model was most confident about, worst first."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    proba = np.asarray(y_proba, dtype=float)
    wrong = np.flatnonzero(truth != predicted)
    if wrong.size:
        confidence = proba[wrong, predicted[wrong]] if proba.size else np.zeros(wrong.size)
        wrong = wrong[np.argsort(-confidence)][:MAX_MISCLASSIFIED]
    ncols = 4
    nrows = max(1, int(np.ceil(max(1, wrong.size) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)
    if wrong.size == 0:
        ax = axes[0][0]
        ax.set_visible(True)
        ax.text(0.5, 0.5, "nothing misclassified", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    for i, index in enumerate(wrong.tolist()):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        ax.imshow(images[int(index)])
        ax.set_title(f"{labels[int(truth[index])]} -> {labels[int(predicted[index])]}",
                     color=INK, fontsize=8, loc="left")
    fig.suptitle("Worst mistakes", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def displayable(x: torch.Tensor) -> np.ndarray:
    """Undo the per-channel normalisation enough to look at: (N, H, W, 3) in [0, 1]."""
    array = x.permute(0, 2, 3, 1).cpu().numpy()
    low = array.min(axis=(1, 2, 3), keepdims=True)
    high = array.max(axis=(1, 2, 3), keepdims=True)
    return np.clip((array - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def save_figures(record: dict, plots_dir: Path, split: str, images) -> list[str]:
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    labels = [str(c) for c in (record.get("classes") or sorted({*y_true, *y_pred}))]
    return [
        save(confusion_figure(y_true, y_pred, labels), plots_dir, f"{split}_confusion",
             "confusion"),
        save(per_class_figure(y_true, y_pred, labels), plots_dir, f"{split}_per_class",
             "per_class"),
        save(misclassified_figure(images, y_true, y_pred, record["y_proba"], labels),
             plots_dir, f"{split}_misclassified", "misclassified"),
    ]


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    # A caption below may contain a non-ASCII character; on Windows a piped stdout
    # otherwise defaults to the console codepage and mangles it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on one split.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--split", default="val", choices=list(SPLITS))
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path relative to --project")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()

    run_id = None
    checkpoint = args.checkpoint
    if checkpoint is None and args.split == "test":
        checkpoint, run_id = best_checkpoint(project_dir)
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    checkpoint_path = project_dir / checkpoint
    if not checkpoint_path.exists():
        print(f"error: no checkpoint at {checkpoint_path}; run train.py first", flush=True)
        return 1

    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    task_type = data["task_type"]
    metric = metric_for(project_dir, task_type)
    n_classes = len(data["classes"])
    device = pick_device()
    model = load_checkpoint(project_dir, checkpoint_path, config, data).to(device)

    metrics = read_json(project_dir / METRICS_FILE, default=None)
    started_at = metrics.get("started_at") if isinstance(metrics, dict) else None

    pair = data[args.split]
    ev = evaluate_split(model, pair, device, metric, n_classes)
    record = eval_record(args.split, task_type, metric, data["classes"], ev, started_at)
    record["checkpoint"] = Path(checkpoint).as_posix()
    record["run_id"] = run_id
    record["figures"] = save_figures(record, project_dir / "plots", args.split,
                                     displayable(pair[0]))
    out = project_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{args.split} {metric}={record['value']:.4f} loss={record['loss']:.4f} "
        f"({len(record['y_true'])} images, checkpoint {record['checkpoint']}) -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
