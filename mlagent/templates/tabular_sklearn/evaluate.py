"""Score a saved model on one split and draw its evaluation figures.

Run it from the project folder:

    python evaluate.py                                  the validation split, latest checkpoint
    python evaluate.py --split test                     the test split, best run's checkpoint
    python evaluate.py --split test --checkpoint P      a specific checkpoint

Writes `eval_{split}.json` (the numbers plus the raw predictions, so the figures can be
redrawn without retraining) and PNGs into `plots/`. `train.py` imports `evaluate_split`
from here (which calls `compute_metric` and `full_proba` internally) so the two scripts
can never disagree about what a metric means.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from data import load_data
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
)

# --- settings ---
SCRIPT_NAME = "evaluate.py"
PROJECT_DIR = Path(".")
CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
RUNS_FILE = "runs.jsonl"
METRICS_FILE = "metrics.json"
DEFAULT_CHECKPOINT = "checkpoints/best.joblib"
SPLITS = ("val", "test")
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"tabular_classification": "accuracy", "tabular_regression": "rmse"}
EPS = 1e-12
MAX_ROC_CLASSES = 8

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
    return str(spec.get("metric") or DEFAULT_METRIC[task_type])


# --- scoring ---
def compute_metric(name: str, y_true, y_pred) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if name == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"unknown metric {name!r}")


def full_proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with one column per class even if a class was absent from training."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    out = np.clip(out, EPS, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def evaluate_split(model, X, y, task_type: str, metric: str, n_classes: int | None) -> dict:
    y_pred = model.predict(X)
    if task_type == "tabular_classification":
        proba = full_proba(model, X, n_classes)
        loss = float(log_loss(y, proba, labels=list(range(n_classes))))
        y_proba = proba.tolist()
    else:
        loss = float(mean_squared_error(y, y_pred))
        y_proba = None
    return {
        "loss": loss,
        "value": compute_metric(metric, y, y_pred),
        "y_true": np.asarray(y).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": y_proba,
    }


def eval_record(
    split: str, task_type: str, metric: str, classes, ev: dict, started_at: str | None
) -> dict:
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
    task_type = (read_json(project_dir / "data_meta.json", default={}) or {}).get(
        "task_type", "tabular_classification"
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


def show(fig) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is not None:
        display(fig)


def save(fig, plots_dir: Path, name: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig)
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


def roc_pr_figure(y_true, y_proba, labels: list[str]):
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    n = p.shape[1]
    if n == 2:
        curves = [(1, p[:, 1], labels[1])]
    else:
        curves = [(k, p[:, k], labels[k]) for k in range(min(n, MAX_ROC_CLASSES))]
    drawn = 0
    for (k, score, label), colour in zip(curves, SERIES, strict=False):
        positive = (y == k).astype(int)
        if positive.sum() in (0, len(positive)):
            continue
        drawn += 1
        fpr, tpr, _ = roc_curve(positive, score)
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, label=label)
        precision, recall, _ = precision_recall_curve(positive, score)
        ax_pr.plot(recall, precision, color=colour, linewidth=2, label=label)
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    suffix = f" (showing {len(curves)} of {n} classes)" if n > 2 and len(curves) < n else ""
    frame(ax_roc, "ROC curve" + suffix)
    ax_roc.set_xlabel("false positive rate", color=INK_2, fontsize=8)
    ax_roc.set_ylabel("true positive rate", color=INK_2, fontsize=8)
    frame(ax_pr, "Precision-recall curve")
    ax_pr.set_xlabel("recall", color=INK_2, fontsize=8)
    ax_pr.set_ylabel("precision", color=INK_2, fontsize=8)
    if drawn > 1:
        ax_roc.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        ax_pr.legend(frameon=False, fontsize=8, labelcolor=INK_2)
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


def pred_vs_actual_figure(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=SURFACE)
    ax.scatter(y, p, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
    frame(ax, "Predicted vs actual")
    ax.set_xlabel("actual", color=INK_2, fontsize=8)
    ax.set_ylabel("predicted", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def residual_figure(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    residuals = y - p
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    ax_hist.hist(residuals, bins=min(30, max(5, len(residuals) // 5)), color=SERIES[0])
    frame(ax_hist, "Residual distribution")
    ax_hist.set_xlabel("actual - predicted", color=INK_2, fontsize=8)
    ax_scatter.scatter(p, residuals, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    ax_scatter.axhline(0, color=AXIS, linestyle="--", linewidth=1)
    frame(ax_scatter, "Residuals vs predicted")
    ax_scatter.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax_scatter.set_ylabel("residual", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def save_figures(record: dict, plots_dir: Path, split: str) -> list[str]:
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    if record["task_type"] == "tabular_classification":
        labels = [str(c) for c in (record.get("classes") or sorted({*y_true, *y_pred}))]
        names = [save(confusion_figure(y_true, y_pred, labels), plots_dir,
                      f"{split}_confusion")]
        if record.get("y_proba"):
            names.append(save(roc_pr_figure(y_true, record["y_proba"], labels), plots_dir,
                              f"{split}_roc_pr"))
        names.append(save(per_class_figure(y_true, y_pred, labels), plots_dir,
                          f"{split}_per_class"))
        return names
    return [
        save(pred_vs_actual_figure(y_true, y_pred), plots_dir, f"{split}_pred_vs_actual"),
        save(residual_figure(y_true, y_pred), plots_dir, f"{split}_residuals"),
    ]


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`. `%run script.py --flag`
    is different: there argv[0] is this script's own name, not the launcher, so its flags
    are still parsed.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
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
    n_classes = len(data["classes"]) if data["classes"] is not None else None
    model = joblib.load(checkpoint_path)
    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]

    metrics = read_json(project_dir / METRICS_FILE, default=None)
    started_at = metrics.get("started_at") if isinstance(metrics, dict) else None

    ev = evaluate_split(model, X, y, task_type, metric, n_classes)
    record = eval_record(args.split, task_type, metric, data["classes"], ev, started_at)
    record["checkpoint"] = Path(checkpoint).as_posix()
    record["run_id"] = run_id
    record["figures"] = save_figures(record, project_dir / "plots", args.split)
    out = project_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{args.split} {metric}={record['value']:.4f} loss={record['loss']:.4f} "
        f"({len(record['y_true'])} rows, checkpoint {record['checkpoint']}) -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
