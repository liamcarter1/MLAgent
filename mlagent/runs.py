"""A finished training run: log it to runs.jsonl and freeze its outputs.

`runlog.py` owns the runs.jsonl file. This module owns the step from "train.py and
evaluate.py have finished" to "the run is a numbered entry with archived figures, metrics
and checkpoint", which the train stage and the tune stage both take.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mlagent import config as cfg
from mlagent.project import Project, read_json_file, write_json_file
from mlagent.runlog import append_run, read_runs

EVAL_VAL_FILE = "eval_val.json"
CURVES_FIGURE = "training_curves.png"
BEST_CHECKPOINT = "best.joblib"


@dataclass
class LoggedRun:
    entry: dict
    figures: list[Path] = field(default_factory=list)
    new: bool = True


def build_run_entry(
    metrics: dict, checkpoint: str | None = None, applied_diff: dict | None = None
) -> dict:
    """One `runs.jsonl` line, derived from the metrics.json that train.py wrote."""
    epochs = metrics.get("epochs") or []
    last = epochs[-1] if epochs else {}
    ok = metrics.get("status") == "done"
    return {
        "started_at": metrics.get("started_at"),
        "status": "done" if ok else "failed",
        "config": dict(metrics.get("config") or {}),
        "epochs_run": len(epochs),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_metric": metrics.get("best_val_metric"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "seconds": metrics.get("seconds"),
        "error": metrics.get("error"),
        "applied_diff": dict(applied_diff) if applied_diff else None,
        "checkpoint": checkpoint if ok else None,
    }


def archive_run(project: Project, run_id: int) -> list[Path]:
    """Freeze this run's figures and checkpoint under `run{N}` names."""
    archived: list[Path] = []
    curves = project.plots_dir / CURVES_FIGURE
    if curves.exists():
        target = project.plots_dir / f"run{run_id}_training.png"
        shutil.copy2(curves, target)
        archived.append(target)
    for source in sorted(project.plots_dir.glob("val_*.png")):
        target = project.plots_dir / f"run{run_id}_{source.name}"
        shutil.copy2(source, target)
        archived.append(target)
    best = project.checkpoints_dir / BEST_CHECKPOINT
    if best.exists():
        shutil.copy2(best, project.checkpoints_dir / f"run{run_id}.joblib")
    return archived


def run_metrics_path(project: Project, run_id: int) -> Path:
    return project.runs_dir / f"run{run_id}_metrics.json"


def archive_metrics(project: Project, run_id: int, metrics: dict) -> Path:
    """Copy metrics.json to `runs/run{N}_metrics.json` so the next run cannot overwrite it."""
    return write_json_file(run_metrics_path(project, run_id), metrics)


def read_run_metrics(project: Project, runs: list[dict]) -> dict[int, dict]:
    """The archived metrics per run id, for the runs that have an archive."""
    found: dict[int, dict] = {}
    for run in runs:
        run_id = run.get("run_id")
        if not isinstance(run_id, int):
            continue
        data = read_json_file(run_metrics_path(project, run_id))
        if isinstance(data, dict):
            found[run_id] = data
    return found


def _archived_figures(project: Project, run_id: int) -> list[Path]:
    return sorted(
        project.plots_dir.glob(f"run{run_id}_*.png"),
        key=lambda p: (not p.name.endswith("_training.png"), p.name),
    )


def run_problem(project: Project) -> str | None:
    """Why the run in metrics.json cannot be logged yet; None when it can or already was."""
    metrics = project.read_json(cfg.METRICS_FILE)
    if not isinstance(metrics, dict) or not metrics.get("started_at"):
        return (
            "I can't see a finished run in `metrics.json` yet. Run the `train.py` cell, "
            "then run this cell again."
        )
    started = metrics["started_at"]
    if any(r.get("started_at") == started for r in read_runs(project.runs_path)):
        return None
    if metrics.get("status") == "done":
        eval_data = project.read_json(EVAL_VAL_FILE)
        if not isinstance(eval_data, dict) or eval_data.get("started_at") != started:
            return (
                "The `train.py` run finished, but `eval_val.json` doesn't match it yet. "
                "Run the `evaluate.py` cell, then run this cell again."
            )
    return None


def log_finished_run(project: Project, applied_diff: dict | None = None) -> LoggedRun | None:
    """Log the run in metrics.json once, archiving its figures, metrics and checkpoint.

    Returns None exactly when `run_problem` reports a problem. A run that was already
    logged comes back with `new=False` and the figures archived for it the first time.
    """
    if run_problem(project) is not None:
        return None
    metrics = project.read_json(cfg.METRICS_FILE)
    runs = read_runs(project.runs_path)
    existing = next((r for r in runs if r.get("started_at") == metrics["started_at"]), None)
    if existing is not None:
        return LoggedRun(entry=existing, figures=_archived_figures(project, existing["run_id"]),
                         new=False)
    ok = metrics.get("status") == "done"
    run_id = len(runs) + 1
    figures = archive_run(project, run_id) if ok else []
    archive_metrics(project, run_id, metrics)
    checkpoint = (
        f"checkpoints/run{run_id}.joblib"
        if ok and (project.checkpoints_dir / f"run{run_id}.joblib").exists()
        else None
    )
    entry = append_run(project.runs_path, build_run_entry(metrics, checkpoint, applied_diff))
    return LoggedRun(entry=entry, figures=figures, new=True)
