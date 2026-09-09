"""Train stage: hand the user train.py and evaluate.py, then log and explain the run."""

from __future__ import annotations

import shutil
from pathlib import Path

from mlagent import config as cfg
from mlagent.runlog import append_run, read_runs
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.templates_io import CODE_FILES

EVAL_VAL_FILE = "eval_val.json"
CURVES_FIGURE = "training_curves.png"
BEST_CHECKPOINT = "best.joblib"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_run_entry(metrics: dict, checkpoint: str | None = None) -> dict:
    """One `runs.jsonl` line, entirely derived from the metrics.json that train.py wrote."""
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
        "applied_diff": None,
        "checkpoint": checkpoint if ok else None,
    }


def archive_run(project, run_id: int) -> list[Path]:
    """Freeze this run's outputs under `run{N}` names so later runs cannot overwrite them."""
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


class TrainStage(ScriptStageBase):
    name = "train"

    def is_complete(self, ctx: StageContext) -> bool:
        runs = read_runs(ctx.project.runs_path)
        if not any(r.get("status") == "done" for r in runs):
            return False
        metrics = ctx.project.read_json(cfg.METRICS_FILE) or {}
        started = metrics.get("started_at")
        if not started:
            return True
        return any(r.get("started_at") == started for r in runs)

    def prepare(self, ctx: StageContext) -> Handoff:
        project = ctx.project
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not all((project.root / f).exists() for f in CODE_FILES):
            raise RuntimeError("training project not found; run the codegen stage first")
        spec = ctx.spec()
        ctx.display(
            "Training runs on the [[CPU]] for tabular data, so there is no [[compute unit]] "
            f"cost gate for this run. `train.py` runs {config.get('epochs')} [[epoch]]s, "
            "redrawing the loss and metric curves as it goes, and saves the best model to "
            "`checkpoints/best.joblib`. `evaluate.py` then scores that model on the "
            f"[[validation set]] and draws the {spec.metric} figures. Run both cells."
        )
        ctx.teaching().preamble("train", {"config": config, "metric": spec.metric})
        return Handoff(
            stage=self.name,
            commands=[["train.py"], ["evaluate.py"]],
            outputs=[cfg.METRICS_FILE, EVAL_VAL_FILE],
        )

    def debrief(self, ctx: StageContext) -> None:
        project = ctx.project
        metrics = project.read_json(cfg.METRICS_FILE)
        if not isinstance(metrics, dict) or not metrics.get("started_at"):
            ctx.display(
                "I can't see a finished run in `metrics.json` yet. Run the `train.py` cell, "
                "then run this cell again."
            )
            return
        runs = read_runs(project.runs_path)
        entry = next(
            (r for r in runs if r.get("started_at") == metrics["started_at"]), None
        )
        if entry is None:
            ok = metrics.get("status") == "done"
            if ok:
                eval_data = project.read_json(EVAL_VAL_FILE)
                if (
                    not isinstance(eval_data, dict)
                    or eval_data.get("started_at") != metrics["started_at"]
                ):
                    ctx.display(
                        "The `train.py` run finished, but `eval_val.json` doesn't match it "
                        "yet. Run the `evaluate.py` cell, then run this cell again."
                    )
                    return
            run_id = len(runs) + 1
            figures = archive_run(project, run_id) if ok else []
            checkpoint = (
                f"checkpoints/run{run_id}.joblib"
                if ok and (project.checkpoints_dir / f"run{run_id}.joblib").exists()
                else None
            )
            entry = append_run(project.runs_path, build_run_entry(metrics, checkpoint))
        else:
            figures = sorted(
                project.plots_dir.glob(f"run{entry['run_id']}_*.png"),
                key=lambda p: (not p.name.endswith("_training.png"), p.name),
            )

        run_id = entry["run_id"]
        if entry["status"] != "done":
            ctx.display(
                f"Run {run_id} failed: {entry['error']}. Fix the cause (the traceback is in "
                "the `train.py` cell's output) and run the cells again."
            )
            return

        spec = ctx.spec()
        eval_data = project.read_json(EVAL_VAL_FILE) or {}
        ctx.display(self._narrative(ctx, spec, entry, metrics, eval_data, figures))

    def _narrative(self, ctx: StageContext, spec, entry: dict, metrics: dict,
                   eval_data: dict, figures: list[Path]) -> str:
        summary = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "run_id": entry["run_id"],
            "model_type": metrics.get("model_type"),
            "epochs": metrics.get("epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "stopped_early": metrics.get("stopped_early"),
            "validation": {"metric": eval_data.get("metric"), "value": eval_data.get("value"),
                           "loss": eval_data.get("loss")},
        }
        target = spec.target_value
        target_str = f"{target:g}" if isinstance(target, int | float) else _fmt(target)
        headline = (
            f"Run {entry['run_id']} finished: best validation {spec.metric} "
            f"{_fmt(entry['best_val_metric'])} at epoch {_fmt(entry['best_epoch'])} "
            f"(target {target_str})."
        )
        fallback = (
            "Look at the [[loss]] curves: if validation loss rises while training loss "
            "keeps falling, the model is [[overfitting]]."
        )
        narrative = ctx.teaching().debrief("train", summary, figures, fallback=fallback)
        return headline + "\n\n" + narrative
