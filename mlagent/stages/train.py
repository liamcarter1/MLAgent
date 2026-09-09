"""Train stage: hand the user train.py and evaluate.py, then log and explain the run."""

from __future__ import annotations

from pathlib import Path

from mlagent import config as cfg
from mlagent.runlog import read_runs
from mlagent.runs import EVAL_VAL_FILE, log_finished_run, run_problem
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.templates_io import CODE_FILES

__all__ = ["EVAL_VAL_FILE", "TrainStage"]


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


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
        # metrics.json/eval_val.json are derived outputs the cells regenerate; drop any
        # stale copy from an earlier run so it can't satisfy outputs_ready's mtime check
        # before the user has actually rerun train.py and evaluate.py this time.
        (project.root / cfg.METRICS_FILE).unlink(missing_ok=True)
        (project.root / EVAL_VAL_FILE).unlink(missing_ok=True)
        return Handoff(
            stage=self.name,
            commands=[["train.py"], ["evaluate.py"]],
            outputs=[cfg.METRICS_FILE, EVAL_VAL_FILE],
        )

    def debrief(self, ctx: StageContext) -> None:
        project = ctx.project
        problem = run_problem(project)
        if problem:
            ctx.display(problem)
            return
        logged = log_finished_run(project)
        if logged is None:
            return
        entry = logged.entry
        run_id = entry["run_id"]
        if entry["status"] != "done":
            ctx.display(
                f"Run {run_id} failed: {entry['error']}. Fix the cause (the traceback is in "
                "the `train.py` cell's output) and run the cells again."
            )
            return

        spec = ctx.spec()
        metrics = project.read_json(cfg.METRICS_FILE) or {}
        eval_data = project.read_json(EVAL_VAL_FILE) or {}
        ctx.display(self._narrative(ctx, spec, entry, metrics, eval_data, logged.figures))

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
