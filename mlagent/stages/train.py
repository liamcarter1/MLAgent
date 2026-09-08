"""Train stage: run train.py in a subprocess with a live plot; log and evaluate the run."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mlagent import config as cfg
from mlagent.llm import LLMError, ask_text
from mlagent.plots import present, present_evaluation, training_curves
from mlagent.prompts_io import audience, load_prompt
from mlagent.runlog import append_run, read_runs
from mlagent.runner import RunResult, run_training
from mlagent.stages.base import StageContext
from mlagent.templates_io import CODE_FILES

EVAL_VAL_FILE = "eval_val.json"
LOG_TAIL_SHOWN = 15


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class IPythonDisplay:
    """Updates a single Colab display cell in place.

    The first call creates a display handle; every later call updates that same
    handle instead of `clear_output(wait=True)`, which would wipe the whole cell
    transcript (intake questions, audit report, etc.) above the plot.
    """

    def __init__(self) -> None:
        self._handle = None

    def __call__(self, fig: Figure) -> None:
        try:
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            return
        if get_ipython() is None:
            return
        if self._handle is None:
            self._handle = display(fig, display_id=True)
        else:
            self._handle.update(fig)


class LivePlotter:
    """Redraws the training curves each time metrics.json gains an epoch."""

    def __init__(self, metric: str, display_fig: Callable[[Figure], None] | None = None):
        self.metric = metric
        self.updates = 0
        self._display = display_fig or IPythonDisplay()

    def update(self, metrics: dict) -> None:
        self.updates += 1
        epochs = metrics.get("epochs") or []
        if not epochs:
            return
        fig = training_curves(epochs, self.metric)
        try:
            self._display(fig)
        finally:
            plt.close(fig)


def build_run_entry(
    config: dict, result: RunResult, started_at: str, checkpoint: str | None = None
) -> dict:
    metrics = result.metrics or {}
    epochs = metrics.get("epochs") or []
    last = epochs[-1] if epochs else {}
    error = metrics.get("error")
    if error is None and result.timed_out:
        error = "timed out"
    elif error is None and result.returncode != 0:
        error = f"exit code {result.returncode}"
    return {
        "started_at": started_at,
        "status": "done" if result.ok else "failed",
        "config": dict(config),
        "epochs_run": len(epochs),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_metric": metrics.get("best_val_metric"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "seconds": result.seconds,
        "error": error,
        "applied_diff": None,
        "checkpoint": checkpoint if result.ok else None,
    }


class TrainStage:
    name = "train"

    def __init__(
        self,
        runner: Callable[..., RunResult] = run_training,
        python: str = sys.executable,
        timeout: float | None = None,
        display_fig: Callable[[Figure], None] | None = None,
        poll_seconds: float = 1.0,
    ):
        self.runner = runner
        self.python = python
        self.timeout = timeout
        self.display_fig = display_fig
        self.poll_seconds = poll_seconds

    def is_complete(self, ctx: StageContext) -> bool:
        return any(r.get("status") == "done" for r in read_runs(ctx.project.runs_path))

    def run(self, ctx: StageContext) -> None:
        project = ctx.project
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not all((project.root / f).exists() for f in CODE_FILES):
            raise RuntimeError("training project not found; run the codegen stage first")
        spec = ctx.spec()
        timeout = self.timeout
        if timeout is None:
            timeout = max(60.0, spec.minutes_per_run * 60 * 3)
        ctx.display(
            "Training runs on the [[CPU]] for tabular data, so there is no [[compute unit]] "
            "cost gate for this run. Watch the curves update each [[epoch]] (capped at "
            f"{timeout / 60:.0f} minutes)."
        )
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        plotter = LivePlotter(spec.metric, display_fig=self.display_fig)
        result = self.runner(
            project.root,
            on_line=lambda line: print(line, end=""),
            on_metrics=plotter.update,
            python=self.python,
            timeout=timeout,
            poll_seconds=self.poll_seconds,
        )
        run_id = len(read_runs(project.runs_path)) + 1
        checkpoint = None
        if result.ok:
            src = project.checkpoints_dir / "best.joblib"
            if src.exists():
                dst = project.checkpoints_dir / f"run{run_id}.joblib"
                shutil.copy2(src, dst)
                checkpoint = f"checkpoints/run{run_id}.joblib"
        entry = append_run(
            project.runs_path, build_run_entry(config, result, started_at, checkpoint)
        )
        run_id = entry["run_id"]
        if not result.ok:
            tail = "".join(result.log_tail[-LOG_TAIL_SHOWN:]).rstrip()
            ctx.display(
                f"Run {run_id} failed ({entry['error']}). Last lines of output:\n\n"
                f"```\n{tail}\n```\n\nFix the cause and rerun this stage."
            )
            return

        metrics = result.metrics or {}
        present(training_curves(metrics.get("epochs") or [], spec.metric), project.plots_dir,
                f"run{run_id}_training")
        eval_data = project.read_json(EVAL_VAL_FILE) or {}
        figure_paths = present_evaluation(eval_data, project.plots_dir, f"run{run_id}_val")
        ctx.display(self._debrief(ctx, spec, entry, metrics, eval_data, figure_paths))

    def _debrief(self, ctx: StageContext, spec, entry: dict, metrics: dict, eval_data: dict,
                 figure_paths: list[Path]) -> str:
        summary = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "run_id": entry["run_id"],
            "epochs": metrics.get("epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "stopped_early": metrics.get("stopped_early"),
            "validation": {"metric": eval_data.get("metric"), "value": eval_data.get("value"),
                           "loss": eval_data.get("loss")},
            "figures": [p.name for p in figure_paths],
        }
        headline = (
            f"Run {entry['run_id']} finished: best validation {spec.metric} "
            f"{_fmt(entry['best_val_metric'])} at epoch {_fmt(entry['best_epoch'])} "
            f"(target {spec.target_value:g})."
        )
        try:
            narrative = ask_text(
                ctx.llm,
                load_prompt("train", audience=audience(spec.learning_level)),
                json.dumps(summary, default=str),
            )
        except LLMError:
            narrative = "Look at the [[loss]] curves: if validation loss rises while training " \
                        "loss keeps falling, the model is [[overfitting]]."
        return headline + "\n\n" + narrative
