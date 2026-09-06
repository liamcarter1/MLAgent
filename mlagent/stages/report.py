"""Report stage: one held-out test evaluation, then report.md with figures."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from mlagent import config as cfg
from mlagent.llm import LLMError, ask_text
from mlagent.plots import present_evaluation
from mlagent.prompts_io import load_prompt
from mlagent.runlog import best_run, read_runs, summarise
from mlagent.runner import RunResult, run_script
from mlagent.stages.base import StageContext

EVAL_TEST_FILE = "eval_test.json"
LOG_TAIL_SHOWN = 15


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _run_number(path: Path) -> int:
    """Parse the run id out of "run<N>_..." for numeric (not lexicographic) sorting."""
    m = re.search(r"^run(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def render_report(project_name: str, spec: dict, runs: list[dict], best: dict | None,
                  eval_test: dict, lessons: str, figures: list[Path]) -> str:
    metric = spec.get("metric", "")
    best_cfg = (best or {}).get("config") or {}
    lines = [
        f"# {project_name}: training report",
        "",
        f"**Goal:** {spec.get('goal', '')}",
        "",
        f"**Task:** {spec.get('task_type', '')} | **Metric:** {metric} | "
        f"**Target:** {_fmt(spec.get('target_value'))}",
        "",
        "## Run history",
        "",
        summarise(runs, metric),
        "",
        "## Best configuration",
        "",
        f"Run {(best or {}).get('run_id', '-')} with validation {metric} "
        f"{_fmt((best or {}).get('best_val_metric'))}:",
        "",
        "```json",
        json.dumps(best_cfg, indent=2, sort_keys=True),
        "```",
        "",
        "## Held-out test result",
        "",
        f"Test {eval_test.get('metric', metric)}: **{_fmt(eval_test.get('value'))}** "
        f"(loss {_fmt(eval_test.get('loss'))}). Evaluated once on the test split.",
        "",
        "## What we learned",
        "",
        lessons.strip(),
        "",
        "## Figures",
        "",
    ]
    for path in figures:
        rel = Path("plots") / path.name
        lines.append(f"![{path.stem}]({rel.as_posix()})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class ReportStage:
    name = "report"

    def __init__(
        self,
        runner: Callable[..., RunResult] = run_script,
        python: str = sys.executable,
        timeout: float | None = None,
        poll_seconds: float = 0.5,
    ):
        self.runner = runner
        self.python = python
        self.timeout = timeout
        self.poll_seconds = poll_seconds

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.report_path.exists()

    def run(self, ctx: StageContext) -> None:
        project = ctx.project
        spec = ctx.spec()
        runs = read_runs(project.runs_path)
        best = best_run(runs, spec.metric)
        if best is None:
            ctx.display("No successful training run yet; run the train stage first.")
            return
        ctx.display(
            f"The best run so far is run {best['run_id']} with validation {spec.metric} "
            f"{_fmt(best.get('best_val_metric'))}. The [[test set]] has been untouched until now; "
            "evaluating on it once gives an honest estimate of real-world performance."
        )
        if not ctx.questioner.confirm(
            "Evaluate the best model on the held-out test set now and write the report?",
            default=True,
        ):
            ctx.display("Skipped. Rerun this stage when you have finished tuning.")
            return

        args = ["train.py", "--eval-test"]
        checkpoint = best.get("checkpoint")
        if checkpoint and (project.root / checkpoint).exists():
            args += ["--checkpoint", checkpoint]
        result = self.runner(
            project.root,
            args,
            python=self.python,
            timeout=self.timeout,
            poll_seconds=self.poll_seconds,
        )
        if not result.ok:
            tail = "".join(result.log_tail[-LOG_TAIL_SHOWN:]).rstrip()
            ctx.display(f"Test evaluation failed. Last lines of output:\n\n```\n{tail}\n```")
            return
        eval_test = project.read_json(EVAL_TEST_FILE) or {}
        test_figures = present_evaluation(eval_test, project.plots_dir, "test")
        run_figures = sorted(project.plots_dir.glob("run*_training.png"), key=_run_number)

        lessons = self._lessons(ctx, spec, runs, best, eval_test)
        report = render_report(
            project.name, spec.to_dict(), runs, best,
            eval_test, lessons, run_figures + test_figures,
        )
        project.report_path.write_text(report, encoding="utf-8")
        ctx.display(
            f"Test {spec.metric}: **{_fmt(eval_test.get('value'))}** "
            f"(target {spec.target_value:g}).\n\n{lessons}\n\n"
            f"Report written to `{cfg.REPORT_FILE}` in the project folder."
        )

    def _lessons(self, ctx: StageContext, spec, runs: list[dict], best: dict,
                 eval_test: dict) -> str:
        summary = {
            "spec": spec.to_dict(),
            "runs": [
                {k: r.get(k) for k in ("run_id", "status", "config", "best_epoch",
                                        "best_val_metric", "final_train_loss",
                                        "final_val_loss", "error")}
                for r in runs
            ],
            "best_run_id": best.get("run_id"),
            "test": {"metric": eval_test.get("metric"), "value": eval_test.get("value"),
                     "loss": eval_test.get("loss")},
        }
        try:
            return ask_text(ctx.llm, load_prompt("report"), json.dumps(summary, default=str))
        except LLMError:
            return (
                f"Best validation {spec.metric} was {_fmt(best.get('best_val_metric'))}; "
                f"the [[test set]] gave {_fmt(eval_test.get('value'))}. A large gap between "
                "them means the model does not [[generalise]] well."
            )
