"""Report stage: one held-out test evaluation, then report.md with figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mlagent import config as cfg
from mlagent.runlog import best_run, read_runs, summarise
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext

EVAL_TEST_FILE = "eval_test.json"
EVAL_TEST_COMMAND = ["evaluate.py", "--split", "test"]


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
    ]
    checkpoint = eval_test.get("checkpoint")
    if checkpoint:
        lines.append(f"Checkpoint evaluated: `{checkpoint}`.")
    lines += [
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


def _load_runs_and_best(project, spec) -> tuple[list[dict], dict | None]:
    """The one place that decides "the best run": shared by `is_complete`, prepare and debrief.

    Narrows the candidate set to done runs with a non-null recorded `checkpoint` field,
    matching `evaluate.py`'s own `best_checkpoint()` filter (it checks the run entry, not
    whether the checkpoint file still exists on disk). Without this, a done run recorded
    with `checkpoint: None` could be "best" here but never actually evaluated by
    `evaluate.py`, leaving `debrief` refusing forever. The full `runs` list is still
    returned unnarrowed for the report's run-history table.
    """
    runs = read_runs(project.runs_path)
    candidates = [r for r in runs if r.get("status") == "done" and r.get("checkpoint")]
    return runs, best_run(candidates, spec.metric)


class ReportStage(ScriptStageBase):
    name = "report"

    def is_complete(self, ctx: StageContext) -> bool:
        project = ctx.project
        if not (project.report_path.exists() and project.exists(EVAL_TEST_FILE)):
            return False
        meta = project.read_json(cfg.REPORT_META_FILE)
        if not isinstance(meta, dict) or "best_run" not in meta:
            return False
        _runs, best = _load_runs_and_best(project, ctx.spec())
        if best is None:
            return False
        return meta["best_run"] == best.get("run_id")

    def prepare(self, ctx: StageContext) -> Handoff | None:
        project = ctx.project
        spec = ctx.spec()
        _runs, best = _load_runs_and_best(project, spec)
        if best is None:
            ctx.display("No successful training run yet; run the train stage first.")
            return None
        best_run_id = best["run_id"]
        existing = project.read_json(EVAL_TEST_FILE)
        meta = project.read_json(cfg.REPORT_META_FILE)
        meta_best = meta.get("best_run") if isinstance(meta, dict) else None

        if existing is not None and meta_best == best_run_id:
            ctx.display(
                f"The [[test set]] was already evaluated for run {best_run_id}; I will rewrite "
                "the report with the current run history instead of touching it again."
            )
            return None

        if existing is None:
            detail = (
                "The [[test set]] has been untouched until now; evaluating on it once gives "
                "an honest estimate of real-world performance."
            )
        elif meta_best is None:
            detail = (
                "An earlier test evaluation exists but its run is unknown; run "
                f"{best_run_id} is the best model, so it will be evaluated once more."
            )
        else:
            detail = (
                f"The test set was last evaluated for run {meta_best}; run {best_run_id} is now "
                "the best model, so it needs evaluating once more."
            )
        ctx.display(
            f"The best run so far is run {best_run_id} with validation {spec.metric} "
            f"{_fmt(best.get('best_val_metric'))}. {detail}"
        )
        if not ctx.questioner.confirm(
            "Evaluate the best model on the held-out test set now and write the report?",
            default=True,
        ):
            ctx.display("Skipped. Rerun this stage when you have finished tuning.")
            return None
        ctx.display(
            "Run the next cell. `evaluate.py --split test` loads run "
            f"{best_run_id}'s checkpoint and scores it once on the rows no model has seen."
        )
        # eval_test.json is derived output the cell regenerates; drop any stale copy so
        # a leftover from an earlier (worse) run can't satisfy outputs_ready's mtime check
        # or be mistaken by debrief for a fresh evaluation of the new best run.
        (project.root / EVAL_TEST_FILE).unlink(missing_ok=True)
        return Handoff(
            stage=self.name, commands=[list(EVAL_TEST_COMMAND)], outputs=[EVAL_TEST_FILE]
        )

    def debrief(self, ctx: StageContext) -> None:
        project = ctx.project
        spec = ctx.spec()
        runs, best = _load_runs_and_best(project, spec)
        if best is None:
            return
        eval_test = project.read_json(EVAL_TEST_FILE)
        if not isinstance(eval_test, dict):
            # Either the user hasn't run the handed-off cell yet (the orchestrator's
            # waiting message already told them to) or prepare() itself declined/skipped
            # and said why; either way there is nothing new to say here.
            return
        run_id = eval_test.get("run_id")
        if run_id is not None and run_id != best["run_id"]:
            ctx.display(
                f"`eval_test.json` was produced for run {run_id}, but run {best['run_id']} is "
                "the best model. Run the `evaluate.py --split test` cell again so the report "
                "describes the model you would actually ship."
            )
            return
        self._write_report(ctx, project, spec, runs, best, eval_test)

    def _write_report(self, ctx: StageContext, project, spec, runs: list[dict], best: dict,
                      eval_test: dict) -> None:
        test_figures = [
            project.plots_dir / str(name) for name in (eval_test.get("figures") or [])
        ]
        run_figures = sorted(project.plots_dir.glob("run*_training.png"), key=_run_number)

        lessons = self._lessons(ctx, spec, runs, best, eval_test, test_figures)
        report = render_report(
            project.name, spec.to_dict(), runs, best,
            eval_test, lessons, run_figures + test_figures,
        )
        project.report_path.write_text(report, encoding="utf-8")
        project.write_json(
            cfg.REPORT_META_FILE, {"best_run": best["run_id"], "n_runs": len(runs)}
        )
        run_id = eval_test.get("run_id")
        source = (
            f"run {run_id}" if run_id is not None else f"checkpoint `{eval_test.get('checkpoint')}`"
        )
        ctx.display(
            f"Test {spec.metric} for {source}: **{_fmt(eval_test.get('value'))}** "
            f"(target {spec.target_value:g}).\n\n{lessons}\n\n"
            f"Report written to `{cfg.REPORT_FILE}` in the project folder."
        )

    def _lessons(self, ctx: StageContext, spec, runs: list[dict], best: dict,
                 eval_test: dict, figures: list[Path]) -> str:
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
        fallback = (
            f"Best validation {spec.metric} was {_fmt(best.get('best_val_metric'))}; "
            f"the [[test set]] gave {_fmt(eval_test.get('value'))}. A large gap between "
            "them means the model does not [[generalise]] well."
        )
        return ctx.teaching().debrief("report", summary, figures, fallback=fallback)
