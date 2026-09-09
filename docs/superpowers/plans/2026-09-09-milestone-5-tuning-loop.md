# Milestone 5: Tuning Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the first training run, a new `tune` stage diagnoses the run history, proposes one to three ranked configuration changes, applies the one the user picks, has the user rerun `train.py` and `evaluate.py`, logs the new run with the diff that produced it, draws comparison figures, and goes round again until the user stops, the target is met, or `max_rounds` is reached.

**Architecture:** Three pure modules do the work and are unit-tested alone: `mlagent/runs.py` (log a finished run and archive its figures, metrics and checkpoint; shared by train and tune), `mlagent/diagnose.py` (deterministic curve diagnosis, heuristic proposals, `apply_proposal`), and two comparison figures in `mlagent/plots.py`. `mlagent/stages/tune.py` is a two-phase script stage whose handoff reuses the train cells; its `debrief` returns `True` to make the orchestrator re-prepare it in the same `orch.run()`. `tune_state.json` carries the loop across Colab resets. The config edit menu moves from codegen into `templates_io.edit_config` so both stages share it.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, matplotlib (Agg in tests), pytest, ruff. Tests never hit the network: every LLM call takes an `LLM` and tests pass `FakeLLM`.

**Spec:** `docs/superpowers/specs/2026-09-09-milestone-5-tuning-loop-design.md` (all six decisions confirmed by the user on 2026-09-09). Its parent is `docs/superpowers/specs/2026-09-08-milestone-4-learning-mode-design.md`.

## Global Constraints

- Pipeline order after this milestone: `intake -> data -> clean -> codegen -> train -> tune -> report`.
- Generated scripts (`mlagent/templates/**`) are NOT modified by this milestone. No script patching, ever; a failed run gets a configuration proposal.
- The tune handoff is exactly `Handoff("tune", [["train.py"], ["evaluate.py"]], ["metrics.json", "eval_val.json"])`.
- `tune_state.json` shape: `{"round": int, "max_rounds": int, "decision": "continue" | "stopped" | "target_met" | "rounds_exhausted", "pending": null | {"round": int, "applied_diff": {key: {"from": old, "to": new}}, "reason": str, "diagnosis": str, "proposed_at": iso}, "history": [{"round": int, "diagnosis": str, "applied_diff": {...}, "run_id": int, "improved": bool}]}`.
- `runs.jsonl` entry gains `applied_diff`: `null` for a run the tuner did not produce, else the `{key: {"from", "to"}}` dict. Every other key is unchanged.
- `runs/run{N}_metrics.json` is a verbatim copy of `metrics.json` at debrief time, written for done AND failed runs. `runs/` is created by `project.ensure_dirs()`.
- The `propose_diffs` tool schema is `{"proposals": [{"rank": int, "changes": object, "reason": str, "expected": "faster" | "better" | "steadier"}]}` with `maxItems: 3`.
- Every LLM path degrades on `LLMError`: heuristic proposals, fixed narrative, fixed captions. At expert level: no preamble, no primer, no figure notes.
- Comparison figures are drawn agent-side in `mlagent/plots.py` and saved as `plots/compare_curves.png` and `plots/compare_runs.png`, overwritten each round. No `compare.py` template.
- The tune notebook cell has no `#@param` fields; its one choice uses the console questioner. `max_rounds` comes from `Spec.max_rounds` (intake).
- Prompts live in `mlagent/prompts/*.md`, never inline in Python; every stage prompt ends with `Audience: {audience}`.
- Write text files with `encoding="utf-8"`; `pathlib` everywhere. `ruff check .` clean; `python -m pytest` green; the chart tests also pass under `-W error::DeprecationWarning`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W
  ```

## Rulings made while planning (deviations from the spec's letter)

1. `log_finished_run` returns a `LoggedRun` dataclass (`entry`, `figures`, `new`) rather than a bare tuple, and a sibling `run_problem(project) -> str | None` supplies the "not ready" message; the spec's `| None` return is kept (None exactly when `run_problem` is not None). Reason: the train stage's "already logged, narrate again" branch needs to know the run was not new.
2. The metrics archive is written for failed runs too (`archive_metrics`), so a `failed_run` diagnosis can quote the error and the comparison chart can show what epochs ran. Figures and checkpoints are still archived only for done runs.
3. The orchestrator does not know the name `tune_state.json`. `reset(name)` calls an optional `on_reset(ctx)` hook on every stage from `name` onwards; `TuneStage.on_reset` deletes the file. `reset("train")` therefore clears the loop, as the spec requires, without a hard-coded filename.
4. `compare_curves(runs, metrics_by_run)` takes the run entries as well as the metrics, because the legend labels come from each run's `applied_diff`. `compare_runs(runs, metric, target)` matches the spec.
5. The comparison figures are saved with `plots.save_and_close` (new) rather than `present`, because `Teaching.debrief` already displays them with captions; `present` would show each figure twice in Colab.
6. Proposals are shown with a `proposal_table` (key, now, proposed, description) rather than `config_table`, which has no "now" column.
7. When no proposal survives coercion (all collapse to no change), the stage records `decision = "stopped"` and says so, so the report stage can run; the user keeps the Train again cell.
8. `config_table` moves to `templates_io` alongside `edit_config`; codegen imports both (its public name `codegen.config_table` still resolves).

## File structure

| File | Responsibility |
|---|---|
| `mlagent/config.py` | add `RUNS_DIRNAME = "runs"`, `TUNE_STATE_FILE = "tune_state.json"` |
| `mlagent/project.py` | add `runs_dir`; `ensure_dirs` creates it |
| `mlagent/runs.py` (new) | `LoggedRun`, `build_run_entry`, `archive_run`, `archive_metrics`, `run_metrics_path`, `read_run_metrics`, `run_problem`, `log_finished_run` |
| `mlagent/stages/train.py` | debrief via `runs.log_finished_run`; keeps `_narrative` |
| `mlagent/diagnose.py` (new) | `Diagnosis`, `Proposal`, `meets_target`, `diagnose`, `heuristic_proposals`, `diff_config`, `apply_proposal` |
| `mlagent/plots.py` | `run_label`, `compare_curves`, `compare_runs`, `save_and_close` |
| `mlagent/captions.py` | captions for `compare_curves`, `compare_runs` |
| `mlagent/templates_io.py` | `config_table`, `edit_config` (moved from codegen) |
| `mlagent/stages/codegen.py` | imports the two above; `_edit_config` and its `config_table` removed |
| `mlagent/stages/base.py` | `stage_debrief` returns the debrief's value; `stage_reset` hook |
| `mlagent/orchestrator.py` | re-prepare on truthy debrief; `reset` calls `stage_reset` |
| `mlagent/prompts/tune.md`, `tune_debrief.md`, `teaching/tuning.md` (new) | the tuner's prompts and primer |
| `mlagent/stages/tune.py` (new) | `TuneStage` |
| `mlagent/colab.py`, `scripts/build_notebook.py`, `notebooks/ML_Training_Agent.ipynb` | wiring and the Tune cell |
| `docs/colab-smoke.md`, `CLAUDE.md` | Milestone 5 checklist; architecture notes |
| tests | `test_runs.py`, `test_diagnose.py`, `test_tune_stage.py` (new); `test_plots.py`, `test_captions.py`, `test_templates_io.py`, `test_orchestrator.py`, `test_train_stage.py`, `test_colab.py`, `test_pipeline_e2e.py`, `test_prompts_io.py` (modified) |

---

### Task 1: `mlagent/runs.py` — one place that logs a finished run

**Files:**
- Create: `mlagent/runs.py`
- Modify: `mlagent/config.py`, `mlagent/project.py`, `mlagent/stages/train.py`
- Test: `tests/test_runs.py` (new), `tests/test_train_stage.py`, `tests/test_project.py`

**Interfaces:**
- Consumes: `runlog.read_runs`, `runlog.append_run`, `Project.read_json/write_json`, `project.write_json_file`, `project.read_json_file`.
- Produces (used by Tasks 7 and 8):
  - `EVAL_VAL_FILE = "eval_val.json"`, `CURVES_FIGURE`, `BEST_CHECKPOINT`
  - `@dataclass LoggedRun(entry: dict, figures: list[Path], new: bool)`
  - `build_run_entry(metrics: dict, checkpoint: str | None = None, applied_diff: dict | None = None) -> dict`
  - `archive_run(project, run_id: int) -> list[Path]`
  - `run_metrics_path(project, run_id: int) -> Path`
  - `archive_metrics(project, run_id: int, metrics: dict) -> Path`
  - `read_run_metrics(project, runs: list[dict]) -> dict[int, dict]`
  - `run_problem(project) -> str | None`
  - `log_finished_run(project, applied_diff: dict | None = None) -> LoggedRun | None`
  - `config.RUNS_DIRNAME`, `config.TUNE_STATE_FILE`, `Project.runs_dir`

- [ ] **Step 1: Add the constants and the runs directory**

In `mlagent/config.py`, after `REPORT_META_FILE = "report_meta.json"`:

```python
RUNS_DIRNAME = "runs"
TUNE_STATE_FILE = "tune_state.json"
```

In `mlagent/project.py`, after the `checkpoints_dir` property:

```python
    @property
    def runs_dir(self) -> Path:
        """Per-run archives (`run{N}_metrics.json`); runs.jsonl itself is `runs_path`."""
        return self.root / config.RUNS_DIRNAME
```

and change `ensure_dirs` to:

```python
    def ensure_dirs(self) -> None:
        for d in (self.root, self.data_raw, self.data_clean, self.plots_dir,
                  self.checkpoints_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
```

Append to `tests/test_project.py`:

```python
def test_ensure_dirs_creates_the_runs_archive_dir(tmp_path):
    from mlagent.project import Project

    p = Project(tmp_path / "proj")
    p.ensure_dirs()
    assert p.runs_dir == tmp_path / "proj" / "runs" and p.runs_dir.is_dir()
```

- [ ] **Step 2: Write the failing tests for `runs.py`**

Create `tests/test_runs.py`:

```python
from __future__ import annotations

from mlagent import runs
from mlagent.runlog import read_runs

STARTED = "2026-09-09T10:00:00.000000+00:00"


def write_fake_run(project, started=STARTED, status="done", epochs=3, best_val=0.8):
    """Fake what train.py and evaluate.py leave behind, without running them."""
    metrics = {
        "status": status, "started_at": started, "model_type": "gradient_boosting",
        "task_type": "tabular_classification", "metric": "accuracy",
        "config": {"model_type": "gradient_boosting", "epochs": epochs, "learning_rate": 0.1},
        "epochs": [
            {"epoch": i + 1, "train_loss": 1.0 / (i + 1), "val_loss": 1.1 / (i + 1),
             "train_metric": 0.5, "val_metric": 0.6, "seconds": 0.1}
            for i in range(epochs)
        ],
        "best_epoch": epochs if status == "done" else None,
        "best_val_metric": best_val if status == "done" else None,
        "stopped_early": False,
        "error": None if status == "done" else "ValueError: boom",
        "seconds": 0.3,
    }
    project.write_json("metrics.json", metrics)
    if status == "done":
        project.write_json("eval_val.json", {"started_at": started, "metric": "accuracy",
                                             "value": best_val, "loss": 0.4})
        (project.plots_dir / "training_curves.png").write_bytes(b"png")
        (project.plots_dir / "val_confusion.png").write_bytes(b"png")
        (project.checkpoints_dir / "best.joblib").write_bytes(b"model")
    return metrics


def test_no_metrics_is_a_problem(project):
    assert "metrics.json" in runs.run_problem(project)
    assert runs.log_finished_run(project) is None


def test_stale_eval_val_is_a_problem(project):
    write_fake_run(project)
    project.write_json("eval_val.json", {"started_at": "other"})
    assert "evaluate.py" in runs.run_problem(project)
    assert runs.log_finished_run(project) is None
    assert read_runs(project.runs_path) == []


def test_logging_a_done_run_archives_everything(project):
    metrics = write_fake_run(project)
    diff = {"learning_rate": {"from": 0.2, "to": 0.1}}
    logged = runs.log_finished_run(project, applied_diff=diff)
    assert logged.new is True
    entry = logged.entry
    assert entry["run_id"] == 1 and entry["status"] == "done"
    assert entry["applied_diff"] == diff
    assert entry["checkpoint"] == "checkpoints/run1.joblib"
    assert (project.checkpoints_dir / "run1.joblib").exists()
    assert [p.name for p in logged.figures] == ["run1_training.png", "run1_val_confusion.png"]
    assert project.read_json("runs/run1_metrics.json") == metrics
    assert runs.run_metrics_path(project, 1) == project.runs_dir / "run1_metrics.json"
    assert runs.run_problem(project) is None  # already logged is not a problem


def test_logging_the_same_run_twice_returns_the_existing_entry(project):
    write_fake_run(project)
    first = runs.log_finished_run(project)
    second = runs.log_finished_run(project)
    assert second.new is False and second.entry == first.entry
    assert [p.name for p in second.figures] == [p.name for p in first.figures]
    assert len(read_runs(project.runs_path)) == 1


def test_a_failed_run_archives_metrics_but_not_figures(project):
    (project.plots_dir / "training_curves.png").write_bytes(b"leftover")
    (project.checkpoints_dir / "best.joblib").write_bytes(b"leftover")
    write_fake_run(project, status="failed")
    logged = runs.log_finished_run(project)
    assert logged.entry["status"] == "failed" and logged.entry["checkpoint"] is None
    assert logged.entry["applied_diff"] is None
    assert logged.figures == []
    assert not (project.plots_dir / "run1_training.png").exists()
    assert not (project.checkpoints_dir / "run1.joblib").exists()
    assert project.read_json("runs/run1_metrics.json")["status"] == "failed"


def test_read_run_metrics_returns_only_archived_runs(project):
    write_fake_run(project)
    runs.log_finished_run(project)
    write_fake_run(project, started="2026-09-09T11:00:00.000000+00:00", epochs=2)
    runs.log_finished_run(project)
    history = read_runs(project.runs_path)
    assert [r["run_id"] for r in history] == [1, 2]
    by_run = runs.read_run_metrics(project, history)
    assert set(by_run) == {1, 2}
    assert len(by_run[1]["epochs"]) == 3 and len(by_run[2]["epochs"]) == 2
    (project.runs_dir / "run1_metrics.json").unlink()
    assert set(runs.read_run_metrics(project, history)) == {2}


def test_build_run_entry_carries_applied_diff():
    entry = runs.build_run_entry({"status": "done", "started_at": "t", "epochs": []},
                                 checkpoint="checkpoints/run1.joblib",
                                 applied_diff={"epochs": {"from": 5, "to": 10}})
    assert entry["applied_diff"] == {"epochs": {"from": 5, "to": 10}}
    assert entry["checkpoint"] == "checkpoints/run1.joblib"
    assert runs.build_run_entry({"status": "failed", "epochs": []})["applied_diff"] is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_runs.py tests/test_project.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.runs'` (and the project test fails on the missing `runs_dir` until Step 1 is applied).

- [ ] **Step 4: Write `mlagent/runs.py`**

```python
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
```

- [ ] **Step 5: Point the train stage at `runs.py`**

Replace the top of `mlagent/stages/train.py` (everything above `class TrainStage`) with:

```python
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
```

(`build_run_entry`, `archive_run`, `CURVES_FIGURE`, `BEST_CHECKPOINT` and the `shutil` import leave this file.) Replace `TrainStage.debrief` with:

```python
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
```

`is_complete`, `prepare` and `_narrative` are unchanged.

In `tests/test_train_stage.py` change the import line

```python
from mlagent.stages.train import EVAL_VAL_FILE, TrainStage, archive_run, build_run_entry
```

to

```python
from mlagent.runs import archive_run, build_run_entry
from mlagent.stages.train import EVAL_VAL_FILE, TrainStage
```

and in `test_real_training_run_is_logged_archived_and_debriefed`, after the `run1.joblib` assertion, add:

```python
    assert project.read_json("runs/run1_metrics.json")["started_at"] == runs[0]["started_at"]
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_runs.py tests/test_project.py tests/test_train_stage.py tests/test_report_stage.py -v`
Expected: PASS. Then `ruff check .` clean.

- [ ] **Step 7: Commit**

```bash
git add mlagent/runs.py mlagent/config.py mlagent/project.py mlagent/stages/train.py tests/test_runs.py tests/test_project.py tests/test_train_stage.py
git commit -m "feat: runs.log_finished_run logs and archives a finished run for train and tune

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 2: `mlagent/diagnose.py` — diagnosis, heuristic proposals, apply_proposal

**Files:**
- Create: `mlagent/diagnose.py`
- Test: `tests/test_diagnose.py` (new)

**Interfaces:**
- Consumes: `runlog.HIGHER_IS_BETTER`, `runlog.best_run`, `runlog.is_better`, `templates_io.schema_for`, `templates_io.default_config`, `templates_io.coerce_config`, `Spec` (`metric`, `target_value`).
- Produces (used by Task 7):
  - `LABELS`, `EXPECTATIONS = ("faster", "better", "steadier")`
  - `@dataclass Diagnosis(label: str, evidence: dict, latest_run_id: int | None, best_run_id: int | None, improved: bool)`
  - `@dataclass Proposal(rank: int, changes: dict[str, object], reason: str, expected: str = "better")`
  - `meets_target(metric: str, value, target) -> bool`
  - `diagnose(runs: list[dict], metrics_by_run: dict[int, dict], spec) -> Diagnosis` (raises `ValueError` on an empty history)
  - `heuristic_proposals(diagnosis: Diagnosis, config: dict, schema: dict) -> list[Proposal]`
  - `diff_config(old: dict, new: dict) -> dict[str, dict]`
  - `apply_proposal(config: dict, proposal: Proposal, nested_schema: dict) -> tuple[dict, dict, list[str]]` (raises `ValueError` on an unknown `model_type`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnose.py`:

```python
from __future__ import annotations

import pytest

from mlagent.diagnose import (
    LABELS,
    Diagnosis,
    Proposal,
    apply_proposal,
    diagnose,
    diff_config,
    heuristic_proposals,
    meets_target,
)
from mlagent.spec import Spec
from mlagent.templates_io import load_schema, schema_for, validate_config

NESTED = load_schema("tabular_sklearn")


def spec(metric="accuracy", target=0.9) -> Spec:
    return Spec(goal="g", task_type="tabular_classification", metric=metric,
                target_value=target, data_source="synthetic", minutes_per_run=5,
                max_rounds=3, gpu="none")


def run(run_id, val, train=None, status="done", best_val=0.7, error=None, stopped_early=False,
        best_epoch=None):
    """A runs.jsonl entry plus its archived metrics, from a list of per-epoch losses."""
    train = train if train is not None else [v * 0.9 for v in val]
    epochs = [{"epoch": i + 1, "train_loss": t, "val_loss": v, "train_metric": 0.5,
               "val_metric": 0.6, "seconds": 0.1} for i, (t, v) in enumerate(zip(train, val))]
    if best_epoch is None:
        best_epoch = 1 + min(range(len(val)), key=lambda i: val[i]) if val else None
    entry = {"run_id": run_id, "status": status, "best_val_metric": best_val if status == "done"
             else None, "best_epoch": best_epoch, "epochs_run": len(val), "error": error,
             "applied_diff": None, "checkpoint": f"checkpoints/run{run_id}.joblib"
             if status == "done" else None, "config": {}}
    metrics = {"status": status, "epochs": epochs, "best_epoch": best_epoch,
               "stopped_early": stopped_early}
    return entry, metrics


def history(*pairs):
    runs = [entry for entry, _m in pairs]
    by_run = {entry["run_id"]: m for entry, m in pairs}
    return runs, by_run


def test_labels_are_the_seven_from_the_spec():
    assert set(LABELS) == {"overfitting", "underfitting", "learning_rate_too_high", "plateau",
                           "failed_run", "improving", "target_met"}


def test_meets_target_respects_metric_direction():
    assert meets_target("accuracy", 0.9, 0.9) and not meets_target("accuracy", 0.89, 0.9)
    assert meets_target("rmse", 4.0, 5.0) and not meets_target("rmse", 6.0, 5.0)
    assert not meets_target("accuracy", None, 0.9)


def test_empty_history_raises():
    with pytest.raises(ValueError):
        diagnose([], {}, spec())


def test_failed_latest_run_is_failed_run():
    runs, by_run = history(run(1, [1.0, 0.8, 0.7]),
                           run(2, [1.0], status="failed", error="ValueError: boom"))
    d = diagnose(runs, by_run, spec())
    assert d.label == "failed_run" and d.evidence["error"] == "ValueError: boom"
    assert d.evidence["n_failed"] == 1 and d.latest_run_id == 2 and d.best_run_id == 1


def test_target_met_beats_curve_shape():
    runs, by_run = history(run(1, [1.0, 0.9, 1.2, 1.4], best_val=0.95))
    assert diagnose(runs, by_run, spec(target=0.9)).label == "target_met"


def test_overfitting_when_validation_rises_while_training_falls():
    val = [1.0, 0.8, 0.7, 0.72, 0.8, 0.9]
    train = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    runs, by_run = history(run(1, val, train))
    d = diagnose(runs, by_run, spec())
    assert d.label == "overfitting"
    assert d.evidence["val_trend"] > 0 and d.evidence["train_trend"] < 0
    assert d.evidence["best_is_last"] is False


def test_underfitting_when_still_falling_at_the_end():
    val = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    runs, by_run = history(run(1, val))
    d = diagnose(runs, by_run, spec())
    assert d.label == "underfitting" and d.evidence["best_is_last"] is True


def test_improving_when_the_latest_run_beat_the_previous_best_and_is_still_falling():
    runs, by_run = history(run(1, [1.0, 0.9, 0.85], best_val=0.7),
                           run(2, [1.0, 0.8, 0.6, 0.5], best_val=0.8))
    d = diagnose(runs, by_run, spec())
    assert d.label == "improving" and d.improved is True and d.best_run_id == 2


def test_learning_rate_too_high_when_validation_oscillates():
    val = [1.0, 0.6, 1.1, 0.5, 1.2, 0.55, 1.3]
    runs, by_run = history(run(1, val, [v * 0.9 for v in val]))
    d = diagnose(runs, by_run, spec())
    assert d.label == "learning_rate_too_high" and d.evidence["oscillation"] >= 0.5


def test_plateau_when_flat():
    val = [1.0, 0.7, 0.6, 0.6, 0.6, 0.6]
    runs, by_run = history(run(1, val, [v * 0.95 for v in val]))
    assert diagnose(runs, by_run, spec()).label == "plateau"


def test_too_few_epochs_reads_as_underfitting():
    runs, by_run = history(run(1, [1.0, 0.9]))
    assert diagnose(runs, by_run, spec()).label == "underfitting"


def test_missing_metrics_archive_falls_back_to_plateau():
    runs, _by_run = history(run(1, [1.0, 0.9, 0.8]))
    assert diagnose(runs, {}, spec()).label == "plateau"


@pytest.mark.parametrize("family", ["gradient_boosting", "random_forest", "linear"])
@pytest.mark.parametrize("label", [lab for lab in LABELS if lab != "target_met"])
def test_heuristic_proposals_stay_inside_the_schema(family, label):
    flat = schema_for(NESTED, family)
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = family
    d = Diagnosis(label=label, evidence={"error": "boom"}, latest_run_id=1, best_run_id=1,
                  improved=(label == "improving"))
    proposals = heuristic_proposals(d, config, flat)
    assert len(proposals) == 1 and proposals[0].rank == 1
    assert proposals[0].reason and proposals[0].expected in ("faster", "better", "steadier")
    new_config, diff, _notes = apply_proposal(config, proposals[0], NESTED)
    assert diff, (family, label)
    family_after = new_config["model_type"]
    assert validate_config(new_config, schema_for(NESTED, family_after)) == []


def test_target_met_has_no_heuristic():
    d = Diagnosis(label="target_met", evidence={}, latest_run_id=1, best_run_id=1)
    assert heuristic_proposals(d, {"model_type": "linear"}, schema_for(NESTED, "linear")) == []


def test_diff_config_lists_changed_added_and_removed_keys():
    assert diff_config({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == {
        "b": {"from": 2, "to": 3}, "c": {"from": None, "to": 4}}
    assert diff_config({"a": 1}, {})["a"] == {"from": 1, "to": None}


def test_apply_proposal_coerces_and_reports_the_diff():
    flat = schema_for(NESTED, "gradient_boosting")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = "gradient_boosting"
    proposal = Proposal(rank=1, changes={"learning_rate": 5.0, "bogus": 1, "epochs": "20"},
                        reason="r")
    new_config, diff, notes = apply_proposal(config, proposal, NESTED)
    assert new_config["learning_rate"] == 1.0 and new_config["epochs"] == 20
    assert "bogus" not in new_config
    assert diff == {"learning_rate": {"from": 0.1, "to": 1.0}, "epochs": {"from": 10, "to": 20}}
    assert any("Lowered learning_rate" in n for n in notes)
    assert any("bogus" in n for n in notes)


def test_apply_proposal_with_no_effective_change_has_an_empty_diff():
    flat = schema_for(NESTED, "linear")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = "linear"
    proposal = Proposal(rank=1, changes={"alpha": config["alpha"]}, reason="r")
    _new, diff, _notes = apply_proposal(config, proposal, NESTED)
    assert diff == {}


def test_family_switch_fills_the_target_family_defaults_and_keeps_common_keys():
    flat = schema_for(NESTED, "linear")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config.update({"model_type": "linear", "epochs": 7, "seed": 3})
    proposal = Proposal(rank=1, changes={"model_type": "random_forest", "trees_per_epoch": 50},
                        reason="r")
    new_config, diff, _notes = apply_proposal(config, proposal, NESTED)
    assert new_config["model_type"] == "random_forest"
    assert new_config["epochs"] == 7 and new_config["seed"] == 3
    assert new_config["trees_per_epoch"] == 50 and "alpha" not in new_config
    assert validate_config(new_config, schema_for(NESTED, "random_forest")) == []
    assert diff["model_type"] == {"from": "linear", "to": "random_forest"}
    assert diff["alpha"]["to"] is None and diff["trees_per_epoch"]["from"] is None


def test_apply_proposal_rejects_an_unknown_family():
    config = {"model_type": "linear"}
    with pytest.raises(ValueError):
        apply_proposal(config, Proposal(rank=1, changes={"model_type": "svm"}, reason="r"),
                       NESTED)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diagnose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.diagnose'`.

- [ ] **Step 3: Write `mlagent/diagnose.py`**

```python
"""Read a run history and say what the curves show; propose the next change without an LLM.

Pure and deterministic. The tune stage calls `diagnose` before asking Claude for proposals,
falls back to `heuristic_proposals` when Claude is unavailable, and turns any proposal,
Claude's or ours, into a validated config plus the diff that produced it with
`apply_proposal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlagent.runlog import HIGHER_IS_BETTER, best_run, is_better
from mlagent.templates_io import coerce_config, default_config, schema_for

LABELS = (
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "plateau",
    "failed_run",
    "improving",
    "target_met",
)
EXPECTATIONS = ("faster", "better", "steadier")

# Relative change in validation loss over the last third of the epochs that still counts
# as flat; a relative train/val gap at the best epoch that counts as overfitting; the share
# of epoch-to-epoch validation-loss changes that flip sign before we call it oscillation.
TREND_FLAT = 0.02
GAP_LARGE = 0.25
OSCILLATING = 0.5
MIN_EPOCHS_TO_JUDGE = 3
MIN_EPOCHS_FOR_OSCILLATION = 4


@dataclass
class Diagnosis:
    label: str
    evidence: dict = field(default_factory=dict)
    latest_run_id: int | None = None
    best_run_id: int | None = None
    improved: bool = False


@dataclass
class Proposal:
    rank: int
    changes: dict[str, object]
    reason: str
    expected: str = "better"


def meets_target(metric: str, value, target) -> bool:
    if not isinstance(value, int | float) or not isinstance(target, int | float):
        return False
    if isinstance(value, bool) or isinstance(target, bool):
        return False
    if HIGHER_IS_BETTER.get(metric, True):
        return value >= target
    return value <= target


def _trend(values: list[float]) -> float:
    """Relative change from the start of the last third to the end; negative is falling."""
    if len(values) < 2:
        return 0.0
    window = values[-max(2, -(-len(values) // 3)):]
    first, last = window[0], window[-1]
    return (last - first) / max(abs(first), 1e-9)


def _oscillation(values: list[float]) -> float:
    """Share of consecutive changes that flip direction; 1.0 is a perfect zigzag."""
    deltas = [b - a for a, b in zip(values, values[1:]) if b != a]
    if len(deltas) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(deltas, deltas[1:]) if (a > 0) != (b > 0))
    return flips / (len(deltas) - 1)


def diagnose(runs: list[dict], metrics_by_run: dict[int, dict], spec) -> Diagnosis:
    """One label for the latest run, with the numbers that justify it."""
    if not runs:
        raise ValueError("no runs to diagnose")
    latest = runs[-1]
    latest_id = latest.get("run_id")
    best = best_run(runs, spec.metric)
    best_id = best.get("run_id") if best else None
    previous_best = best_run([r for r in runs if r is not latest], spec.metric)
    improved = bool(
        latest.get("status") == "done"
        and previous_best is not None
        and is_better(spec.metric, latest.get("best_val_metric"),
                      previous_best.get("best_val_metric"))
    )
    evidence: dict = {
        "n_runs": len(runs),
        "n_failed": sum(1 for r in runs if r.get("status") != "done"),
        "best_val_metric": best.get("best_val_metric") if best else None,
        "previous_best": previous_best.get("best_val_metric") if previous_best else None,
        "target_value": spec.target_value,
    }
    if latest.get("status") != "done":
        evidence["error"] = str(latest.get("error") or "unknown error")
        return Diagnosis("failed_run", evidence, latest_id, best_id, False)
    if best is not None and meets_target(spec.metric, best.get("best_val_metric"),
                                         spec.target_value):
        return Diagnosis("target_met", evidence, latest_id, best_id, improved)

    metrics = metrics_by_run.get(latest_id) or {}
    epochs = [e for e in (metrics.get("epochs") or []) if isinstance(e, dict)]
    val = [float(e["val_loss"]) for e in epochs if e.get("val_loss") is not None]
    train = [float(e["train_loss"]) for e in epochs if e.get("train_loss") is not None]
    n = len(val)
    best_epoch = latest.get("best_epoch") or n
    evidence.update({
        "n_epochs": n,
        "best_epoch": best_epoch,
        "best_is_last": bool(n) and best_epoch == n,
        "stopped_early": bool(metrics.get("stopped_early")),
    })
    if n == 0:
        return Diagnosis("plateau", evidence, latest_id, best_id, improved)
    if n < MIN_EPOCHS_TO_JUDGE:
        evidence["val_trend"] = round(_trend(val), 4)
        label = "improving" if improved else "underfitting"
        return Diagnosis(label, evidence, latest_id, best_id, improved)

    val_trend, train_trend, osc = _trend(val), _trend(train), _oscillation(val)
    i = min(max(int(best_epoch) - 1, 0), n - 1)
    gap = (val[i] - train[i]) / max(abs(val[i]), 1e-9) if i < len(train) else 0.0
    evidence.update({
        "val_trend": round(val_trend, 4),
        "train_trend": round(train_trend, 4),
        "gap": round(gap, 4),
        "oscillation": round(osc, 4),
    })
    if osc >= OSCILLATING and n >= MIN_EPOCHS_FOR_OSCILLATION:
        label = "learning_rate_too_high"
    elif (val_trend > TREND_FLAT and train_trend < 0) or (gap > GAP_LARGE and best_epoch < n):
        label = "overfitting"
    elif best_epoch == n and val_trend < -TREND_FLAT:
        label = "improving" if improved else "underfitting"
    else:
        label = "plateau"
    return Diagnosis(label, evidence, latest_id, best_id, improved)


def heuristic_proposals(diagnosis: Diagnosis, config: dict, schema: dict) -> list[Proposal]:
    """One fixed proposal per label and family, used when Claude is unavailable.

    Values are scaled from the current config; `apply_proposal` clamps them to the schema.
    """
    label = diagnosis.label
    family = str(config.get("model_type", ""))
    changes: dict[str, object] = {}
    expected = "better"

    def scaled(key: str, factor: float) -> None:
        current = config.get(key)
        if isinstance(current, int | float) and not isinstance(current, bool):
            changes[key] = current * factor

    if label == "target_met":
        return []
    if label == "overfitting":
        expected = "steadier"
        if family == "gradient_boosting":
            scaled("min_samples_leaf", 2)
            scaled("learning_rate", 0.5)
            reason = ("Validation [[loss]] turned upward while training loss kept falling: "
                      "the trees are fitting noise. Larger leaves and a smaller "
                      "[[learning rate]] make each tree more cautious.")
        elif family == "random_forest":
            scaled("min_samples_leaf", 3)
            scaled("max_features", 0.5)
            reason = ("The forest is memorising rows. Requiring more rows per leaf and "
                      "showing each tree fewer columns makes the trees disagree in useful "
                      "ways instead of all copying the noise.")
        else:
            current = config.get("alpha")
            changes["alpha"] = current * 10 if isinstance(current, int | float) and current else 0.001
            reason = ("A linear model overfits when its weights grow large; a stronger "
                      "[[regularisation]] penalty (`alpha`) keeps them small.")
    elif label in ("underfitting", "improving"):
        scaled("epochs", 1.5)
        if family == "gradient_boosting":
            scaled("learning_rate", 1.5)
        elif family == "random_forest":
            scaled("trees_per_epoch", 2)
        reason = (
            "The last change helped and validation loss was still falling when training "
            "stopped, so keep going in the same direction with more epochs."
            if label == "improving"
            else "Validation loss was still falling at the last [[epoch]]: the model had not "
                 "finished learning. More epochs (and a bigger step per epoch) let it."
        )
    elif label == "learning_rate_too_high":
        expected = "steadier"
        if family == "random_forest":
            scaled("trees_per_epoch", 2)
            reason = ("A forest has no learning rate; the validation loss jumps because each "
                      "epoch adds too few trees to average out the noise. Doubling the trees "
                      "per epoch smooths it.")
        else:
            scaled("learning_rate", 0.3)
            reason = ("Validation loss jumps up and down instead of settling: each update "
                      "overshoots. A much smaller [[learning rate]] takes smaller steps.")
    elif label == "plateau":
        if family == "gradient_boosting":
            scaled("max_leaf_nodes", 2)
            scaled("learning_rate", 0.5)
            reason = ("The curve has flattened. Bigger trees can learn finer structure; a "
                      "smaller learning rate stops them overshooting while they do.")
        elif family == "random_forest":
            scaled("max_features", 1.5)
            scaled("trees_per_epoch", 2)
            reason = ("The forest has flattened. Letting each tree see more columns and "
                      "adding more trees per epoch gives it more to average over.")
        else:
            changes["model_type"] = "gradient_boosting"
            reason = ("A linear model has flattened out: it cannot learn interactions "
                      "between columns. [[Gradient boosting]] can, so switch family.")
    else:  # failed_run
        expected = "steadier"
        if family == "random_forest":
            scaled("trees_per_epoch", 0.5)
            reason = ("The run failed; fewer trees per epoch make each step cheaper and "
                      "get past most memory or timeout failures.")
        else:
            scaled("learning_rate", 0.3)
            scaled("epochs", 0.5)
            reason = ("The run failed, most often from losses blowing up. A much smaller "
                      "[[learning rate]] and fewer epochs are the gentlest retry.")
    if not changes:
        return []
    return [Proposal(rank=1, changes=changes, reason=reason, expected=expected)]


def diff_config(old: dict, new: dict) -> dict[str, dict]:
    """`{key: {"from": old, "to": new}}` for every key whose value differs, in old's order."""
    diff: dict[str, dict] = {}
    for key in [*old, *(k for k in new if k not in old)]:
        before, after = old.get(key), new.get(key)
        if before != after:
            diff[key] = {"from": before, "to": after}
    return diff


def apply_proposal(config: dict, proposal: Proposal, nested_schema: dict) -> tuple[dict, dict, list[str]]:
    """Coerce a proposal against the (possibly new) family's schema; return the config, the
    diff from the current config, and the coercion notes. Raises ValueError on an unknown
    `model_type`."""
    changes = dict(proposal.changes or {})
    family = str(changes.get("model_type") or config.get("model_type") or "")
    flat = schema_for(nested_schema, family)
    if family != config.get("model_type"):
        base = default_config(flat)
        for key in nested_schema.get("common") or {}:
            if key in config and key != "model_type":
                base[key] = config[key]
    else:
        base = dict(config)
    merged = {**base, **changes, "model_type": family}
    new_config, notes = coerce_config(merged, flat)
    return new_config, diff_config(config, new_config), notes
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_diagnose.py -v`
Expected: PASS (24 tests including the parametrised grid). If a parametrised heuristic case yields an empty diff, adjust that heuristic's scale factor (the schema bounds are in `mlagent/templates/tabular_sklearn/config_schema.json`), not the test.

- [ ] **Step 5: Commit**

```bash
git add mlagent/diagnose.py tests/test_diagnose.py
git commit -m "feat: deterministic run diagnosis, heuristic proposals and apply_proposal

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 3: Comparison figures and their captions

**Files:**
- Modify: `mlagent/plots.py`, `mlagent/captions.py`
- Test: `tests/test_plots.py`, `tests/test_captions.py`

**Interfaces:**
- Produces (used by Task 7):
  - `plots.run_label(run: dict) -> str`
  - `plots.compare_curves(runs: list[dict], metrics_by_run: dict[int, dict]) -> Figure`
  - `plots.compare_runs(runs: list[dict], metric: str, target: float | None = None) -> Figure`
  - `plots.save_and_close(fig: Figure, plots_dir: Path, name: str) -> Path`
  - `captions.CAPTIONS["compare_curves"]`, `captions.CAPTIONS["compare_runs"]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plots.py`:

```python
import warnings  # noqa: E402


def _run(run_id, val, status="done", best_val=0.7, diff=None):
    epochs = [{"epoch": i + 1, "val_loss": v, "train_loss": v * 0.9} for i, v in enumerate(val)]
    entry = {"run_id": run_id, "status": status, "best_epoch": 1 + val.index(min(val)) if val
             else None, "best_val_metric": best_val if status == "done" else None,
             "applied_diff": diff}
    return entry, {"epochs": epochs}


def _history(*pairs):
    return [e for e, _m in pairs], {e["run_id"]: m for e, m in pairs}


def test_run_label_names_the_change_that_produced_the_run():
    assert plots.run_label({"run_id": 1, "applied_diff": None}) == "run 1"
    assert plots.run_label({"run_id": 2, "applied_diff": {
        "learning_rate": {"from": 0.1, "to": 0.05}}}) == "run 2 (learning_rate 0.05)"
    assert plots.run_label({"run_id": 3, "applied_diff": {
        "model_type": {"from": "linear", "to": "random_forest"},
        "alpha": {"from": 0.0001, "to": None},
        "trees_per_epoch": {"from": None, "to": 20}}}) == (
        "run 3 (model_type random_forest, +2 more)")


def test_comparison_figures_render_for_one_three_and_failed_runs(tmp_path):
    one = _history(_run(1, [1.0, 0.8, 0.7]))
    three = _history(_run(1, [1.0, 0.8, 0.7]),
                     _run(2, [1.0, 0.7, 0.5, 0.45], best_val=0.8,
                          diff={"epochs": {"from": 3, "to": 4}}),
                     _run(3, [1.0, 0.9], best_val=0.6, diff={"epochs": {"from": 4, "to": 2}}))
    failed = _history(_run(1, [1.0, 0.8, 0.7]),
                      _run(2, [1.0], status="failed", diff={"learning_rate": {"from": 0.1,
                                                                             "to": 0.9}}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i, (runs, by_run) in enumerate((one, three, failed)):
            curves = plots.compare_curves(runs, by_run)
            bars = plots.compare_runs(runs, "accuracy", target=0.9)
            a = plots.save_and_close(curves, tmp_path, f"curves{i}")
            b = plots.save_and_close(bars, tmp_path, f"bars{i}")
            assert a.exists() and b.exists()
            assert not plt.fignum_exists(curves.number)
    # a history with no archived metrics still draws (an empty-state message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.save_and_close(plots.compare_curves(one[0], {}), tmp_path, "empty")


def test_compare_curves_has_one_legend_entry_per_run_with_epochs():
    runs, by_run = _history(_run(1, [1.0, 0.8]), _run(2, [1.0, 0.7], diff={"epochs": {
        "from": 2, "to": 3}}))
    fig = plots.compare_curves(runs, by_run)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["run 1", "run 2 (epochs 3)"]
    plt.close(fig)


def test_compare_runs_draws_a_target_line_and_labels_every_run():
    runs, _by_run = _history(_run(1, [1.0]), _run(2, [1.0], status="failed"))
    fig = plots.compare_runs(runs, "rmse", target=5.0)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["run 1", "run 2"]
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    assert "rmse" in ax.get_xlabel()
    plt.close(fig)
```

In `tests/test_captions.py` add `"compare_curves", "compare_runs"` to `EXPECTED_KINDS` and append:

```python
def test_comparison_captions_match_by_stem():
    assert caption_for("plots/compare_curves.png") == CAPTIONS["compare_curves"]
    assert caption_for("plots/compare_runs.png") == CAPTIONS["compare_runs"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_plots.py tests/test_captions.py -v`
Expected: FAIL with `AttributeError: module 'mlagent.plots' has no attribute 'run_label'` and the captions set mismatch.

- [ ] **Step 3: Add the captions**

In `mlagent/captions.py`, add to `CAPTIONS` after `"residuals"`:

```python
    "compare_curves": (
        "One line per training run: validation [[loss]] per [[epoch]], with a dot at each "
        "run's best epoch. The legend names the change that produced each run. The lowest "
        "line whose dot sits at its right-hand end still had room to improve."
    ),
    "compare_runs": (
        "One bar per run: the best validation score it reached, with your target as the "
        "dashed line. A hollow bar is a run that failed. Compare the labels to see which "
        "change moved the bar, and in which direction."
    ),
```

- [ ] **Step 4: Add the figures to `mlagent/plots.py`**

Update the module docstring to:

```python
"""The house chart style, plus the agent-side run-comparison figures.

Every per-run figure the user sees is drawn by a template script that carries its own copy
of this palette. What lives here is the palette, the axis style, and the two figures that
compare runs, which only the assistant can draw because only it has the run history."""
```

Append after `present`:

```python
def save_and_close(fig: Figure, plots_dir: Path, name: str) -> Path:
    """Save without displaying: the teaching layer shows the file with its caption."""
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    plt.close(fig)
    return path


def _fmt_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_label(run: dict) -> str:
    """`run 2 (learning_rate 0.05)`: the run number plus the change that produced it."""
    run_id = run.get("run_id", "?")
    diff = run.get("applied_diff") or {}
    if not isinstance(diff, dict) or not diff:
        return f"run {run_id}"
    keys = list(diff)
    change = diff[keys[0]]
    to = change.get("to") if isinstance(change, dict) else change
    label = f"run {run_id} ({keys[0]} {_fmt_value(to)}"
    if len(keys) > 1:
        label += f", +{len(keys) - 1} more"
    return label + ")"


def compare_curves(runs: list[dict], metrics_by_run: dict[int, dict]) -> Figure:
    """Validation loss per epoch for every run, best epoch marked, one colour per run."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    drawn = 0
    for i, run in enumerate(runs):
        metrics = metrics_by_run.get(run.get("run_id")) or {}
        epochs = [e for e in (metrics.get("epochs") or [])
                  if isinstance(e, dict) and e.get("val_loss") is not None]
        if not epochs:
            continue
        colour = SERIES[i % len(SERIES)]
        xs = [e["epoch"] for e in epochs]
        ys = [e["val_loss"] for e in epochs]
        ax.plot(xs, ys, color=colour, linewidth=2, label=run_label(run))
        best = run.get("best_epoch")
        if best in xs:
            ax.plot([best], [ys[xs.index(best)]], marker="o", markersize=6, color=colour,
                    linestyle="none")
        drawn += 1
    if drawn:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no epochs recorded yet", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("validation loss", color=INK_2, fontsize=9)
    style_axes(ax, "Validation loss per epoch, every run")
    fig.tight_layout()
    return fig


def compare_runs(runs: list[dict], metric: str, target: float | None = None) -> Figure:
    """A horizontal bar per run of its best validation metric; failed runs hollow."""
    fig, ax = plt.subplots(figsize=(7.5, max(2.4, 0.55 * len(runs) + 1.4)))
    values = [r.get("best_val_metric") for r in runs
              if r.get("status") == "done" and r.get("best_val_metric") is not None]
    span = max([abs(v) for v in values] + [abs(target) if target is not None else 0.0, 1e-9])
    ys = list(range(len(runs)))
    for y, run in zip(ys, runs):
        colour = SERIES[y % len(SERIES)]
        value = run.get("best_val_metric")
        if run.get("status") == "done" and value is not None:
            ax.barh(y, value, color=colour, height=0.6)
            ax.text(value, y, f" {value:.4g}", va="center", fontsize=8, color=INK_2)
        else:
            ax.barh(y, span, height=0.6, fill=False, edgecolor=colour, linestyle="--")
            ax.text(span / 2, y, "failed", ha="center", va="center", fontsize=8, color=MUTED)
    if target is not None:
        ax.axvline(target, color=INK_2, linestyle="--", linewidth=1, label=f"target {target:g}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(ys)
    ax.set_yticklabels([run_label(r) for r in runs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"best validation {metric}", color=INK_2, fontsize=9)
    style_axes(ax, "Best validation score per run", grid_axis="x")
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Run the tests, including the warning-free check**

Run: `python -m pytest tests/test_plots.py tests/test_captions.py tests/test_template_profile.py tests/test_template_evaluate.py tests/test_template_train.py tests/test_template_clean.py -v -W error::DeprecationWarning`
Expected: PASS (the template caption-parity tests still pass because they compare only their own kinds).

- [ ] **Step 6: Commit**

```bash
git add mlagent/plots.py mlagent/captions.py tests/test_plots.py tests/test_captions.py
git commit -m "feat: agent-side compare_curves and compare_runs figures with captions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 4: Move the config edit menu and `config_table` into `templates_io`

**Files:**
- Modify: `mlagent/templates_io.py`, `mlagent/stages/codegen.py`
- Test: `tests/test_templates_io.py`, `tests/test_codegen_stage.py` (unchanged, must still pass)

**Interfaces:**
- Produces (used by Task 7):
  - `templates_io.config_table(config: dict, schema: dict) -> str` (moved verbatim)
  - `templates_io.edit_config(questioner, config: dict, schema: dict, display=lambda text: None) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_templates_io.py`:

```python
from mlagent.templates_io import config_table, edit_config  # noqa: E402
from mlagent.ui.questions import ScriptedQuestioner  # noqa: E402


def _gb_config():
    nested = load_schema("tabular_sklearn")
    flat = schema_for(nested, "gradient_boosting")
    config = default_config(flat)
    config["model_type"] = "gradient_boosting"
    return config, flat


def test_edit_config_changes_one_value_and_stops_at_done():
    config, flat = _gb_config()
    q = ScriptedQuestioner(["epochs = 10", "20", "Done"])
    edited = edit_config(q, config, flat)
    assert edited["epochs"] == 20 and config["epochs"] == 10  # the input is not mutated
    assert "model_type" not in " ".join(q.asked)  # choice keys are not on the menu


def test_edit_config_reverts_a_value_the_schema_rejects():
    config, flat = _gb_config()
    shown: list[str] = []
    q = ScriptedQuestioner(["learning_rate = 0.1", "5", "Done"])
    edited = edit_config(q, config, flat, display=shown.append)
    assert edited["learning_rate"] == 0.1
    assert any("not allowed" in s for s in shown)


def test_edit_config_zero_means_none_for_nullable_keys():
    config, flat = _gb_config()
    config["max_depth"] = 4
    q = ScriptedQuestioner(["max_depth = 4", "0", "Done"])
    assert edit_config(q, config, flat)["max_depth"] is None


def test_config_table_lists_every_key_with_its_description():
    config, flat = _gb_config()
    table = config_table(config, flat)
    assert table.startswith("| key | value | what it does |")
    assert "| learning_rate | 0.1 |" in table and "| max_depth | none |" in table
```

(`load_schema`, `schema_for`, `default_config` are already imported at the top of that file; if not, add them to the existing import.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_templates_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'config_table'`.

- [ ] **Step 3: Move the two functions**

Append to `mlagent/templates_io.py`:

```python
def config_table(config: dict, schema: dict) -> str:
    """Markdown table of a config: key, value, and the schema's one-line description."""
    rows = ["| key | value | what it does |", "|---|---|---|"]
    for key, value in config.items():
        desc = schema.get(key, {}).get("description", "")
        if value is None:
            shown = "none"
        elif isinstance(value, float):
            shown = f"{value:g}"
        else:
            shown = str(value)
        rows.append(f"| {key} | {shown} | {desc} |")
    return "\n".join(rows)


def edit_config(questioner, config: dict, schema: dict, display=lambda text: None) -> dict:
    """Let the user change numeric values one at a time until they pick Done.

    Choice-typed keys (`model_type`) are not offered: the questioner can only prompt for
    numbers, and a family switch is the codegen stage's or the tuner's job. A value the
    schema rejects is reported through `display` and reverted.
    """
    config = dict(config)
    while True:
        editable = [k for k in config if schema.get(k, {}).get("type") != "choice"]
        options = [f"{k} = {config[k]}" for k in editable] + ["Done"]
        pick = questioner.choice("Which value do you want to change?", options, allow_other=False)
        if pick == "Done":
            return config
        key = pick.split(" = ", 1)[0]
        rule = schema.get(key)
        if rule is None:
            continue
        nullable = bool(rule.get("nullable"))
        hint = " (0 means no limit)" if nullable else ""
        current = config.get(key)
        value = questioner.number(
            f"New value for {key}{hint}: {rule.get('description', '')}",
            default=0 if current is None else current,
            minimum=0 if nullable else rule.get("min"),
            maximum=rule.get("max"),
        )
        if nullable and value == 0:
            config[key] = None
        elif rule.get("type") == "integer":
            config[key] = int(round(value))
        else:
            config[key] = float(value)
        problems = validate_config(config, schema)
        if problems:
            display("That value is not allowed: " + "; ".join(problems))
            config[key] = current
```

In `mlagent/stages/codegen.py`: delete the module-level `config_table` function and the `_edit_config` method; add `config_table, edit_config` to the `from mlagent.templates_io import (...)` list; change the call site to

```python
        if not ctx.questioner.confirm("Happy with this configuration? (No lets you change values)"):
            config = edit_config(ctx.questioner, config, schema, ctx.display)
            ctx.project.write_json(cfg.CONFIG_FILE, config)
            ctx.display("Updated configuration:\n\n" + config_table(config, schema))
```

`config_table` stays importable as `mlagent.stages.codegen.config_table` through that import (it is used in `prepare`, so ruff will not flag it).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_templates_io.py tests/test_codegen_stage.py -v && ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates_io.py mlagent/stages/codegen.py tests/test_templates_io.py
git commit -m "refactor: config_table and the edit_config menu move to templates_io for the tuner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 5: Orchestrator re-prepares a stage whose debrief returns `True`; reset hook

**Files:**
- Modify: `mlagent/stages/base.py`, `mlagent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces (used by Task 7):
  - `Stage.debrief(ctx) -> bool | None`; a truthy return means "round complete, prepare me again in this same `run()`".
  - `stage_debrief(stage, ctx) -> object` returns that value.
  - `stage_reset(stage, ctx) -> None` calls `stage.on_reset(ctx)` when the stage defines it.
  - `Orchestrator.reset(name)` calls `stage_reset` on every stage from `name` onwards (in pipeline order) before saving state.
  - `Orchestrator.debrief(name)` (the forced path) also honours a truthy return by clearing `prepared`/`handoff` for that stage.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orchestrator.py`:

```python
class LoopingStage(ScriptStage):
    """A two-phase stage that asks to be re-prepared `rounds` times before finishing."""

    def __init__(self, name: str, rounds: int):
        super().__init__(name)
        self.rounds = rounds
        self.resets = 0

    def debrief(self, ctx: StageContext):
        self.debriefed += 1
        if self.debriefed < self.rounds:
            return True
        ctx.project.write_json(f"{self.name}.json", {"ok": True})
        return None

    def on_reset(self, ctx: StageContext) -> None:
        self.resets += 1
        (ctx.project.root / f"{self.name}.json").unlink(missing_ok=True)


def test_a_truthy_debrief_re_prepares_the_stage_in_the_same_run(project):
    loop, after = LoopingStage("loop", rounds=3), RecordingStage("after")
    orch = Orchestrator(make_ctx(project), [loop, after])
    # Round 1: prepare, wait for the user.
    assert orch.run() == [] and orch.waiting().stage == "loop" and loop.prepared == 1
    user_runs(project, "loop")
    # Round 2: debrief returns True -> prepared again, new handoff, still waiting.
    assert orch.run() == [] and loop.debriefed == 1 and loop.prepared == 2
    assert orch.waiting().stage == "loop"
    assert "loop" in project.read_json("state.json")["prepared"]
    user_runs(project, "loop")
    assert orch.run() == [] and loop.debriefed == 2 and loop.prepared == 3
    user_runs(project, "loop")
    # Final round: debrief returns None, stage completes, the next stage runs.
    assert orch.run() == ["loop", "after"] and loop.debriefed == 3 and loop.prepared == 3
    assert orch.completed() == ["loop", "after"]


def test_outputs_from_the_previous_round_do_not_satisfy_the_new_handoff(project):
    """After a re-prepare the stale output must not be mistaken for the next round's."""
    loop = LoopingStage("loop", rounds=2)
    orch = Orchestrator(make_ctx(project), [loop])
    orch.run()
    user_runs(project, "loop")
    orch.run()  # debrief 1 -> True -> prepare 2; prepare rewrites loop.py (newer than output)
    assert loop.prepared == 2
    assert orch.run() == [] and loop.debriefed == 1  # still waiting on round 2's output


def test_reset_calls_on_reset_for_the_stage_and_every_later_one(project):
    a, loop, c = RecordingStage("a"), LoopingStage("loop", rounds=1), LoopingStage("c", 1)
    orch = Orchestrator(make_ctx(project), [a, loop, c])
    orch.run()
    user_runs(project, "loop")
    orch.run()
    user_runs(project, "c")
    assert orch.run() == ["loop", "c"] or orch.completed() == ["a", "loop", "c"]
    orch.reset("loop")
    assert loop.resets == 1 and c.resets == 1
    assert orch.completed() == ["a"]
    assert not project.exists("loop.json") and not project.exists("c.json")


def test_forced_debrief_returning_true_clears_the_handoff(project):
    loop = LoopingStage("loop", rounds=2)
    orch = Orchestrator(make_ctx(project), [loop])
    orch.run()
    orch.debrief("loop")  # returns True: the round is over, the stage must prepare again
    state = project.read_json("state.json")
    assert state["handoff"] is None and "loop" not in state["prepared"]
    assert orch.run() == [] and loop.prepared == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: the four new tests FAIL (`loop.prepared == 2` assertion fails; `resets` stays 0).

- [ ] **Step 3: Change `mlagent/stages/base.py`**

In the `Stage` protocol change the debrief line to:

```python
    def debrief(self, ctx: StageContext) -> bool | None: ...
```

Replace `stage_debrief` with, and add `stage_reset` after it:

```python
def stage_debrief(stage, ctx: StageContext) -> object:
    """Run a stage's second phase. A truthy return asks the orchestrator to prepare the
    stage again in the same call (a tuning round is over, the next one starts)."""
    debrief = getattr(stage, "debrief", None)
    if debrief is None:
        return None
    return debrief(ctx)


def stage_reset(stage, ctx: StageContext) -> None:
    """Let a stage forget its own loop state when it, or an earlier stage, is reset."""
    on_reset = getattr(stage, "on_reset", None)
    if on_reset is not None:
        on_reset(ctx)
```

Update the module docstring's last sentence to: "`debrief` reads whatever those cells produced, shows it, and writes the stage's completion artifact; returning `True` asks to be prepared again at once."

- [ ] **Step 4: Change `mlagent/orchestrator.py`**

Add `stage_reset` to the `from mlagent.stages.base import (...)` list. Replace `reset`:

```python
    def reset(self, stage_name: str) -> None:
        """Forget this stage and every later one; they rerun even if their artifacts exist."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            raise ValueError(f"unknown stage {stage_name!r}; known: {names}")
        idx = names.index(stage_name)
        for stage in self.stages[idx:]:
            stage_reset(stage, self.ctx)
        state = self._state()
        state["completed"] = [n for n in state["completed"] if n in names[:idx]]
        state["prepared"] = [n for n in state["prepared"] if n in names[:idx]]
        state["forced"] = names[idx:]
        state["current"] = None
        state["handoff"] = None
        self._save(state)
```

Add a helper and use it in both debrief paths:

```python
    def _forget_handoff(self, name: str) -> None:
        """The round is over: drop the stage's handoff so the next run() prepares it again."""
        state = self._state()
        state["prepared"] = [n for n in state["prepared"] if n != name]
        state["handoff"] = None
        self._save(state)
```

Replace `debrief(name)`:

```python
    def debrief(self, name: str) -> None:
        """Force a stage's second phase, skipping the freshness check.

        The escape hatch for when Drive's mtimes lie about outputs the user really did make.
        """
        stage = self._stage(name)
        previous = self.ctx.stage
        self.ctx.stage = name
        try:
            if stage_debrief(stage, self.ctx):
                self._forget_handoff(name)
            elif stage.is_complete(self.ctx):
                self.mark_complete(name)
        finally:
            self.ctx.stage = previous
```

Replace the `if not done:` block inside `_run` with:

```python
            if not done:
                self.ctx.stage = stage.name
                while True:
                    state = self._state()
                    state["current"] = stage.name
                    self._save(state)
                    if stage.name in state["prepared"]:
                        handoff = Handoff.from_dict(state.get("handoff"))
                    else:
                        self.ctx.display(f"**Stage: {stage.name}**")
                        handoff = stage_prepare(stage, self.ctx)
                        state = self._state()
                        state["handoff"] = handoff.to_dict() if handoff is not None else None
                        if handoff is not None:
                            state["prepared"] = [*state["prepared"], stage.name]
                        self._save(state)
                    if handoff is not None and not stage_outputs_ready(stage, self.ctx, handoff):
                        self.ctx.display(waiting_message(handoff))
                        return ran
                    if not stage_debrief(stage, self.ctx):
                        break
                    # The stage asked to go round again: forget this handoff and prepare
                    # afresh, so the next proposal appears right under this debrief.
                    self._forget_handoff(stage.name)
                if stage.is_complete(self.ctx):
                    self.mark_complete(stage.name)
                    ran.append(stage.name)
                else:
                    self.ctx.display(
                        f"Stage {stage.name} did not finish; rerun orch.run() to continue."
                    )
                    return ran
```

Update the module docstring to mention the loop: `"""Run stages in order, checkpointing to state.json so a Colab reset can resume. A stage whose debrief returns True is prepared again in the same call (the tuning loop)."""`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_orchestrator.py tests/test_pipeline_e2e.py -v && ruff check .`
Expected: PASS. (`test_outputs_from_the_previous_round…` passes because `ScriptStage.prepare` rewrites `loop.py`, making the old `loop_out.json` older than the script; on a filesystem with coarse mtimes the `MTIME_TOLERANCE` may let it through — if that test is flaky on Windows, have `LoopingStage.prepare` also `unlink` its output file, which is exactly what the real tune stage does.)

- [ ] **Step 6: Commit**

```bash
git add mlagent/stages/base.py mlagent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: a debrief returning True re-prepares its stage in the same run; reset hook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 6: The tuner's prompts and primer

**Files:**
- Create: `mlagent/prompts/tune.md`, `mlagent/prompts/tune_debrief.md`, `mlagent/prompts/teaching/tuning.md`
- Test: `tests/test_prompts_io.py`

**Interfaces:**
- Produces (used by Task 7): `load_prompt("tune", audience=...)`, `load_prompt("tune_debrief", audience=...)`, `material("tuning", level)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts_io.py`:

```python
def test_tune_prompts_load_and_take_the_audience():
    from mlagent.prompts_io import audience, load_prompt

    tune = load_prompt("tune", audience=audience("beginner"))
    assert "propose_diffs" in tune and "{audience}" not in tune
    assert "minutes" in tune  # asks for changes that respect minutes_per_run
    debrief = load_prompt("tune_debrief", audience=audience("expert"))
    assert "write_debrief" in debrief and "{audience}" not in debrief


def test_tuning_primer_is_level_fenced():
    from mlagent.teaching import material

    beginner = material("tuning", "beginner")
    expert = material("tuning", "expert")
    assert "learning rate" in beginner.lower() and "overfitting" in beginner.lower()
    assert len(beginner) > len(expert)
    assert "<!--" not in beginner and "<!--" not in expert
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_prompts_io.py -v`
Expected: FAIL with `FileNotFoundError` for `tune.md`.

- [ ] **Step 3: Write `mlagent/prompts/tune.md`**

```
You are the tuning stage of an ML training assistant that runs inside Google Colab. The training code is a fixed, tested template; you never change code, only the values in `config.json`. One training run has just been diagnosed and the user will pick ONE of your proposals, rerun the same `train.py` and `evaluate.py`, and come back.

You receive JSON with: the project spec (task type, metric, target value, minutes per run, max rounds); the diagnosis (a label from overfitting, underfitting, learning_rate_too_high, plateau, failed_run, improving, plus the evidence numbers); a table of every run so far (status, best validation metric, best epoch, epochs run, seconds, and the config change that produced it); the current config; the flat schema for its model family (key, type, min, max, default, description); and the list of model families the template supports.

Call `propose_diffs` exactly once with one to three proposals, best first:
- `rank`: 1 for the change you would make yourself.
- `changes`: only keys from the schema, with values inside min/max. Change one or two keys per proposal; the user should be able to see what each proposal tests. To switch model family, set `model_type` to another supported family and (optionally) keys from that family; its other keys take defaults.
- `reason`: under 70 words, tied to the evidence you were given (quote a number). Say what you expect the curve to do differently.
- `expected`: `faster` (same score in less time), `better` (higher score), or `steadier` (less noise, less overfitting).

Rules: keep every proposal inside the minutes-per-run budget (a run's seconds are in the table; do not more than double epochs or trees at once). Do not repeat a change that a previous run already tried unless the evidence says it helped. After a failed run, propose the gentlest configuration that avoids the error. Do not propose changes that would coerce to no change.

Wrap technical terms in double square brackets like [[learning rate]] or [[overfitting]] so the user can click them. Do not write code and do not restate the raw JSON.

Audience: {audience}
```

- [ ] **Step 4: Write `mlagent/prompts/tune_debrief.md`**

```
You are the tuning stage of an ML training assistant that runs inside Google Colab. A training run that applied one configuration change has just finished. You receive JSON with the project spec (task type, metric, target value), the round number, the change that was applied and why, the diagnosis that motivated it, the new run's numbers (status, best validation value, best epoch, epochs run, whether it stopped early, error), the previous best run's numbers, whether the new run beat it, and a short table of every run.

Write a short debrief for a learner (under 160 words):

- One sentence on what changed and what happened: the new best validation value versus the previous best, and whether the target is now met.
- One or two sentences on whether the change did what was expected, reading the comparison figures: did the validation curve fall faster, settle sooner, or stop rising?
- If the run failed, say plainly what the error suggests.
- End with one sentence on what to try next, or say that the target is met, or that the rounds are used up.

Wrap technical terms in double square brackets like [[validation loss]] so the user can click them. Use the numbers given; do not invent any. Do not restate the raw JSON.

Call `write_debrief` exactly once. `narrative` is the text above; `figure_notes` maps each figure filename you were given to one sentence about what THIS run history shows in it (leave it empty when you were given no figures).

Audience: {audience}
```

- [ ] **Step 5: Write `mlagent/prompts/teaching/tuning.md`**

```
### Tuning: reading the curves, then turning one knob

<!--level:beginner,intermediate-->
Tuning means changing the settings in `config.json` and training again. The assistant
looks at the [[validation loss]] curve from the last run, names what it sees, and proposes
one change at a time so you can tell what each change did.
<!--/level-->

#### The four shapes a curve can take

- **Overfitting.** Training loss keeps falling, validation loss turns upward. The model is
  memorising rows. Make it more cautious: bigger leaves, stronger regularisation, a smaller
  [[learning rate]].
- **Underfitting.** Both losses are still falling when training stops. Give it more: more
  [[epoch]]s, a bigger step, more capacity.
- **Learning rate too high.** Validation loss jumps up and down. Each update overshoots;
  take smaller steps.
- **Plateau.** Both losses are flat. More of the same will not help; change the shape of
  the model (bigger trees, more columns per tree) or switch family.

<!--level:beginner-->
Losses are the model's own score for being wrong: lower is better, and the training loss is
always a little flattering because the model has seen those rows. The validation loss is the
honest one, which is why the diagnosis reads that curve.
<!--/level-->

#### What each family's knobs do

**Gradient boosting.** `learning_rate` is how much of each new tree's correction is applied
(smaller is steadier, needs more epochs); `iters_per_epoch` is trees added per epoch;
`max_leaf_nodes` and `max_depth` are how big a tree can grow (bigger learns finer detail and
overfits sooner); `min_samples_leaf` is how many rows a leaf must hold (bigger is more
cautious); `l2_regularization` shrinks leaf values (bigger is more cautious).

**Random forest.** `trees_per_epoch` is how many trees each epoch adds (more is smoother);
`max_depth` and `min_samples_leaf` limit each tree as above; `max_features` is the share of
columns each tree may look at (smaller makes trees disagree more, which usually helps).

**Linear / logistic regression.** `learning_rate` is the step size; `alpha` is the
[[regularisation]] strength (bigger keeps the weights small and fights overfitting).

<!--level:beginner,intermediate-->
`epochs` and `early_stopping_patience` are shared: an epoch is one round of learning, and
patience is how many epochs without improvement the run tolerates before stopping.
<!--/level-->
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_prompts_io.py tests/test_teaching.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mlagent/prompts/tune.md mlagent/prompts/tune_debrief.md mlagent/prompts/teaching/tuning.md tests/test_prompts_io.py
git commit -m "feat: tune, tune_debrief prompts and the tuning primer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 7: `TuneStage`

**Files:**
- Create: `mlagent/stages/tune.py`
- Test: `tests/test_tune_stage.py` (new)

**Interfaces:**
- Consumes: everything produced by Tasks 1–6.
- Produces (used by Task 8): `TuneStage` (name `"tune"`), `TUNE_COMMANDS = [["train.py"], ["evaluate.py"]]`, `TUNE_OUTPUTS = ["metrics.json", "eval_val.json"]`, `EDIT_LABEL`, `STOP_LABEL`, `apply_label(n) -> "Apply proposal n"`, `load_tune_state(project, max_rounds) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tune_stage.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import TrainStage
from mlagent.stages.tune import (
    EDIT_LABEL,
    STOP_LABEL,
    TUNE_COMMANDS,
    TUNE_OUTPUTS,
    TuneStage,
    apply_label,
)
from mlagent.ui.questions import ScriptedQuestioner

SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


def make_ctx(project, llm=None, answers=()):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append,
                       display_figure=lambda path, caption="": figures.append((path, caption)))
    return ctx, shown, figures


def run_cells(project, handoff):
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command], cwd=str(project.root),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def with_run_one(project, target=0.999):
    """Codegen with gradient boosting, a small config, and one real logged run."""
    ctx, _s, _f = make_ctx(project, answers=["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(SMALL)
    project.write_json("config.json", cfg)
    spec = project.read_json("spec.json")
    spec["target_value"] = target
    project.write_json("spec.json", spec)
    ctx, _s, _f = make_ctx(project)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert len(runlog.read_runs(project.runs_path)) == 1
    return project


def one_proposal(changes, reason="Slower steps.", expected="steadier"):
    return [("tool", "propose_diffs", {"proposals": [
        {"rank": 1, "changes": changes, "reason": reason, "expected": expected}]})]


def test_labels():
    assert apply_label(2) == "Apply proposal 2"
    assert TUNE_COMMANDS == [["train.py"], ["evaluate.py"]]
    assert TUNE_OUTPUTS == ["metrics.json", "eval_val.json"]


def test_a_guided_round_applies_the_proposal_and_logs_run_two(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([
        [("text", "Tuning is about to start.")],           # round-1 preamble
        one_proposal({"learning_rate": 0.05}),              # propose_diffs
        [("tool", "write_debrief", {"narrative": "Run 2 was [[steadier]].",
                                    "figure_notes": {"compare_curves.png": "Lower line."}})],
    ])
    ctx, shown, figures = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="tune", commands=TUNE_COMMANDS, outputs=TUNE_OUTPUTS)
    text = "\n".join(shown)
    assert "round 1 of 3" in text.lower()
    assert "Proposal 1" in text and "Slower steps." in text
    assert "| learning_rate | 0.1 | 0.05 |" in text
    assert project.read_json("config.json")["learning_rate"] == 0.05
    state = project.read_json("tune_state.json")
    assert state["round"] == 0 and state["decision"] == "continue"
    assert state["pending"]["round"] == 1
    assert state["pending"]["applied_diff"] == {"learning_rate": {"from": 0.1, "to": 0.05}}
    assert state["pending"]["diagnosis"] in ("overfitting", "underfitting",
                                              "learning_rate_too_high", "plateau")
    assert not project.exists("metrics.json") and not project.exists("eval_val.json")
    assert not stage.outputs_ready(ctx, handoff)
    prompt = llm.calls[1]["messages"][0]["content"]
    assert '"label"' in prompt and '"schema"' in prompt and '"model_types"' in prompt

    run_cells(project, handoff)
    assert stage.outputs_ready(ctx, handoff)
    assert stage.debrief(ctx) is True
    runs = runlog.read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[1]["applied_diff"] == {"learning_rate": {"from": 0.1, "to": 0.05}}
    assert runs[1]["config"]["learning_rate"] == 0.05
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()
    names = [Path(p).name for p, _c in figures]
    assert names[:2] == ["compare_curves.png", "compare_runs.png"]
    assert "run2_training.png" in names
    assert all(caption for _p, caption in figures)
    assert "Lower line." in dict((Path(p).name, c) for p, c in figures)["compare_curves.png"]
    assert "Run 2 was [[steadier]]." in "\n".join(shown)
    assert "Run 2 finished" in "\n".join(shown)
    state = project.read_json("tune_state.json")
    assert state["round"] == 1 and state["pending"] is None
    assert state["decision"] == "continue"
    assert state["history"] == [{"round": 1, "diagnosis": state["history"][0]["diagnosis"],
                                 "applied_diff": {"learning_rate": {"from": 0.1, "to": 0.05}},
                                 "run_id": 2, "improved": state["history"][0]["improved"]}]
    assert not stage.is_complete(ctx)
    # A second prepare in the same loop does not repeat the preamble or the primer.
    llm2 = FakeLLM([one_proposal({"epochs": 4})])
    ctx2, shown2, _f = make_ctx(project, llm2, answers=[STOP_LABEL])
    assert stage.prepare(ctx2) is None
    assert "round 2 of 3" in "\n".join(shown2).lower()
    assert "Tuning is about to start." not in "\n".join(shown2)


def test_stop_ends_the_loop_with_a_heuristic_when_the_llm_is_down(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _f = make_ctx(project, answers=[STOP_LABEL])  # FakeLLM([]) -> LLMError
    stage = TuneStage()
    assert stage.prepare(ctx) is None
    text = "\n".join(shown)
    assert "Proposal 1" in text  # the heuristic still produced one
    assert project.read_json("tune_state.json")["decision"] == "stopped"
    assert stage.is_complete(ctx)
    assert project.read_json("config.json")["learning_rate"] == 0.1  # nothing applied
    # Once decided, prepare only explains why and asks nothing.
    ctx2, shown2, _f = make_ctx(project)
    assert stage.prepare(ctx2) is None
    assert "stopped" in "\n".join(shown2).lower()


def test_heuristic_proposal_is_applied_when_the_llm_is_down(clean_project):
    project = with_run_one(clean_project)
    before = project.read_json("config.json")
    ctx, _shown, _f = make_ctx(project, answers=[apply_label(1)])
    handoff = TuneStage().prepare(ctx)
    assert handoff is not None
    after = project.read_json("config.json")
    assert after != before and after["model_type"] == "gradient_boosting"
    assert project.read_json("tune_state.json")["pending"]["applied_diff"]


def test_target_met_ends_the_loop_without_asking(clean_project):
    project = with_run_one(clean_project, target=0.0)  # any accuracy meets 0.0
    ctx, shown, _f = make_ctx(project)  # no scripted answers: a question would raise
    stage = TuneStage()
    assert stage.prepare(ctx) is None
    assert project.read_json("tune_state.json")["decision"] == "target_met"
    assert "target" in "\n".join(shown).lower()
    assert stage.is_complete(ctx)


def test_rounds_exhausted_after_the_last_allowed_round(clean_project):
    project = with_run_one(clean_project)
    spec = project.read_json("spec.json")
    spec["max_rounds"] = 1
    project.write_json("spec.json", spec)
    llm = FakeLLM([[("text", "pre")], one_proposal({"epochs": 4})])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    assert stage.debrief(ctx) is None  # debrief narrative falls back (script exhausted)
    state = project.read_json("tune_state.json")
    assert state["round"] == 1 and state["decision"] == "rounds_exhausted"
    assert stage.is_complete(ctx)
    assert "rounds" in "\n".join(shown).lower()


def test_edit_path_changes_the_proposal_before_applying(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")], one_proposal({"learning_rate": 0.05})])
    ctx, shown, _f = make_ctx(project, llm,
                              answers=[EDIT_LABEL, "epochs = 3", "2", "Done"])
    handoff = TuneStage().prepare(ctx)
    assert handoff is not None
    config = project.read_json("config.json")
    assert config["learning_rate"] == 0.05 and config["epochs"] == 2
    diff = project.read_json("tune_state.json")["pending"]["applied_diff"]
    assert diff == {"epochs": {"from": 3, "to": 2}, "learning_rate": {"from": 0.1, "to": 0.05}}
    assert "edited" in project.read_json("tune_state.json")["pending"]["reason"].lower()


def test_three_proposals_are_ranked_and_the_second_can_be_chosen(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")], [("tool", "propose_diffs", {"proposals": [
        {"rank": 2, "changes": {"epochs": 4}, "reason": "second", "expected": "better"},
        {"rank": 1, "changes": {"learning_rate": 0.05}, "reason": "first",
         "expected": "steadier"},
        {"rank": 3, "changes": {"learning_rate": 0.1}, "reason": "no-op", "expected": "faster"},
        {"rank": 4, "changes": {"min_samples_leaf": 40}, "reason": "fourth",
         "expected": "steadier"},
    ]})]])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(2)])
    TuneStage().prepare(ctx)
    text = "\n".join(shown)
    assert text.index("first") < text.index("second") < text.index("fourth")
    assert "no-op" not in text  # coerced to no change -> dropped
    assert project.read_json("config.json")["epochs"] == 4


def test_family_switch_produces_a_valid_config_and_a_run(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")],
                   one_proposal({"model_type": "random_forest"}, reason="Forest time.")])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    config = project.read_json("config.json")
    assert config["model_type"] == "random_forest" and "trees_per_epoch" in config
    assert "learning_rate" not in config and config["epochs"] == 3
    run_cells(project, handoff)
    assert stage.debrief(ctx) is True
    runs = runlog.read_runs(project.runs_path)
    assert runs[1]["config"]["model_type"] == "random_forest"
    assert runs[1]["applied_diff"]["model_type"] == {"from": "gradient_boosting",
                                                     "to": "random_forest"}


def test_debrief_without_a_pending_round_does_nothing(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _f = make_ctx(project)
    assert TuneStage().debrief(ctx) is None
    assert shown == [] and not project.exists("tune_state.json")


def test_debrief_before_the_cells_ran_explains_and_waits(clean_project):
    project = with_run_one(clean_project)
    ctx, _s, _f = make_ctx(project, answers=[apply_label(1)])
    stage = TuneStage()
    stage.prepare(ctx)  # unlinks metrics.json
    ctx2, shown, _f = make_ctx(project)
    assert stage.debrief(ctx2) is None
    assert "train.py" in "\n".join(shown)
    assert project.read_json("tune_state.json")["pending"] is not None


def test_on_reset_forgets_the_loop(clean_project):
    project = with_run_one(clean_project)
    project.write_json("tune_state.json", {"round": 2, "max_rounds": 3, "decision": "stopped",
                                           "pending": None, "history": []})
    ctx, _s, _f = make_ctx(project)
    TuneStage().on_reset(ctx)
    assert not project.exists("tune_state.json")


def test_expert_level_skips_preamble_and_primer(clean_project):
    project = with_run_one(clean_project)
    spec = project.read_json("spec.json")
    spec["learning_level"] = "expert"
    project.write_json("spec.json", spec)
    llm = FakeLLM([one_proposal({"epochs": 4})])  # no preamble turn
    ctx, shown, _f = make_ctx(project, llm, answers=[STOP_LABEL])
    TuneStage().prepare(ctx)
    assert "What each family's knobs do" not in "\n".join(shown)
    assert len(llm.calls) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tune_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.tune'`.

- [ ] **Step 3: Write `mlagent/stages/tune.py`**

```python
"""Tune stage: diagnose the run history, apply one approved change, rerun the train cells.

One round is `prepare` (diagnosis, proposals, one question, new config.json, handoff) then
`debrief` (log the run with its diff, draw the comparison figures, narrate). `debrief`
returns True to have the orchestrator prepare the next round in the same call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mlagent import config as cfg
from mlagent import plots
from mlagent.diagnose import (
    EXPECTATIONS,
    Diagnosis,
    Proposal,
    apply_proposal,
    diagnose,
    diff_config,
    heuristic_proposals,
    meets_target,
)
from mlagent.llm import LLMError, ToolSpec
from mlagent.prompts_io import audience, load_prompt
from mlagent.runlog import best_run, is_better, read_runs, summarise
from mlagent.runs import EVAL_VAL_FILE, log_finished_run, read_run_metrics, run_problem
from mlagent.spec import Spec
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.teaching import EXPERT, material
from mlagent.templates_io import (
    TEMPLATE_FOR_TASK,
    config_table,
    edit_config,
    load_schema,
    model_types,
    schema_for,
)

TUNE_COMMANDS = [["train.py"], ["evaluate.py"]]
TUNE_OUTPUTS = [cfg.METRICS_FILE, EVAL_VAL_FILE]
MAX_PROPOSALS = 3
EDIT_LABEL = "Edit a proposal first"
STOP_LABEL = "Stop tuning and write the report"
COMPARE_CURVES = "compare_curves"
COMPARE_RUNS = "compare_runs"
CONTINUE = "continue"

DECISION_TEXT = {
    "stopped": (
        "Tuning is stopped at your request. The report stage evaluates the best run on the "
        "[[test set]]; to try another change by hand, edit `config.json` and use the "
        "*Train again* cell."
    ),
    "target_met": (
        "The best run meets your target, so tuning is done. The report stage scores it once "
        "on the [[test set]]. You can still edit `config.json` and use the *Train again* "
        "cell if you want to keep experimenting."
    ),
    "rounds_exhausted": (
        "That was the last tuning round you allowed at intake (`max_rounds`), so tuning is "
        "done. The report stage evaluates the best run; `orch.reset('tune')` starts a fresh "
        "loop if you want more rounds."
    ),
}

DIAGNOSIS_TEXT = {
    "overfitting": (
        "**Diagnosis: [[overfitting]].** Validation [[loss]] turned upward while training "
        "loss kept falling, so the model is learning noise that does not carry over to new "
        "rows."
    ),
    "underfitting": (
        "**Diagnosis: [[underfitting]].** Validation loss was still falling at the last "
        "[[epoch]]: the model had not finished learning when training stopped."
    ),
    "learning_rate_too_high": (
        "**Diagnosis: the [[learning rate]] looks too high.** Validation loss jumps up and "
        "down from epoch to epoch instead of settling."
    ),
    "plateau": (
        "**Diagnosis: a [[plateau]].** Validation loss has flattened; more of the same will "
        "not help, so the model needs a different shape or more signal."
    ),
    "failed_run": (
        "**Diagnosis: the last run failed** ({error}). A gentler configuration usually gets "
        "past this."
    ),
    "improving": (
        "**Diagnosis: still improving.** The last change helped and the curve was still "
        "falling at the end, so there is more to gain in the same direction."
    ),
    "target_met": "**Diagnosis: target met.**",
}

PROPOSE_TOOL = ToolSpec(
    name="propose_diffs",
    description=(
        "Propose one to three configuration changes, best first. Each `changes` object maps "
        "config keys to new values; `reason` is shown to the user; `expected` says what the "
        "change should improve."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "maxItems": MAX_PROPOSALS,
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "changes": {"type": "object"},
                        "reason": {"type": "string"},
                        "expected": {"type": "string", "enum": list(EXPECTATIONS)},
                    },
                    "required": ["rank", "changes", "reason", "expected"],
                },
            }
        },
        "required": ["proposals"],
    },
    handler=lambda inp: "recorded",
)


def apply_label(n: int) -> str:
    return f"Apply proposal {n}"


def _fmt(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def empty_tune_state(max_rounds: int) -> dict:
    return {"round": 0, "max_rounds": int(max_rounds), "decision": CONTINUE, "pending": None,
            "history": []}


def load_tune_state(project, max_rounds: int) -> dict:
    """tune_state.json with every key present; a missing or corrupt file is a fresh loop."""
    state = project.read_json(cfg.TUNE_STATE_FILE)
    base = empty_tune_state(max_rounds)
    if isinstance(state, dict) and isinstance(state.get("history"), list):
        base.update(state)
    return base


def describe(diagnosis: Diagnosis) -> str:
    """The diagnosis in plain words, with the numbers that back it."""
    text = DIAGNOSIS_TEXT.get(diagnosis.label, DIAGNOSIS_TEXT["plateau"]).format(
        error=diagnosis.evidence.get("error", "unknown error")
    )
    ev = diagnosis.evidence
    parts = []
    if isinstance(ev.get("val_trend"), int | float):
        parts.append(f"validation loss changed {ev['val_trend']:+.1%} over the last third")
    if isinstance(ev.get("gap"), int | float):
        parts.append(f"train/validation gap {ev['gap']:.1%} at the best epoch")
    if isinstance(ev.get("oscillation"), int | float):
        parts.append(f"{ev['oscillation']:.0%} of epoch-to-epoch changes flipped direction")
    if ev.get("stopped_early"):
        parts.append("training stopped early")
    return text + (" (" + "; ".join(parts) + ")" if parts else "")


def proposal_table(diff: dict, schema: dict) -> str:
    rows = ["| key | now | proposed | what it does |", "|---|---|---|---|"]
    for key, change in diff.items():
        desc = schema.get(key, {}).get("description", "")
        rows.append(f"| {key} | {_fmt(change.get('from'))} | {_fmt(change.get('to'))} | {desc} |")
    return "\n".join(rows)


class TuneStage(ScriptStageBase):
    name = "tune"

    # --- lifecycle -----------------------------------------------------------------------
    def is_complete(self, ctx: StageContext) -> bool:
        state = ctx.project.read_json(cfg.TUNE_STATE_FILE)
        return isinstance(state, dict) and state.get("decision") not in (None, CONTINUE)

    def on_reset(self, ctx: StageContext) -> None:
        (ctx.project.root / cfg.TUNE_STATE_FILE).unlink(missing_ok=True)

    # --- phase 1 -------------------------------------------------------------------------
    def prepare(self, ctx: StageContext) -> Handoff | None:
        project = ctx.project
        spec = ctx.spec()
        runs = read_runs(project.runs_path)
        if not any(r.get("status") == "done" for r in runs):
            ctx.display("There is no successful training run to tune yet; finish the train "
                        "stage first.")
            return None
        state = load_tune_state(project, spec.max_rounds)
        if state["decision"] != CONTINUE:
            ctx.display(DECISION_TEXT[state["decision"]])
            return None
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not config.get("model_type"):
            raise RuntimeError("config.json not found; run the codegen stage first")
        nested = load_schema(TEMPLATE_FOR_TASK[spec.task_type])

        diagnosis = diagnose(runs, read_run_metrics(project, runs), spec)
        if diagnosis.label == "target_met":
            state["decision"] = "target_met"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display(DECISION_TEXT["target_met"])
            return None

        round_no = state["round"] + 1
        level = ctx.learning_level()
        if state["round"] == 0:
            ctx.teaching().preamble("tune", {
                "max_rounds": spec.max_rounds, "metric": spec.metric,
                "target_value": spec.target_value, "model_type": config.get("model_type"),
                "best_val_metric": diagnosis.evidence.get("best_val_metric"),
            })
            if level != EXPERT:
                ctx.display(material("tuning", level))
        ctx.display(f"**Tuning round {round_no} of {state['max_rounds']}.**\n\n"
                    + summarise(runs, spec.metric))
        ctx.display(describe(diagnosis))

        applied = self._propose(ctx, spec, diagnosis, runs, config, nested)
        if not applied:
            state["decision"] = "stopped"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display("I have no change to propose for this run, so tuning stops here. "
                        + DECISION_TEXT["stopped"])
            return None
        for proposal, _new_config, diff, notes in applied:
            flat = schema_for(nested, str(_new_config["model_type"]))
            text = (f"**Proposal {proposal.rank}** (expected: {proposal.expected})\n\n"
                    f"{proposal.reason}\n\n{proposal_table(diff, flat)}")
            if notes:
                text += "\n\n" + " ".join(notes)
            ctx.display(text)

        chosen = self._choose(ctx, applied, config, nested)
        if chosen is None:
            state["decision"] = "stopped"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display(DECISION_TEXT["stopped"])
            return None
        new_config, diff, reason = chosen
        project.write_json(cfg.CONFIG_FILE, new_config)
        state["pending"] = {
            "round": round_no,
            "applied_diff": diff,
            "reason": reason,
            "diagnosis": diagnosis.label,
            "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        project.write_json(cfg.TUNE_STATE_FILE, state)
        # The cells regenerate these; a stale copy must not satisfy outputs_ready.
        (project.root / cfg.METRICS_FILE).unlink(missing_ok=True)
        (project.root / EVAL_VAL_FILE).unlink(missing_ok=True)
        changed = ", ".join(f"`{k}` {_fmt(v['from'])} -> {_fmt(v['to'])}" for k, v in diff.items())
        ctx.display(f"Applied to `config.json`: {changed}. Run the `train.py` and "
                    "`evaluate.py` cells again, then run this cell to see whether it helped.")
        return Handoff(stage=self.name, commands=[list(c) for c in TUNE_COMMANDS],
                       outputs=list(TUNE_OUTPUTS))

    def _propose(self, ctx: StageContext, spec: Spec, diagnosis: Diagnosis, runs: list[dict],
                 config: dict, nested: dict) -> list[tuple[Proposal, dict, dict, list[str]]]:
        """Proposals from Claude, else the heuristic; each coerced, no-ops dropped, re-ranked."""
        raw = self._ask_llm(ctx, spec, diagnosis, runs, config, nested)
        if not raw:
            flat = schema_for(nested, str(config["model_type"]))
            raw = heuristic_proposals(diagnosis, config, flat)
        applied: list[tuple[Proposal, dict, dict, list[str]]] = []
        for proposal in raw:
            try:
                new_config, diff, notes = apply_proposal(config, proposal, nested)
            except ValueError:
                continue
            if not diff:
                continue
            applied.append((proposal, new_config, diff, notes))
            if len(applied) == MAX_PROPOSALS:
                break
        for rank, (proposal, _c, _d, _n) in enumerate(applied, start=1):
            proposal.rank = rank
        return applied

    def _ask_llm(self, ctx: StageContext, spec: Spec, diagnosis: Diagnosis, runs: list[dict],
                 config: dict, nested: dict) -> list[Proposal]:
        store: dict = {}

        def record(inp: dict) -> str:
            store["proposals"] = inp.get("proposals")
            return "recorded"

        tool = ToolSpec(name=PROPOSE_TOOL.name, description=PROPOSE_TOOL.description,
                        input_schema=PROPOSE_TOOL.input_schema, handler=record)
        payload = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value, "minutes_per_run": spec.minutes_per_run,
                     "max_rounds": spec.max_rounds},
            "diagnosis": {"label": diagnosis.label, "evidence": diagnosis.evidence,
                          "improved": diagnosis.improved,
                          "latest_run_id": diagnosis.latest_run_id,
                          "best_run_id": diagnosis.best_run_id},
            "runs": [{k: r.get(k) for k in ("run_id", "status", "best_val_metric", "best_epoch",
                                            "epochs_run", "seconds", "applied_diff", "error")}
                     for r in runs],
            "config": config,
            "schema": schema_for(nested, str(config["model_type"])),
            "model_types": model_types(nested),
        }
        try:
            ctx.llm.run(
                load_prompt("tune", audience=audience(ctx.learning_level())),
                [{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            return []
        proposals: list[Proposal] = []
        for i, item in enumerate(store.get("proposals") or [], start=1):
            if not isinstance(item, dict) or not isinstance(item.get("changes"), dict):
                continue
            expected = item.get("expected")
            rank = item.get("rank")
            proposals.append(Proposal(
                rank=int(rank) if isinstance(rank, int | float) else i,
                changes=dict(item["changes"]),
                reason=str(item.get("reason") or ""),
                expected=expected if expected in EXPECTATIONS else "better",
            ))
        proposals.sort(key=lambda p: p.rank)
        return proposals

    def _choose(self, ctx: StageContext, applied: list, config: dict, nested: dict):
        """Apply / edit / stop. Returns (new_config, diff, reason) or None to stop."""
        while True:
            options = [apply_label(i) for i in range(1, len(applied) + 1)] + [EDIT_LABEL, STOP_LABEL]
            answer = ctx.questioner.choice("What shall we do?", options, allow_other=False,
                                           key="tune.action")
            if answer == STOP_LABEL:
                return None
            if answer == EDIT_LABEL:
                idx = 0
                if len(applied) > 1:
                    pick = ctx.questioner.choice(
                        "Which proposal do you want to edit?",
                        [f"Proposal {i}" for i in range(1, len(applied) + 1)],
                        allow_other=False, key="tune.edit_which",
                    )
                    idx = self._index(pick, len(applied))
                proposal, new_config, _diff, _notes = applied[idx]
                flat = schema_for(nested, str(new_config["model_type"]))
                edited = edit_config(ctx.questioner, new_config, flat, ctx.display)
                diff = diff_config(config, edited)
                if not diff:
                    ctx.display("That leaves the configuration unchanged; pick a proposal or stop.")
                    continue
                ctx.display("Edited configuration:\n\n" + config_table(edited, flat))
                return edited, diff, f"{proposal.reason} (edited by you)"
            idx = self._index(answer, len(applied))
            proposal, new_config, diff, _notes = applied[idx]
            return new_config, diff, proposal.reason

    @staticmethod
    def _index(answer: str, count: int) -> int:
        try:
            idx = int(str(answer).rsplit(" ", 1)[-1]) - 1
        except ValueError:
            idx = 0
        return min(max(idx, 0), count - 1)

    # --- phase 2 -------------------------------------------------------------------------
    def debrief(self, ctx: StageContext) -> bool | None:
        project = ctx.project
        spec = ctx.spec()
        state = load_tune_state(project, spec.max_rounds)
        pending = state.get("pending")
        if not isinstance(pending, dict) or not project.exists(cfg.TUNE_STATE_FILE):
            return None
        problem = run_problem(project)
        if problem:
            ctx.display(problem)
            return None
        logged = log_finished_run(project, applied_diff=pending.get("applied_diff"))
        if logged is None:
            return None
        entry = logged.entry
        runs = read_runs(project.runs_path)
        metrics_by_run = read_run_metrics(project, runs)
        previous = [r for r in runs if r.get("run_id") != entry["run_id"]]
        previous_best = best_run(previous, spec.metric)
        best = best_run(runs, spec.metric)
        improved = bool(
            entry["status"] == "done" and previous_best is not None
            and is_better(spec.metric, entry.get("best_val_metric"),
                          previous_best.get("best_val_metric"))
        )

        figures = [
            plots.save_and_close(plots.compare_curves(runs, metrics_by_run), project.plots_dir,
                                 COMPARE_CURVES),
            plots.save_and_close(plots.compare_runs(runs, spec.metric, spec.target_value),
                                 project.plots_dir, COMPARE_RUNS),
            *logged.figures,
        ]
        payload = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "round": pending.get("round"),
            "applied_diff": pending.get("applied_diff"),
            "reason": pending.get("reason"),
            "diagnosis_before": pending.get("diagnosis"),
            "run": {k: entry.get(k) for k in ("run_id", "status", "best_val_metric", "best_epoch",
                                              "epochs_run", "seconds", "error")},
            "stopped_early": bool((metrics_by_run.get(entry["run_id"]) or {}).get("stopped_early")),
            "previous_best": ({k: previous_best.get(k) for k in ("run_id", "best_val_metric")}
                              if previous_best else None),
            "improved": improved,
            "runs": [{k: r.get(k) for k in ("run_id", "status", "best_val_metric", "applied_diff")}
                     for r in runs],
        }
        fallback = (
            "Compare the new run's line with the earlier ones in the first figure, and its "
            "bar with the others in the second: the change helped if its bar moved towards "
            "the dashed target line."
        )
        narrative = ctx.teaching().debrief("tune_debrief", payload, figures, fallback=fallback)
        ctx.display(self._headline(spec, entry, previous_best, improved) + "\n\n" + narrative)

        state["round"] = int(pending.get("round") or state["round"] + 1)
        state["history"].append({
            "round": state["round"], "diagnosis": pending.get("diagnosis"),
            "applied_diff": pending.get("applied_diff"), "run_id": entry["run_id"],
            "improved": improved,
        })
        state["pending"] = None
        best_value = best.get("best_val_metric") if best else None
        if meets_target(spec.metric, best_value, spec.target_value):
            state["decision"] = "target_met"
        elif state["round"] >= state["max_rounds"]:
            state["decision"] = "rounds_exhausted"
        else:
            state["decision"] = CONTINUE
        project.write_json(cfg.TUNE_STATE_FILE, state)
        if state["decision"] != CONTINUE:
            ctx.display(DECISION_TEXT[state["decision"]])
            return None
        return True

    @staticmethod
    def _headline(spec: Spec, entry: dict, previous_best: dict | None, improved: bool) -> str:
        run_id = entry["run_id"]
        if entry["status"] != "done":
            return (f"Run {run_id} failed: {entry.get('error')}. I will propose a gentler "
                    "configuration next.")
        value = _fmt(entry.get("best_val_metric"))
        if previous_best is None:
            return f"Run {run_id} finished: best validation {spec.metric} {value}."
        prev = _fmt(previous_best.get("best_val_metric"))
        verdict = "better than" if improved else "not better than"
        return (f"Run {run_id} finished: best validation {spec.metric} {value}, {verdict} the "
                f"previous best {prev} (run {previous_best['run_id']}).")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_tune_stage.py -v && ruff check .`
Expected: PASS (each test runs `train.py`/`evaluate.py` for real; the file takes a minute or two). If the FakeLLM script order in a test does not match the calls (`llm.calls`), fix the stage's call order to match the spec (preamble → propose → debrief), not the test.

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/tune.py tests/test_tune_stage.py
git commit -m "feat: the tune stage — diagnose, propose, apply, rerun, compare, repeat

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

### Task 8: Wire the stage into Colab, the notebook, the end-to-end test and the docs

**Files:**
- Modify: `mlagent/colab.py`, `scripts/build_notebook.py`, `notebooks/ML_Training_Agent.ipynb` (regenerated), `docs/colab-smoke.md`, `CLAUDE.md`
- Test: `tests/test_colab.py`, `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: `TuneStage`, `TUNE_COMMANDS`, `STOP_LABEL`, `apply_label` from Task 7.

- [ ] **Step 1: Update the tests first**

In `tests/test_colab.py`:
- In `test_make_context_and_start` the expected stage list becomes `["intake", "data", "clean", "codegen", "train", "tune", "report"]`.
- In `test_cell_source_and_script_cells` the expected key set gains `"tune"`.
- Replace `test_handoff_commands_match_what_the_stages_return` with:

```python
def test_handoff_commands_match_what_the_stages_return(tmp_path):
    from mlagent import colab
    from mlagent.stages.clean import CLEAN_PY
    from mlagent.stages.report import EVAL_TEST_COMMAND
    from mlagent.stages.tune import TUNE_COMMANDS

    assert colab.HANDOFF_COMMANDS["clean"] == [[CLEAN_PY]]
    assert colab.HANDOFF_COMMANDS["train"] == [["train.py"], ["evaluate.py"]]
    assert colab.HANDOFF_COMMANDS["tune"] == TUNE_COMMANDS
    assert colab.HANDOFF_COMMANDS["report"] == [list(EVAL_TEST_COMMAND)]
```

- Append:

```python
def test_notebook_has_a_tune_cell_between_evaluate_and_report():
    sources = [c["source"] for c in _load_notebook_cells() if c.get("cell_type") == "code"]
    tune = next(i for i, s in enumerate(sources) if "orch.run(until='tune')" in s)
    assert sources[tune - 1] == "%load evaluate.py"
    assert "orch.run(until='report')" in sources[tune + 1]
    assert "#@param" not in sources[tune]
    assert sources[tune].startswith("#@title 6. Tune")
    assert "#@title 7. Report" in sources[tune + 1]
```

(The `checked == 23` and `form_cells == 4` assertions stay as they are: the Tune cell has no form fields.)

In `tests/test_pipeline_e2e.py`:
- Add `from mlagent.stages.tune import STOP_LABEL, TuneStage, apply_label`.
- Add `"tune.action": STOP_LABEL,` to `FORM_ANSWERS` (after `"codegen.model_type"`).
- `ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "tune", "report"]`.
- In `make_orchestrator` add `TuneStage()` between `TrainStage()` and `ReportStage()`, and give it a `questioner=None` parameter: `questioner=questioner or AutoApproveQuestioner([])`.
- In `test_full_pipeline_runs_through_handoffs`, after the `runs.jsonl` assertions add:

```python
    tune_state = project.read_json("tune_state.json")
    assert tune_state["decision"] == "stopped" and tune_state["history"] == []
```

- In `test_a_second_training_run_is_logged_and_the_report_re_triggers` change the second advance to `ran = advance(orch, project, answers=FORM_ANSWERS)` and the assertion to `assert ran == ["train", "tune", "report"]`, and add `assert not project.exists("tune_state.json") or project.read_json("tune_state.json")["history"] == []` right after `orch.reset("train")` (the reset hook cleared the loop).
- Append:

```python
def test_one_guided_round_then_stop_and_the_report_scores_the_new_best(project, advance):
    answers = {k: v for k, v in FORM_ANSWERS.items() if k != "tune.action"}
    answers["intake.target_value"] = 1.0   # unreachable: the loop must not end on target_met
    answers["data.noise"] = 0.3
    orch = make_orchestrator(
        project, questioner=AutoApproveQuestioner([apply_label(1), STOP_LABEL]))
    ran = advance(orch, project, answers=answers)
    assert ran == ALL_STAGES
    runs = read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["applied_diff"] is None and runs[1]["applied_diff"]
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    state = project.read_json("tune_state.json")
    assert state["decision"] == "stopped" and state["round"] == 1
    assert [h["run_id"] for h in state["history"]] == [2]
    best = max(runs, key=lambda r: r["best_val_metric"])
    assert project.read_json("report_meta.json")["best_run"] == best["run_id"]
    assert project.read_json("eval_test.json")["run_id"] == best["run_id"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_colab.py tests/test_pipeline_e2e.py -v`
Expected: FAIL (stage list, key set, missing tune cell, `ran != ALL_STAGES`).

- [ ] **Step 3: Wire `mlagent/colab.py`**

Add `from mlagent.stages.tune import TuneStage` (keep imports sorted for ruff `I`). In `HANDOFF_COMMANDS` insert after `"train"`:

```python
    "tune": [["train.py"], ["evaluate.py"]],
```

In `start`, the stage list becomes:

```python
    return Orchestrator(
        ctx,
        [IntakeStage(), DataStage(), CleanStage(), CodegenStage(), TrainStage(), TuneStage(),
         ReportStage()],
    )
```

`PURGED_MODULES` is unchanged (the tuner adds no script).

- [ ] **Step 4: Update `scripts/build_notebook.py`**

In the roadmap markdown cell:
- Change "The assistant leads you through six steps. Steps 1-4 each have a form cell you fill in. Step 5 (train) is two script cells you run: `train.py`, then `evaluate.py`. Step 6 (report) is a cell you just run." to "The assistant leads you through seven steps. Steps 1-4 each have a form cell you fill in. Step 5 (train) is two script cells you run: `train.py`, then `evaluate.py`. Steps 6 (tune) and 7 (report) are cells you just run; tune sends you back to the two train cells once per round."
- After the "5. **Train.**" line insert:

```
        "6. **Tune.** *You do:* pick one of the assistant's proposed changes (or stop), rerun "
        "the two train cells, and run the tune cell again. *You get:* a diagnosis of the "
        "curves, a comparison chart of every run, and a new run logged with the change that "
        "made it. Repeat until you stop, hit your target, or use up MAX_ROUNDS.\n"
```

- Renumber the report line to "7. **Report.**".
- In the "**If you get stuck.**" paragraph change "*Train again* (near the bottom) reruns training after you edit `config.json` by hand." to "*Train again* (near the bottom) reruns training after you edit `config.json` by hand; the tune step is the guided way to do the same thing."

Replace the report cell and add the tune cell so the sequence reads:

```python
    script_cell("train", 0),
    script_cell("train", 1),
    code(
        "#@title 6. Tune",
        "# The assistant reads the run history, diagnoses the curves and proposes one to",
        "# three changes. Pick one in the output below (or edit it, or stop), run the",
        "# train.py and evaluate.py cells above again, then run this cell again to see",
        "# whether it helped. It keeps going until you stop, the target is met, or",
        "# MAX_ROUNDS is used up.",
        "orch.run(until='tune')",
    ),
    code(
        "#@title 7. Report",
        "# Score the best model once on the test rows it has never seen, then write "
        "report.md. The assistant asks before touching the test set.",
        "orch.run(until='report')",
    ),
```

Replace the "Train again" markdown cell text with:

```
        "## Train again\n\n"
        "For a manual experiment: edit `config.json` in the project folder yourself, then "
        "run the cell below. (The *6. Tune* cell is the guided alternative: the assistant "
        "proposes the change for you.) This cell resets the train stage, the tuning loop and "
        "the report, then re-prepares training, naming the `train.py` and `evaluate.py` cells "
        "above for you to run again. Run those two cells, then the *6. Tune* cell to log the "
        "new run and start a fresh tuning loop, or the `orch.run()` cell to go straight to the "
        "report. `orch.waiting()` says which cells the assistant is still waiting on; "
        "`orch.debrief('train')` forces the debrief if Drive's timestamps lag; "
        "`orch.reset('tune')` restarts only the tuning loop."
```

Then regenerate: `python scripts/build_notebook.py` (expect "22 cells").

- [ ] **Step 5: Add the Milestone 5 section to `docs/colab-smoke.md`**

Append:

```markdown
## Milestone 5 (Tuning loop)

Run on a project that has finished item 8 of the Milestone 4 list (one logged run), at
beginner level unless an item says otherwise.

1. **One guided round.** Run the `6. Tune` cell: it prints the tuning primer, a run table,
   a one-line diagnosis with numbers in brackets, one to three proposals each with a
   now/proposed table, and asks `What shall we do?` in an `input()` box. Answer
   `Apply proposal 1`. `config.json` on Drive changes accordingly and the cell names the
   `train.py` and `evaluate.py` cells. Run both, then run the `6. Tune` cell again: two
   comparison figures appear first (validation loss per run, best score per run) with
   captions, then the run-2 figures, then a narrative saying whether the change helped;
   `runs.jsonl` gains run 2 with an `applied_diff`, `runs/run2_metrics.json` exists, and the
   next round's proposals follow straight after, in the same cell output.
2. **Edit a proposal.** In that next round answer `Edit a proposal first`, change one value
   in the menu, pick `Done`, and confirm the applied diff lists both the proposal's change
   and yours.
3. **Stop.** Answer `Stop tuning and write the report`. The cell says tuning is stopped and
   finishes. `tune_state.json` shows `"decision": "stopped"`. The `7. Report` cell then
   names the best run (which may now be run 2 or 3) and asks before scoring the test set.
4. **Target met exits without a question.** `orch.reset('intake')` is too much; instead
   edit `spec.json`'s `target_value` down to a value run 1 already beat, run
   `orch.reset('tune'); orch.run(until='tune')`: the cell says the target is met and asks
   nothing.
5. **Rounds run out.** Set `MAX_ROUNDS` to 1 at intake on a fresh project (or edit
   `spec.json`), reset the tune stage, apply one proposal and rerun: after the debrief the
   cell says the last allowed round is done, and `7. Report` proceeds.
6. **A failed run gets a gentler proposal.** Edit `config.json` by hand to a value that
   breaks training (for gradient boosting, `learning_rate: 1.0` with `epochs: 30` usually
   produces a non-finite loss), use *Train again*, then the `6. Tune` cell: the diagnosis
   reads "the last run failed" with the error, and proposal 1 lowers the learning rate.
7. **Expert level is terse.** Switch `learning_level` to `expert` in `spec.json`, reset the
   tune stage: no primer, no preamble, proposals with one-line reasons, comparison figures
   with the fixed caption only.
8. **Resetting train clears the loop.** *Train again* after a tuning loop: `tune_state.json`
   disappears from Drive and the next `6. Tune` run starts at round 1 with the full history
   in its run table.
```

- [ ] **Step 6: Update `CLAUDE.md`**

- In "What this is": pipeline becomes `intake -> data -> clean -> codegen -> train -> tune -> report`; add the Milestone 5 spec path to the design-spec list.
- In the Pipeline bullet, after the `train` sentence insert: "`tune` (Milestone 5) reads `runs.jsonl` and `runs/run{N}_metrics.json`, diagnoses the latest run with `mlagent/diagnose.py`, asks Claude for `propose_diffs` (falling back to `heuristic_proposals`), lets the user apply, edit (`templates_io.edit_config`) or stop, writes the new `config.json` and `tune_state.json` (`pending`), and hands off the same `[["train.py"], ["evaluate.py"]]`; its debrief logs the run with `applied_diff` via `mlagent/runs.py::log_finished_run`, draws `plots.compare_curves`/`compare_runs`, and returns `True` to be re-prepared in the same `orch.run()` until the user stops, the target is met or `max_rounds` is reached (`decision` in `tune_state.json`). `reset('train')` clears the loop through the stage's `on_reset` hook."
- In the orchestrator bullet add: "A stage's `debrief` may return `True` to be prepared again in the same call; `reset(name)` calls each later stage's optional `on_reset(ctx)`."
- Replace the `mlagent/plots.py` sentence with: "`mlagent/plots.py` keeps the palette, `style_axes`, `save_figure`, `present`, `save_and_close` and the two agent-side comparison figures; every per-run figure is drawn by a template."
- In Commands add: `python -m pytest tests/test_tune_stage.py -v   # the tuning loop, run for real`.

- [ ] **Step 7: Run everything**

Run: `python scripts/build_notebook.py && python -m pytest && ruff check . && python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py`
Expected: notebook written with 22 cells; all tests PASS; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add mlagent/colab.py scripts/build_notebook.py notebooks/ML_Training_Agent.ipynb docs/colab-smoke.md CLAUDE.md tests/test_colab.py tests/test_pipeline_e2e.py
git commit -m "feat: tune stage wired into Colab, the notebook (6. Tune), the e2e test and the smoke checklist

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SrLtKbAAckQMMjjS3fKH1W"
```

---

## Self-review

**Spec coverage.** Pipeline order and handoff (Tasks 7, 8). `prepare` bullets: reads runs/metrics/state (7), loop-over and target-met exits (7), `Diagnosis` and the seven labels (2), round-1 preamble and primer (6, 7), one LLM call with `propose_diffs` and coercion/no-op dropping/heuristic fallback (2, 7), table display (7), one `choice` with apply/edit/stop and `edit_config` moved (4, 7), apply writes config + pending + handoff (7). `debrief` bullets: `log_finished_run` in `runs.py` used by both stages (1, 7), metrics archive (1), agent-side comparison figures through `Teaching.debrief("tune_debrief")` (3, 6, 7), state update and decision (7), returns True (7). Orchestrator re-prepare and reset (5). Completion and `reset("tune")`/`reset("train")` (5, 7). Report unchanged (its `_load_runs_and_best` still narrows to done runs with checkpoints). Files and contracts (1, 2, 7). Figures (3). Notebook and Colab (8). Learning levels: expert skips preamble/primer (7), figure notes suppressed at expert by `Teaching.show_figures` (existing). Testing section: each named test file has a task. Risks: the prompt asks for changes within the minutes budget (6).

**Placeholder scan.** No TBD/TODO; every code step has its code; the only "adjust if" notes name the exact file to touch.

**Type consistency.** `log_finished_run(project, applied_diff=None) -> LoggedRun | None` (1) is called that way in 7. `diagnose(runs, metrics_by_run, spec)`, `heuristic_proposals(diagnosis, config, schema)`, `apply_proposal(config, proposal, nested) -> (config, diff, notes)`, `diff_config(old, new)` (2) match their uses in 7. `plots.compare_curves(runs, metrics_by_run)`, `compare_runs(runs, metric, target)`, `save_and_close(fig, plots_dir, name)` (3) match 7. `edit_config(questioner, config, schema, display)` and `config_table(config, schema)` (4) match 7. `stage_debrief` returning the value and `stage_reset` (5) are what 7's `on_reset` and `debrief -> bool | None` rely on. `TUNE_COMMANDS`, `STOP_LABEL`, `apply_label` (7) are what 8 imports.
