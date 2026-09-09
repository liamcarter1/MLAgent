from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlagent import config as cfg
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.report import EVAL_TEST_FILE, ReportStage, render_report
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner


def run_cells(project, handoff):
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command], cwd=str(project.root),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def trained(project):
    ctx = StageContext(project=project, llm=FakeLLM([]),
                       questioner=ScriptedQuestioner(["Gradient boosting", "y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().prepare(ctx)
    config = project.read_json("config.json")
    config.update({"epochs": 2, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", config)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    return project


def make_ctx(project, llm=None, answers=("y",)):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append,
                       display_figure=lambda path, caption="": figures.append((path, caption)))
    return ctx, shown, figures


def test_report_hands_off_the_test_evaluation_then_writes_markdown(clean_project):
    project = trained(clean_project)
    llm = FakeLLM([[("text", "The [[test set]] score was close to validation.")]])
    ctx, shown, figures = make_ctx(project, llm)
    stage = ReportStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="report",
                              commands=[["evaluate.py", "--split", "test"]],
                              outputs=["eval_test.json"])
    assert any("test set" in s.lower() for s in shown)
    assert not stage.outputs_ready(ctx, handoff)

    run_cells(project, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    record = json.loads((project.root / EVAL_TEST_FILE).read_text(encoding="utf-8"))
    assert record["run_id"] == 1
    report = project.report_path.read_text(encoding="utf-8")
    assert report.startswith("# ")
    assert "| run | status |" in report
    assert "## Held-out test result" in report and "accuracy" in report
    assert "## Best configuration" in report and "model_type" in report
    assert "## What we learned" in report and "[[test set]]" in report
    assert "plots/test_confusion.png" in report and "plots/run1_training.png" in report
    assert project.read_json(cfg.REPORT_META_FILE)["best_run"] == 1
    assert {Path(p).name for p, _c in figures} >= {"test_confusion.png"}
    assert all(caption for _p, caption in figures)


def test_learning_level_reaches_the_debrief_prompt(clean_project):
    project = trained(clean_project)
    spec = project.read_json("spec.json")
    project.write_json("spec.json", {**spec, "learning_level": "beginner"})
    llm = FakeLLM([[("text", "The test set score was close to validation.")]])
    ctx, _shown, _figures = make_ctx(project, llm)
    stage = ReportStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    stage.debrief(ctx)
    system = llm.calls[-1]["system"]
    assert "new to machine learning" in system
    assert "You are the report stage of an ML training assistant" in system


def test_declining_the_confirm_skips_without_a_handoff(clean_project):
    project = trained(clean_project)
    ctx, shown, _figures = make_ctx(project, answers=("n",))
    stage = ReportStage()
    assert stage.prepare(ctx) is None
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("Skipped" in s for s in shown)
    assert not any("evaluate.py" in s for s in shown)


def test_rewrites_the_report_without_touching_test_again(clean_project):
    project = trained(clean_project)
    ctx, _shown, _figures = make_ctx(project)
    stage = ReportStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    first = project.report_path.read_text(encoding="utf-8")

    ctx2, shown2, _figures2 = make_ctx(project)
    assert stage.prepare(ctx2) is None  # already evaluated for the best run
    stage.debrief(ctx2)
    assert any("already evaluated" in s.lower() for s in shown2)
    assert project.report_path.read_text(encoding="utf-8") == first


def test_stale_eval_test_is_refused(clean_project):
    project = trained(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = ReportStage()
    run_cells(project, stage.prepare(ctx))
    record = json.loads((project.root / EVAL_TEST_FILE).read_text(encoding="utf-8"))
    record["run_id"] = 99
    (project.root / EVAL_TEST_FILE).write_text(json.dumps(record), encoding="utf-8")
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("run 99" in s for s in shown)


def test_prepare_drops_a_stale_eval_test_before_the_new_handoff(clean_project):
    project = trained(clean_project)
    record = {"run_id": 99, "checkpoint": "checkpoints/run99.joblib", "metric": "accuracy",
              "value": 0.1, "loss": 1.0, "figures": []}
    (project.root / EVAL_TEST_FILE).write_text(json.dumps(record), encoding="utf-8")
    project.write_json(cfg.REPORT_META_FILE, {"best_run": 99, "n_runs": 99})

    ctx, _shown, _figures = make_ctx(project)
    stage = ReportStage()
    handoff = stage.prepare(ctx)
    assert handoff is not None
    assert not (project.root / EVAL_TEST_FILE).exists()
    assert not stage.outputs_ready(ctx, handoff)


def test_run_id_none_names_the_checkpoint_instead_of_a_run(clean_project):
    project = trained(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = ReportStage()
    run_cells(project, stage.prepare(ctx))
    record = json.loads((project.root / EVAL_TEST_FILE).read_text(encoding="utf-8"))
    record["run_id"] = None
    record["checkpoint"] = "checkpoints/run1.joblib"
    (project.root / EVAL_TEST_FILE).write_text(json.dumps(record), encoding="utf-8")

    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("checkpoints/run1.joblib" in s for s in shown)
    report = project.report_path.read_text(encoding="utf-8")
    assert "checkpoints/run1.joblib" in report


def test_run_number_sorts_numerically_not_lexicographically(clean_project):
    from mlagent.stages.report import _run_number

    plots_dir = clean_project.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run2_training.png", "run10_training.png"):
        (plots_dir / name).touch()
    ordered = sorted(plots_dir.glob("run*_training.png"), key=_run_number)
    assert [p.name for p in ordered] == ["run2_training.png", "run10_training.png"]


def test_no_successful_run_yet(clean_project):
    ctx, shown, _figures = make_ctx(clean_project)
    stage = ReportStage()
    assert stage.prepare(ctx) is None
    assert any("train" in s.lower() for s in shown)


def test_best_run_candidates_exclude_done_runs_without_a_checkpoint(clean_project):
    from mlagent.runlog import append_run
    from mlagent.stages.report import _load_runs_and_best

    project = clean_project
    append_run(project.runs_path,
               {"status": "done", "best_val_metric": 0.99, "checkpoint": None})
    append_run(project.runs_path,
               {"status": "done", "best_val_metric": 0.5, "checkpoint": "checkpoints/run2.joblib"})

    ctx, _shown, _figures = make_ctx(project)
    runs, best = _load_runs_and_best(project, ctx.spec())

    assert len(runs) == 2  # the full history is returned unnarrowed
    assert best is not None and best["run_id"] == 2


def test_render_report_shape():
    report = render_report(
        "demo", {"goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
                 "target_value": 0.9},
        [{"run_id": 1, "status": "done", "best_val_metric": 0.8}],
        {"run_id": 1, "config": {"model_type": "linear"}, "best_val_metric": 0.8},
        {"metric": "accuracy", "value": 0.78, "loss": 0.5, "checkpoint": "checkpoints/run1.joblib"},
        "Lessons here.", [Path("plots/test_confusion.png")],
    )
    assert "# demo: training report" in report
    assert "![test_confusion](plots/test_confusion.png)" in report
    assert "checkpoints/run1.joblib" in report
