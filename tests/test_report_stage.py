from __future__ import annotations

from pathlib import Path

from mlagent.llm import FakeLLM
from mlagent.runner import RunResult
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.report import ReportStage, render_report
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner


def trained(project):
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner(["y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().run(ctx)
    cfg = project.read_json("config.json")
    cfg.update({"epochs": 2, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", cfg)
    TrainStage(poll_seconds=0.05).run(ctx)
    assert TrainStage().is_complete(ctx)
    return project


def make_ctx(project, llm=None, answers=("y",)):
    shown = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append)
    return ctx, shown


def test_report_evaluates_test_once_and_writes_markdown(clean_project):
    project = trained(clean_project)
    llm = FakeLLM([[("text", "The [[test set]] score was close to validation.")]])
    ctx, shown = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    assert project.exists("eval_test.json")
    report = project.report_path.read_text(encoding="utf-8")
    assert report.startswith("# ")
    assert "| run | status |" in report
    assert "## Held-out test result" in report and "accuracy" in report
    assert "## Best configuration" in report and "learning_rate" in report
    assert "## What we learned" in report and "[[test set]]" in report
    assert "![" in report and "plots/test_confusion.png" in report
    assert "plots/run1_training.png" in report
    assert sorted(p.name for p in project.plots_dir.glob("test_*.png")) == [
        "test_confusion.png", "test_per_class.png", "test_roc_pr.png"
    ]
    assert any("[[test set]]" in s for s in shown)


def test_declining_the_confirm_leaves_stage_incomplete(clean_project):
    project = trained(clean_project)
    ctx, _ = make_ctx(project, answers=["n"])
    stage = ReportStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert not project.exists("eval_test.json")


def test_no_successful_run_stops_early(clean_project):
    ctx, shown = make_ctx(clean_project)
    stage = ReportStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert any("train" in s.lower() for s in shown)


def test_eval_failure_is_shown(clean_project):
    project = trained(clean_project)

    def fake_runner(root, args, **kwargs):
        return RunResult(returncode=1, metrics=None, log_tail=["KeyError: 'x'\n"], seconds=0.1)

    ctx, shown = make_ctx(project)
    stage = ReportStage(runner=fake_runner)
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert any("KeyError" in s for s in shown)


def test_render_report_structure():
    runs = [{"run_id": 1, "status": "done", "epochs_run": 2, "best_epoch": 2,
             "best_val_metric": 0.8, "final_train_loss": 0.3, "final_val_loss": 0.5,
             "seconds": 1.0, "config": {"learning_rate": 0.1}}]
    spec = {"goal": "Predict churn", "task_type": "tabular_classification",
            "metric": "accuracy", "target_value": 0.9}
    eval_test = {"metric": "accuracy", "value": 0.78, "loss": 0.55}
    text = render_report("demo", spec, runs, runs[0], eval_test, "Lessons here.",
                         [Path("plots/test_confusion.png")])
    assert text.splitlines()[0] == "# demo: training report"
    assert "Predict churn" in text and "0.78" in text and "0.9" in text
    assert "Lessons here." in text and "![test_confusion](plots/test_confusion.png)" in text
