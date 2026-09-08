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


def test_report_evaluates_best_runs_checkpoint(clean_project):
    """F2/F3: the report must evaluate the BEST run's checkpoint, not the last run's."""
    from mlagent.runlog import best_run, read_runs

    project = trained(clean_project)  # run 1
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner(["y"]),
                       explainer=None, display=lambda s: None)
    cfg = project.read_json("config.json")
    cfg.update({"epochs": 1, "iters_per_epoch": 1, "early_stopping_patience": 0})
    project.write_json("config.json", cfg)
    TrainStage(poll_seconds=0.05).run(ctx)  # run 2, worse/different config

    runs = read_runs(project.runs_path)
    assert len(runs) == 2
    spec_metric = ctx.spec().metric
    best = best_run(runs, spec_metric)
    assert best is not None and best["checkpoint"]

    recorded_args = []

    def recording_runner(root, args, **kwargs):
        recorded_args.append(args)
        from mlagent.runner import run_script
        return run_script(root, args, **kwargs)

    ctx2, _shown = make_ctx(project)
    stage = ReportStage(runner=recording_runner, poll_seconds=0.05)
    stage.run(ctx2)
    assert stage.is_complete(ctx2)
    assert recorded_args == [["train.py", "--eval-test", "--checkpoint", best["checkpoint"]]]


def test_run_number_sorts_numerically_not_lexicographically(clean_project):
    from mlagent.stages.report import _run_number

    plots_dir = clean_project.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run2_training.png", "run10_training.png"):
        (plots_dir / name).touch()
    ordered = sorted(plots_dir.glob("run*_training.png"), key=_run_number)
    assert [p.name for p in ordered] == ["run2_training.png", "run10_training.png"]


def test_report_meta_records_best_run_after_success(clean_project):
    project = trained(clean_project)
    llm = FakeLLM([[("text", "Lessons here.")]])
    ctx, _ = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    meta = project.read_json("report_meta.json")
    assert meta["best_run"] == 1
    assert meta["n_runs"] == 1


def test_better_run_after_report_makes_stage_incomplete_and_declining_leaves_it_incomplete(
    clean_project,
):
    from mlagent.runlog import append_run

    project = trained(clean_project)  # run 1
    llm = FakeLLM([[("text", "Lessons for run 1.")]])
    ctx, _ = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    run1 = project.read_json("report_meta.json")
    assert run1["best_run"] == 1

    # A better run 2 appears (accuracy is "higher is better" for this fixture's spec).
    append_run(project.runs_path, {"status": "done", "best_val_metric": 0.99, "config": {}})
    assert not stage.is_complete(ctx)

    # Declining the re-evaluation leaves the stage incomplete and the report untouched.
    ctx2, shown2 = make_ctx(project, answers=["n"])
    stage.run(ctx2)
    assert not stage.is_complete(ctx2)
    assert any("last evaluated for run 1" in s and "run 2 is now the best model" in s
               for s in shown2)
    assert not any("untouched" in s for s in shown2)


def test_worse_run_keeps_complete_and_rewrites_without_asking_or_evaluating(clean_project):
    from mlagent.runlog import append_run

    project = trained(clean_project)  # run 1
    llm = FakeLLM([[("text", "Lessons for run 1.")]])
    ctx, _ = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    meta_before = project.read_json("report_meta.json")
    assert meta_before["n_runs"] == 1

    # A worse run 2 appears; run 1 stays best.
    append_run(project.runs_path, {"status": "done", "best_val_metric": 0.01, "config": {}})
    assert stage.is_complete(ctx)

    def failing_runner(root, args, **kwargs):
        raise AssertionError("eval runner must not be invoked when the best run is unchanged")

    ctx2, shown2 = make_ctx(project, llm=FakeLLM([[("text", "Lessons for both runs.")]]),
                             answers=[])  # no answers available: confirm must not be called
    stage2 = ReportStage(runner=failing_runner, poll_seconds=0.05)
    stage2.run(ctx2)
    assert stage2.is_complete(ctx2)
    assert any("already evaluated for run 1" in s for s in shown2)
    meta_after = project.read_json("report_meta.json")
    assert meta_after["best_run"] == 1
    assert meta_after["n_runs"] == 2
    report = project.report_path.read_text(encoding="utf-8")
    assert "| 1 |" in report and "| 2 |" in report


def test_missing_meta_with_existing_eval_test_says_run_unknown(clean_project):
    from mlagent.runlog import append_run

    project = trained(clean_project)  # run 1
    llm = FakeLLM([[("text", "Lessons for run 1.")]])
    ctx, _ = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    stage.run(ctx)
    assert stage.is_complete(ctx)

    # Simulate a project created before report_meta.json existed: eval_test.json is
    # present but there is no sidecar recording which run it belongs to.
    project.report_meta_path.unlink()

    # A better run 2 appears.
    append_run(project.runs_path, {"status": "done", "best_val_metric": 0.99, "config": {}})

    ctx2, shown2 = make_ctx(project, answers=["n"])
    stage.run(ctx2)
    assert not any("run None" in s for s in shown2)
    assert any("run is unknown" in s for s in shown2)
    assert not any("untouched" in s for s in shown2)


def test_confirm_wording_says_untouched_only_when_no_eval_test(clean_project):
    project = trained(clean_project)
    ctx, shown = make_ctx(project, answers=["n"])
    stage = ReportStage()
    stage.run(ctx)
    assert any("untouched" in s for s in shown)
    assert not any("last evaluated for run" in s for s in shown)


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
