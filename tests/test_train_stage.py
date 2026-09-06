from __future__ import annotations

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.runner import RunResult
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import LivePlotter, TrainStage, build_run_entry
from mlagent.ui.questions import ScriptedQuestioner


def prepared(project):
    """Run codegen with defaults so the training project exists."""
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner(["y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().run(ctx)
    cfg = project.read_json("config.json")
    cfg.update({"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", cfg)
    return project


def make_ctx(project, llm=None, answers=()):
    shown = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append)
    return ctx, shown


def test_real_training_run_logs_and_plots(clean_project, capsys):
    project = prepared(clean_project)
    llm = FakeLLM([[("text", "Best [[validation accuracy]] beat the target.")]])
    ctx, shown = make_ctx(project, llm)
    figs = []
    stage = TrainStage(display_fig=figs.append, poll_seconds=0.05)
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["run_id"] == 1
    assert runs[0]["epochs_run"] == 3 and runs[0]["best_val_metric"] is not None
    assert runs[0]["config"]["epochs"] == 3
    assert runs[0]["checkpoint"] == "checkpoints/run1.joblib"
    assert (project.checkpoints_dir / "run1.joblib").exists()
    assert project.metrics_path.exists() and project.exists("eval_val.json")
    names = sorted(p.name for p in project.plots_dir.glob("run1_*.png"))
    assert names == ["run1_training.png", "run1_val_confusion.png", "run1_val_per_class.png",
                     "run1_val_roc_pr.png"]
    assert len(figs) >= 1  # live plot redrawn at least once
    text = "\n".join(shown)
    assert "[[validation accuracy]]" in text
    assert "cost" in text.lower() and "cpu" in text.lower()
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "best_epoch" in prompt
    # F8: subprocess stdout streams to the cell instead of being discarded.
    assert "epoch 1/" in capsys.readouterr().out


def test_failed_run_is_logged_and_stage_incomplete(clean_project):
    project = prepared(clean_project)

    def fake_runner(root, **kwargs):
        return RunResult(returncode=1, metrics={"status": "failed", "epochs": [],
                                                "error": "ValueError: boom"},
                         log_tail=["Traceback\n", "ValueError: boom\n"], seconds=0.2,
                         expects_metrics=True)

    ctx, shown = make_ctx(project)
    stage = TrainStage(runner=fake_runner)
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert runs[0]["status"] == "failed" and "boom" in runs[0]["error"]
    assert any("boom" in s for s in shown)


def test_llm_failure_still_completes(clean_project):
    project = prepared(clean_project)
    ctx, shown = make_ctx(project, FakeLLM([]))
    TrainStage(poll_seconds=0.05).run(ctx)
    assert TrainStage().is_complete(ctx)
    assert any("best" in s.lower() for s in shown)


def test_missing_config_raises(clean_project):
    ctx, _ = make_ctx(clean_project)
    try:
        TrainStage().run(ctx)
    except RuntimeError as exc:
        assert "codegen" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_live_plotter_and_run_entry():
    figs = []
    plotter = LivePlotter("accuracy", display_fig=figs.append)
    plotter.update({"epochs": []})
    plotter.update({"epochs": [{"epoch": 1, "train_loss": 1, "val_loss": 1,
                                "train_metric": 0.5, "val_metric": 0.5}]})
    assert plotter.updates == 2 and len(figs) == 1
    result = RunResult(returncode=0, seconds=1.5, expects_metrics=True, metrics={
        "status": "done", "best_epoch": 2, "best_val_metric": 0.9,
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
                   {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4}]})
    entry = build_run_entry({"epochs": 2}, result, "2026-09-06T10:00:00")
    assert entry["status"] == "done" and entry["epochs_run"] == 2
    assert entry["final_val_loss"] == 0.4 and entry["seconds"] == 1.5
    assert entry["applied_diff"] is None and entry["error"] is None
    assert entry["checkpoint"] is None  # no checkpoint path passed
    entry_with_checkpoint = build_run_entry(
        {"epochs": 2}, result, "2026-09-06T10:00:00", checkpoint="checkpoints/run3.joblib"
    )
    assert entry_with_checkpoint["checkpoint"] == "checkpoints/run3.joblib"
    failed_result = RunResult(returncode=1, expects_metrics=True, metrics={"status": "failed"})
    failed_entry = build_run_entry(
        {}, failed_result, "2026-09-06T10:00:00", checkpoint="checkpoints/run4.joblib"
    )
    assert failed_entry["checkpoint"] is None  # never recorded for a failed run


def test_timeout_defaults_to_minutes_per_run_ceiling(clean_project):
    project = prepared(clean_project)
    calls = []

    def fake_runner(root, **kwargs):
        calls.append(kwargs)
        return RunResult(returncode=1, metrics={"status": "failed", "epochs": [],
                                                "error": "boom"},
                         log_tail=[], seconds=0.1, expects_metrics=True)

    ctx, _shown = make_ctx(project)
    stage = TrainStage(runner=fake_runner)
    stage.run(ctx)
    assert len(calls) == 1
    spec = ctx.spec()
    assert spec.minutes_per_run == 5
    assert calls[0]["timeout"] == max(60.0, spec.minutes_per_run * 60 * 3)


def test_headline_tolerates_missing_best(clean_project):
    ctx, _shown = make_ctx(clean_project, FakeLLM([]))
    spec = ctx.spec()
    entry = {"run_id": 1, "best_val_metric": None, "best_epoch": None}
    text = TrainStage()._debrief(ctx, spec, entry, {}, {}, [])
    assert "Run 1" in text
