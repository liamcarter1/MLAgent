from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import EVAL_VAL_FILE, TrainStage, archive_run, build_run_entry
from mlagent.ui.questions import ScriptedQuestioner

SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


def prepared(project):
    """Run codegen with defaults so the training project exists."""
    ctx = StageContext(project=project, llm=FakeLLM([]),
                       questioner=ScriptedQuestioner(["Gradient boosting", "y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(SMALL)
    project.write_json("config.json", cfg)
    return project


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


def test_real_training_run_is_logged_archived_and_debriefed(clean_project):
    project = prepared(clean_project)
    # Two calls: prepare()'s preamble, then debrief()'s narrative.
    llm = FakeLLM([
        [("text", "Training is about to start.")],
        [("text", "Best [[validation accuracy]] beat the target.")],
    ])
    ctx, shown, figures = make_ctx(project, llm)
    stage = TrainStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="train", commands=[["train.py"], ["evaluate.py"]],
                              outputs=["metrics.json", "eval_val.json"])
    text = "\n".join(shown)
    assert "cost" in text.lower() and "cpu" in text.lower()
    assert not stage.outputs_ready(ctx, handoff)

    run_cells(project, handoff)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["run_id"] == 1
    assert runs[0]["epochs_run"] == 3 and runs[0]["best_val_metric"] is not None
    assert runs[0]["config"]["epochs"] == 3
    assert runs[0]["checkpoint"] == "checkpoints/run1.joblib"
    assert runs[0]["started_at"]
    assert (project.checkpoints_dir / "run1.joblib").exists()
    names = sorted(p.name for p in project.plots_dir.glob("run1_*.png"))
    assert names == ["run1_training.png", "run1_val_confusion.png",
                     "run1_val_per_class.png", "run1_val_roc_pr.png"]
    shown_figures = [Path(p).name for p, _c in figures]
    assert shown_figures[0] == "run1_training.png"
    assert set(shown_figures) == set(names)
    assert all(caption for _p, caption in figures)
    assert "[[validation accuracy]]" in "\n".join(shown)
    prompt = llm.calls[-1]["messages"][0]["content"]  # the debrief call, not the preamble
    assert "best_epoch" in prompt


def test_a_second_run_is_logged_as_run_two(clean_project):
    project = prepared(clean_project)
    ctx, _shown, _figures = make_ctx(project)
    stage = TrainStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    # The user edits config.json and runs the cells again without asking the agent first.
    cfg = project.read_json("config.json")
    cfg["epochs"] = 2
    project.write_json("config.json", cfg)
    run_cells(project, handoff)
    assert not stage.is_complete(ctx)  # the new run is not logged yet

    stage.debrief(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["started_at"] != runs[1]["started_at"]
    assert (project.plots_dir / "run2_training.png").exists()
    assert stage.is_complete(ctx)


def test_debriefing_twice_does_not_log_the_same_run_twice(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    stage.debrief(ctx)
    assert len(runlog.read_runs(project.runs_path)) == 1


def test_failed_run_is_logged_and_the_stage_stays_incomplete(clean_project):
    project = prepared(clean_project)
    project.ensure_dirs()
    # Leftover outputs from an earlier successful run must not be archived under run1's
    # name just because a later run's metrics.json happens to say "failed".
    (project.plots_dir / "training_curves.png").write_bytes(b"leftover")
    (project.checkpoints_dir / "best.joblib").write_bytes(b"leftover")
    project.write_json("metrics.json", {
        "status": "failed", "started_at": "2026-09-08T10:00:00.000000+00:00",
        "model_type": "gradient_boosting", "config": {"epochs": 3}, "epochs": [],
        "error": "ValueError: boom", "seconds": 0.2,
    })
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage()
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert runs[0]["status"] == "failed" and "boom" in runs[0]["error"]
    assert runs[0]["checkpoint"] is None
    assert any("boom" in s for s in shown)
    assert not list(project.plots_dir.glob("run1_*.png"))
    assert not (project.checkpoints_dir / "run1.joblib").exists()


def test_debrief_without_metrics_says_so(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    TrainStage().debrief(ctx)
    assert any("train.py" in s for s in shown)


def test_prepare_without_a_config_raises(clean_project):
    ctx, _shown, _figures = make_ctx(clean_project)
    try:
        TrainStage().prepare(ctx)
    except RuntimeError as exc:
        assert "codegen" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_build_run_entry_reads_everything_from_metrics():
    metrics = {
        "status": "done", "started_at": "2026-09-08T10:00:00.000000+00:00",
        "config": {"epochs": 2}, "best_epoch": 2, "best_val_metric": 0.9, "seconds": 1.5,
        "error": None,
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
                   {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4}],
    }
    entry = build_run_entry(metrics, "checkpoints/run3.joblib")
    assert entry["status"] == "done" and entry["epochs_run"] == 2
    assert entry["final_val_loss"] == 0.4 and entry["seconds"] == 1.5
    assert entry["applied_diff"] is None and entry["error"] is None
    assert entry["checkpoint"] == "checkpoints/run3.joblib"
    assert entry["started_at"] == metrics["started_at"]

    failed = build_run_entry({"status": "failed", "error": "boom", "epochs": []},
                             "checkpoints/run4.joblib")
    assert failed["status"] == "failed" and failed["checkpoint"] is None


def test_archive_run_copies_curves_checkpoint_and_val_figures(project):
    project.ensure_dirs()
    (project.plots_dir / "training_curves.png").write_bytes(b"a")
    (project.plots_dir / "val_confusion.png").write_bytes(b"b")
    (project.plots_dir / "val_roc_pr.png").write_bytes(b"c")
    (project.checkpoints_dir / "best.joblib").write_bytes(b"d")
    archived = archive_run(project, 7)
    assert [p.name for p in archived] == ["run7_training.png", "run7_val_confusion.png",
                                          "run7_val_roc_pr.png"]
    assert (project.checkpoints_dir / "run7.joblib").exists()


def test_debrief_refuses_a_stale_eval_val_json(clean_project):
    project = prepared(clean_project)
    ctx, shown, figures = make_ctx(project)
    stage = TrainStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert len(runlog.read_runs(project.runs_path)) == 1

    # Rerun only train.py: eval_val.json now belongs to the previous run.
    run_cells(project, Handoff(stage="train", commands=[["train.py"]], outputs=[]))
    shown.clear()
    figures.clear()
    stage.debrief(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 1
    assert not stage.is_complete(ctx)
    assert any("evaluate.py" in s for s in shown)
    assert not figures

    run_cells(project, Handoff(stage="train", commands=[["evaluate.py"]], outputs=[]))
    stage.debrief(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert stage.is_complete(ctx)


def test_narrative_headline_survives_a_non_numeric_target_value(clean_project):
    project = prepared(clean_project)
    ctx, _shown, _figures = make_ctx(project, FakeLLM([]))
    stage = TrainStage()

    class FakeSpec:
        task_type = "tabular_classification"
        metric = "accuracy"
        target_value = None

    entry = {"run_id": 1, "best_val_metric": 0.8, "best_epoch": 2}
    metrics = {"model_type": "gradient_boosting", "epochs": [], "best_epoch": 2}
    text = stage._narrative(ctx, FakeSpec(), entry, metrics, {}, [])
    assert "Run 1 finished" in text
    assert "target -" in text


def test_learning_level_reaches_the_debrief_prompt(clean_project):
    project = prepared(clean_project)
    spec = project.read_json("spec.json")
    project.write_json("spec.json", {**spec, "learning_level": "beginner"})
    llm = FakeLLM([[("text", "x")], [("text", "done")]])
    ctx, _shown, _figures = make_ctx(project, llm)
    stage = TrainStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    stage.debrief(ctx)
    assert "new to machine learning" in llm.calls[-1]["system"]


def test_llm_failure_still_completes(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project, FakeLLM([]))
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("best" in s.lower() for s in shown)
    assert json.loads(project.metrics_path.read_text(encoding="utf-8"))["status"] == "done"
    assert project.exists(EVAL_VAL_FILE)
