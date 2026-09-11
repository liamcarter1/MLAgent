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


def test_archive_run_preserves_a_pt_checkpoint_suffix(project):
    from mlagent.runs import archive_run

    project.ensure_dirs()
    (project.checkpoints_dir / "best.pt").write_bytes(b"weights")
    (project.plots_dir / "training_curves.png").write_bytes(b"png")
    (project.plots_dir / "val_misclassified.png").write_bytes(b"png")

    figures = archive_run(project, 3, checkpoint="checkpoints/best.pt")
    assert (project.checkpoints_dir / "run3.pt").exists()
    assert not (project.checkpoints_dir / "run3.joblib").exists()
    assert {p.name for p in figures} == {"run3_training.png", "run3_val_misclassified.png"}


def test_archive_run_falls_back_to_the_legacy_joblib_name(project):
    from mlagent.runs import archive_run

    project.ensure_dirs()
    (project.checkpoints_dir / "best.joblib").write_bytes(b"model")
    archive_run(project, 1)
    assert (project.checkpoints_dir / "run1.joblib").exists()


def test_log_finished_run_records_the_pt_checkpoint(project):
    from mlagent.runs import log_finished_run

    project.ensure_dirs()
    started = "2026-09-10T12:00:00.000000+00:00"
    project.write_json("metrics.json", {
        "status": "done", "started_at": started, "model_type": "small_cnn",
        "task_type": "image_classification", "config": {"model_type": "small_cnn"},
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_metric": 0.7,
                    "val_metric": 0.65}],
        "best_epoch": 1, "best_val_metric": 0.65, "seconds": 1.0, "error": None,
        "checkpoint": "checkpoints/best.pt",
    })
    project.write_json("eval_val.json", {"started_at": started, "value": 0.65})
    (project.checkpoints_dir / "best.pt").write_bytes(b"weights")

    logged = log_finished_run(project)
    assert logged is not None and logged.new is True
    assert logged.entry["checkpoint"] == "checkpoints/run1.pt"
    assert (project.checkpoints_dir / "run1.pt").exists()
