from __future__ import annotations

from mlagent import runlog


def test_append_and_read_round_trip(tmp_path):
    path = tmp_path / "runs.jsonl"
    assert runlog.read_runs(path) == []
    first = runlog.append_run(path, {"status": "done", "best_val_metric": 0.8})
    second = runlog.append_run(path, {"status": "failed"})
    assert first["run_id"] == 1 and second["run_id"] == 2
    runs = runlog.read_runs(path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["best_val_metric"] == 0.8


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "runs.jsonl"
    runlog.append_run(path, {"status": "done"})
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    runlog.append_run(path, {"status": "done"})
    assert [r["run_id"] for r in runlog.read_runs(path)] == [1, 2]


def test_best_run_respects_metric_direction():
    runs = [
        {"run_id": 1, "status": "done", "best_val_metric": 0.7},
        {"run_id": 2, "status": "done", "best_val_metric": 0.9},
        {"run_id": 3, "status": "failed", "best_val_metric": 0.99},
    ]
    assert runlog.best_run(runs, "accuracy")["run_id"] == 2
    assert runlog.best_run(runs, "rmse")["run_id"] == 1
    assert runlog.best_run([], "accuracy") is None


def test_is_better():
    assert runlog.is_better("accuracy", 0.9, 0.8)
    assert not runlog.is_better("accuracy", 0.8, 0.9)
    assert runlog.is_better("rmse", 1.0, 2.0)
    assert runlog.is_better("rmse", 1.0, None)
    assert not runlog.is_better("rmse", None, 1.0)


def test_summarise_markdown_table():
    runs = [{"run_id": 1, "status": "done", "epochs_run": 5, "best_epoch": 4,
             "best_val_metric": 0.8123456, "final_train_loss": 0.3, "final_val_loss": 0.4,
             "seconds": 2.5}]
    table = runlog.summarise(runs, "accuracy")
    assert table.splitlines()[0].startswith("| run | status |")
    assert "val accuracy" in table
    assert "| 1 | done | 5 | 4 | 0.8123 |" in table
    assert runlog.summarise([], "accuracy") == "_No runs yet._"


def test_project_paths(project):
    assert project.config_path == project.root / "config.json"
    assert project.runs_path == project.root / "runs.jsonl"
    assert project.metrics_path == project.root / "metrics.json"
    assert project.report_path == project.root / "report.md"
