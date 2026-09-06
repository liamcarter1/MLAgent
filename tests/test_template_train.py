from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
CODE_FILES = ("data.py", "model.py", "train.py")


def install(project, config: dict) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "train.py", *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=120,
    )


SMALL = {"epochs": 4, "iters_per_epoch": 3, "learning_rate": 0.2, "seed": 1,
         "early_stopping_patience": 0}


def test_classification_run_writes_metrics_checkpoint_and_eval(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done"
    assert metrics["metric"] == "accuracy" and metrics["higher_is_better"] is True
    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2, 3, 4]
    assert all(e["train_loss"] > 0 and e["val_loss"] > 0 for e in metrics["epochs"])
    assert 1 <= metrics["best_epoch"] <= 4
    assert metrics["best_val_metric"] == max(e["val_metric"] for e in metrics["epochs"])
    assert metrics["seconds_per_epoch"] > 0
    assert (root / "checkpoints" / "best.joblib").exists()
    ev = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert ev["split"] == "val" and ev["task_type"] == "tabular_classification"
    assert len(ev["y_true"]) == len(ev["y_pred"]) == len(ev["y_proba"]) == metrics["n_val"]
    assert len(ev["y_proba"][0]) == 2
    assert "epoch 4/4" in proc.stdout
    assert not (root / "eval_test.json").exists()


def test_regression_run_and_eval_test(regression_project):
    root = install(regression_project, SMALL)
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric"] == "rmse" and metrics["higher_is_better"] is False
    assert metrics["best_val_metric"] == min(e["val_metric"] for e in metrics["epochs"])
    proc = run(root, "--eval-test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ev = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert ev["split"] == "test" and ev["y_proba"] is None
    assert len(ev["y_true"]) == metrics["n_test"]
    assert ev["value"] > 0


def test_early_stopping_stops_before_all_epochs(clean_project):
    # A huge learning rate on a tiny model plateaus quickly; patience 1 must stop early.
    root = install(clean_project, {**SMALL, "epochs": 30, "learning_rate": 1.0,
                                   "max_leaf_nodes": 2, "early_stopping_patience": 1})
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stopped_early"] is True
    assert len(metrics["epochs"]) < 30


def test_dry_run_prints_timing_and_writes_nothing(clean_project):
    root = install(clean_project, SMALL)  # SMALL has iters_per_epoch: 3
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = json.loads(proc.stdout.strip().splitlines()[-1])
    assert line["seconds_per_round"] >= 0 and line["n_train"] > 0
    assert line["seconds_per_epoch"] >= line["seconds_per_round"]
    # one dry-run round timed, then scaled by the configured iters_per_epoch (3)
    assert line["seconds_per_epoch"] == round(line["seconds_per_round"] * 3, 4)
    assert not (root / "metrics.json").exists()
    assert not (root / "checkpoints" / "best.joblib").exists()


def test_failure_is_recorded_in_metrics(clean_project):
    root = install(clean_project, SMALL)
    meta = clean_project.read_json("data_meta.json")
    meta["target"] = "missing_column"
    clean_project.write_json("data_meta.json", meta)
    proc = run(root)
    assert proc.returncode == 1
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert "missing_column" in metrics["error"]
    required_keys = {
        "task_type", "metric", "config", "epochs", "best_epoch", "best_val_metric",
        "seconds_per_epoch"
    }
    assert set(metrics) >= required_keys


def test_eval_test_without_checkpoint_fails_clearly(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root, "--eval-test")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()


def test_eval_test_with_explicit_checkpoint(regression_project):
    root = install(regression_project, SMALL)
    assert run(root).returncode == 0
    # Copy the checkpoint under a different name and point --eval-test at it explicitly.
    src = root / "checkpoints" / "best.joblib"
    dst = root / "checkpoints" / "run1.joblib"
    shutil.copy2(src, dst)
    proc = run(root, "--eval-test", "--checkpoint", "checkpoints/run1.joblib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (root / "eval_test.json").exists()


def test_eval_test_with_missing_explicit_checkpoint_names_it(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root, "--eval-test", "--checkpoint", "checkpoints/does_not_exist.joblib")
    assert proc.returncode == 1
    combined = (proc.stdout + proc.stderr)
    assert "does_not_exist.joblib" in combined


def test_failed_run_does_not_inherit_previous_run_metrics(clean_project):
    """A successful run followed by a broken one must not report the earlier run's
    epochs/best metric as the failed run's own (final-review finding F1)."""
    root = install(clean_project, SMALL)
    proc = run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert first["status"] == "done" and first["epochs"]

    meta = clean_project.read_json("data_meta.json")
    meta["target"] = "missing_column"
    clean_project.write_json("data_meta.json", meta)
    proc = run(root)
    assert proc.returncode == 1

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["best_epoch"] is None
    assert metrics["best_val_metric"] is None
    assert metrics["epochs"] == []


def load_train_module():
    """Import train.py as a standalone module (it is not a package member)."""
    template_dir = Path("mlagent/templates/tabular_sklearn").resolve()
    sys.path.insert(0, str(template_dir))
    try:
        spec = importlib.util.spec_from_file_location("train_module", template_dir / "train.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        return train_module
    finally:
        sys.path.pop(0)


def test_full_proba_handles_class_absent_from_training():
    train_module = load_train_module()
    X = np.random.default_rng(0).random((20, 2))
    y = np.array([0, 1] * 10)
    model = HistGradientBoostingClassifier(max_iter=2, random_state=0)
    model.fit(X, y)

    result = train_module.evaluate(
        model, X, y, "tabular_classification", "accuracy", n_classes=3
    )

    assert np.isfinite(result["loss"]), "loss should be finite"
    y_proba = np.array(result["y_proba"])
    assert y_proba.shape == (20, 3), "proba should have 20 rows, 3 classes"
    row_sums = y_proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "proba rows should sum to 1"


def test_write_json_retries_on_permission_error(tmp_path, monkeypatch):
    train_module = load_train_module()
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("simulated Windows replace-while-open race")
        return real_replace(src, dst)

    monkeypatch.setattr(train_module.os, "replace", flaky_replace)
    target = tmp_path / "m.json"
    train_module.write_json(target, {"a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert calls["n"] == 3
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_cleans_temp_file_when_retries_exhausted(tmp_path, monkeypatch):
    train_module = load_train_module()

    def always_raises(src, dst):
        raise PermissionError("simulated persistent Windows replace-while-open race")

    monkeypatch.setattr(train_module.os, "replace", always_raises)
    monkeypatch.setattr(train_module.time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError):
        train_module.write_json(tmp_path / "m.json", {"a": 1})

    assert list(tmp_path.iterdir()) == []
