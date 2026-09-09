from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mlagent import captions

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")


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


SMALL = {"model_type": "gradient_boosting", "epochs": 4, "iters_per_epoch": 3,
         "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
         "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
         "l2_regularization": 0.0}


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
    assert "epoch 4/4" in proc.stdout
    assert metrics["started_at"] and metrics["started_at"].endswith("+00:00")
    assert metrics["model_type"] == "gradient_boosting"
    assert (root / "plots" / "training_curves.png").exists()
    assert not (root / "eval_val.json").exists()
    assert not (root / "eval_test.json").exists()


def test_regression_run(regression_project):
    root = install(regression_project, SMALL)
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric"] == "rmse" and metrics["higher_is_better"] is False
    assert metrics["best_val_metric"] == min(e["val_metric"] for e in metrics["epochs"])


def test_early_stopping_stops_before_all_epochs(clean_project):
    # A huge learning rate on a tiny model plateaus quickly; patience 1 must stop early.
    root = install(clean_project, {**SMALL, "epochs": 30, "learning_rate": 1.0,
                                   "max_leaf_nodes": 2, "early_stopping_patience": 1})
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stopped_early"] is True
    assert len(metrics["epochs"]) < 30


def test_dry_run_prints_timing_and_writes_nothing(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = json.loads(proc.stdout.strip().splitlines()[-1])
    assert line["seconds_per_epoch"] >= 0 and line["n_train"] > 0
    assert not (root / "metrics.json").exists()
    assert not (root / "checkpoints" / "best.joblib").exists()


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "linear"])
def test_dry_run_works_for_every_family(clean_project, model_type):
    configs = {
        "gradient_boosting": SMALL,
        "random_forest": {"model_type": "random_forest", "epochs": 2, "trees_per_epoch": 5,
                          "max_depth": 4, "min_samples_leaf": 1, "max_features": 0.8,
                          "seed": 1, "early_stopping_patience": 0},
        "linear": {"model_type": "linear", "epochs": 3, "learning_rate": 0.05,
                   "alpha": 0.0001, "seed": 1, "early_stopping_patience": 0},
    }
    root = install(clean_project, configs[model_type])
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = json.loads(proc.stdout.strip().splitlines()[-1])
    # rounded to 4 dp: a sub-100 microsecond epoch can legitimately round to 0.0
    assert line["seconds_per_epoch"] >= 0
    assert line["n_train"] > 0
    assert not (root / "metrics.json").exists()


def test_linear_run_config_has_no_iters_per_epoch_key(clean_project):
    root = install(clean_project, {"model_type": "linear", "epochs": 2, "learning_rate": 0.05,
                                   "alpha": 0.0001, "seed": 1, "early_stopping_patience": 0})
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert "iters_per_epoch" not in metrics["config"]


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


def test_each_run_gets_a_new_started_at(clean_project):
    root = install(clean_project, SMALL)
    assert run(root).returncode == 0
    first = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["started_at"]
    assert run(root).returncode == 0
    second = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["started_at"]
    assert first and second and first != second


def test_template_captions_match_mlagent_captions():
    train_module = load_train_module()
    assert train_module.CAPTIONS == {"training_curves": captions.CAPTIONS["training_curves"]}


def test_train_script_shape(clean_project):
    root = install(clean_project, SMALL)
    source = (root / "train.py").read_text(encoding="utf-8")
    assert "--eval-test" not in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert "# --- settings ---" in source


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


def test_cli_argv_ignores_kernel_launchers_but_parses_run_and_script_argv(monkeypatch):
    train_module = load_train_module()

    monkeypatch.setattr(sys, "argv", ["/x/ipykernel_launcher.py", "-f", "k.json"])
    assert train_module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["/x/colab_kernel_launcher.py", "-f", "k.json"])
    assert train_module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["train.py", "--dry-run"])
    assert train_module.cli_argv() == ["--dry-run"]

    monkeypatch.setattr(sys, "argv", ["/some/dir/train.py", "--dry-run"])
    assert train_module.cli_argv() == ["--dry-run"]

    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert train_module.cli_argv() == []


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
