from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
SMALL = {"model_type": "gradient_boosting", "epochs": 3, "iters_per_epoch": 3,
         "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
         "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
         "l2_regularization": 0.0}


def install(project, config=None) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config or SMALL)
    return project.root


def run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=180,
    )


def train(project) -> Path:
    root = install(project)
    proc = run(root, "train.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root


def test_evaluate_val_writes_record_and_classification_figures(clean_project):
    root = train(clean_project)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert record["split"] == "val" and record["task_type"] == "tabular_classification"
    assert record["metric"] == "accuracy" and 0.0 <= record["value"] <= 1.0
    assert record["checkpoint"] == "checkpoints/best.joblib"
    assert record["run_id"] is None
    assert len(record["y_true"]) == len(record["y_pred"]) == len(record["y_proba"])
    assert len(record["y_proba"][0]) == 2
    assert set(record["figures"]) == {"val_confusion.png", "val_roc_pr.png",
                                      "val_per_class.png"}
    for name in record["figures"]:
        assert (root / "plots" / name).exists()


def test_evaluate_test_uses_the_best_run_checkpoint(clean_project):
    root = train(clean_project)
    shutil.copy(root / "checkpoints" / "best.joblib", root / "checkpoints" / "run1.joblib")
    (root / "runs.jsonl").write_text(
        json.dumps({"run_id": 1, "status": "done", "best_val_metric": 0.7,
                    "checkpoint": "checkpoints/run1.joblib"}) + "\n"
        + json.dumps({"run_id": 2, "status": "failed", "best_val_metric": 0.99,
                      "checkpoint": None}) + "\n",
        encoding="utf-8",
    )
    proc = run(root, "evaluate.py", "--split", "test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["split"] == "test"
    assert record["checkpoint"] == "checkpoints/run1.joblib"
    assert record["run_id"] == 1
    assert "test_confusion.png" in record["figures"]


def test_evaluate_regression_figures_and_explicit_checkpoint(regression_project):
    root = train(regression_project)
    proc = run(root, "evaluate.py", "--split", "test",
               "--checkpoint", "checkpoints/best.joblib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["y_proba"] is None and record["metric"] == "rmse"
    assert set(record["figures"]) == {"test_pred_vs_actual.png", "test_residuals.png"}
    assert record["run_id"] is None


def test_evaluate_without_a_checkpoint_fails_clearly(clean_project):
    root = install(clean_project)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()


def load_evaluate_module():
    """Import evaluate.py as a standalone module (it is not a package member).

    evaluate.py does `from data import load_data`, so the template directory must be on
    sys.path while it is exec'd.
    """
    import importlib.util

    sys.path.insert(0, str(TEMPLATE))
    try:
        spec = importlib.util.spec_from_file_location("evaluate", TEMPLATE / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["evaluate"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_full_proba_handles_a_class_absent_from_training():
    module = load_evaluate_module()

    class Stub:
        classes_ = np.array([0, 2])

        def predict_proba(self, X):
            return np.tile([0.25, 0.75], (len(X), 1))

    proba = module.full_proba(Stub(), np.zeros((4, 2)), 3)
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] < 1e-6


def test_evaluate_script_shape(clean_project):
    root = install(clean_project)
    source = (root / "evaluate.py").read_text(encoding="utf-8")
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert "import mlagent" not in source


@pytest.mark.parametrize("model_type", ["random_forest", "linear"])
def test_other_families_train_and_evaluate(clean_project, model_type):
    configs = {
        "random_forest": {"model_type": "random_forest", "epochs": 2, "trees_per_epoch": 5,
                          "max_depth": 4, "min_samples_leaf": 1, "max_features": 0.8,
                          "seed": 1, "early_stopping_patience": 0},
        "linear": {"model_type": "linear", "epochs": 3, "learning_rate": 0.05,
                   "alpha": 0.0001, "seed": 1, "early_stopping_patience": 0},
    }
    root = install(clean_project, configs[model_type])
    assert run(root, "train.py").returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == model_type
    assert run(root, "evaluate.py").returncode == 0


def test_cli_argv_ignores_ipykernel_launcher_but_parses_run_and_script_argv(monkeypatch):
    module = load_evaluate_module()

    monkeypatch.setattr(sys, "argv", ["/x/ipykernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--split", "test"])
    assert module.cli_argv() == ["--split", "test"]

    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    assert module.cli_argv() == []
