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

TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
pytest.importorskip("torch")

TINY = {"model_type": "tiny_cnn", "epochs": 2, "batch_size": 16, "learning_rate": 0.01,
        "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1, "augment": "basic",
        "dropout": 0.1}
SMALL = {**TINY, "model_type": "small_cnn", "epochs": 1, "augment": "none"}
RESNET = {"model_type": "resnet18", "epochs": 1, "batch_size": 16, "learning_rate": 0.01,
          "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1, "augment": "none",
          "pretrained": "none", "freeze_backbone": "no"}
BLOWN_UP = {**TINY, "learning_rate": 0.1, "epochs": 3}

# "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
CPU_ONLY = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}


def install(project, config) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=600, env=CPU_ONLY,
    )


def train(project, config) -> Path:
    root = install(project, config)
    proc = run(root, "train.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root


def load_evaluate_module(root: Path):
    """Import evaluate.py in-process, forcing fresh `data`/`model` imports.

    `evaluate.py` does `from data import ...` and `from model import ...`; if another
    template's test already cached modules named `data` or `model` under those bare
    names, a stale one would satisfy the import instead of this project's own. Swap the
    cache out for the duration of the import and restore it afterwards.
    """
    sys.path.insert(0, str(root))
    saved = {name: sys.modules.pop(name, None) for name in ("data", "model")}
    try:
        spec = importlib.util.spec_from_file_location("img_evaluate", root / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["img_evaluate"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


def test_train_writes_the_full_metrics_contract(clean_image_project):
    root = train(clean_image_project, TINY)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    for key in ("status", "started_at", "model_type", "task_type", "metric",
                "higher_is_better", "config", "classes", "n_train", "n_val", "n_test",
                "epochs", "best_epoch", "best_val_metric", "stopped_early", "error",
                "seconds", "seconds_per_epoch", "device", "checkpoint"):
        assert key in metrics, key
    assert metrics["status"] == "done"
    assert metrics["model_type"] == "tiny_cnn"
    assert metrics["task_type"] == "image_classification"
    assert metrics["metric"] == "accuracy" and metrics["higher_is_better"] is True
    assert metrics["device"] == "cpu"
    assert metrics["checkpoint"] == "checkpoints/best.pt"
    assert len(metrics["epochs"]) == 2
    for row in metrics["epochs"]:
        assert set(row) >= {"epoch", "train_loss", "val_loss", "train_metric", "val_metric"}
    assert 1 <= metrics["best_epoch"] <= 2
    assert (root / "checkpoints" / "best.pt").exists()
    assert (root / "plots" / "training_curves.png").exists()


def test_evaluate_val_writes_the_record_and_all_three_figures(clean_image_project):
    root = train(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert record["split"] == "val" and record["task_type"] == "image_classification"
    assert record["metric"] == "accuracy" and 0.0 <= record["value"] <= 1.0
    assert record["checkpoint"] == "checkpoints/best.pt"
    assert record["run_id"] is None
    assert len(record["y_true"]) == len(record["y_pred"]) == len(record["y_proba"])
    assert len(record["y_proba"][0]) == 3
    assert set(record["figures"]) == {"val_confusion.png", "val_per_class.png",
                                      "val_misclassified.png"}
    for name in record["figures"]:
        assert (root / "plots" / name).exists()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert record["started_at"] == metrics["started_at"]


def test_evaluate_prints_a_caption_under_the_misclassified_grid(clean_image_project):
    root = train(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    stripped = captions.CAPTIONS["misclassified"].replace("[[", "").replace("]]", "")
    assert proc.stdout.count("How to read this: " + stripped) == 1


def test_template_captions_match_mlagent_captions(clean_image_project):
    root = install(clean_image_project, TINY)
    module = load_evaluate_module(root)
    kinds = ["confusion", "per_class", "misclassified"]
    assert module.CAPTIONS == {k: captions.CAPTIONS[k] for k in kinds}


def test_small_cnn_trains_and_evaluates(clean_image_project):
    root = train(clean_image_project, SMALL)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == "small_cnn"
    assert run(root, "evaluate.py").returncode == 0


def test_resnet18_without_pretrained_weights_trains_and_evaluates(clean_image_project):
    pytest.importorskip("torchvision")
    root = train(clean_image_project, RESNET)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == "resnet18"
    assert metrics["config"]["pretrained"] == "none"
    assert run(root, "evaluate.py").returncode == 0


def test_evaluate_test_uses_the_best_runs_checkpoint(clean_image_project):
    root = train(clean_image_project, TINY)
    shutil.copy(root / "checkpoints" / "best.pt", root / "checkpoints" / "run1.pt")
    (root / "runs.jsonl").write_text(
        json.dumps({"run_id": 1, "status": "done", "best_val_metric": 0.7,
                    "checkpoint": "checkpoints/run1.pt"}) + "\n"
        + json.dumps({"run_id": 2, "status": "failed", "best_val_metric": 0.99,
                      "checkpoint": None}) + "\n",
        encoding="utf-8",
    )
    proc = run(root, "evaluate.py", "--split", "test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["checkpoint"] == "checkpoints/run1.pt"
    assert record["run_id"] == 1
    assert "test_confusion.png" in record["figures"]


def test_a_non_finite_loss_is_recorded_as_a_failed_run(clean_image_project):
    root = install(clean_image_project, BLOWN_UP)
    (root / "data.py").write_text(
        (root / "data.py").read_text(encoding="utf-8").replace(
            "    validate(data)\n    return data",
            "    validate(data)\n"
            "    data['train'] = (data['train'][0] * float('inf'), data['train'][1])\n"
            "    return data",
        ),
        encoding="utf-8",
    )
    proc = run(root, "train.py")
    assert proc.returncode == 1
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["error"]
    assert metrics["checkpoint"] is None


def test_evaluate_without_a_checkpoint_fails_clearly(clean_image_project):
    root = install(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()


def test_the_scripts_follow_the_generated_script_conventions():
    for name in ("train.py", "evaluate.py"):
        source = (TEMPLATE / name).read_text(encoding="utf-8")
        assert "import mlagent" not in source and "from mlagent" not in source
        assert "import PIL" not in source and "from PIL" not in source
        assert "# --- settings ---" in source
        assert "def cli_argv()" in source
        assert "sys.exit(0)" not in source.split("if __name__")[0]
        assert source.count("\n# --- ") >= 4
        assert "torchvision" not in source
