"""`python train.py --dry-run` in both template families, run for real as a subprocess.

Nothing here touches a GPU: CUDA_VISIBLE_DEVICES="-1" (not "", which unsets the variable
on Windows) is in every child environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TABULAR_TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
IMAGE_TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
CPU_ONLY = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
DRY_RUN_KEYS = {"device", "gpu_name", "batches_per_epoch", "seconds_per_batch",
                "seconds_per_epoch", "n_train", "script"}

GB = {"model_type": "gradient_boosting", "epochs": 4, "iters_per_epoch": 3,
      "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
      "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
      "l2_regularization": 0.0}
RF = {"model_type": "random_forest", "epochs": 2, "trees_per_epoch": 5, "max_depth": 4,
      "min_samples_leaf": 1, "max_features": 0.8, "seed": 1, "early_stopping_patience": 0}
LINEAR = {"model_type": "linear", "epochs": 3, "learning_rate": 0.05, "alpha": 0.0001,
          "seed": 1, "early_stopping_patience": 0}


def install(project, template: Path, config: dict) -> Path:
    for name in CODE_FILES:
        shutil.copy(template / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "train.py", *args], cwd=str(root), capture_output=True, text=True,
        encoding="utf-8", timeout=600, env=CPU_ONLY,
    )


def record(root: Path) -> dict:
    return json.loads((root / "dry_run.json").read_text(encoding="utf-8"))


def assert_wrote_nothing_else(root: Path) -> None:
    assert not (root / "metrics.json").exists()
    assert not (root / "checkpoints" / "best.joblib").exists()
    assert not (root / "checkpoints" / "best.pt").exists()
    assert not (root / "plots" / "training_curves.png").exists()


@pytest.mark.parametrize("config", [GB, RF, LINEAR], ids=["gb", "rf", "linear"])
def test_the_tabular_dry_run_writes_the_whole_contract(clean_project, config):
    root = install(clean_project, TABULAR_TEMPLATE, config)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = record(root)
    assert set(data) == DRY_RUN_KEYS
    assert data["device"] == "cpu" and data["gpu_name"] is None
    assert data["batches_per_epoch"] == 1
    assert data["seconds_per_batch"] >= 0          # a 4dp round can legitimately be 0.0
    assert data["seconds_per_epoch"] == pytest.approx(
        data["seconds_per_batch"] * data["batches_per_epoch"])
    assert data["n_train"] > 0
    assert data["script"] == "train.py"
    assert_wrote_nothing_else(root)


def test_the_tabular_dry_run_prints_a_human_line(clean_project):
    root = install(clean_project, TABULAR_TEMPLATE, GB)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    # Asserting on the last non-blank line, not that it is the only line: a stricter
    # exactly-one-line assertion is brittle against any incidental print (warnings,
    # library banners) that lands on stdout ahead of it.
    assert lines[-1].startswith("Dry run: 3 batches in ")
    assert "s per batch" in lines[-1] and "batches per epoch" in lines[-1]


def test_a_broken_tabular_dry_run_exits_non_zero_with_the_error_on_stderr(clean_project):
    root = install(clean_project, TABULAR_TEMPLATE, GB)
    (root / "data" / "clean" / "data.csv").unlink()
    proc = run(root, "--dry-run")
    assert proc.returncode != 0
    assert "error:" in proc.stderr.lower() or "Error" in proc.stderr
    assert not (root / "dry_run.json").exists()
    assert_wrote_nothing_else(root)


def test_the_tabular_dry_run_section_is_in_the_walkthrough():
    from mlagent.codewalk import split_sections

    source = (TABULAR_TEMPLATE / "train.py").read_text(encoding="utf-8")
    titles = [title for title, _body in split_sections(source)]
    assert "Dry run" in titles


torch = pytest.importorskip("torch")

TINY_CNN = {"model_type": "tiny_cnn", "epochs": 2, "batch_size": 16, "learning_rate": 0.01,
            "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1,
            "augment": "basic", "dropout": 0.3}


def test_the_image_dry_run_writes_the_whole_contract(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = record(root)
    assert set(data) == DRY_RUN_KEYS
    assert data["device"] == "cpu" and data["gpu_name"] is None   # CUDA_VISIBLE_DEVICES=-1
    # 60 images, 70% train split, batch_size 16 -> 3 batches per epoch.
    assert data["batches_per_epoch"] == 3
    assert data["seconds_per_batch"] > 0
    assert data["seconds_per_epoch"] == pytest.approx(
        data["seconds_per_batch"] * data["batches_per_epoch"])
    assert data["n_train"] == 42
    assert data["script"] == "train.py"
    assert_wrote_nothing_else(root)


def test_the_image_dry_run_prints_a_human_line(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    # Last non-blank line, not the only line -- see the tabular test's comment above.
    assert lines[-1].startswith("Dry run: 3 batches in ")
    assert "on cpu;" in lines[-1] and "3 batches per epoch" in lines[-1]


def test_the_image_dry_run_leaves_the_model_untrained_on_disk(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    assert run(root, "--dry-run").returncode == 0
    assert not (root / "checkpoints").exists() or not any(
        (root / "checkpoints").glob("*.pt"))


def test_a_broken_image_dry_run_exits_non_zero_with_the_error_on_stderr(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    (root / "data" / "clean" / "data.npz").unlink()
    proc = run(root, "--dry-run")
    assert proc.returncode != 0
    assert proc.stderr.strip()
    assert not (root / "dry_run.json").exists()
    assert_wrote_nothing_else(root)


def test_a_plain_image_run_still_ignores_stray_kernel_flags():
    source = (IMAGE_TEMPLATE / "train.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--dry-run", action="store_true"' in source
    assert "def cli_argv()" in source


def test_the_image_dry_run_section_is_in_the_walkthrough():
    from mlagent.codewalk import split_sections

    source = (IMAGE_TEMPLATE / "train.py").read_text(encoding="utf-8")
    titles = [title for title, _body in split_sections(source)]
    assert "Dry run" in titles
