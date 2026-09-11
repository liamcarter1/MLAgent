from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlagent.templates_io import default_config, load_schema, schema_for, validate_config

TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py")
torch = pytest.importorskip("torch")


def install(project) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    return project.root


def load_module(root: Path, name: str):
    """Import a template file as a standalone module, with the project root on sys.path."""
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(f"img_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"img_{name}"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))


def test_the_schema_has_only_choice_integer_and_number_types():
    schema = load_schema("image_torch")
    for group in (schema["common"], *schema["models"].values()):
        for key, rule in group.items():
            assert rule["type"] in ("choice", "integer", "number"), key


def test_every_family_default_config_validates():
    schema = load_schema("image_torch")
    assert sorted(schema["models"]) == ["resnet18", "small_cnn", "tiny_cnn"]
    for family in schema["models"]:
        flat = schema_for(schema, family)
        config = {**default_config(flat), "model_type": family}
        assert validate_config(config, flat) == []


def test_the_common_keys_and_ranges_match_the_spec():
    common = load_schema("image_torch")["common"]
    assert common["model_type"]["default"] == "small_cnn"
    assert common["model_type"]["choices"] == ["tiny_cnn", "small_cnn", "resnet18"]
    assert (common["epochs"]["min"], common["epochs"]["max"]) == (1, 50)
    assert (common["batch_size"]["min"], common["batch_size"]["max"]) == (8, 256)
    assert common["learning_rate"]["min"] == pytest.approx(1e-5)
    assert common["learning_rate"]["max"] == pytest.approx(1e-1)
    assert (common["weight_decay"]["min"], common["weight_decay"]["max"]) == (0.0, 0.1)
    assert common["augment"]["choices"] == ["none", "basic"]
    assert common["augment"]["default"] == "basic"


def test_the_model_specific_keys_match_the_spec():
    models = load_schema("image_torch")["models"]
    assert set(models["tiny_cnn"]) == {"dropout"}
    assert set(models["small_cnn"]) == {"dropout"}
    assert (models["small_cnn"]["dropout"]["min"], models["small_cnn"]["dropout"]["max"]) == (
        0.0, 0.7
    )
    assert models["resnet18"]["pretrained"]["choices"] == ["imagenet", "none"]
    assert models["resnet18"]["freeze_backbone"]["choices"] == ["yes", "no"]
    assert models["resnet18"]["freeze_backbone"]["default"] == "no"


def test_data_loads_splits_normalises_and_never_overlaps(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    config = {"model_type": "tiny_cnn", "batch_size": 8, "seed": 1, "augment": "basic"}
    loaded = data.load_data(root, config)

    assert loaded["task_type"] == "image_classification"
    assert loaded["classes"] == clean_image_project.read_json("data_meta.json")["class_labels"]
    assert loaded["image_size"] == 32 and loaded["n_channels"] == 3
    total = loaded["n_train"] + loaded["n_val"] + loaded["n_test"]
    assert total == 60
    assert loaded["n_train"] > loaded["n_val"] > 0 and loaded["n_test"] > 0
    x_train, y_train = loaded["train"]
    assert x_train.shape == (loaded["n_train"], 3, 32, 32)
    assert x_train.dtype == torch.float32 and y_train.dtype == torch.int64
    assert float(x_train.min()) > -10.0 and float(x_train.max()) < 10.0
    assert len(set(y_train.tolist())) >= 2


def test_the_split_is_frozen_by_split_seed_not_by_the_model_seed(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    a = data.load_data(root, {"model_type": "tiny_cnn", "seed": 1})
    b = data.load_data(root, {"model_type": "tiny_cnn", "seed": 999})
    assert torch.equal(a["val"][1], b["val"][1])
    assert torch.equal(a["test"][0], b["test"][0])


def test_tiny_classes_go_entirely_to_train_with_a_warning(clean_image_project, capsys):
    root = install(clean_image_project)
    data = load_module(root, "data")
    # class 0: 1 image, class 1: 2 images, class 2: a normal 20-image class.
    labels = np.array([0] + [1, 1] + [2] * 20, dtype=np.int64)
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    parts = data.stratified_split(labels, splits, seed=1)

    all_indices = [int(i) for name in ("train", "val", "test") for i in parts[name]]
    assert sorted(all_indices) == list(range(len(labels)))  # no overlap, none dropped
    assert set(np.flatnonzero(labels == 0).tolist()) <= set(parts["train"].tolist())
    assert set(np.flatnonzero(labels == 1).tolist()) <= set(parts["train"].tolist())
    assert not (set(np.flatnonzero(labels == 0).tolist()) & set(parts["val"].tolist()))
    assert not (set(np.flatnonzero(labels == 0).tolist()) & set(parts["test"].tolist()))
    assert not (set(np.flatnonzero(labels == 1).tolist()) & set(parts["val"].tolist()))
    assert not (set(np.flatnonzero(labels == 1).tolist()) & set(parts["test"].tolist()))

    output = capsys.readouterr().out
    assert "warning" in output and "class 0" in output and "class 1" in output


def test_shift_batch_pads_instead_of_wrapping(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    x = torch.zeros(1, 3, 8, 8)
    x[:, :, :, 0] = 1.0  # left column is 1.0, right column is 0.0

    shifted_right = data.shift_batch(x, dy=0, dx=3)
    assert shifted_right.shape == x.shape
    # The three left-most columns are zero padding, not the wrapped right edge.
    assert torch.equal(shifted_right[:, :, :, :3], torch.zeros(1, 3, 8, 3))
    # The original left column (value 1.0) now sits at its shifted position.
    assert torch.all(shifted_right[:, :, :, 3] == 1.0)

    shifted_left = data.shift_batch(x, dy=0, dx=-3)
    assert shifted_left.shape == x.shape
    # Shifting the 1.0 column off the left edge must not glue it onto the right edge.
    assert torch.all(shifted_left[:, :, :, -3:] == 0.0)


def test_make_loader_and_augment_batch_keep_the_shape(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    loaded = data.load_data(root, {"model_type": "tiny_cnn", "batch_size": 8})
    loader = data.make_loader(loaded["train"], batch_size=8, shuffle=True, seed=0)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1:] == (3, 32, 32) and batch_y.dtype == torch.int64
    generator = torch.Generator().manual_seed(0)
    augmented = data.augment_batch(batch_x, generator)
    assert augmented.shape == batch_x.shape
    assert augmented.dtype == batch_x.dtype


def test_pick_device_reports_cpu_without_cuda(clean_image_project, monkeypatch):
    root = install(clean_image_project)
    data = load_module(root, "data")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert data.pick_device() == "cpu"


@pytest.mark.parametrize("model_type", ["tiny_cnn", "small_cnn"])
def test_the_cnn_families_forward_to_the_right_shape(clean_image_project, model_type):
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": model_type, "dropout": 0.2, "seed": 3}, n_classes=4, image_size=32
    )
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 4)


def _imports_torchvision(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any("torchvision" in alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return bool(node.module) and "torchvision" in node.module
    return False


def test_the_cnn_families_never_import_torchvision(clean_image_project):
    root = install(clean_image_project)
    tree = ast.parse((root / "model.py").read_text(encoding="utf-8"))

    # No top-level (module-scope) statement imports torchvision.
    assert not any(_imports_torchvision(node) for node in tree.body)

    # The resnet18 builder function imports it lazily, inside its own body.
    resnet_builders = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and "resnet" in node.name.lower()
    ]
    assert resnet_builders, "expected a resnet18 builder function"
    assert any(
        _imports_torchvision(inner)
        for builder in resnet_builders
        for inner in ast.walk(builder)
    )


def test_resnet18_builds_without_pretrained_weights(clean_image_project):
    pytest.importorskip("torchvision")
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": "resnet18", "pretrained": "none", "freeze_backbone": "no", "seed": 3},
        n_classes=3, image_size=32,
    )
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_freeze_backbone_freezes_everything_but_the_final_layer(clean_image_project):
    pytest.importorskip("torchvision")
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": "resnet18", "pretrained": "none", "freeze_backbone": "yes", "seed": 3},
        n_classes=3, image_size=32,
    )
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_build_model_rejects_an_unknown_family(clean_image_project):
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    with pytest.raises(ValueError, match="unknown model_type"):
        model_module.build_model({"model_type": "vit"}, n_classes=3, image_size=32)


def test_the_scripts_follow_the_generated_script_conventions():
    for name in CODE_FILES:
        source = (TEMPLATE / name).read_text(encoding="utf-8")
        assert "import mlagent" not in source and "from mlagent" not in source
        assert "import PIL" not in source and "from PIL" not in source
        assert "# --- settings ---" in source
        assert "sys.exit(0)" not in source
        assert source.count("\n# --- ") >= 3


def test_data_py_runs_as_a_script_without_arguments(clean_image_project):
    root = install(clean_image_project)
    proc = subprocess.run(
        [sys.executable, "data.py"], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=180,
        # "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
        env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "train" in proc.stdout
