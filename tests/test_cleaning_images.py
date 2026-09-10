from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mlagent import captions
from mlagent.cleaning_images import (
    STEPS_MARKER,
    apply_steps,
    describe_step,
    render_clean_py,
)
from mlagent.imageset import read_pair, write_pair
from mlagent.synth.images import SynthImageConfig, generate

TEMPLATE = Path("mlagent/templates/image_common").resolve()


def make_set(n=24, size=32, n_classes=3):
    return generate(SynthImageConfig(n_images=n, image_size=size, n_classes=n_classes,
                                     seed=6, noise=0.05))


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_image_common_{name}",
                                                   TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_image_common_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_apply_steps_with_no_steps_returns_an_equal_copy():
    s = make_set()
    out = apply_steps(s, [])
    assert out.n_images == s.n_images
    assert np.array_equal(out.images, s.images)


def test_apply_steps_drops_the_named_indices_and_reindexes():
    s = make_set(n=12)
    out = apply_steps(s, [{"op": "drop_indices", "params": {"indices": [0, 5, 11],
                                                            "reason": "duplicate images"}}])
    assert out.n_images == 9
    assert out.manifest["index"].tolist() == list(range(9))
    kept = [i for i in range(12) if i not in (0, 5, 11)]
    assert np.array_equal(out.images, s.images[kept])
    assert out.class_names == s.class_names


def test_apply_steps_merges_two_drop_steps_without_double_counting():
    s = make_set(n=10)
    out = apply_steps(s, [
        {"op": "drop_indices", "params": {"indices": [1, 2], "reason": "duplicates"}},
        {"op": "drop_indices", "params": {"indices": [2, 3], "reason": "blank images"}},
    ])
    assert out.n_images == 7
    assert np.array_equal(out.images, s.images[[0, 4, 5, 6, 7, 8, 9]])


def test_apply_steps_rejects_an_unknown_op():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        apply_steps(make_set(), [{"op": "drop_columns", "params": {"columns": ["a"]}}])


def test_describe_step_reads_as_a_sentence():
    text = describe_step({"op": "drop_indices",
                          "params": {"indices": [1, 2, 3], "reason": "duplicate images"}})
    assert "3" in text and "duplicate images" in text


def test_render_clean_py_substitutes_the_steps_and_keeps_the_conventions():
    steps = [{"op": "drop_indices", "params": {"indices": [0, 1], "reason": "blank images"}}]
    source = render_clean_py(steps)
    assert STEPS_MARKER not in source
    assert '"drop_indices"' in source
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5


def test_template_captions_match_mlagent_captions():
    module = load_module("clean")
    assert module.CAPTIONS == {
        "clean_before_after_classes": captions.CAPTIONS["clean_before_after_classes"]
    }


def test_the_rendered_script_runs_for_real_and_writes_the_clean_pair(project):
    s = make_set(n=24, size=32, n_classes=3)
    write_pair(s, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "raw_path": "data/raw/data.npz", "image_size": 32, "n_channels": 3,
    })
    steps = [{"op": "drop_indices", "params": {"indices": [0, 1, 2], "reason": "duplicates"}}]
    (project.root / "clean.py").write_text(render_clean_py(steps), encoding="utf-8")

    result = subprocess.run([sys.executable, "clean.py"], cwd=str(project.root),
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr

    cleaned = read_pair(project.data_clean)
    assert cleaned.n_images == 21
    assert np.array_equal(cleaned.images, s.images[3:])
    assert cleaned.class_names == s.class_names

    profile = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert profile["before"]["n_images"] == 24
    assert profile["after"]["n_images"] == 21
    assert profile["steps"] == steps
    assert profile["figures"] == ["clean_before_after_classes.png"]
    assert (project.plots_dir / "clean_before_after_classes.png").exists()
    assert "How to read this:" in result.stdout


def test_the_rendered_script_leaves_the_raw_pair_untouched(project):
    s = make_set(n=16, size=32, n_classes=2)
    write_pair(s, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "raw_path": "data/raw/data.npz", "image_size": 32, "n_channels": 3,
    })
    (project.root / "clean.py").write_text(
        render_clean_py([{"op": "drop_indices",
                          "params": {"indices": [0], "reason": "blank images"}}]),
        encoding="utf-8",
    )
    npz_before = (project.data_raw / "data.npz").read_bytes()
    manifest_before = (project.data_raw / "manifest.csv").read_bytes()
    subprocess.run([sys.executable, "clean.py"], cwd=str(project.root), check=True,
                   capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert (project.data_raw / "data.npz").read_bytes() == npz_before
    assert (project.data_raw / "manifest.csv").read_bytes() == manifest_before
    assert read_pair(project.data_raw).n_images == 16
