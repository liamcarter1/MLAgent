from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from mlagent import captions
from mlagent.imageset import write_pair
from mlagent.synth.images import SynthImageConfig, generate
from mlagent.templates_io import copy_shared

TEMPLATE = Path("mlagent/templates/image_common").resolve()
IMAGE_PROFILE = "image_common/profile.py"
KINDS = ("thumbnails", "class_balance", "intensity", "class_means")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_image_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_image_{name}"] = module
    spec.loader.exec_module(module)
    return module


def image_project(project, n_images=30, image_size=32, n_classes=3, duplicate_fraction=0.2,
                  blank_fraction=0.1):
    imageset = generate(SynthImageConfig(
        n_images=n_images, image_size=image_size, n_classes=n_classes, seed=4,
        duplicate_fraction=duplicate_fraction, blank_fraction=blank_fraction, noise=0.05,
    ))
    write_pair(imageset, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label",
        "task_type": "image_classification",
        "modality": "image",
        "source": "synthetic",
        "raw_path": "data/raw/data.npz",
        "raw_n_rows": imageset.n_images,
        "raw_n_cols": image_size * image_size * 3,
        "image_size": image_size,
        "n_channels": 3,
    })
    copy_shared(IMAGE_PROFILE, project.root)
    return project, imageset


def run_profile(project, args=()):
    result = subprocess.run(
        [sys.executable, "profile.py", *args],
        cwd=str(project.root), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_the_copied_script_never_imports_mlagent(project):
    copy_shared(IMAGE_PROFILE, project.root)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5


def test_profile_writes_every_key_and_all_four_figures(project):
    project, imageset = image_project(project)
    run_profile(project)
    written = json.loads((project.root / "profile_raw.json").read_text(encoding="utf-8"))

    assert written["n_images"] == imageset.n_images
    assert written["image_size"] == 32
    assert written["n_channels"] == 3
    assert written["n_classes"] == 3
    assert sum(written["class_counts"].values()) == imageset.n_images
    assert written["duplicate_images"] >= 1
    assert written["blank_images"] >= 1
    assert len(written["channel_mean"]) == 3 and len(written["channel_std"]) == 3
    for key in ("source_width", "source_height"):
        assert set(written[key]) == {"min", "median", "max"}
    assert written["figures"] == [
        "raw_thumbnails.png", "raw_class_balance.png", "raw_intensity.png",
        "raw_class_means.png",
    ]
    for name in written["figures"]:
        assert (project.plots_dir / name).exists()


def test_profile_tag_and_input_flags_write_the_clean_profile(project):
    project, imageset = image_project(project)
    write_pair(imageset, project.data_clean)
    run_profile(project, ["--input", "data/clean/data.npz", "--tag", "clean"])
    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["figures"][0] == "clean_thumbnails.png"
    assert (project.plots_dir / "clean_class_means.png").exists()


def test_profile_prints_a_caption_under_each_figure(project):
    project, _imageset = image_project(project)
    result = run_profile(project)
    for kind in KINDS:
        stripped = captions.CAPTIONS[kind].replace("[[", "").replace("]]", "")
        assert result.stdout.count("How to read this: " + stripped) == 1


def test_template_captions_match_mlagent_captions():
    module = load_module("profile")
    assert module.CAPTIONS == {k: captions.CAPTIONS[k] for k in KINDS}


def test_cli_argv_ignores_kernel_launchers_but_parses_script_argv(monkeypatch):
    module = load_module("profile")
    monkeypatch.setattr(sys, "argv", ["/x/colab_kernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []
    monkeypatch.setattr(sys, "argv", ["profile.py", "--tag", "clean"])
    assert module.cli_argv() == ["--tag", "clean"]
