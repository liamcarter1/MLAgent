from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mlagent import captions
from mlagent.profile import profile_dataframe
from mlagent.templates_io import COMMON_FILES, copy_common

TEMPLATE = Path("mlagent/templates/common").resolve()


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_common_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_common_{name}"] = module
    spec.loader.exec_module(module)
    return module


def run_profile(project, args=()):
    result = subprocess.run(
        [sys.executable, "profile.py", *args],
        cwd=str(project.root), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_copy_common_writes_the_scripts(project):
    written = copy_common(COMMON_FILES, project.root)
    assert [p.name for p in written] == list(COMMON_FILES)
    assert (project.root / "profile.py").exists()
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source


def test_profile_script_matches_the_library_profile(clean_project):
    project = clean_project
    (project.data_raw).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(project.data_clean / "data.csv")
    df.to_csv(project.data_raw / "data.csv", index=False)
    copy_common(COMMON_FILES, project.root)
    run_profile(project)

    written = json.loads((project.root / "profile_raw.json").read_text(encoding="utf-8"))
    figures = written.pop("figures")
    assert written == profile_dataframe(df, "target")
    assert set(figures) == {
        "raw_histograms.png", "raw_missing.png", "raw_class_balance.png",
        "raw_correlation.png",
    }
    for name in figures:
        assert (project.plots_dir / name).exists()


def test_profile_script_tag_and_input_flags_and_regression_target(regression_project):
    project = regression_project
    copy_common(COMMON_FILES, project.root)
    run_profile(project, ["--input", "data/clean/data.csv", "--tag", "clean"])
    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["target"]["kind"] == "numeric"
    assert "clean_target_distribution.png" in written["figures"]
    assert (project.plots_dir / "clean_target_distribution.png").exists()
    assert not (project.plots_dir / "clean_class_balance.png").exists()


def test_template_captions_match_mlagent_captions():
    module = load_module("profile")
    kinds = ["histograms", "missing", "class_balance", "target_distribution", "correlation"]
    assert module.CAPTIONS == {k: captions.CAPTIONS[k] for k in kinds}


def test_profile_script_prints_a_caption_under_each_figure(clean_project):
    project = clean_project
    (project.data_raw).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(project.data_clean / "data.csv")
    df.to_csv(project.data_raw / "data.csv", index=False)
    copy_common(COMMON_FILES, project.root)
    result = run_profile(project)
    for kind in ("histograms", "missing", "class_balance", "correlation"):
        stripped = captions.CAPTIONS[kind].replace("[[", "").replace("]]", "")
        line = "How to read this: " + stripped
        assert result.stdout.count(line) == 1


def test_profile_script_has_walkthrough_sections_and_an_argv_guard(project):
    copy_common(COMMON_FILES, project.root)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5


def test_cli_argv_ignores_kernel_launchers_but_parses_run_and_script_argv(monkeypatch):
    module = load_module("profile")

    monkeypatch.setattr(sys, "argv", ["/x/ipykernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["/x/colab_kernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["profile.py", "--tag", "clean"])
    assert module.cli_argv() == ["--tag", "clean"]

    monkeypatch.setattr(sys, "argv", ["/some/dir/profile.py", "--tag", "clean"])
    assert module.cli_argv() == ["--tag", "clean"]

    monkeypatch.setattr(sys, "argv", ["profile.py"])
    assert module.cli_argv() == []
