import json
from pathlib import Path

import pytest

from mlagent.project import Project, read_json_file, write_json_file


def test_paths_and_dirs(tmp_path: Path):
    p = Project(tmp_path / "projects" / "cats")
    assert p.name == "cats"
    p.ensure_dirs()
    assert p.data_raw.is_dir()
    assert p.data_clean.is_dir()
    assert p.plots_dir.is_dir()
    assert p.checkpoints_dir.is_dir()
    assert p.spec_path == p.root / "spec.json"
    assert p.state_path == p.root / "state.json"
    assert p.glossary_path == p.root / "glossary.json"


def test_json_roundtrip(project: Project):
    assert project.read_json("spec.json") is None
    assert project.read_json("spec.json", default={}) == {}
    path = project.write_json("spec.json", {"goal": "classify cats", "n": 3})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["goal"] == "classify cats"
    assert project.read_json("spec.json")["n"] == 3
    assert project.exists("spec.json")


def test_write_json_is_atomic_no_temp_files_left(project: Project):
    project.write_json("spec.json", {"a": 1})
    leftovers = [p for p in project.root.iterdir() if p.name != "spec.json" and p.suffix == ".tmp"]
    assert leftovers == []
    assert project.read_json("spec.json") == {"a": 1}


def test_read_json_tolerates_corrupt_file_and_warns(project: Project):
    path = project.root / "spec.json"
    path.write_text('{"goal": "cats", "task_type": ', encoding="utf-8")
    with pytest.warns(UserWarning, match="spec.json"):
        result = project.read_json("spec.json", default={})
    assert result == {}


def test_read_json_file_tolerates_missing_and_corrupt(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert read_json_file(missing, default="fallback") == "fallback"

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning, match="corrupt.json"):
        assert read_json_file(corrupt, default="fallback") == "fallback"


def test_write_json_file_atomic_roundtrip(tmp_path: Path):
    path = tmp_path / "sub" / "data.json"
    write_json_file(path, {"x": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


def test_ensure_dirs_creates_the_runs_archive_dir(tmp_path):
    from mlagent.project import Project

    p = Project(tmp_path / "proj")
    p.ensure_dirs()
    assert p.runs_dir == tmp_path / "proj" / "runs" and p.runs_dir.is_dir()
