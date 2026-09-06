import json
from pathlib import Path

from mlagent.project import Project


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
