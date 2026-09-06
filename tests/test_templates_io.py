from __future__ import annotations

import pytest

from mlagent import templates_io as tio


def test_template_dir_and_schema():
    d = tio.template_dir("tabular_sklearn")
    assert d.is_dir()
    for name in tio.CODE_FILES:
        assert (d / name).exists()
    schema = tio.load_schema("tabular_sklearn")
    assert set(schema) >= {"learning_rate", "epochs", "iters_per_epoch", "seed"}
    with pytest.raises(FileNotFoundError):
        tio.template_dir("no_such_template")


def test_default_config_matches_schema_defaults():
    schema = tio.load_schema("tabular_sklearn")
    cfg = tio.default_config(schema)
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10 and cfg["max_depth"] is None
    assert tio.validate_config(cfg, schema) == []


def test_validate_config_reports_problems():
    schema = tio.load_schema("tabular_sklearn")
    cfg = tio.default_config(schema)
    cfg["learning_rate"] = 5.0
    cfg["epochs"] = "ten"
    cfg["bogus"] = 1
    del cfg["seed"]
    problems = tio.validate_config(cfg, schema)
    assert any("learning_rate" in p for p in problems)
    assert any("epochs" in p for p in problems)
    assert any("bogus" in p for p in problems)
    assert any("seed" in p for p in problems)


def test_coerce_config_clamps_and_drops():
    schema = tio.load_schema("tabular_sklearn")
    cfg, notes = tio.coerce_config(
        {"learning_rate": 9, "epochs": 20.0, "max_depth": None, "bogus": 3, "seed": "7"}, schema
    )
    assert cfg["learning_rate"] == 1.0
    assert cfg["epochs"] == 20 and isinstance(cfg["epochs"], int)
    assert cfg["max_depth"] is None
    assert cfg["seed"] == 7
    assert "bogus" not in cfg
    assert cfg["iters_per_epoch"] == 10  # default kept
    assert any("learning_rate" in n for n in notes) and any("bogus" in n for n in notes)
    assert tio.validate_config(cfg, schema) == []


def test_copy_template(project):
    paths = tio.copy_template("tabular_sklearn", project.root)
    assert [p.name for p in paths] == list(tio.CODE_FILES)
    assert all(p.exists() for p in paths)
    assert "load_data" in (project.root / "data.py").read_text(encoding="utf-8")
    assert not (project.root / "config_schema.json").exists()


def test_validate_rejects_nan_and_inf():
    schema = tio.load_schema("tabular_sklearn")
    cfg = tio.default_config(schema)
    cfg["learning_rate"] = float("nan")
    assert any("learning_rate" in p for p in tio.validate_config(cfg, schema))
    cfg["learning_rate"] = float("inf")
    assert any("learning_rate" in p for p in tio.validate_config(cfg, schema))


def test_coerce_drops_nan_bool_and_non_dict():
    schema = tio.load_schema("tabular_sklearn")
    cfg, notes = tio.coerce_config({"learning_rate": float("nan"), "seed": True}, schema)
    assert cfg["learning_rate"] == 0.1 and cfg["seed"] == 42
    assert any("learning_rate" in n for n in notes) and any("seed" in n for n in notes)
    assert tio.validate_config(cfg, schema) == []
    cfg, notes = tio.coerce_config(["not", "a", "dict"], schema)
    assert cfg == tio.default_config(schema) and notes
    cfg, notes = tio.coerce_config(None, schema)
    assert cfg == tio.default_config(schema) and notes == []
