from __future__ import annotations

import pytest

from mlagent import templates_io as tio


def test_template_dir_and_schema():
    d = tio.template_dir("tabular_sklearn")
    assert d.is_dir()
    for name in tio.CODE_FILES:
        assert (d / name).exists()
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
    assert set(schema) >= {"learning_rate", "epochs", "iters_per_epoch", "seed"}
    with pytest.raises(FileNotFoundError):
        tio.template_dir("no_such_template")


def test_default_config_matches_schema_defaults():
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
    cfg = tio.default_config(schema)
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10 and cfg["max_depth"] is None
    assert tio.validate_config(cfg, schema) == []


def test_validate_config_reports_problems():
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
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
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
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
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
    cfg = tio.default_config(schema)
    cfg["learning_rate"] = float("nan")
    assert any("learning_rate" in p for p in tio.validate_config(cfg, schema))
    cfg["learning_rate"] = float("inf")
    assert any("learning_rate" in p for p in tio.validate_config(cfg, schema))


def test_coerce_drops_nan_bool_and_non_dict():
    schema = tio.schema_for(tio.load_schema("tabular_sklearn"), "gradient_boosting")
    cfg, notes = tio.coerce_config({"learning_rate": float("nan"), "seed": True}, schema)
    assert cfg["learning_rate"] == 0.1 and cfg["seed"] == 42
    assert any("learning_rate" in n for n in notes) and any("seed" in n for n in notes)
    assert tio.validate_config(cfg, schema) == []
    cfg, notes = tio.coerce_config(["not", "a", "dict"], schema)
    assert cfg == tio.default_config(schema) and notes
    cfg, notes = tio.coerce_config(None, schema)
    assert cfg == tio.default_config(schema) and notes == []


def test_common_files_are_locatable_and_copied(tmp_path):
    assert tio.COMMON_DIR.is_dir()
    assert tio.common_file("profile.py").exists()
    written = tio.copy_common(tio.COMMON_FILES, tmp_path)
    assert [p.name for p in written] == list(tio.COMMON_FILES)
    with pytest.raises(FileNotFoundError):
        tio.common_file("no_such_script.py")


def test_schema_is_nested_and_schema_for_flattens_it():
    schema = tio.load_schema("tabular_sklearn")
    assert set(schema) == {"common", "models"}
    assert set(schema["common"]) == {"model_type", "epochs", "early_stopping_patience", "seed"}
    assert tio.model_types(schema) == ["gradient_boosting", "random_forest", "linear"]

    flat = tio.schema_for(schema, "random_forest")
    assert set(flat) == {"model_type", "epochs", "early_stopping_patience", "seed",
                         "trees_per_epoch", "max_depth", "min_samples_leaf", "max_features"}
    assert "learning_rate" not in flat
    assert tio.schema_for(schema, "linear")["alpha"]["default"] == 0.0001
    with pytest.raises(ValueError):
        tio.schema_for(schema, "quantum")


def test_choice_rules_validate_and_coerce():
    schema = tio.load_schema("tabular_sklearn")
    flat = tio.schema_for(schema, "gradient_boosting")
    cfg = tio.default_config(flat)
    assert cfg["model_type"] == "gradient_boosting"
    assert tio.validate_config(cfg, flat) == []

    bad = {**cfg, "model_type": "quantum"}
    assert any("model_type" in p for p in tio.validate_config(bad, flat))
    bad_type = {**cfg, "model_type": 3}
    assert any("model_type" in p for p in tio.validate_config(bad_type, flat))

    coerced, notes = tio.coerce_config({"model_type": "quantum", "learning_rate": 0.05}, flat)
    assert coerced["model_type"] == "gradient_boosting"
    assert coerced["learning_rate"] == 0.05
    assert any("model_type" in n for n in notes)


from mlagent.templates_io import (  # noqa: E402
    config_table,
    default_config,
    edit_config,
    load_schema,
    schema_for,
)
from mlagent.ui.questions import ScriptedQuestioner  # noqa: E402


def _gb_config():
    nested = load_schema("tabular_sklearn")
    flat = schema_for(nested, "gradient_boosting")
    config = default_config(flat)
    config["model_type"] = "gradient_boosting"
    return config, flat


def test_edit_config_changes_one_value_and_stops_at_done():
    config, flat = _gb_config()
    q = ScriptedQuestioner(["epochs = 10", "20", "Done"])
    edited = edit_config(q, config, flat)
    assert edited["epochs"] == 20 and config["epochs"] == 10  # the input is not mutated
    assert "model_type" not in " ".join(q.asked)  # choice keys are not on the menu


def test_edit_config_reverts_a_value_the_schema_rejects():
    config, flat = _gb_config()
    shown: list[str] = []
    q = ScriptedQuestioner(["learning_rate = 0.1", "5", "Done"])
    edited = edit_config(q, config, flat, display=shown.append)
    assert edited["learning_rate"] == 0.1
    assert any("not allowed" in s for s in shown)


def test_edit_config_zero_means_none_for_nullable_keys():
    config, flat = _gb_config()
    config["max_depth"] = 4
    q = ScriptedQuestioner(["max_depth = 4", "0", "Done"])
    assert edit_config(q, config, flat)["max_depth"] is None


def test_config_table_lists_every_key_with_its_description():
    config, flat = _gb_config()
    table = config_table(config, flat)
    assert table.startswith("| key | value | what it does |")
    assert "| learning_rate | 0.1 |" in table and "| max_depth | none |" in table


def test_shared_file_and_copy_shared_round_trip(tmp_path):
    from mlagent.templates_io import copy_shared, shared_file

    assert shared_file("common/profile.py").is_file()
    written = copy_shared("common/profile.py", tmp_path)
    assert written == tmp_path / "profile.py"
    assert "SCRIPT_NAME" in written.read_text(encoding="utf-8")


def test_copy_shared_can_rename_the_destination(tmp_path):
    from mlagent.templates_io import copy_shared

    written = copy_shared("common/profile.py", tmp_path, name="profile.py")
    assert written.name == "profile.py"


def test_shared_file_rejects_a_missing_path():
    import pytest

    from mlagent.templates_io import shared_file

    with pytest.raises(FileNotFoundError):
        shared_file("common/nope.py")
