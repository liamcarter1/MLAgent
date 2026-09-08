from __future__ import annotations

import json

from mlagent.llm import FakeLLM
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage, check_data, config_table
from mlagent.templates_io import CODE_FILES, load_schema
from mlagent.ui.questions import ScriptedQuestioner


def make_ctx(project, llm, answers):
    shown = []
    ctx = StageContext(project=project, llm=llm, questioner=ScriptedQuestioner(answers),
                       explainer=None, display=shown.append)
    return ctx, shown


def test_llm_proposal_is_coerced_and_written(clean_project):
    llm = FakeLLM([
        [("tool", "propose_config", {"config": {"learning_rate": 0.05, "epochs": 500,
                                                 "bogus": 1},
                                      "rationale": "Small data, so a lower [[learning rate]]."})],
        [("text", "done")],
    ])
    ctx, shown = make_ctx(clean_project, llm, ["y"])  # confirm: happy with config
    stage = CodegenStage()
    assert not stage.is_complete(ctx)
    stage.prepare(ctx)
    assert stage.is_complete(ctx)
    for name in CODE_FILES:
        assert (clean_project.root / name).exists()
    cfg = clean_project.read_json("config.json")
    assert cfg["learning_rate"] == 0.05
    assert cfg["epochs"] == 100  # clamped to the schema maximum
    assert "bogus" not in cfg
    assert cfg["seed"] == 42
    text = "\n".join(shown)
    assert "[[learning rate]]" in text
    assert "Lowered epochs" in text
    assert llm.calls and llm.calls[0]["tools"][0].name == "propose_config"
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "config_schema" in prompt or "learning_rate" in prompt


def test_llm_failure_falls_back_to_defaults(clean_project):
    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10
    assert any("defaults" in s for s in shown)


def test_user_edits_config_values(clean_project):
    answers = [
        "n",                         # not happy with the config
        "epochs = 10",               # pick the key
        "3",                         # new value
        "max_depth = None",          # pick nullable key
        "4",                         # new value
        "Done",
    ]
    ctx, _ = make_ctx(clean_project, FakeLLM([]), answers)
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["epochs"] == 3 and cfg["max_depth"] == 4


def test_bad_data_stops_stage_without_writing(clean_project):
    meta = clean_project.read_json("data_meta.json")
    meta["target"] = "nope"
    clean_project.write_json("data_meta.json", meta)
    ctx, shown = make_ctx(clean_project, FakeLLM([]), [])
    stage = CodegenStage()
    stage.prepare(ctx)
    assert not stage.is_complete(ctx)
    assert not (clean_project.root / "config.json").exists()
    assert any("nope" in s for s in shown)


def test_unsupported_task_type_is_reported(clean_project):
    spec = clean_project.read_json("spec.json")
    spec["task_type"] = "image_classification"
    clean_project.write_json("spec.json", spec)
    ctx, shown = make_ctx(clean_project, FakeLLM([]), [])
    stage = CodegenStage()
    stage.prepare(ctx)
    assert not stage.is_complete(ctx)
    assert any("image_classification" in s for s in shown)


def test_check_data_and_config_table(clean_project):
    meta = clean_project.read_json("data_meta.json")
    assert check_data(meta, clean_project.root) == []
    meta["splits"] = {"train": 0.5, "val": 0.1, "test": 0.1}
    meta["feature_columns"] = []
    problems = check_data(meta, clean_project.root)
    assert any("split" in p for p in problems) and any("feature" in p for p in problems)
    schema = load_schema("tabular_sklearn")
    table = config_table({"learning_rate": 0.1, "max_depth": None}, schema)
    assert "| learning_rate | 0.1 |" in table and "| max_depth | none |" in table


def test_is_complete_requires_valid_config(clean_project):
    ctx, _ = make_ctx(clean_project, FakeLLM([]), ["y"])
    stage = CodegenStage()
    stage.prepare(ctx)
    cfg = clean_project.read_json("config.json")
    cfg["learning_rate"] = 99
    clean_project.write_json("config.json", cfg)
    assert not stage.is_complete(ctx)
    assert json.loads(clean_project.config_path.read_text(encoding="utf-8"))["learning_rate"] == 99


def test_config_is_flat_for_the_chosen_model(clean_project):
    from mlagent.templates_io import load_schema, schema_for, validate_config

    ctx, _shown = make_ctx(clean_project, FakeLLM([]), ["y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    schema = schema_for(load_schema("tabular_sklearn"), cfg["model_type"])
    assert validate_config(cfg, schema) == []
    assert "models" not in cfg and "common" not in cfg
