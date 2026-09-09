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
        [("tool", "recommend_model",
          {"model_type": "gradient_boosting", "reason": "Enough rows."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {"learning_rate": 0.05, "epochs": 500,
                                                 "bogus": 1},
                                      "rationale": "Small data, so a lower [[learning rate]]."})],
        [("text", "done")],
    ])
    # "Gradient boosting" is the model choice; "y" confirms the proposed config.
    ctx, shown = make_ctx(clean_project, llm, ["Gradient boosting", "y"])
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
    # A later call (the code walkthrough) also uses the LLM, so find the propose_config
    # call specifically rather than assuming it is the last one.
    propose_call = next(
        c for c in llm.calls if c["tools"] and c["tools"][0].name == "propose_config"
    )
    prompt = propose_call["messages"][0]["content"]
    assert "config_schema" in prompt or "learning_rate" in prompt


def test_llm_failure_falls_back_to_defaults(clean_project):
    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10
    assert any("defaults" in s for s in shown)


def test_user_edits_config_values(clean_project):
    answers = [
        "Gradient boosting",         # model choice
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


class RecordingQuestioner(ScriptedQuestioner):
    """A ScriptedQuestioner that remembers the options offered to one particular question."""

    def __init__(self, answers, watched_question: str):
        super().__init__(answers)
        self._watched_question = watched_question
        self.offered_options: list[str] | None = None

    def choice(self, question, options, allow_other=True, key=None):
        if question == self._watched_question:
            self.offered_options = list(options)
        return super().choice(question, options, allow_other=allow_other, key=key)


def test_edit_menu_omits_model_type(clean_project):
    # model_type is a "choice" rule; ctx.questioner.number() cannot prompt for it (it would
    # crash on a string default), so the edit menu must never offer it. Task 10 owns the
    # real model-choice flow.
    questioner = RecordingQuestioner(
        ["Gradient boosting", "n", "Done"], "Which value do you want to change?"
    )
    ctx = StageContext(project=clean_project, llm=FakeLLM([]), questioner=questioner,
                       explainer=None, display=lambda s: None)
    CodegenStage().prepare(ctx)
    assert questioner.offered_options is not None
    assert not any(opt.startswith("model_type") for opt in questioner.offered_options)
    assert "Done" in questioner.offered_options


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
    from mlagent.templates_io import schema_for

    schema = schema_for(load_schema("tabular_sklearn"), "gradient_boosting")
    table = config_table({"learning_rate": 0.1, "max_depth": None}, schema)
    assert "| learning_rate | 0.1 |" in table and "| max_depth | none |" in table


def test_is_complete_requires_valid_config(clean_project):
    ctx, _ = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    stage = CodegenStage()
    stage.prepare(ctx)
    cfg = clean_project.read_json("config.json")
    cfg["learning_rate"] = 99
    clean_project.write_json("config.json", cfg)
    assert not stage.is_complete(ctx)
    assert json.loads(clean_project.config_path.read_text(encoding="utf-8"))["learning_rate"] == 99


def test_config_is_flat_for_the_chosen_model(clean_project):
    from mlagent.templates_io import load_schema, schema_for, validate_config

    ctx, _shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    schema = schema_for(load_schema("tabular_sklearn"), cfg["model_type"])
    assert validate_config(cfg, schema) == []
    assert "models" not in cfg and "common" not in cfg


def test_model_choices_material_is_shown_and_the_recommendation_is_named(clean_project):
    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "random_forest",
                                       "reason": "Small, noisy, mixed [[features]]."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {"trees_per_epoch": 30},
                                      "rationale": "More trees for a small set."})],
        [("text", "done")],
    ])
    ctx, shown = make_ctx(clean_project, llm, ["Random forest", "y"])
    CodegenStage().prepare(ctx)
    text = "\n".join(shown)
    assert "Gradient boosting" in text and "Random forest" in text
    assert "Linear" in text
    assert "Small, noisy" in text
    cfg = clean_project.read_json("config.json")
    assert cfg["model_type"] == "random_forest"
    assert cfg["trees_per_epoch"] == 30
    assert "learning_rate" not in cfg  # not a random-forest key
    asked = [q for q in ctx.questioner.asked if "Which model" in q]
    assert asked and "Random forest" in asked[0]  # the question names the recommendation


def test_ask_me_label_takes_the_recommendation(clean_project):
    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "linear", "reason": "Few rows."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {}, "rationale": "Defaults."})],
        [("text", "done")],
    ])
    ctx, _shown = make_ctx(clean_project, llm, ["Ask me after the explanation", "y"])
    CodegenStage().prepare(ctx)
    assert clean_project.read_json("config.json")["model_type"] == "linear"


def test_fallback_heuristic_when_the_llm_is_unavailable(clean_project):
    from mlagent.stages.codegen import fallback_model

    assert fallback_model({"clean_n_rows": 120}) == "linear"
    assert fallback_model({"clean_n_rows": 5000}) == "gradient_boosting"
    assert fallback_model({}) == "gradient_boosting"
    assert fallback_model({"clean_n_rows": "lots"}) == "gradient_boosting"

    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["model_type"] == "gradient_boosting"
    assert any("defaults" in s for s in shown)


def test_unknown_model_recommendation_falls_back(clean_project):
    from mlagent.stages.codegen import fallback_model

    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "quantum", "reason": "Made up."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {}, "rationale": "Defaults."})],
        [("text", "done")],
    ])
    ctx, _shown = make_ctx(clean_project, llm, ["Ask me after the explanation", "y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["model_type"] == fallback_model(clean_project.read_json("data_meta.json"))


def test_walkthrough_lists_every_generated_file(clean_project):
    # Beginner level: one call per file, each rendered section-by-section. At the
    # default "intermediate" level the walkthrough is one paragraph per file instead
    # (see tests/test_teaching.py for that shape).
    spec = clean_project.read_json("spec.json")
    clean_project.write_json("spec.json", {**spec, "learning_level": "beginner"})
    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    text = "\n".join(shown)
    for name in ("data.py", "model.py", "train.py", "evaluate.py"):
        assert f"### `{name}`" in text
    assert "#### settings" in text


def test_model_choices_material_has_no_level_markers(clean_project):
    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "linear", "reason": "Few rows."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {}, "rationale": "Defaults."})],
        [("text", "done")],
    ])
    ctx, shown = make_ctx(clean_project, llm, ["Ask me after the explanation", "y"])
    CodegenStage().prepare(ctx)
    text = "\n".join(shown)
    assert "<!--" not in text
