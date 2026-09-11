from __future__ import annotations

import json

import pytest

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

    def choice(self, question, options, allow_other=True, key=None, default=None):
        if question == self._watched_question:
            self.offered_options = list(options)
        return super().choice(question, options, allow_other=allow_other, key=key,
                              default=default)


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


def test_unregistered_task_type_is_reported(clean_project):
    spec = clean_project.read_json("spec.json")
    spec["task_type"] = "audio_classification"
    clean_project.write_json("spec.json", spec)
    ctx, _shown = make_ctx(clean_project, FakeLLM([]), [])
    stage = CodegenStage()
    with pytest.raises(ValueError, match="audio_classification"):
        stage.prepare(ctx)
    assert not stage.is_complete(ctx)


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
    # Default (intermediate) level: one call for all files, but each file still gets its
    # own heading and its full fenced source, not just a summary.
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


def test_labels_for_each_family_and_the_back_compat_alias():
    from mlagent.stages.codegen import MODEL_LABELS, label_for, labels_for

    assert set(labels_for("image_torch").values()) == {"tiny_cnn", "small_cnn", "resnet18"}
    assert set(labels_for("tabular_sklearn").values()) == {
        "linear", "random_forest", "gradient_boosting"
    }
    assert MODEL_LABELS == labels_for("tabular_sklearn")
    assert label_for("image_torch", "resnet18") == "Pretrained ResNet-18"
    assert label_for("tabular_sklearn", "linear") == "Linear / logistic regression"


def test_the_image_labels_cover_every_model_type_in_the_schema():
    from mlagent.stages.codegen import labels_for
    from mlagent.templates_io import load_schema, model_types

    assert sorted(labels_for("image_torch").values()) == sorted(
        model_types(load_schema("image_torch"))
    )


def test_fallback_model_is_family_aware():
    from mlagent.stages.codegen import fallback_model

    assert fallback_model({"clean_n_rows": 100}, "tabular_sklearn") == "linear"
    assert fallback_model({"clean_n_rows": 5000}, "tabular_sklearn") == "gradient_boosting"
    assert fallback_model({"clean_n_rows": 100}, "image_torch") == "small_cnn"


def test_check_data_accepts_a_clean_image_project(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    assert check_data(project.read_json("data_meta.json"), project.root) == []


def test_check_data_reports_a_missing_image_file(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    (project.data_clean / "data.npz").unlink()
    problems = check_data(project.read_json("data_meta.json"), project.root)
    assert problems and "data.npz" in problems[0]


def test_check_data_reports_too_few_image_classes(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    meta = {**project.read_json("data_meta.json"), "n_classes": 1}
    problems = check_data(meta, project.root)
    assert any("2 classes" in p for p in problems)


def test_codegen_on_an_image_project_writes_the_image_template_and_config(clean_image_project):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.stages.codegen import CodegenStage
    from mlagent.templates_io import load_schema, schema_for, validate_config
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    project = clean_image_project
    shown: list[str] = []
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner({"codegen.model_type": "Small CNN"},
                                  ScriptedQuestioner(["y"])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    stage = CodegenStage()
    stage.prepare(ctx)

    for name in ("data.py", "model.py", "train.py", "evaluate.py"):
        assert project.exists(name)
    source = (project.root / "model.py").read_text(encoding="utf-8")
    assert "SmallCNN" in source
    config = project.read_json("config.json")
    assert config["model_type"] == "small_cnn"
    flat = schema_for(load_schema("image_torch"), "small_cnn")
    assert validate_config(config, flat) == []
    assert stage.is_complete(ctx) is True
    assert any("CNN" in text for text in shown)


def test_the_image_model_choices_material_is_shown(clean_image_project):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.stages.codegen import CodegenStage
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    project = clean_image_project
    shown: list[str] = []
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner({"codegen.model_type": "Tiny CNN"},
                                  ScriptedQuestioner(["y"])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    CodegenStage().prepare(ctx)
    joined = "\n".join(shown)
    assert "ResNet" in joined and "transfer learning" in joined.lower()
