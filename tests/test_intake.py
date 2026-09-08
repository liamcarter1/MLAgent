import pytest

from mlagent.llm import FakeLLM
from mlagent.spec import Spec
from mlagent.stages.base import StageContext
from mlagent.stages.intake import IntakeStage, collect_draft
from mlagent.ui.questions import ScriptedQuestioner

ANSWERS = [
    "Predict customer churn from account data",  # goal
    "Intermediate - explain the key ideas",       # learning level
    "Tabular classification",                     # task type label
    "accuracy",                                   # metric
    "0.9",                                        # target
    "Synthetic data",                             # data source label
    "10",                                         # minutes per run
    "5",                                          # max rounds
    "No GPU (CPU only)",                          # gpu label
]


def make_ctx(project, llm, answers):
    shown: list[str] = []
    ctx = StageContext(project=project, llm=llm, questioner=ScriptedQuestioner(answers),
                       explainer=None, display=shown.append)
    return ctx, shown


def test_collect_draft_maps_labels_to_codes():
    draft = collect_draft(ScriptedQuestioner(ANSWERS))
    assert draft["task_type"] == "tabular_classification"
    assert draft["data_source"] == "synthetic"
    assert draft["gpu"] == "none"
    assert draft["target_value"] == 0.9
    assert draft["minutes_per_run"] == 10 and draft["max_rounds"] == 5
    assert draft["learning_level"] == "intermediate"


def test_intake_writes_spec_via_tool(project):
    spec_fields = {
        "goal": "Predict customer churn from account data", "task_type": "tabular_classification",
        "metric": "accuracy", "target_value": 0.9, "data_source": "synthetic",
        "minutes_per_run": 10, "max_rounds": 5, "gpu": "none", "notes": "binary target",
    }
    llm = FakeLLM(script=[
        [("tool", "ask_user", {"question": "Is churn binary?", "options": ["yes", "no"]})],
        [("tool", "write_spec", spec_fields)],
        [("text", "Spec saved. Next we get [[training data]].")],
    ])
    ctx, shown = make_ctx(project, llm, ANSWERS + ["yes"])
    stage = IntakeStage()
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    saved = Spec.from_dict(project.read_json("spec.json"))
    assert saved.notes == "binary target"
    assert "Is churn binary?" in ctx.questioner.asked
    assert any("[[training data]]" in s for s in shown)
    assert "intake" in llm.calls[0]["system"].lower()
    assert "churn" in llm.calls[0]["messages"][0]["content"]


def test_intake_falls_back_to_draft_when_model_writes_nothing(project):
    llm = FakeLLM(script=[[("text", "Looks good.")]])
    ctx, _ = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    saved = Spec.from_dict(project.read_json("spec.json"))
    assert saved.goal.startswith("Predict customer churn")


def test_write_spec_tool_rejects_invalid_and_model_can_retry(project):
    bad = {"goal": "x", "task_type": "tabular_regression", "metric": "accuracy", "target_value": 1,
           "data_source": "synthetic", "minutes_per_run": 5, "max_rounds": 2, "gpu": "none"}
    good = {**bad, "metric": "rmse"}
    llm = FakeLLM(
        script=[[("tool", "write_spec", bad)], [("tool", "write_spec", good)], [("text", "ok")]]
    )
    ctx, _ = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    assert Spec.from_dict(project.read_json("spec.json")).metric == "rmse"


def test_is_complete_false_for_corrupt_spec(project):
    project.write_json("spec.json", {"goal": "only"})
    ctx, _ = make_ctx(project, FakeLLM([]), [])
    assert IntakeStage().is_complete(ctx) is False


def test_intake_falls_back_to_draft_when_llm_errors(project):
    llm = FakeLLM(script=[])  # empty script -> LLMError on first .run()
    ctx, shown = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    saved = Spec.from_dict(project.read_json("spec.json"))
    assert saved.goal.startswith("Predict customer churn")
    assert any("Couldn't reach Claude" in s for s in shown)


def test_intake_persists_draft_before_calling_llm(project):
    llm = FakeLLM(script=[])
    ctx, _ = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    draft = project.read_json("draft_spec.json")
    assert draft is not None
    assert draft["goal"].startswith("Predict customer churn")


def test_is_complete_false_for_truncated_spec_json(project):
    (project.root / "spec.json").write_text('{"goal": "cats", "task_type": ', encoding="utf-8")
    ctx, _ = make_ctx(project, FakeLLM([]), [])
    with pytest.warns(UserWarning, match="spec.json"):
        assert IntakeStage().is_complete(ctx) is False


def test_intake_asks_the_learning_level_and_stores_it(project):
    from mlagent.stages.intake import LEVEL_LABELS

    assert LEVEL_LABELS["Beginner - explain everything as we go"] == "beginner"
    ctx, _shown = make_ctx(
        project,
        FakeLLM([]),
        [
            "Predict churn",
            "Beginner - explain everything as we go",
            "Tabular classification",
            "accuracy",
            "0.9",
            "Synthetic data",
            "10",
            "5",
            "No GPU (CPU only)",
        ],
    )
    IntakeStage().run(ctx)
    assert project.read_json("spec.json")["learning_level"] == "beginner"
    assert project.read_json("draft_spec.json")["learning_level"] == "beginner"


def test_write_spec_tool_keeps_users_learning_level_when_model_omits_it(project):
    spec_fields_no_level = {
        "goal": "Predict customer churn from account data", "task_type": "tabular_classification",
        "metric": "accuracy", "target_value": 0.9, "data_source": "synthetic",
        "minutes_per_run": 10, "max_rounds": 5, "gpu": "none",
    }
    llm = FakeLLM(script=[
        [("tool", "write_spec", spec_fields_no_level)],
        [("text", "ok")],
    ])
    answers = list(ANSWERS)
    answers[1] = "Beginner - explain everything as we go"
    ctx, _ = make_ctx(project, llm, answers)
    IntakeStage().run(ctx)
    assert project.read_json("spec.json")["learning_level"] == "beginner"


def test_write_spec_tool_keeps_models_own_learning_level_when_given(project):
    spec_fields_with_level = {
        "goal": "Predict customer churn from account data", "task_type": "tabular_classification",
        "metric": "accuracy", "target_value": 0.9, "data_source": "synthetic",
        "minutes_per_run": 10, "max_rounds": 5, "gpu": "none", "learning_level": "expert",
    }
    llm = FakeLLM(script=[
        [("tool", "write_spec", spec_fields_with_level)],
        [("text", "ok")],
    ])
    answers = list(ANSWERS)
    answers[1] = "Beginner - explain everything as we go"
    ctx, _ = make_ctx(project, llm, answers)
    IntakeStage().run(ctx)
    assert project.read_json("spec.json")["learning_level"] == "expert"
