import pytest

from mlagent.spec import LEARNING_LEVELS, METRICS_FOR_TASK, Spec, SpecError


def make_spec(**overrides) -> Spec:
    base = {
        "goal": "Predict whether a customer churns",
        "task_type": "tabular_classification",
        "metric": "accuracy",
        "target_value": 0.9,
        "data_source": "synthetic",
        "minutes_per_run": 10,
        "max_rounds": 5,
        "gpu": "none",
    }
    base.update(overrides)
    return Spec(**base)


def test_valid_spec_roundtrips():
    s = make_spec(notes="keep it simple")
    assert s.validate() == []
    d = s.to_dict()
    assert d["task_type"] == "tabular_classification"
    assert Spec.from_dict(d) == s


def test_invalid_values_are_reported():
    s = make_spec(task_type="video", metric="bleu", data_source="ftp", gpu="H100",
                  minutes_per_run=0, max_rounds=0, goal="")
    problems = s.validate()
    assert len(problems) == 7
    assert any("task_type" in p for p in problems)


def test_metric_must_match_task():
    s = make_spec(task_type="tabular_regression", metric="accuracy")
    assert any("metric" in p for p in s.validate())
    assert "rmse" in METRICS_FOR_TASK["tabular_regression"]


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(SpecError):
        Spec.from_dict({**make_spec().to_dict(), "bogus": 1})


def test_learning_level_defaults_and_validates():
    s = make_spec()
    assert s.learning_level == "intermediate"
    assert s.validate() == []
    assert LEARNING_LEVELS == ("beginner", "intermediate", "expert")
    bad = make_spec(learning_level="guru")
    assert any("learning_level" in p for p in bad.validate())


def test_learning_level_round_trips_and_is_optional_in_from_dict():
    s = make_spec(learning_level="beginner")
    assert Spec.from_dict(s.to_dict()).learning_level == "beginner"
    legacy = {k: v for k, v in make_spec().to_dict().items() if k != "learning_level"}
    assert Spec.from_dict(legacy).learning_level == "intermediate"
