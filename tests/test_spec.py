import pytest

from mlagent.spec import METRICS_FOR_TASK, Spec, SpecError


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
