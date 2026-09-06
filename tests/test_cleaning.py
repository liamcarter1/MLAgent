import numpy as np
import pandas as pd
import pytest

from mlagent.cleaning import apply_steps, describe_step, render_clean_py


def df():
    return pd.DataFrame({
        "id": [1, 2, 3, 3],
        "x": [1.0, np.nan, 100.0, 100.0],
        "cat": ["A ", "a", "b", "b"],
        "num_text": ["1", "2", "n/a", "n/a"],
        "target": [0, 1, np.nan, np.nan],
    })


def test_each_op():
    step1 = {"op": "drop_columns", "params": {"columns": ["id", "missing_col"]}}
    out = apply_steps(df(), [step1])
    assert list(out.columns) == ["x", "cat", "num_text", "target"]
    out = apply_steps(df(), [{"op": "drop_duplicates", "params": {}}])
    assert len(out) == 3 and list(out.index) == [0, 1, 2]
    step2 = {"op": "fill_missing",
             "params": {"column": "x", "strategy": "median"}}
    out = apply_steps(df(), [step2])
    assert out["x"].isna().sum() == 0 and out.loc[1, "x"] == 100.0
    step3 = {"op": "fill_missing",
             "params": {"column": "cat", "strategy": "mode"}}
    out = apply_steps(df(), [step3])
    assert out["cat"].isna().sum() == 0
    step4 = {"op": "drop_rows_missing_target",
             "params": {"target": "target"}}
    out = apply_steps(df(), [step4])
    assert len(out) == 2
    out = apply_steps(df(), [{"op": "normalise_categories",
                              "params": {"column": "cat"}}])
    assert out["cat"].tolist() == ["a", "a", "b", "b"]
    step5 = {"op": "clip_outliers",
             "params": {"column": "x", "lower": 0.0, "upper": 10.0}}
    out = apply_steps(df(), [step5])
    assert out["x"].max() == 10.0
    out = apply_steps(df(), [{"op": "coerce_numeric",
                              "params": {"column": "num_text"}}])
    assert (out["num_text"].dtype.kind == "f"
            and out["num_text"].isna().sum() == 2)


def test_apply_steps_is_pure_and_ordered():
    original = df()
    steps = [
        {"op": "drop_duplicates", "params": {}},
        {"op": "drop_rows_missing_target", "params": {"target": "target"}},
    ]
    out = apply_steps(original, steps)
    assert len(out) == 2 and len(original) == 4


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        apply_steps(df(), [{"op": "teleport", "params": {}}])


def test_describe_step():
    step1 = {"op": "drop_columns", "params": {"columns": ["id"]}}
    assert "id" in describe_step(step1)
    step2 = {"op": "fill_missing",
             "params": {"column": "x", "strategy": "median"}}
    assert "median" in describe_step(step2)
    assert describe_step({"op": "drop_duplicates", "params": {}})


def test_rendered_clean_py_reproduces_apply_steps():
    steps = [
        {"op": "drop_duplicates", "params": {}},
        {"op": "normalise_categories", "params": {"column": "cat"}},
    ]
    source = render_clean_py(steps)
    namespace: dict = {}
    exec(compile(source, "clean.py", "exec"), namespace)
    pd.testing.assert_frame_equal(namespace["clean"](df()), apply_steps(df(), steps))
    assert namespace["STEPS"] == steps
