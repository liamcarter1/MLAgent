import json

import numpy as np
import pandas as pd

from mlagent.profile import is_categorical_series, profile_dataframe, profile_markdown


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": ["x", "y", "x", None],
        "flag": [True, False, True, True],
        "target": [0, 1, 0, 1],
    })


def test_profile_columns_and_target_counts():
    p = profile_dataframe(sample_df(), target="target")
    assert p["n_rows"] == 4 and p["n_cols"] == 4 and p["duplicate_rows"] == 0
    a = next(c for c in p["columns"] if c["name"] == "a")
    assert a["missing"] == 1 and a["missing_pct"] == 25.0 and a["n_unique"] == 3
    assert a["min"] == 1.0 and a["max"] == 4.0
    b = next(c for c in p["columns"] if c["name"] == "b")
    assert "min" not in b and b["sample"] == ["x", "y"]
    assert p["target"] == {"name": "target", "kind": "categorical", "counts": {"0": 2, "1": 2}}
    json.dumps(p)  # must be serialisable


def test_numeric_target_and_missing_target():
    df = pd.DataFrame({"x": range(50), "y": np.linspace(0, 1, 50)})
    p = profile_dataframe(df, target="y")
    assert p["target"]["kind"] == "numeric" and p["target"]["max"] == 1.0
    assert profile_dataframe(df, target="nope")["target"] is None


def test_is_categorical_series():
    assert is_categorical_series(pd.Series(["a", "b"]))
    assert is_categorical_series(pd.Series([0, 1, 1, 0]))
    assert is_categorical_series(pd.Series([0.0, 1.0, np.nan]))
    assert not is_categorical_series(pd.Series(np.linspace(0, 1, 30)))
    assert not is_categorical_series(pd.Series(range(100)))


def test_profile_markdown_mentions_columns():
    md = profile_markdown(profile_dataframe(sample_df(), target="target"))
    assert "| a |" in md and "target" in md and "4 rows" in md
