"""Contract: every audit-proposed fix must be applicable by cleaning.apply_steps without
raising, whatever the shape of the data - individually and all together in audit order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlagent.audit import audit_tabular
from mlagent.cleaning import apply_steps


def messy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "const": 1,
        "cat": rng.choice(["a", "b", "c"], size=n),
        "target": rng.integers(0, 2, size=n),
    })
    df.loc[:9, "f1"] = np.nan
    df.loc[:3, "cat"] = ["A ", " b", "C", "a "]
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


def all_nan_column_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame({
        "empty": [np.nan] * n,
        "x": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })


def single_class_target_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 50
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "target": [1] * n,
    })


def unique_free_text_column_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 60
    notes = [f"this is note number {i} about something unique" for i in range(n)]
    return pd.DataFrame({
        "notes": notes,
        "x": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })


def datetime_string_column_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 80
    return pd.DataFrame({
        "when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "x": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })


def wide_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 50
    data = {f"c{i}": rng.normal(size=n) for i in range(300)}
    data["target"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


FRAMES = {
    "messy": messy_frame,
    "all_nan_column": all_nan_column_frame,
    "single_class_target": single_class_target_frame,
    "unique_free_text_column": unique_free_text_column_frame,
    "datetime_string_column": datetime_string_column_frame,
    "wide_300_columns": wide_frame,
}


@pytest.mark.parametrize("name", list(FRAMES))
def test_every_fix_applies_individually_and_all_together(name):
    df = FRAMES[name]()
    issues = audit_tabular(df, target="target")
    fixes = [i.fix for i in issues if i.fix is not None]

    for fix in fixes:
        apply_steps(df, [fix])  # must not raise

    apply_steps(df, fixes)  # all together, in audit order, must not raise
