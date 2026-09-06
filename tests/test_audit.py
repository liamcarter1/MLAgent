import json
import warnings

import numpy as np
import pandas as pd

from mlagent.audit import SEVERITY_ORDER, audit_tabular


def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "const": 1,
        "cat": rng.choice(["a", "b", "c"], size=n),
        "mixed": [str(i) for i in range(n)],
        "age": rng.integers(0, 80, size=n),
        "target": rng.integers(0, 2, size=n),
    })
    df.loc[:9, "f1"] = np.nan
    df.loc[:3, "cat"] = ["A ", " b", "C", "a "]
    df.loc[:4, "mixed"] = "n/a"
    df.loc[:1, "age"] = -5
    df.loc[:1, "f2"] = 1000.0
    df["leak"] = df["target"]
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


def kinds(issues):
    return {i.kind for i in issues}


def test_every_planted_issue_is_found_and_sorted():
    issues = audit_tabular(messy_df(), target="target")
    assert kinds(issues) >= {
        "id_column", "missing_values", "duplicate_rows", "constant_column", "mixed_types",
        "inconsistent_categories", "outliers", "target_leakage", "suspicious_values",
    }
    ranks = [SEVERITY_ORDER[i.severity] for i in issues]
    assert ranks == sorted(ranks)
    json.dumps([i.to_dict() for i in issues])


def test_fixes_reference_correct_columns():
    by_kind = {i.kind: i for i in audit_tabular(messy_df(), target="target")}
    expected_id = {"op": "drop_columns", "params": {"columns": ["row_id"]}}
    assert by_kind["id_column"].fix == expected_id
    expected_missing = {"op": "fill_missing",
                        "params": {"column": "f1", "strategy": "median"}}
    assert by_kind["missing_values"].fix == expected_missing
    assert by_kind["duplicate_rows"].fix == {"op": "drop_duplicates", "params": {}}
    assert by_kind["inconsistent_categories"].fix["params"] == {"column": "cat"}
    assert (by_kind["outliers"].fix["op"] == "clip_outliers"
            and by_kind["outliers"].column == "f2")
    expected_mixed = {"op": "coerce_numeric", "params": {"column": "mixed"}}
    assert by_kind["mixed_types"].fix == expected_mixed
    assert by_kind["target_leakage"].column == "leak"
    assert by_kind["suspicious_values"].fix is None


def test_target_checks():
    n = 300
    df = pd.DataFrame({
        "x": np.arange(n, dtype=float),
        "target": [0] * 290 + [1] * 8 + [2] * 2,
    })
    df.loc[0, "target"] = np.nan
    by_kind = {i.kind: i for i in audit_tabular(df, target="target")}
    expected_fix = {"op": "drop_rows_missing_target",
                    "params": {"target": "target"}}
    assert by_kind["target_missing"].fix == expected_fix
    assert by_kind["class_imbalance"].severity == "medium"
    assert "2" in by_kind["rare_classes"].evidence["rare"]


def test_regression_leakage_by_correlation():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    df = pd.DataFrame({
        "good": rng.normal(size=100),
        "leaky": y * 3 + 0.001 * rng.normal(size=100),
        "target": y,
    })
    issues = audit_tabular(df, target="target")
    assert [i.column for i in issues if i.kind == "target_leakage"] == [
        "leaky"
    ]


def test_sparse_column_is_dropped_not_filled():
    df = pd.DataFrame({
        "sparse": [np.nan] * 90 + [1.0] * 10,
        "ok": range(100),
        "target": [0, 1] * 50,
    })
    issue = next(i for i in audit_tabular(df, "target") if i.column == "sparse")
    assert issue.severity == "high" and issue.fix["op"] == "drop_columns"


def test_clean_frame_has_no_issues():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": rng.normal(size=100),
        "target": rng.integers(0, 2, 100),
    })
    assert audit_tabular(df, target="target") == []


def test_numeric_as_text_detected():
    """Entirely numeric text column should flag numeric_as_text issue."""
    df = pd.DataFrame({
        "num_text": ["1", "2", "3", "4", "5"],
        "target": [0, 1, 0, 1, 0],
    })
    by_kind = {i.kind: i for i in audit_tabular(df, target="target")}
    assert "numeric_as_text" in by_kind
    assert by_kind["numeric_as_text"].severity == "low"
    assert by_kind["numeric_as_text"].fix == {
        "op": "coerce_numeric",
        "params": {"column": "num_text"},
    }


def test_free_text_comment_column_is_not_flagged_as_id():
    n = 60
    df = pd.DataFrame({
        "comment": [f"this is a unique sentence number {i} about the order" for i in range(n)],
        "x": np.arange(n, dtype=float),
        "target": [0, 1] * (n // 2),
    })
    kinds_found = kinds(audit_tabular(df, target="target"))
    assert "id_column" not in kinds_found


def test_unique_datetime_string_column_is_not_flagged_as_id():
    n = 60
    df = pd.DataFrame({
        "when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "x": np.arange(n, dtype=float),
        "target": [0, 1] * (n // 2),
    })
    kinds_found = kinds(audit_tabular(df, target="target"))
    assert "id_column" not in kinds_found


def test_row_id_and_user_id_tokens_still_flagged():
    n = 60
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "user_id": [f"u{i}xyz" for i in range(n)],
        "x": np.arange(n, dtype=float),
        "target": [0, 1] * (n // 2),
    })
    by_kind_cols = [
        i.column for i in audit_tabular(df, target="target") if i.kind == "id_column"
    ]
    assert set(by_kind_cols) == {"row_id", "user_id"}


def test_mixed_case_unique_user_id_flags_id_and_inconsistent_categories():
    """user_id is fully unique on raw values (so id_column fires) but has case
    collisions after normalisation (so inconsistent_categories also fires)."""
    n = 40
    values = [f"User{i}" for i in range(n)]
    # introduce case collisions with a couple of other rows' lower-cased forms,
    # without breaking uniqueness of the raw strings themselves.
    values[0] = "user1"  # collides with "User1" once lower-cased
    values[2] = "user3"  # collides with "User3" once lower-cased
    df = pd.DataFrame({
        "user_id": values,
        "x": np.arange(n, dtype=float),
        "target": [0, 1] * (n // 2),
    })
    by_kind = {i.kind: i for i in audit_tabular(df, target="target") if i.column == "user_id"}
    assert "id_column" in by_kind
    assert "inconsistent_categories" in by_kind


def test_high_cardinality_free_text_not_flagged_inconsistent_categories():
    n = 200
    notes = [f"note about order {i} shipped on time" for i in range(n)]
    notes[1] = notes[1].upper()  # one case collision
    df = pd.DataFrame({
        "notes": notes,
        "x": np.arange(n, dtype=float),
        "target": [0, 1] * (n // 2),
    })
    kinds_found = kinds(audit_tabular(df, target="target"))
    assert "inconsistent_categories" not in kinds_found


def test_low_cardinality_messy_categories_still_flagged():
    kinds_found = kinds(audit_tabular(messy_df(), target="target"))
    assert "inconsistent_categories" in kinds_found


def test_constant_numeric_column_no_correlation_warning():
    """Constant numeric column should not raise RuntimeWarning on correlation."""
    df = pd.DataFrame({
        "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        issues = audit_tabular(df, target="target")
    # Should not have raised, and no leakage issue for constant column
    assert not any(i.kind == "target_leakage" for i in issues)
