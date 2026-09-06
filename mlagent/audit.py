"""Cleanliness audit for tabular data.

Each check returns Issues with evidence and a proposed fix.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import pandas as pd
from pandas.api import types as ptypes

from mlagent.profile import is_categorical_series

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ID_NAME = re.compile(r"(^|_)(id|uuid|key)$|^id", re.IGNORECASE)
QUANTITY_NAME = re.compile(r"(age|count|qty|quantity|price|amount|duration)", re.IGNORECASE)


@dataclass
class Issue:
    kind: str
    severity: str
    message: str
    column: str | None = None
    evidence: dict = field(default_factory=dict)
    fix: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _drop(column: str) -> dict:
    return {"op": "drop_columns", "params": {"columns": [column]}}


def _to_str_key(val) -> str:
    """Convert categorical values to strings, handling float whole numbers."""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _features(df: pd.DataFrame, target: str | None) -> list[str]:
    return [str(c) for c in df.columns if c != target]


def _check_target(df: pd.DataFrame, target: str | None) -> list[Issue]:
    if target is None or target not in df.columns:
        return []
    s = df[target]
    out: list[Issue] = []
    n_missing = int(s.isna().sum())
    if n_missing:
        out.append(Issue(
            "target_missing", "high",
            f"{n_missing} rows have no value for the target column '{target}';"
            " they cannot be used for training.",
            target, {"rows": n_missing},
            {"op": "drop_rows_missing_target", "params": {"target": target}},
        ))
    if is_categorical_series(s):
        counts = s.value_counts(dropna=True)
        n = int(counts.sum())
        if n:
            majority = float(counts.iloc[0] / n)
            if majority > 0.9:
                out.append(Issue(
                    "class_imbalance", "medium",
                    f"The majority class makes up {majority:.0%} of rows; accuracy will"
                    " look good even for a useless model.",
                    target, {"majority_fraction": round(majority, 4),
                             "counts": {_to_str_key(k): int(v)
                                       for k, v in counts.head(10).items()}},
                ))
            rare = {_to_str_key(k): int(v) for k, v in counts.items()
                    if v / n < 0.01}
            if rare:
                out.append(Issue(
                    "rare_classes", "medium",
                    f"{len(rare)} class(es) have fewer than 1% of rows; the model"
                    " will struggle to learn them.",
                    target, {"rare": rare},
                ))
    return out


def _check_missing(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        n = int(s.isna().sum())
        if not n:
            continue
        pct = float(s.isna().mean() * 100)
        evidence = {"missing": n, "missing_pct": round(pct, 2)}
        if pct > 50:
            out.append(Issue(
                "missing_values", "high",
                f"'{col}' is missing in {pct:.1f}% of rows; too sparse to be useful.",
                col, evidence, _drop(col),
            ))
        else:
            strategy = "median" if ptypes.is_numeric_dtype(s) else "mode"
            out.append(Issue(
                "missing_values", "medium",
                f"'{col}' is missing in {pct:.1f}% of rows ({n}); most models"
                " cannot handle gaps.",
                col, evidence,
                {"op": "fill_missing", "params": {"column": col, "strategy": strategy}},
            ))
    return out


def _check_duplicates(df: pd.DataFrame, target: str | None) -> list[Issue]:
    n = int(df.duplicated().sum())
    if not n:
        return []
    return [Issue(
        "duplicate_rows", "medium",
        f"{n} rows are exact duplicates; they over-weight those examples and"
        " can leak between train and test splits.",
        None, {"rows": n}, {"op": "drop_duplicates", "params": {}},
    )]


def _check_constant(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col].dropna()
        if s.empty:
            continue
        nunique = int(s.nunique())
        if nunique <= 1:
            out.append(Issue(
                "constant_column", "medium",
                f"'{col}' has a single value; it carries no information.",
                col, {"n_unique": nunique}, _drop(col),
            ))
        elif len(s) >= 100:
            top = float(s.value_counts(normalize=True).iloc[0])
            if top > 0.99:
                out.append(Issue(
                    "near_constant_column", "low",
                    f"'{col}' is the same value in {top:.1%} of rows.",
                    col, {"top_fraction": round(top, 4)}, _drop(col),
                ))
    return out


def _check_mixed_types(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        if not ptypes.is_object_dtype(s):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        numeric_fraction = float(pd.to_numeric(vals, errors="coerce").notna()
                                 .mean())
        if 0.5 <= numeric_fraction < 1.0:
            out.append(Issue(
                "mixed_types", "medium",
                f"'{col}' is mostly numbers stored as text ({numeric_fraction:.0%})"
                " with some non-numeric entries.",
                col, {"numeric_fraction": round(numeric_fraction, 4)},
                {"op": "coerce_numeric", "params": {"column": col}},
            ))
    return out


def _check_categorical_consistency(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        if not ptypes.is_object_dtype(s):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        raw = int(vals.nunique())
        norm = int(vals.str.strip().str.lower().nunique())
        if norm < raw:
            out.append(Issue(
                "inconsistent_categories", "medium",
                f"'{col}' has {raw} spellings for {norm} real categories"
                " (case or whitespace differences).",
                col, {"raw_unique": raw, "normalised_unique": norm},
                {"op": "normalise_categories", "params": {"column": col}},
            ))
    return out


def _check_outliers(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        if not ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s):
            continue
        vals = s.dropna()
        if len(vals) < 20 or vals.nunique() < 10:
            continue
        q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n = int(((vals < lower) | (vals > upper)).sum())
        if n and n / len(vals) > 0.005:
            out.append(Issue(
                "outliers", "low",
                f"'{col}' has {n} extreme values far outside the typical range.",
                col, {"count": n, "lower": lower, "upper": upper},
                {"op": "clip_outliers",
                 "params": {"column": col, "lower": lower, "upper": upper}},
            ))
    return out


def _check_leakage(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)
    t = df[target] if target is not None and target in df.columns else None
    for col in _features(df, target):
        s = df[col]
        if n >= 20 and (ptypes.is_integer_dtype(s) or ptypes.is_object_dtype(s)):
            ratio = s.nunique(dropna=True) / n
            if ratio == 1.0 or (ratio >= 0.95 and ID_NAME.search(col)):
                out.append(Issue(
                    "id_column", "high",
                    f"'{col}' is unique for (almost) every row; it is an"
                    " identifier, not a feature, and a model could memorise it.",
                    col, {"unique_ratio": round(float(ratio), 4)}, _drop(col),
                ))
                continue
        if t is None:
            continue
        if (ptypes.is_numeric_dtype(s) and ptypes.is_numeric_dtype(t)
                and not is_categorical_series(t)):
            corr = s.corr(t)
            if pd.notna(corr) and abs(float(corr)) > 0.98:
                out.append(Issue(
                    "target_leakage", "high",
                    f"'{col}' is almost perfectly correlated with the target"
                    f" (r={corr:.3f}); it probably encodes the answer.",
                    col, {"correlation": round(float(corr), 4)}, _drop(col),
                ))
        elif s.astype(str).equals(t.astype(str)):
            out.append(Issue(
                "target_leakage", "high",
                f"'{col}' is identical to the target column.",
                col, {}, _drop(col),
            ))
    return out


def _check_suspicious_ranges(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        if (not ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s)
                or not QUANTITY_NAME.search(col)):
            continue
        n = int((s.dropna() < 0).sum())
        if n:
            out.append(Issue(
                "suspicious_values", "low",
                f"'{col}' has {n} negative values, which is unusual for a"
                " quantity-like column; check the source.",
                col, {"negative": n},
            ))
    return out


CHECKS: tuple[Callable[[pd.DataFrame, str | None], list[Issue]], ...] = (
    _check_target,
    _check_missing,
    _check_duplicates,
    _check_constant,
    _check_mixed_types,
    _check_categorical_consistency,
    _check_outliers,
    _check_leakage,
    _check_suspicious_ranges,
)


def audit_tabular(df: pd.DataFrame, target: str | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(df, target))
    issues.sort(key=lambda i: SEVERITY_ORDER[i.severity])
    return issues
