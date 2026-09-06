"""Dataset profiling: a JSON-serialisable summary plus a markdown rendering."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

CATEGORICAL_MAX_UNIQUE = 50
MAX_LISTED_CATEGORIES = 20


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python; NaN becomes None."""
    if value is None or value is pd.NA:
        return None
    if pd.isna(value) and isinstance(value, float):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def is_categorical_series(s: pd.Series) -> bool:
    if ptypes.is_bool_dtype(s) or not ptypes.is_numeric_dtype(s):
        return True
    vals = s.dropna()
    if vals.empty or vals.nunique() > CATEGORICAL_MAX_UNIQUE:
        return False
    return bool((vals % 1 == 0).all())


def _column_info(name: str, s: pd.Series) -> dict:
    info: dict[str, Any] = {
        "name": name,
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "missing_pct": round(float(s.isna().mean() * 100), 2),
        "n_unique": int(s.nunique(dropna=True)),
        "sample": [str(v) for v in s.dropna().unique()[:5]],
    }
    if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
        vals = s.dropna()
        if not vals.empty:
            info.update({
                "min": _py(vals.min()),
                "max": _py(vals.max()),
                "mean": _py(round(float(vals.mean()), 6)),
                "std": _py(round(float(vals.std()), 6)) if len(vals) > 1 else 0.0,
            })
    return info


def _target_info(name: str, s: pd.Series) -> dict:
    if is_categorical_series(s):
        all_counts = s.value_counts(dropna=True)
        counts = all_counts.head(MAX_LISTED_CATEGORIES)
        return {
            "name": name,
            "kind": "categorical",
            "n_classes": int(all_counts.shape[0]),
            "total": int(all_counts.sum()),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
    vals = s.dropna()
    if not vals.empty:
        return {
            "name": name,
            "kind": "numeric",
            "min": _py(vals.min()),
            "max": _py(vals.max()),
            "mean": _py(round(float(vals.mean()), 6)),
            "std": _py(round(float(vals.std()), 6)) if len(vals) > 1 else 0.0,
        }
    return {
        "name": name,
        "kind": "numeric",
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3),
        "columns": [_column_info(str(c), df[c]) for c in df.columns],
        "target": None,
    }
    if target is not None and target in df.columns:
        profile["target"] = _target_info(target, df[target])
    return profile


def profile_markdown(profile: dict) -> str:
    def _escape_pipe(s: str) -> str:
        """Escape pipes in markdown table cells."""
        return s.replace("|", "\\|")

    lines = [
        f"### Data profile: {profile['n_rows']} rows × {profile['n_cols']} columns "
        f"({profile['duplicate_rows']} duplicate rows, {profile['memory_mb']} MB)",
        "",
        "| column | type | missing | unique | sample |",
        "|---|---|---|---|---|",
    ]
    for c in profile["columns"]:
        sample = ", ".join(c["sample"])[:60]
        escaped_name = _escape_pipe(c["name"])
        escaped_sample = _escape_pipe(sample)
        row = (
            f"| {escaped_name} | {c['dtype']} | {c['missing_pct']}% | "
            f"{c['n_unique']} | {escaped_sample} |"
        )
        lines.append(row)
    t = profile.get("target")
    if t:
        lines.append("")
        if t["kind"] == "categorical":
            counts = ", ".join(f"{k}: {v}" for k, v in t["counts"].items())
            lines.append(f"Target `{t['name']}` is categorical: {counts}")
        else:
            lines.append(
                f"Target `{t['name']}` is numeric: min {t['min']}, max {t['max']}, "
                f"mean {t['mean']}"
            )
    return "\n".join(lines)
