# ML Training Agent — Milestone 2 (Tabular Data + Cleaning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After intake, the assistant obtains tabular data (synthetic with realistic quirks, a file on Drive, or a HuggingFace dataset), profiles and plots it, audits it for cleanliness, explains each issue, applies the fixes the user approves as a re-runnable `clean.py`, and records target column and split fractions for Milestone 3.

**Architecture:** Two new stages (`data`, `clean`) plug into the existing `Orchestrator`/`StageContext`. Pure, testable modules do the work: `synth/tabular.py` (data generation), `profile.py` (profiling), `audit.py` (checks → `Issue` list), `cleaning.py` (ops → new DataFrame, plus `clean.py` rendering), `plots.py` (matplotlib figures), `datasources/drive.py` and `datasources/hf.py` (acquisition). Stages only orchestrate: ask via `Questioner`, narrate via one `ask_text` call per stage, persist artifacts on the project folder. No LLM tool loops are needed in this milestone.

**Tech Stack:** pandas, numpy, scikit-learn (data generation), matplotlib (Agg in tests), `huggingface_hub` + `datasets` (lazily imported, injectable, never hit in tests), pyarrow/openpyxl for parquet/excel.

**Spec:** `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md` (sections "Stage pipeline" items 2–3, "Plots", "Testing").

## Global Constraints

- Python `>=3.10`; all tests run with `python -m pytest`; `ruff check .` must pass with the rule set enabled in Task 1 (`E, F, W, I, B, UP`, line length 100).
- No network in tests. LLM calls use `FakeLLM`; HuggingFace search/load are injected fakes; Drive is a temp directory.
- Every prompt lives in `mlagent/prompts/*.md`, loaded with `load_prompt`; never inline prompts in Python.
- User-visible text may contain `[[term]]` markup for click-to-explain.
- Raw data is never modified: raw lands in `data/raw/data.csv`, cleaned in `data/clean/data.csv`.
- Cleaning steps are data (`{"op": ..., "params": {...}}`) applied by `mlagent.cleaning.apply_steps`; the generated `clean.py` just stores the steps and calls the same function, so it is re-runnable.
- Charts: static matplotlib; palette constants in `plots.py` (series `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`, sequential blue ramp, diverging blue–neutral–orange); single-series charts have no legend, multi-series charts always do; text uses ink colours, never series colours; figures are saved to `projects/<name>/plots/<name>.png` and displayed only when running under IPython.
- Windows dev machine: `pathlib`, `encoding="utf-8"`. Project conventions: `from __future__ import annotations`, `from collections.abc import Callable`.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2
  ```

---

## File Structure (Milestone 2)

```
pyproject.toml                    (modify: deps, ruff rules)
mlagent/
  ui/questions.py                 (modify: number())
  stages/base.py                  (modify: spec())
  stages/intake.py                (modify: use number())
  synth/__init__.py  synth/tabular.py
  profile.py
  audit.py
  cleaning.py
  plots.py
  datasources/__init__.py  datasources/drive.py  datasources/hf.py
  stages/data.py  stages/clean.py
  prompts/data.md  prompts/clean.md
  colab.py                        (modify: stages list, context snapshot)
scripts/build_notebook.py         (modify: intro note)
notebooks/ML_Training_Agent.ipynb (regenerate)
docs/colab-smoke.md               (modify: M2 section)
CLAUDE.md                         (modify: architecture)
tests/
  conftest.py                     (modify: Agg backend)
  test_questions.py test_orchestrator.py test_intake.py (modify)
  test_synth_tabular.py test_profile.py test_audit.py test_cleaning.py test_plots.py
  test_datasources.py test_data_stage.py test_clean_stage.py test_colab.py (modify)
```

---

### Task 1: Dependencies, lint rules, `Questioner.number`, `StageContext.spec`, notebook note

**Files:**
- Modify: `pyproject.toml`, `mlagent/ui/questions.py`, `mlagent/stages/base.py`, `mlagent/stages/intake.py`, `scripts/build_notebook.py`, `tests/conftest.py`, `tests/test_questions.py`, `tests/test_orchestrator.py`
- Regenerate: `notebooks/ML_Training_Agent.ipynb`

**Interfaces:**
- Produces: `Questioner.number(prompt: str, default: float | None = None, minimum: float | None = None, maximum: float | None = None) -> float` on the protocol, `ConsoleQuestioner`, and `ScriptedQuestioner`; `StageContext.spec() -> Spec` (raises `SpecError` if `spec.json` is missing).
- Removes: `_to_float`, `_to_int` from `mlagent/stages/intake.py`.

- [ ] **Step 1: Update `pyproject.toml`**

Replace the `dependencies` list and add a lint section:
```toml
dependencies = [
    "anthropic>=1.0",
    "markdown>=3.5",
    "ipython>=8.0",
    "pandas>=2.0",
    "numpy>=1.26",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "pyarrow>=14",
    "openpyxl>=3.1",
    "huggingface_hub>=0.23",
    "datasets>=2.19",
]
```
```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]
```
Run `python -m pip install -e ".[dev]"`. Then `ruff check .` and fix every new violation in existing files minimally (import sorting, `UP` modernisations). Do not change behaviour.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_questions.py`:
```python
def test_console_number_validates_and_reprompts():
    q, printed = make_console(["abc", "50", "7"])
    assert q.number("Rows?", default=10, minimum=1, maximum=20) == 7.0
    assert sum("Please enter a number" in line for line in printed) == 2
    q, _ = make_console([""])
    assert q.number("Rows?", default=10) == 10.0


def test_scripted_number_parses_and_defaults():
    q = ScriptedQuestioner(["0.9", ""])
    assert q.number("Target?") == 0.9
    assert q.number("Rounds?", default=5) == 5.0
```

Append to `tests/test_orchestrator.py`:
```python
def test_ctx_spec_reads_spec_or_raises(project):
    from mlagent.spec import Spec, SpecError

    ctx = make_ctx(project)
    with pytest.raises(SpecError):
        ctx.spec()
    project.write_json("spec.json", {
        "goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 2, "gpu": "none", "notes": "",
    })
    assert isinstance(ctx.spec(), Spec)
    assert ctx.spec().metric == "accuracy"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_questions.py tests/test_orchestrator.py -v`
Expected: FAIL with `AttributeError: 'ConsoleQuestioner' object has no attribute 'number'` and `'StageContext' object has no attribute 'spec'`.

- [ ] **Step 4: Implement**

`mlagent/ui/questions.py` — add to the protocol:
```python
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float: ...
```
Add to `ConsoleQuestioner`:
```python
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        bounds = ""
        if minimum is not None or maximum is not None:
            lo = "" if minimum is None else f"{minimum:g}"
            hi = "" if maximum is None else f"{maximum:g}"
            bounds = f" ({lo} to {hi})"
        suffix = f" [{default:g}]" if default is not None else ""
        while True:
            raw = self._input(f"{prompt}{bounds}{suffix} > ").strip()
            if not raw and default is not None:
                return float(default)
            try:
                value = float(raw)
            except ValueError:
                self._print("Please enter a number.")
                continue
            if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                self._print(f"Please enter a number{bounds}.")
                continue
            return value
```
Add to `ScriptedQuestioner`:
```python
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        answer = self._next(prompt).strip()
        if not answer and default is not None:
            return float(default)
        return float(answer)
```

`mlagent/stages/base.py` — add imports `from mlagent import config` and `from mlagent.spec import Spec, SpecError`, and this method on `StageContext`:
```python
    def spec(self) -> Spec:
        data = self.project.read_json(config.SPEC_FILE)
        if not data:
            raise SpecError("spec.json not found; run the intake stage first")
        return Spec.from_dict(data)
```

`mlagent/stages/intake.py` — delete `_to_float` and `_to_int`; in `collect_draft` replace the three numeric questions with:
```python
    target_value = q.number(f"What {metric} value would count as good enough?", default=0.9)
    ...
    minutes = int(q.number("Roughly how many minutes per training run are acceptable?", default=10, minimum=1))
    rounds = int(q.number("How many tuning rounds at most?", default=5, minimum=1))
```
(keep the surrounding lines unchanged; existing intake tests still pass because scripted answers "0.9", "10", "5" parse as numbers).

`scripts/build_notebook.py` — extend the intro markdown cell with one more paragraph:
```python
        "\n\n**Tip:** while a cell is waiting for you to type an answer, clicking a highlighted term does "
        "nothing until that cell finishes. Click terms after a stage completes, or run "
        "`colab.explain('term')` in its own cell."
```
Then run `python scripts/build_notebook.py`.

`tests/conftest.py` — add at the very top, before other imports:
```python
import matplotlib

matplotlib.use("Agg")
```

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -v && ruff check .`
Expected: all pass (previous 64 plus 3 new), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: M2 deps and lint rules; add Questioner.number and StageContext.spec

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 2: Synthetic tabular data with quirks

**Files:**
- Create: `mlagent/synth/__init__.py` (empty), `mlagent/synth/tabular.py`, `tests/test_synth_tabular.py`

**Interfaces:**
- Produces: `TASKS = ("classification", "regression")`, `QUIRKS`, `TARGET = "target"`, `CATEGORIES`, `@dataclass SynthTabularConfig(task="classification", n_samples=1000, n_features=8, n_classes=2, class_balance=0.5, noise=0.1, seed=42, quirks=())` with `validate() -> None` (raises `ValueError`), `generate(cfg) -> pd.DataFrame` (deterministic for a seed; feature columns `f1..fn`, target column `target`).

- [ ] **Step 1: Write the failing test**

`tests/test_synth_tabular.py`:
```python
import pandas as pd
import pytest

from mlagent.synth.tabular import QUIRKS, TARGET, SynthTabularConfig, generate


def test_classification_shape_target_and_determinism():
    cfg = SynthTabularConfig(n_samples=200, n_features=5, seed=1)
    df = generate(cfg)
    assert df.shape == (200, 6)
    assert list(df.columns) == ["f1", "f2", "f3", "f4", "f5", TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    pd.testing.assert_frame_equal(df, generate(cfg))


def test_regression_target_is_continuous():
    df = generate(SynthTabularConfig(task="regression", n_samples=100, n_features=3, seed=2))
    assert df[TARGET].dtype.kind == "f"
    assert df[TARGET].nunique() > 50


def test_class_balance_is_respected():
    df = generate(SynthTabularConfig(n_samples=1000, class_balance=0.9, noise=0.0, seed=3))
    assert df[TARGET].value_counts(normalize=True).iloc[0] > 0.8


def test_all_quirks_are_injected():
    df = generate(SynthTabularConfig(n_samples=200, n_features=4, seed=4, quirks=QUIRKS))
    assert df.columns[0] == "row_id"
    assert df["constant"].nunique() == 1
    assert df["category"].nunique() > 3
    assert df["category"].str.strip().str.lower().nunique() == 3
    assert df["f1"].isna().sum() > 0
    assert df.duplicated().sum() >= 5
    assert df["f1"].max() > 50


def test_whitespace_quirk_implies_categorical():
    df = generate(SynthTabularConfig(n_samples=100, seed=5, quirks=("whitespace",)))
    assert "category" in df.columns


def test_validate_rejects_bad_configs():
    with pytest.raises(ValueError):
        SynthTabularConfig(task="clustering").validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(quirks=("glitter",)).validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(n_features=2, n_classes=10).validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(noise=2.0).validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_synth_tabular.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.synth'`

- [ ] **Step 3: Implement**

`mlagent/synth/__init__.py`: empty.

`mlagent/synth/tabular.py`:
```python
"""Synthetic tabular datasets with optional realistic quirks for the cleaning stage to find."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

TASKS = ("classification", "regression")
QUIRKS = ("missing", "duplicates", "id_column", "constant_column", "outliers", "categorical", "whitespace")
TARGET = "target"
CATEGORIES = ("red", "green", "blue")
MESSY_VARIANTS = (" red", "Red", "green ", "GREEN", "Blue", " blue")


@dataclass
class SynthTabularConfig:
    task: str = "classification"
    n_samples: int = 1000
    n_features: int = 8
    n_classes: int = 2
    class_balance: float = 0.5
    noise: float = 0.1
    seed: int = 42
    quirks: tuple[str, ...] = ()

    def n_informative(self) -> int:
        needed = int(math.ceil(math.log2(2 * self.n_classes)))
        return min(self.n_features, max(2, self.n_features // 2, needed))

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.n_samples < 20:
            raise ValueError("n_samples must be at least 20")
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must be between 0 and 1")
        if not 0.05 <= self.class_balance <= 0.95:
            raise ValueError("class_balance must be between 0.05 and 0.95")
        unknown = set(self.quirks) - set(QUIRKS)
        if unknown:
            raise ValueError(f"unknown quirks: {sorted(unknown)}")
        if self.task == "classification" and 2 ** self.n_informative() < 2 * self.n_classes:
            raise ValueError("too many classes for this many features; add features or reduce classes")


def _base(cfg: SynthTabularConfig) -> pd.DataFrame:
    if cfg.task == "classification":
        n_inf = cfg.n_informative()
        weights = [cfg.class_balance, 1 - cfg.class_balance] if cfg.n_classes == 2 else None
        x, y = make_classification(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            n_informative=n_inf,
            n_redundant=min(2, cfg.n_features - n_inf),
            n_classes=cfg.n_classes,
            weights=weights,
            flip_y=cfg.noise,
            random_state=cfg.seed,
        )
    else:
        x, y = make_regression(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            noise=cfg.noise * 10,
            random_state=cfg.seed,
        )
    df = pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(cfg.n_features)])
    df[TARGET] = y
    return df


def _apply_quirks(df: pd.DataFrame, cfg: SynthTabularConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    quirks = set(cfg.quirks)
    if "whitespace" in quirks:
        quirks.add("categorical")
    features = [c for c in df.columns if c != TARGET]
    n = len(df)
    if "categorical" in quirks:
        df["category"] = rng.choice(CATEGORIES, size=n)
    if "whitespace" in quirks:
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[df.index[idx], "category"] = rng.choice(MESSY_VARIANTS, size=len(idx))
    if "outliers" in quirks:
        idx = rng.choice(n, size=max(1, n // 100), replace=False)
        col = features[0]
        df.loc[df.index[idx], col] = df.loc[df.index[idx], col].abs() * 50 + 100
    if "missing" in quirks:
        for col in features[:3]:
            idx = rng.choice(n, size=max(1, n // 20), replace=False)
            df.loc[df.index[idx], col] = np.nan
    if "constant_column" in quirks:
        df["constant"] = 1
    if "id_column" in quirks:
        df.insert(0, "row_id", np.arange(1000, 1000 + n))
    if "duplicates" in quirks:
        n_dup = max(5, n // 50)
        idx = rng.choice(n, size=n_dup, replace=False)
        df = pd.concat([df, df.iloc[idx]], ignore_index=True)
    return df


def generate(cfg: SynthTabularConfig) -> pd.DataFrame:
    cfg.validate()
    return _apply_quirks(_base(cfg), cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synth_tabular.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/synth tests/test_synth_tabular.py
git commit -m "feat: synthetic tabular data generator with injectable quirks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 3: Dataset profiling

**Files:**
- Create: `mlagent/profile.py`, `tests/test_profile.py`

**Interfaces:**
- Produces: `is_categorical_series(s: pd.Series) -> bool`; `profile_dataframe(df, target: str | None = None) -> dict` (JSON-serialisable; keys `n_rows, n_cols, duplicate_rows, memory_mb, columns[...], target`); `profile_markdown(profile: dict) -> str`.
- Column entry keys: `name, dtype, missing, missing_pct, n_unique, sample` plus `min, max, mean, std` for numeric columns. `target` is `None` or `{"name", "kind": "categorical", "counts": {label: n}}` or `{"name", "kind": "numeric", "min", "max", "mean", "std"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_profile.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.profile'`

- [ ] **Step 3: Implement**

`mlagent/profile.py`:
```python
"""Dataset profiling: a JSON-serialisable summary plus a markdown rendering."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

MAX_CATEGORIES = 20


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python; NaN becomes None."""
    if value is None:
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
    if vals.empty or vals.nunique() > MAX_CATEGORIES:
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
        counts = s.value_counts(dropna=True).head(MAX_CATEGORIES)
        return {"name": name, "kind": "categorical", "counts": {str(k): int(v) for k, v in counts.items()}}
    vals = s.dropna()
    return {
        "name": name,
        "kind": "numeric",
        "min": _py(vals.min()),
        "max": _py(vals.max()),
        "mean": _py(round(float(vals.mean()), 6)),
        "std": _py(round(float(vals.std()), 6)) if len(vals) > 1 else 0.0,
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
    lines = [
        f"### Data profile: {profile['n_rows']} rows × {profile['n_cols']} columns "
        f"({profile['duplicate_rows']} duplicate rows, {profile['memory_mb']} MB)",
        "",
        "| column | type | missing | unique | sample |",
        "|---|---|---|---|---|",
    ]
    for c in profile["columns"]:
        sample = ", ".join(c["sample"])[:60]
        lines.append(f"| {c['name']} | {c['dtype']} | {c['missing_pct']}% | {c['n_unique']} | {sample} |")
    t = profile.get("target")
    if t:
        lines.append("")
        if t["kind"] == "categorical":
            counts = ", ".join(f"{k}: {v}" for k, v in t["counts"].items())
            lines.append(f"Target `{t['name']}` is categorical: {counts}")
        else:
            lines.append(
                f"Target `{t['name']}` is numeric: min {t['min']}, max {t['max']}, mean {t['mean']}"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_profile.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/profile.py tests/test_profile.py
git commit -m "feat: dataset profiling with markdown rendering

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 4: Cleanliness audit

**Files:**
- Create: `mlagent/audit.py`, `tests/test_audit.py`

**Interfaces:**
- Consumes: `is_categorical_series` from `mlagent.profile`.
- Produces: `@dataclass Issue(kind: str, severity: str, message: str, column: str | None = None, evidence: dict = {}, fix: dict | None = None)` with `to_dict()`; `SEVERITY_ORDER`; `audit_tabular(df, target: str | None = None) -> list[Issue]` sorted high → medium → low.
- Fix dicts use the ops defined in Task 5: `drop_columns{columns}`, `drop_duplicates{}`, `fill_missing{column, strategy}`, `drop_rows_missing_target{target}`, `normalise_categories{column}`, `clip_outliers{column, lower, upper}`, `coerce_numeric{column}`.
- Issue kinds: `target_missing, class_imbalance, rare_classes, missing_values, duplicate_rows, constant_column, near_constant_column, mixed_types, inconsistent_categories, outliers, id_column, target_leakage, suspicious_values`.

- [ ] **Step 1: Write the failing test**

`tests/test_audit.py`:
```python
import json

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
    assert by_kind["id_column"].fix == {"op": "drop_columns", "params": {"columns": ["row_id"]}}
    assert by_kind["missing_values"].fix == {"op": "fill_missing", "params": {"column": "f1", "strategy": "median"}}
    assert by_kind["duplicate_rows"].fix == {"op": "drop_duplicates", "params": {}}
    assert by_kind["inconsistent_categories"].fix["params"] == {"column": "cat"}
    assert by_kind["outliers"].fix["op"] == "clip_outliers" and by_kind["outliers"].column == "f2"
    assert by_kind["mixed_types"].fix == {"op": "coerce_numeric", "params": {"column": "mixed"}}
    assert by_kind["target_leakage"].column == "leak"
    assert by_kind["suspicious_values"].fix is None


def test_target_checks():
    n = 300
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "target": [0] * 290 + [1] * 8 + [2] * 2})
    df.loc[0, "target"] = np.nan
    by_kind = {i.kind: i for i in audit_tabular(df, target="target")}
    assert by_kind["target_missing"].fix == {"op": "drop_rows_missing_target", "params": {"target": "target"}}
    assert by_kind["class_imbalance"].severity == "medium"
    assert "2" in by_kind["rare_classes"].evidence["rare"]


def test_regression_leakage_by_correlation():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    df = pd.DataFrame({"good": rng.normal(size=100), "leaky": y * 3 + 0.001 * rng.normal(size=100), "target": y})
    issues = audit_tabular(df, target="target")
    assert [i.column for i in issues if i.kind == "target_leakage"] == ["leaky"]


def test_sparse_column_is_dropped_not_filled():
    df = pd.DataFrame({"sparse": [np.nan] * 90 + [1.0] * 10, "ok": range(100), "target": [0, 1] * 50})
    issue = next(i for i in audit_tabular(df, "target") if i.column == "sparse")
    assert issue.severity == "high" and issue.fix["op"] == "drop_columns"


def test_clean_frame_has_no_issues():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "target": rng.integers(0, 2, 100)})
    assert audit_tabular(df, target="target") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.audit'`

- [ ] **Step 3: Implement**

`mlagent/audit.py`:
```python
"""Cleanliness audit for tabular data: each check returns Issues with evidence and a proposed fix."""

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
            f"{n_missing} rows have no value for the target column '{target}'; they cannot be used for training.",
            target, {"rows": n_missing}, {"op": "drop_rows_missing_target", "params": {"target": target}},
        ))
    if is_categorical_series(s):
        counts = s.value_counts(dropna=True)
        n = int(counts.sum())
        if n:
            majority = float(counts.iloc[0] / n)
            if majority > 0.9:
                out.append(Issue(
                    "class_imbalance", "medium",
                    f"The majority class makes up {majority:.0%} of rows; accuracy will look good even for a useless model.",
                    target, {"majority_fraction": round(majority, 4),
                             "counts": {str(k): int(v) for k, v in counts.head(10).items()}},
                ))
            rare = {str(k): int(v) for k, v in counts.items() if v / n < 0.01}
            if rare:
                out.append(Issue(
                    "rare_classes", "medium",
                    f"{len(rare)} class(es) have fewer than 1% of rows; the model will struggle to learn them.",
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
                f"'{col}' is missing in {pct:.1f}% of rows; too sparse to be useful.", col, evidence, _drop(col),
            ))
        else:
            strategy = "median" if ptypes.is_numeric_dtype(s) else "mode"
            out.append(Issue(
                "missing_values", "medium",
                f"'{col}' is missing in {pct:.1f}% of rows ({n}); most models cannot handle gaps.",
                col, evidence, {"op": "fill_missing", "params": {"column": col, "strategy": strategy}},
            ))
    return out


def _check_duplicates(df: pd.DataFrame, target: str | None) -> list[Issue]:
    n = int(df.duplicated().sum())
    if not n:
        return []
    return [Issue(
        "duplicate_rows", "medium",
        f"{n} rows are exact duplicates; they over-weight those examples and can leak between train and test splits.",
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
                "constant_column", "medium", f"'{col}' has a single value; it carries no information.",
                col, {"n_unique": nunique}, _drop(col),
            ))
        elif len(s) >= 100:
            top = float(s.value_counts(normalize=True).iloc[0])
            if top > 0.99:
                out.append(Issue(
                    "near_constant_column", "low", f"'{col}' is the same value in {top:.1%} of rows.",
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
        numeric_fraction = float(pd.to_numeric(vals, errors="coerce").notna().mean())
        if 0.5 <= numeric_fraction < 1.0:
            out.append(Issue(
                "mixed_types", "medium",
                f"'{col}' is mostly numbers stored as text ({numeric_fraction:.0%}) with some non-numeric entries.",
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
                f"'{col}' has {raw} spellings for {norm} real categories (case or whitespace differences).",
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
                "outliers", "low", f"'{col}' has {n} extreme values far outside the typical range.",
                col, {"count": n, "lower": lower, "upper": upper},
                {"op": "clip_outliers", "params": {"column": col, "lower": lower, "upper": upper}},
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
                    f"'{col}' is unique for (almost) every row; it is an identifier, not a feature, and a model could memorise it.",
                    col, {"unique_ratio": round(float(ratio), 4)}, _drop(col),
                ))
                continue
        if t is None:
            continue
        if ptypes.is_numeric_dtype(s) and ptypes.is_numeric_dtype(t) and not is_categorical_series(t):
            corr = s.corr(t)
            if pd.notna(corr) and abs(float(corr)) > 0.98:
                out.append(Issue(
                    "target_leakage", "high",
                    f"'{col}' is almost perfectly correlated with the target (r={corr:.3f}); it probably encodes the answer.",
                    col, {"correlation": round(float(corr), 4)}, _drop(col),
                ))
        elif s.astype(str).equals(t.astype(str)):
            out.append(Issue(
                "target_leakage", "high", f"'{col}' is identical to the target column.", col, {}, _drop(col),
            ))
    return out


def _check_suspicious_ranges(df: pd.DataFrame, target: str | None) -> list[Issue]:
    out: list[Issue] = []
    for col in _features(df, target):
        s = df[col]
        if not ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s) or not QUANTITY_NAME.search(col):
            continue
        n = int((s.dropna() < 0).sum())
        if n:
            out.append(Issue(
                "suspicious_values", "low",
                f"'{col}' has {n} negative values, which is unusual for a quantity-like column; check the source.",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audit.py -v`
Expected: 6 passed. If `test_every_planted_issue_is_found_and_sorted` misses `outliers`, confirm the fixture plants two outliers (rows 0–1) so the count exceeds 0.5% after duplication; do not loosen the threshold.

- [ ] **Step 5: Commit**

```bash
git add mlagent/audit.py tests/test_audit.py
git commit -m "feat: tabular cleanliness audit producing issues with proposed fixes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 5: Cleaning operations and `clean.py` rendering

**Files:**
- Create: `mlagent/cleaning.py`, `tests/test_cleaning.py`

**Interfaces:**
- Produces: `OPS: dict[str, Callable]`; `apply_steps(df, steps: list[dict]) -> pd.DataFrame` (never mutates input; raises `ValueError` on unknown op); `describe_step(step: dict) -> str`; `render_clean_py(steps) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_cleaning.py`:
```python
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
    out = apply_steps(df(), [{"op": "drop_columns", "params": {"columns": ["id", "missing_col"]}}])
    assert list(out.columns) == ["x", "cat", "num_text", "target"]
    out = apply_steps(df(), [{"op": "drop_duplicates", "params": {}}])
    assert len(out) == 3 and list(out.index) == [0, 1, 2]
    out = apply_steps(df(), [{"op": "fill_missing", "params": {"column": "x", "strategy": "median"}}])
    assert out["x"].isna().sum() == 0 and out.loc[1, "x"] == 100.0
    out = apply_steps(df(), [{"op": "fill_missing", "params": {"column": "cat", "strategy": "mode"}}])
    assert out["cat"].isna().sum() == 0
    out = apply_steps(df(), [{"op": "drop_rows_missing_target", "params": {"target": "target"}}])
    assert len(out) == 2
    out = apply_steps(df(), [{"op": "normalise_categories", "params": {"column": "cat"}}])
    assert out["cat"].tolist() == ["a", "a", "b", "b"]
    out = apply_steps(df(), [{"op": "clip_outliers", "params": {"column": "x", "lower": 0.0, "upper": 10.0}}])
    assert out["x"].max() == 10.0
    out = apply_steps(df(), [{"op": "coerce_numeric", "params": {"column": "num_text"}}])
    assert out["num_text"].dtype.kind == "f" and out["num_text"].isna().sum() == 2


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
    assert "id" in describe_step({"op": "drop_columns", "params": {"columns": ["id"]}})
    assert "median" in describe_step({"op": "fill_missing", "params": {"column": "x", "strategy": "median"}})
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cleaning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.cleaning'`

- [ ] **Step 3: Implement**

`mlagent/cleaning.py`:
```python
"""Cleaning steps as data. apply_steps() runs them; render_clean_py() writes a re-runnable script."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _fill_missing(df: pd.DataFrame, column: str, strategy: str = "median", value=None) -> pd.DataFrame:
    if column not in df.columns:
        return df
    s = df[column]
    if strategy == "median":
        fill = s.median()
    elif strategy == "mode":
        mode = s.mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else value
    elif strategy == "constant":
        fill = value
    else:
        raise ValueError(f"unknown fill strategy {strategy!r}")
    out = df.copy()
    out[column] = s.fillna(fill)
    return out


def _drop_rows_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.dropna(subset=[target]).reset_index(drop=True)


def _normalise_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    s = out[column]
    out[column] = s.where(s.isna(), s.astype(str).str.strip().str.lower())
    return out


def _clip_outliers(df: pd.DataFrame, column: str, lower: float, upper: float) -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].clip(lower=lower, upper=upper)
    return out


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


OPS: dict[str, Callable[..., pd.DataFrame]] = {
    "drop_columns": _drop_columns,
    "drop_duplicates": _drop_duplicates,
    "fill_missing": _fill_missing,
    "drop_rows_missing_target": _drop_rows_missing_target,
    "normalise_categories": _normalise_categories,
    "clip_outliers": _clip_outliers,
    "coerce_numeric": _coerce_numeric,
}


def apply_steps(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    out = df
    for step in steps:
        op = step.get("op")
        if op not in OPS:
            raise ValueError(f"unknown cleaning op {op!r}")
        out = OPS[op](out, **step.get("params", {}))
    return out


def describe_step(step: dict) -> str:
    op = step.get("op")
    p = step.get("params", {})
    if op == "drop_columns":
        return "drop column(s) " + ", ".join(f"`{c}`" for c in p.get("columns", []))
    if op == "drop_duplicates":
        return "drop exact duplicate rows"
    if op == "fill_missing":
        return f"fill missing values in `{p.get('column')}` with the {p.get('strategy')}"
    if op == "drop_rows_missing_target":
        return f"drop rows where `{p.get('target')}` is missing"
    if op == "normalise_categories":
        return f"trim whitespace and lower-case the categories in `{p.get('column')}`"
    if op == "clip_outliers":
        return f"clip `{p.get('column')}` to the range {p.get('lower'):.4g} to {p.get('upper'):.4g}"
    if op == "coerce_numeric":
        return f"convert `{p.get('column')}` to numbers (non-numeric entries become missing)"
    return f"{op} {p}"


CLEAN_PY_TEMPLATE = '''"""Generated by mlagent: the cleaning steps approved for this project, re-runnable on new data."""

import json

import pandas as pd

from mlagent.cleaning import apply_steps

STEPS = json.loads(r"""__STEPS__""")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return apply_steps(df, STEPS)


if __name__ == "__main__":
    import sys

    src, dst = sys.argv[1], sys.argv[2]
    clean(pd.read_csv(src)).to_csv(dst, index=False)
'''


def render_clean_py(steps: list[dict]) -> str:
    return CLEAN_PY_TEMPLATE.replace("__STEPS__", json.dumps(steps, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cleaning.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/cleaning.py tests/test_cleaning.py
git commit -m "feat: cleaning ops as data with re-runnable clean.py rendering

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 6: Plots

**Files:**
- Create: `mlagent/plots.py`, `tests/test_plots.py`

**Interfaces:**
- Produces: palette constants `SERIES, SEQUENTIAL, DIVERGING, INK, INK_2, MUTED, GRID, AXIS, SURFACE`; figure builders returning `matplotlib.figure.Figure`: `feature_histograms(df, columns=None, max_cols=12)`, `class_balance(counts: dict[str, int], title=...)`, `target_distribution(series, title=...)`, `missing_matrix(df, max_rows=500)`, `correlation_heatmap(df, max_cols=20)`, `outlier_boxplots(df, columns, max_cols=8)`, `before_after_missing(before: dict, after: dict)`; `save_figure(fig, path) -> Path`; `present(fig, plots_dir: Path, name: str) -> Path` (saves `<plots_dir>/<name>.png`, displays under IPython, closes the figure).

- [ ] **Step 1: Write the failing test**

`tests/test_plots.py`:
```python
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mlagent import plots
from mlagent.profile import profile_dataframe


def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
        "c": rng.integers(0, 5, size=60),
        "cat": rng.choice(["x", "y"], size=60),
        "target": rng.integers(0, 2, size=60),
    }).mask(rng.random((60, 5)) < 0.05)


def test_every_builder_returns_a_figure_with_axes():
    frame = df()
    before = profile_dataframe(frame, "target")
    after = profile_dataframe(frame.dropna(), "target")
    figs = [
        plots.feature_histograms(frame),
        plots.class_balance({"0": 30, "1": 25}),
        plots.target_distribution(frame["a"]),
        plots.missing_matrix(frame),
        plots.correlation_heatmap(frame),
        plots.outlier_boxplots(frame, ["a", "b"]),
        plots.before_after_missing(before, after),
    ]
    for fig in figs:
        assert isinstance(fig, Figure) and fig.axes
    assert plots.before_after_missing(before, after).axes[0].get_legend() is not None
    assert plots.class_balance({"0": 1}).axes[0].get_legend() is None


def test_correlation_with_one_numeric_column_does_not_crash():
    fig = plots.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]}))
    assert isinstance(fig, Figure)


def test_present_saves_png_and_closes(tmp_path):
    import matplotlib.pyplot as plt

    plt.close("all")
    fig = plots.class_balance({"0": 3, "1": 4})
    path = plots.present(fig, tmp_path / "plots", "balance")
    assert path == tmp_path / "plots" / "balance.png" and path.stat().st_size > 1000
    import matplotlib.pyplot as plt

    assert not plt.get_fignums()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plots.py -v`
Expected: FAIL with `ImportError: cannot import name 'plots'`

- [ ] **Step 3: Implement**

`mlagent/plots.py`:
```python
"""All matplotlib figures. Static charts: thin marks, recessive grid, text in ink colours."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pandas.api import types as ptypes

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#f3a17f", "#d95926"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
DIV_CMAP = LinearSegmentedColormap.from_list("mlagent_div", DIVERGING)


def _style(ax, title: str | None = None, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def _numeric_columns(df: pd.DataFrame, columns=None) -> list[str]:
    cols = list(columns) if columns is not None else list(df.columns)
    return [c for c in cols if c in df.columns and ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])]


def feature_histograms(df: pd.DataFrame, columns=None, max_cols: int = 12) -> Figure:
    numeric = _numeric_columns(df, columns)[:max_cols]
    n = max(1, len(numeric))
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, col in zip(axes.flat, numeric, strict=False):
        ax.hist(df[col].dropna(), bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        _style(ax, col)
    for ax in list(axes.flat)[len(numeric):]:
        ax.set_visible(False)
    if not numeric:
        axes[0][0].set_visible(True)
        axes[0][0].text(0.5, 0.5, "no numeric columns", ha="center", color=MUTED)
    fig.suptitle("Feature distributions", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def class_balance(counts: dict[str, int], title: str = "Target class balance") -> Figure:
    fig = plt.figure(figsize=(5, 0.5 * max(3, len(counts)) + 1.2), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    labels = [str(k) for k in counts]
    values = [int(v) for v in counts.values()]
    total = sum(values) or 1
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v} ({v / total:.0%})", va="center", color=INK_2, fontsize=8)
    _style(ax, title, grid_axis="x")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3 if values else 1)
    fig.tight_layout()
    return fig


def target_distribution(series: pd.Series, title: str = "Target distribution") -> Figure:
    fig = plt.figure(figsize=(5, 2.8), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.hist(series.dropna(), bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    _style(ax, title)
    fig.tight_layout()
    return fig


def missing_matrix(df: pd.DataFrame, max_rows: int = 500) -> Figure:
    sample = df if len(df) <= max_rows else df.sample(max_rows, random_state=0).sort_index()
    mat = sample.isna().to_numpy().T.astype(float)
    fig = plt.figure(figsize=(7, 0.28 * len(df.columns) + 1.5), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels([str(c) for c in df.columns], fontsize=8, color=INK_2)
    ax.set_xlabel(f"rows (showing {len(sample)} of {len(df)})", color=MUTED, fontsize=8)
    _style(ax, "Missing values (dark = missing)", grid_axis="none")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> Figure:
    numeric = df[_numeric_columns(df)[:max_cols]]
    fig = plt.figure(figsize=(6, 5), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    if numeric.shape[1] < 2:
        ax.text(0.5, 0.5, "need at least two numeric columns", ha="center", color=MUTED)
        ax.set_axis_off()
        return fig
    corr = numeric.corr().to_numpy()
    im = ax.imshow(corr, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ticks = range(numeric.shape[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(numeric.columns, rotation=60, ha="right", fontsize=8, color=INK_2)
    ax.set_yticklabels(numeric.columns, fontsize=8, color=INK_2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=MUTED, labelsize=8)
    _style(ax, "Correlation between numeric columns", grid_axis="none")
    fig.tight_layout()
    return fig


def outlier_boxplots(df: pd.DataFrame, columns, max_cols: int = 8) -> Figure:
    cols = _numeric_columns(df, columns)[:max_cols]
    fig = plt.figure(figsize=(6, 0.5 * max(2, len(cols)) + 1.2), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    data = [df[c].dropna().to_numpy() for c in cols]
    if data:
        box = ax.boxplot(data, vert=False, tick_labels=cols, patch_artist=True, widths=0.5,
                         flierprops={"marker": ".", "markersize": 4, "markerfacecolor": SERIES[1],
                                     "markeredgecolor": SERIES[1]})
        for patch in box["boxes"]:
            patch.set_facecolor(SEQUENTIAL[1])
            patch.set_edgecolor(SERIES[0])
        for key in ("whiskers", "caps", "medians"):
            for line in box[key]:
                line.set_color(SERIES[0])
    _style(ax, "Value ranges and outliers", grid_axis="x")
    fig.tight_layout()
    return fig


def before_after_missing(before: dict, after: dict) -> Figure:
    names = [c["name"] for c in before["columns"]]
    after_pct = {c["name"]: c["missing_pct"] for c in after["columns"]}
    b = [c["missing_pct"] for c in before["columns"]]
    a = [after_pct.get(n, 0.0) for n in names]
    y = np.arange(len(names))
    fig = plt.figure(figsize=(6, 0.35 * max(3, len(names)) + 1.4), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.barh(y - 0.18, b, height=0.34, color=SERIES[0], label="before cleaning")
    ax.barh(y + 0.18, a, height=0.34, color=SERIES[1], label="after cleaning")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK_2)
    ax.set_xlabel("% missing", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Missing values before and after cleaning", grid_axis="x")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    return path


def _show(fig: Figure) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
    except Exception:  # noqa: BLE001 - display is best-effort outside notebooks
        pass


def present(fig: Figure, plots_dir: Path, name: str) -> Path:
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    _show(fig)
    plt.close(fig)
    return path
```
If ruff (rule `B`) complains about the `noqa`, keep the `except Exception` and drop the noqa comment. If matplotlib warns that `labels=` in `boxplot` is deprecated in favour of `tick_labels=`, use `tick_labels=cols` (matplotlib ≥ 3.9) so test output stays pristine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_plots.py -v -W error::DeprecationWarning`
Expected: 3 passed with no warnings.

- [ ] **Step 5: Commit**

```bash
git add mlagent/plots.py tests/test_plots.py
git commit -m "feat: matplotlib figures for data profiling and cleaning

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 7: Data sources (Drive files, HuggingFace Hub)

**Files:**
- Create: `mlagent/datasources/__init__.py` (empty), `mlagent/datasources/drive.py`, `mlagent/datasources/hf.py`, `tests/test_datasources.py`

**Interfaces:**
- Produces: `drive.TABLE_EXTS`, `drive.list_candidates(roots, exts=TABLE_EXTS, max_depth=3, limit=50) -> list[Path]`, `drive.load_table(path) -> pd.DataFrame`; `hf.HFDataset(id, downloads, likes, description)`, `hf.search_datasets(query, limit=8, api=None) -> list[HFDataset]`, `hf.load_tabular(dataset_id, split="train", config=None, loader=None) -> pd.DataFrame`.
- `api` defaults to `huggingface_hub.HfApi()` and `loader` to `datasets.load_dataset`, both imported lazily inside the function so tests never import them.

- [ ] **Step 1: Write the failing test**

`tests/test_datasources.py`:
```python
from types import SimpleNamespace

import pandas as pd
import pytest

from mlagent.datasources import drive, hf


def test_list_candidates_respects_depth_skips_and_limit(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "deep" / "er" / "est").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "est" / "far.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "deep" / "near.parquet").write_bytes(b"")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("no", encoding="utf-8")
    found = drive.list_candidates([tmp_path, tmp_path / "missing"], max_depth=2)
    assert [p.name for p in found] == ["a.csv", "near.parquet"]
    assert drive.list_candidates([tmp_path], limit=1) == [tmp_path / "a.csv"]


def test_load_table_by_extension(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    tsv = tmp_path / "t.tsv"
    tsv.write_text("a\tb\n1\t2\n", encoding="utf-8")
    pq = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(pq)
    for path in (csv, tsv, pq):
        assert drive.load_table(path).to_dict("records") == [{"a": 1, "b": 2}]
    with pytest.raises(ValueError):
        drive.load_table(tmp_path / "t.xyz")


class FakeApi:
    def __init__(self):
        self.calls = []

    def list_datasets(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(id="org/churn", downloads=1200, likes=5, tags=["task:tabular", "csv"], description=None),
            SimpleNamespace(id="org/other", downloads=None, likes=None, tags=None, description="Some data"),
        ]


def test_search_datasets_maps_results():
    api = FakeApi()
    results = hf.search_datasets("churn", limit=2, api=api)
    assert api.calls[0]["search"] == "churn" and api.calls[0]["limit"] == 2
    assert results[0] == hf.HFDataset(id="org/churn", downloads=1200, likes=5, description="csv")
    assert results[1].downloads == 0 and results[1].description == "Some data"


def test_load_tabular_with_split_and_dict_fallback():
    frame = pd.DataFrame({"a": [1]})

    def loader_ok(dataset_id, **kwargs):
        assert kwargs == {"split": "train"}
        return SimpleNamespace(to_pandas=lambda: frame)

    assert hf.load_tabular("x/y", loader=loader_ok).equals(frame)

    class DictLike(dict):
        pass

    def loader_dict(dataset_id, **kwargs):
        if "split" in kwargs:
            raise ValueError("bad split")
        return DictLike(validation=SimpleNamespace(to_pandas=lambda: frame))

    assert hf.load_tabular("x/y", loader=loader_dict).equals(frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datasources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.datasources'`

- [ ] **Step 3: Implement**

`mlagent/datasources/__init__.py`: empty.

`mlagent/datasources/drive.py`:
```python
"""Find and load tabular files on the mounted Drive (or any directory)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def list_candidates(roots, exts=TABLE_EXTS, max_depth: int = 3, limit: int = 50) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - base
            if depth >= max_depth:
                dirnames[:] = []
            else:
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if Path(name).suffix.lower() in exts:
                    found.append(Path(dirpath) / name)
                    if len(found) >= limit:
                        return found
    return found


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"unsupported file type {suffix!r}; use CSV, TSV, Parquet, or Excel")
```

`mlagent/datasources/hf.py`:
```python
"""Search and load tabular datasets from the HuggingFace Hub (lazy imports, injectable for tests)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HFDataset:
    id: str
    downloads: int
    likes: int
    description: str


def search_datasets(query: str, limit: int = 8, api=None) -> list[HFDataset]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    results = api.list_datasets(search=query, sort="downloads", direction=-1, limit=limit)
    out: list[HFDataset] = []
    for item in results:
        description = getattr(item, "description", None) or ""
        if not description:
            tags = getattr(item, "tags", None) or []
            description = ", ".join(t for t in tags if ":" not in t)
        out.append(HFDataset(
            id=str(item.id),
            downloads=int(getattr(item, "downloads", 0) or 0),
            likes=int(getattr(item, "likes", 0) or 0),
            description=description[:120],
        ))
    return out


def load_tabular(dataset_id: str, split: str = "train", config: str | None = None, loader=None) -> pd.DataFrame:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    kwargs = {"name": config} if config else {}
    try:
        ds = loader(dataset_id, split=split, **kwargs)
    except ValueError:
        ds = loader(dataset_id, **kwargs)
    if hasattr(ds, "keys") and not hasattr(ds, "to_pandas"):
        key = "train" if "train" in ds else next(iter(ds.keys()))
        ds = ds[key]
    return ds.to_pandas()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_datasources.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/datasources tests/test_datasources.py
git commit -m "feat: Drive file discovery and HuggingFace dataset search/load

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 8: Data stage

**Files:**
- Create: `mlagent/stages/data.py`, `mlagent/prompts/data.md`, `tests/test_data_stage.py`

**Interfaces:**
- Consumes: `StageContext` (`.spec()`, `.questioner`, `.llm`, `.display`, `.project`), `synth.tabular`, `profile`, `plots`, `datasources`, `ask_text`, `LLMError`, `load_prompt`.
- Produces: `RAW_FILE = "data.csv"`, `META_FILE = "data_meta.json"`, `PROFILE_RAW_FILE = "profile_raw.json"`, `DEFAULT_QUIRKS`, `class DataStage(search_roots=None, hf_search=search_datasets, hf_load=load_tabular)` with `name = "data"`. Artifacts: `data/raw/data.csv`, `profile_raw.json`, `data_meta.json` = `{"source", "target", "raw_path", "n_rows", "n_cols", ...}` plus `synth_config` / `source_path` / `hf_id` depending on source. Plots saved: `raw_histograms.png`, `raw_missing.png`, `raw_class_balance.png` or `raw_target_distribution.png`, `raw_correlation.png` (when ≥ 2 numeric columns).

- [ ] **Step 1: Write the failing test**

`tests/test_data_stage.py`:
```python
import pandas as pd
import pytest

from mlagent.datasources.hf import HFDataset
from mlagent.llm import FakeLLM
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE, PROFILE_RAW_FILE, RAW_FILE, DataStage
from mlagent.ui.questions import ScriptedQuestioner

SPEC = {
    "goal": "Predict churn", "task_type": "tabular_classification", "metric": "accuracy",
    "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5, "max_rounds": 2,
    "gpu": "none", "notes": "",
}


def make_ctx(project, answers, source="synthetic", task="tabular_classification", llm=None):
    project.write_json("spec.json", {**SPEC, "data_source": source, "task_type": task})
    shown: list[str] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([[("text", "Narrative [[class balance]]")]]),
                       questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append)
    return ctx, shown


def test_synthetic_classification_writes_artifacts_and_plots(project):
    ctx, shown = make_ctx(project, ["300", "5", "2", "0.6", "0.1", "y"])
    stage = DataStage()
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    df = pd.read_csv(project.data_raw / RAW_FILE)
    assert len(df) > 300 and "row_id" in df.columns and "target" in df.columns
    meta = project.read_json(META_FILE)
    assert meta["source"] == "synthetic" and meta["target"] == "target"
    assert meta["synth_config"]["n_samples"] == 300 and meta["synth_config"]["class_balance"] == 0.6
    assert project.read_json(PROFILE_RAW_FILE)["target"]["kind"] == "categorical"
    for name in ("raw_histograms", "raw_missing", "raw_class_balance", "raw_correlation"):
        assert (project.plots_dir / f"{name}.png").exists()
    assert any("[[class balance]]" in s for s in shown)
    assert any("Data profile" in s for s in shown)


def test_synthetic_regression_without_quirks(project):
    ctx, _ = make_ctx(project, ["200", "4", "0.2", "n"], task="tabular_regression")
    DataStage().run(ctx)
    df = pd.read_csv(project.data_raw / RAW_FILE)
    assert df.shape == (200, 5) and (project.plots_dir / "raw_target_distribution.png").exists()


def test_drive_source_lists_files_and_asks_target(project, tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    csv = root / "customers.csv"
    csv.write_text("age,income,churned\n30,100,0\n40,200,1\n50,300,0\n", encoding="utf-8")
    ctx, _ = make_ctx(project, [str(csv), "churned"], source="drive")
    DataStage(search_roots=[root]).run(ctx)
    meta = project.read_json(META_FILE)
    assert meta == {**meta, "source": "drive", "target": "churned", "source_path": str(csv), "n_rows": 3}
    assert "Which column is the target" in ctx.questioner.asked[-1]


def test_drive_source_missing_file_raises(project, tmp_path):
    ctx, _ = make_ctx(project, [str(tmp_path / "nope.csv")], source="drive")
    with pytest.raises(FileNotFoundError):
        DataStage(search_roots=[tmp_path]).run(ctx)


def test_huggingface_source_searches_picks_and_loads(project):
    searches: list[str] = []

    def fake_search(query, limit=8):
        searches.append(query)
        return [] if len(searches) == 1 else [HFDataset("org/churn", 10, 1, "customer churn")]

    def fake_load(dataset_id):
        assert dataset_id == "org/churn"
        return pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    ctx, shown = make_ctx(project, ["nothing", "churn", "org/churn — 10 downloads — customer churn", "y"],
                          source="huggingface")
    DataStage(hf_search=fake_search, hf_load=fake_load).run(ctx)
    assert searches == ["nothing", "churn"]
    assert project.read_json(META_FILE)["hf_id"] == "org/churn"
    assert any("No datasets found" in s for s in shown)


def test_image_task_is_not_supported_yet(project):
    ctx, _ = make_ctx(project, [], task="image_classification")
    with pytest.raises(NotImplementedError):
        DataStage().run(ctx)


def test_llm_failure_still_completes(project):
    ctx, shown = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"], llm=FakeLLM([]))
    DataStage().run(ctx)
    assert DataStage().is_complete(ctx)
    assert any("Couldn't reach Claude" in s for s in shown)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.data'`

- [ ] **Step 3: Implement**

`mlagent/prompts/data.md`:
```markdown
You are the data stage of an ML training assistant that runs inside Google Colab. The user has just obtained a dataset and you have its profile.

Write a short narrative (under 150 words) for the user:

1. What the data looks like in one sentence (rows, columns, target).
2. Two or three things worth noticing before training, drawn from the profile: class balance or target range, columns with missing values, columns that look like identifiers or constants, categorical columns that will need encoding. Say why each matters in plain language.
3. One sentence on what happens next: an automatic cleanliness audit that proposes fixes for approval.

Wrap technical terms in double square brackets like [[class imbalance]] so the user can click them for an explanation. Do not repeat the full table; the user already sees it.
```

`mlagent/stages/data.py`:
```python
"""Data stage: obtain raw tabular data, profile it, plot it, and narrate what to notice."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlagent import plots
from mlagent.datasources.drive import list_candidates, load_table
from mlagent.datasources.hf import load_tabular, search_datasets
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe, profile_markdown
from mlagent.prompts_io import load_prompt
from mlagent.stages.base import StageContext
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate

RAW_FILE = "data.csv"
META_FILE = "data_meta.json"
PROFILE_RAW_FILE = "profile_raw.json"
DEFAULT_QUIRKS = ("missing", "duplicates", "id_column", "categorical", "whitespace", "outliers")
DEFAULT_SEARCH_ROOTS = (Path("/content/drive/MyDrive"), Path("/content"))
TABULAR_TASKS = {"tabular_classification": "classification", "tabular_regression": "regression"}


class DataStage:
    name = "data"

    def __init__(self, search_roots=None, hf_search=search_datasets, hf_load=load_tabular):
        self.search_roots = list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        self.hf_search = hf_search
        self.hf_load = hf_load

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        return bool(meta and meta.get("target")) and (ctx.project.data_raw / RAW_FILE).exists()

    def run(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        if spec.task_type not in TABULAR_TASKS:
            raise NotImplementedError(
                f"{spec.task_type} data is not supported yet (image tasks arrive in Milestone 5)"
            )
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx)
        else:
            df, target, meta = self._huggingface(ctx)

        ctx.project.data_raw.mkdir(parents=True, exist_ok=True)
        path = ctx.project.data_raw / RAW_FILE
        df.to_csv(path, index=False)
        profile = profile_dataframe(df, target)
        ctx.project.write_json(PROFILE_RAW_FILE, profile)
        meta.update({"target": target, "raw_path": str(path), "n_rows": int(len(df)), "n_cols": int(df.shape[1])})
        ctx.project.write_json(META_FILE, meta)

        ctx.display(profile_markdown(profile))
        self._plots(ctx, df, profile)
        self._narrate(ctx, spec.to_dict(), profile)

    def _synthetic(self, ctx: StageContext, task: str):
        q = ctx.questioner
        n_samples = int(q.number("How many rows?", default=1000, minimum=100, maximum=200000))
        n_features = int(q.number("How many numeric features?", default=8, minimum=2, maximum=100))
        n_classes, class_balance = 2, 0.5
        if task == "classification":
            n_classes = int(q.number("How many classes?", default=2, minimum=2, maximum=10))
            if n_classes == 2:
                class_balance = q.number(
                    "Fraction of rows in the majority class (0.5 = balanced)?",
                    default=0.5, minimum=0.5, maximum=0.95,
                )
        noise = q.number("Label/measurement noise (0 = clean, 0.3 = very noisy)?", default=0.1, minimum=0.0, maximum=1.0)
        inject = q.confirm(
            "Inject realistic data problems (missing values, duplicates, an ID column, messy categories, "
            "outliers) so the cleaning stage has work to do?",
            default=True,
        )
        cfg = SynthTabularConfig(
            task=task, n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            class_balance=class_balance, noise=noise, seed=42, quirks=DEFAULT_QUIRKS if inject else (),
        )
        df = generate(cfg)
        return df, TARGET, {"source": "synthetic", "synth_config": asdict(cfg)}

    def _ask_target(self, ctx: StageContext, df: pd.DataFrame) -> str:
        cols = [str(c) for c in df.columns]
        return ctx.questioner.choice("Which column is the target (what you want to predict)?", cols, allow_other=False)

    def _drive(self, ctx: StageContext):
        q = ctx.questioner
        candidates = list_candidates(self.search_roots)
        if candidates:
            answer = q.choice("Which file holds your data? (pick one or type a full path)",
                              [str(p) for p in candidates], allow_other=True)
        else:
            answer = q.text("No CSV/Parquet/Excel files found. Enter the full path to your data file")
        path = Path(answer.strip())
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        df = load_table(path)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "drive", "source_path": str(path)}

    def _huggingface(self, ctx: StageContext):
        q = ctx.questioner
        results = []
        for _ in range(3):
            query = q.text("Describe the dataset you want (a few keywords, e.g. 'credit card fraud')")
            results = self.hf_search(query)
            if results:
                break
            ctx.display(f"No datasets found for '{query}'. Try different words.")
        if not results:
            raise RuntimeError("no HuggingFace datasets found after 3 searches")
        labels = [f"{r.id} — {r.downloads:,} downloads — {r.description[:60]}" for r in results]
        pick = q.choice("Which dataset?", labels, allow_other=False)
        chosen = results[labels.index(pick)]
        ctx.display(f"Downloading **{chosen.id}** from the HuggingFace Hub…")
        df = self.hf_load(chosen.id)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "huggingface", "hf_id": chosen.id}

    def _plots(self, ctx: StageContext, df: pd.DataFrame, profile: dict) -> None:
        pdir = ctx.project.plots_dir
        plots.present(plots.feature_histograms(df), pdir, "raw_histograms")
        plots.present(plots.missing_matrix(df), pdir, "raw_missing")
        t = profile.get("target")
        if t and t["kind"] == "categorical":
            plots.present(plots.class_balance(t["counts"]), pdir, "raw_class_balance")
        elif t:
            plots.present(plots.target_distribution(df[t["name"]]), pdir, "raw_target_distribution")
        if df.select_dtypes("number").shape[1] >= 2:
            plots.present(plots.correlation_heatmap(df), pdir, "raw_correlation")

    def _narrate(self, ctx: StageContext, spec: dict, profile: dict) -> None:
        prompt = "Project spec:\n" + json.dumps(spec, indent=2) + "\n\nData profile:\n" + json.dumps(profile, indent=2)
        try:
            text = ask_text(ctx.llm, load_prompt("data"), prompt)
        except LLMError as exc:
            text = f"(Couldn't reach Claude for a narrative: {exc}) Data saved. Next: the [[data cleaning]] audit."
        ctx.display(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_stage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/data.py mlagent/prompts/data.md tests/test_data_stage.py
git commit -m "feat: data stage (synthetic, Drive, HuggingFace) with profile, plots, narrative

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 9: Clean stage

**Files:**
- Create: `mlagent/stages/clean.py`, `mlagent/prompts/clean.md`, `tests/test_clean_stage.py`

**Interfaces:**
- Consumes: `META_FILE`, `RAW_FILE` from `stages.data`; `audit_tabular`, `Issue`; `apply_steps`, `describe_step`, `render_clean_py`; `profile_dataframe`; `plots.before_after_missing`, `plots.present`.
- Produces: `CLEAN_FILE = "data.csv"`, `AUDIT_FILE = "audit.json"`, `PROFILE_CLEAN_FILE = "profile_clean.json"`, `CLEAN_PY = "clean.py"`, `class CleanStage` with `name = "clean"`. Artifacts: `data/clean/data.csv`, `clean.py`, `audit.json` = `{"issues": [...], "decisions": [{"kind","column","fix","approved"}], "steps": [...]}`, `profile_clean.json`, `plots/clean_before_after_missing.png`, and `data_meta.json` gains `splits: {"train","val","test"}`, `clean_path`, `dropped_columns`.

- [ ] **Step 1: Write the failing test**

`tests/test_clean_stage.py`:
```python
import json

import numpy as np
import pandas as pd

from mlagent.audit import audit_tabular
from mlagent.llm import FakeLLM
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CLEAN_FILE, CLEAN_PY, PROFILE_CLEAN_FILE, CleanStage
from mlagent.stages.data import META_FILE, RAW_FILE
from mlagent.ui.questions import ScriptedQuestioner

SPEC = {
    "goal": "Predict churn", "task_type": "tabular_classification", "metric": "accuracy",
    "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5, "max_rounds": 2,
    "gpu": "none", "notes": "",
}


def messy_df() -> pd.DataFrame:
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


def prepare(project, df):
    project.write_json("spec.json", SPEC)
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / RAW_FILE, index=False)
    project.write_json(META_FILE, {"source": "synthetic", "target": "target"})


def make_ctx(project, answers, llm=None):
    shown: list[str] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([[("text", "Report card [[missing values]]")]]),
                       questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append)
    return ctx, shown


def test_approved_fixes_are_applied_and_recorded(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    assert n_fixable >= 4
    ctx, shown = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"])
    stage = CleanStage()
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert "row_id" not in cleaned.columns and "const" not in cleaned.columns
    assert cleaned.duplicated().sum() == 0 and cleaned["f1"].isna().sum() == 0
    assert set(cleaned["cat"].unique()) == {"a", "b", "c"}
    audit = project.read_json(AUDIT_FILE)
    assert len(audit["decisions"]) == len(audit["issues"]) and len(audit["steps"]) == n_fixable
    assert all(d["approved"] for d in audit["decisions"] if d["fix"])
    meta = project.read_json(META_FILE)
    assert meta["splits"] == {"train": 0.7, "val": 0.15, "test": 0.15} and meta["dropped_columns"] == []
    assert project.read_json(PROFILE_CLEAN_FILE)["n_rows"] == len(cleaned)
    assert (project.plots_dir / "clean_before_after_missing.png").exists()
    assert any("[[missing values]]" in s for s in shown) and any("Cleaning summary" in s for s in shown)
    namespace: dict = {}
    exec(compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec"), namespace)
    pd.testing.assert_frame_equal(namespace["clean"](df).reset_index(drop=True), cleaned, check_dtype=False)


def test_skipped_fixes_and_manual_drops(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    ctx, _ = make_ctx(project, ["n"] * n_fixable + ["f2, nope", "0.8", "0.1"])
    CleanStage().run(ctx)
    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert "row_id" in cleaned.columns and "f2" not in cleaned.columns
    audit = project.read_json(AUDIT_FILE)
    assert audit["steps"] == [{"op": "drop_columns", "params": {"columns": ["f2"]}}]
    assert project.read_json(META_FILE)["dropped_columns"] == ["f2"]


def test_clean_data_has_no_issues_and_splits_are_adjusted(project):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "target": rng.integers(0, 2, 100)})
    prepare(project, df)
    ctx, shown = make_ctx(project, ["", "0.9", "0.3"])
    CleanStage().run(ctx)
    assert any("no problems" in s for s in shown)
    splits = project.read_json(META_FILE)["splits"]
    assert splits["test"] >= 0.05 and abs(sum(splits.values()) - 1.0) < 1e-6
    assert json.loads((project.root / AUDIT_FILE).read_text(encoding="utf-8"))["issues"] == []


def test_llm_failure_does_not_block_cleaning(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    ctx, shown = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"], llm=FakeLLM([]))
    CleanStage().run(ctx)
    assert any("Couldn't reach Claude" in s for s in shown)
    assert (project.data_clean / CLEAN_FILE).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clean_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.clean'`

- [ ] **Step 3: Implement**

`mlagent/prompts/clean.md`:
```markdown
You are the data-cleaning stage of an ML training assistant that runs inside Google Colab. An automatic audit has found issues in the user's dataset. You receive the issues as JSON (kind, severity, column, message, evidence, proposed fix), plus the project spec and a profile summary.

Write a report card for the user (under 250 words):

- Start with one sentence on the overall state of the data.
- Then one short paragraph per issue, in the order given: what was found, why it matters for THIS project's goal and metric, and whether you agree with the proposed fix (say so if you would do something different, and why).
- End with one sentence telling the user they will now be asked to approve or skip each fix.

Wrap technical terms in double square brackets like [[data leakage]] so the user can click them. Be concrete and plain-spoken; do not restate the raw JSON.
```

`mlagent/stages/clean.py`:
```python
"""Clean stage: audit the raw data, explain the issues, apply approved fixes, record splits."""

from __future__ import annotations

import json

import pandas as pd

from mlagent import plots
from mlagent.audit import Issue, audit_tabular
from mlagent.cleaning import apply_steps, describe_step, render_clean_py
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe
from mlagent.prompts_io import load_prompt
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE, RAW_FILE

CLEAN_FILE = "data.csv"
AUDIT_FILE = "audit.json"
PROFILE_CLEAN_FILE = "profile_clean.json"
CLEAN_PY = "clean.py"
MIN_TEST_FRACTION = 0.05


class CleanStage:
    name = "clean"

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE) or {}
        return (
            (ctx.project.data_clean / CLEAN_FILE).exists()
            and ctx.project.exists(AUDIT_FILE)
            and bool(meta.get("splits"))
        )

    def run(self, ctx: StageContext) -> None:
        meta = ctx.project.read_json(META_FILE) or {}
        target = meta.get("target")
        if not target:
            raise RuntimeError("data_meta.json has no target; run the data stage first")
        df = pd.read_csv(ctx.project.data_raw / RAW_FILE)
        before = profile_dataframe(df, target)

        issues = audit_tabular(df, target)
        decisions = self._review_issues(ctx, issues, before)
        steps = [d["fix"] for d in decisions if d["approved"] and d["fix"]]
        cleaned = apply_steps(df, steps)

        drops = self._ask_drops(ctx, cleaned, target)
        if drops:
            step = {"op": "drop_columns", "params": {"columns": drops}}
            steps.append(step)
            cleaned = apply_steps(cleaned, [step])
        splits = self._ask_splits(ctx)

        ctx.project.data_clean.mkdir(parents=True, exist_ok=True)
        clean_path = ctx.project.data_clean / CLEAN_FILE
        cleaned.to_csv(clean_path, index=False)
        (ctx.project.root / CLEAN_PY).write_text(render_clean_py(steps), encoding="utf-8")
        after = profile_dataframe(cleaned, target)
        ctx.project.write_json(PROFILE_CLEAN_FILE, after)
        ctx.project.write_json(AUDIT_FILE, {
            "issues": [i.to_dict() for i in issues], "decisions": decisions, "steps": steps,
        })
        meta.update({"splits": splits, "clean_path": str(clean_path), "dropped_columns": drops})
        ctx.project.write_json(META_FILE, meta)

        plots.present(plots.before_after_missing(before, after), ctx.project.plots_dir, "clean_before_after_missing")
        ctx.display(self._summary(before, after, steps))

    def _review_issues(self, ctx: StageContext, issues: list[Issue], profile: dict) -> list[dict]:
        if not issues:
            ctx.display("The audit found no problems. Nice and clean.")
            return []
        self._explain(ctx, issues, profile)
        decisions: list[dict] = []
        for issue in issues:
            where = f" in `{issue.column}`" if issue.column else ""
            line = f"**[{issue.severity}] {issue.kind}**{where}: {issue.message}"
            if issue.fix:
                ctx.display(f"{line}\n\nProposed fix: {describe_step(issue.fix)}")
                approved = ctx.questioner.confirm("Apply this fix?", default=issue.severity != "low")
            else:
                ctx.display(f"{line}\n\nNo automatic fix; noted for the modelling stage.")
                approved = False
            decisions.append({"kind": issue.kind, "column": issue.column, "fix": issue.fix, "approved": approved})
        return decisions

    def _explain(self, ctx: StageContext, issues: list[Issue], profile: dict) -> None:
        payload = json.dumps({
            "issues": [i.to_dict() for i in issues],
            "spec": ctx.spec().to_dict(),
            "profile_summary": {
                "n_rows": profile["n_rows"], "n_cols": profile["n_cols"], "target": profile.get("target"),
            },
        }, indent=2, default=str)
        try:
            text = ask_text(ctx.llm, load_prompt("clean"), payload)
        except LLMError as exc:
            text = f"(Couldn't reach Claude for the report card: {exc}) Here are the issues found:"
        ctx.display(text)

    def _ask_drops(self, ctx: StageContext, df: pd.DataFrame, target: str) -> list[str]:
        cols = [str(c) for c in df.columns if c != target]
        raw = ctx.questioner.text(
            "Any other columns to drop before training? (comma-separated names, or leave blank)", default="",
        )
        chosen = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in chosen if c not in cols]
        if unknown:
            ctx.display(f"Ignoring unknown columns: {', '.join(unknown)}")
        return [c for c in chosen if c in cols]

    def _ask_splits(self, ctx: StageContext) -> dict:
        q = ctx.questioner
        train = q.number("Fraction of rows for training?", default=0.7, minimum=0.5, maximum=0.9)
        val = q.number("Fraction for validation (used during tuning)?", default=0.15, minimum=0.05, maximum=0.3)
        test = round(1 - train - val, 4)
        if test < MIN_TEST_FRACTION:
            val = round(max(MIN_TEST_FRACTION, 1 - MIN_TEST_FRACTION - train), 4)
            test = round(1 - train - val, 4)
            ctx.display(
                f"Adjusted so the [[test set]] keeps at least {MIN_TEST_FRACTION:.0%}: "
                f"train {train:g}, validation {val:g}, test {test:g}."
            )
        return {"train": round(train, 4), "val": round(val, 4), "test": test}

    def _summary(self, before: dict, after: dict, steps: list[dict]) -> str:
        lines = [
            "### Cleaning summary",
            "",
            f"- Rows: {before['n_rows']} → {after['n_rows']}",
            f"- Columns: {before['n_cols']} → {after['n_cols']}",
            f"- Steps applied: {len(steps)}",
        ]
        lines.extend(f"  - {describe_step(s)}" for s in steps)
        lines += [
            "",
            "Cleaned data saved to `data/clean/data.csv`; the steps are in `clean.py` so they can be re-run "
            "on new data. Next: generating the [[training pipeline]].",
        ]
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clean_stage.py -v`
Expected: 4 passed. Note for `test_approved_fixes_are_applied_and_recorded`: the reproduction check compares `clean(df)` on the original frame with the saved CSV; `check_dtype=False` tolerates CSV round-trip dtype changes.

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/clean.py mlagent/prompts/clean.md tests/test_clean_stage.py
git commit -m "feat: clean stage with audit report card, approved fixes, clean.py, and splits

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 10: Wire stages into Colab, docs

**Files:**
- Modify: `mlagent/colab.py`, `tests/test_colab.py`, `docs/colab-smoke.md`, `CLAUDE.md`

**Interfaces:**
- `colab.start` returns an `Orchestrator` over `[IntakeStage(), DataStage(), CleanStage()]`; `_context_snapshot` also includes `data_meta` (from `data_meta.json`) and `audit` summary (issue kinds from `audit.json`, if present) so explanations can reference the current data.

- [ ] **Step 1: Write the failing test**

In `tests/test_colab.py`, change the stage-name assertion in `test_make_context_and_start` to:
```python
    assert [s.name for s in orch.stages] == ["intake", "data", "clean"]
```
and add:
```python
def test_context_snapshot_includes_data_meta_and_audit(tmp_path):
    ctx = colab.make_context("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    ctx.project.write_json("data_meta.json", {"target": "y"})
    ctx.project.write_json("audit.json", {"issues": [{"kind": "missing_values"}], "decisions": [], "steps": []})
    snap = ctx.explainer.context_provider()
    assert snap["data_meta"] == {"target": "y"}
    assert snap["audit_issue_kinds"] == ["missing_values"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_colab.py -v`
Expected: 2 failures (stage list, missing snapshot keys).

- [ ] **Step 3: Implement**

`mlagent/colab.py`: import `DataStage` from `mlagent.stages.data` and `CleanStage` from `mlagent.stages.clean`; in `start` build `Orchestrator(ctx, [IntakeStage(), DataStage(), CleanStage()])`; extend `_context_snapshot` to return, in addition to the existing keys:
```python
        "data_meta": project.read_json("data_meta.json"),
        "audit_issue_kinds": [i.get("kind") for i in (project.read_json("audit.json") or {}).get("issues", [])],
```

`docs/colab-smoke.md` — append:
```markdown
## Milestone 2 (Tabular data + cleaning)
- [ ] Fresh project, intake with data source "Synthetic data": the data stage asks rows/features/classes/balance/noise/quirks and shows a profile table, histograms, missing-value matrix, class balance, correlation heatmap, and a narrative with clickable terms.
- [ ] The clean stage shows a report card, then asks to approve each fix (ID column, duplicates, missing values, messy categories, outliers). Approve all; the summary shows fewer rows/columns; `projects/<name>/clean.py`, `data/clean/data.csv`, `audit.json` exist on Drive.
- [ ] Split question: enter 0.9 and 0.3 and see the adjustment message.
- [ ] New project with data source "Upload or Google Drive path": upload a CSV to Drive, confirm it appears in the file list, pick it, pick the target column.
- [ ] New project with data source "HuggingFace Hub dataset": search "iris" (or any small tabular set), pick one, pick the target; the audit runs on it.
- [ ] Runtime reset after the data stage: `orch.run()` skips intake and data, resumes at clean.
- [ ] Click a term inside the report card; the explanation mentions the current dataset (target column or issue).
```

`CLAUDE.md` — in Architecture, after the orchestrator bullet, add:
```markdown
- Pipeline so far: `intake` → `data` → `clean` (`mlagent/stages/`). `data` writes `data/raw/data.csv`, `profile_raw.json`, `data_meta.json` (source, target). `clean` runs `audit.py`, lets the user approve fixes, writes `data/clean/data.csv`, `clean.py`, `audit.json`, `profile_clean.json`, and adds `splits` to `data_meta.json`. Raw data is never modified.
- Pure modules do the work and are unit-tested without stages: `synth/tabular.py`, `profile.py`, `audit.py` (checks → `Issue` with a proposed fix dict), `cleaning.py` (fix dicts → `apply_steps`; `render_clean_py` writes a re-runnable script that calls the same function), `plots.py` (all matplotlib figures; `present()` saves to `plots/` and displays under IPython), `datasources/drive.py`, `datasources/hf.py` (HuggingFace calls are lazy imports and injectable; tests never hit the network).
```
In Commands, add: `python -m pytest -W error::DeprecationWarning tests/test_plots.py   # keep chart output warning-free`.

- [ ] **Step 4: Run the full suite and lint**

Run: `python -m pytest -v && ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/colab.py tests/test_colab.py docs/colab-smoke.md CLAUDE.md
git commit -m "feat: wire data and clean stages into the Colab entry point; docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

## Milestone verification

1. `python -m pytest` green and `ruff check .` clean (with the expanded rule set).
2. Milestone 2 section of `docs/colab-smoke.md` run in a real Colab session (user).
3. Milestone 3 (tabular training) will read `data_meta.json` (`target`, `splits`, `clean_path`) and `data/clean/data.csv`; the `DataStage`/`CleanStage` artifact names above are the contract.
