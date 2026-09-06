# ML Training Agent — Milestone 3 (Tabular Training End-to-End) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After cleaning, the assistant writes a real, self-contained training project (`data.py`, `model.py`, `train.py`, `config.json`) onto the project folder, runs it in a subprocess with a live loss/metric plot, records the run in `runs.jsonl`, shows evaluation plots on the validation split, and, when the user is done, evaluates once on the held-out test split and writes `report.md` with figures.

**Architecture:** Three new stages (`codegen`, `train`, `report`) plug into the existing `Orchestrator`/`StageContext`. Pure modules do the work: `templates/tabular_sklearn/` (the generated files, copied verbatim and parameterised only by `config.json`), `templates_io.py` (template copy, config schema, validation), `runner.py` (subprocess execution with stdout streaming and `metrics.json` polling), `runlog.py` (`runs.jsonl`), and new figure functions in `plots.py`. The LLM proposes starting hyperparameters through a tool call bounded by the template's config schema and writes short narratives; it never edits code in this milestone.

**Tech Stack:** scikit-learn `HistGradientBoosting{Classifier,Regressor}` with `warm_start=True` (so "epochs" are boosting rounds and real train/validation loss curves exist), joblib (ships with scikit-learn) for checkpoints, pandas, numpy, matplotlib (Agg in tests), subprocess + threading for the runner.

**Spec:** `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md` (sections "Stage pipeline" items 4, 5 and 7, "Plots", "Cost gate" (skipped for CPU), "Error handling", "Testing").

## Global Constraints

- Python `>=3.10`; all tests run with `python -m pytest`; `ruff check .` must pass (`E, F, W, I, B, UP`, line length 100). Template files under `mlagent/templates/` are linted too.
- No network in tests. LLM calls use `FakeLLM`. Training in tests runs for real on tiny data (a few hundred rows) on CPU and must finish in seconds.
- Every prompt lives in `mlagent/prompts/*.md`, loaded with `load_prompt`; never inline prompts in Python.
- User-visible text may contain `[[term]]` markup for click-to-explain.
- Raw data is never modified. Generated training code reads `data/clean/data.csv` through `data_meta.json` (`clean_path`).
- The generated project (`data.py`, `model.py`, `train.py`) must import only the standard library, numpy, pandas, scikit-learn and joblib. It never imports `mlagent`, so it runs anywhere the data folder exists.
- Only `config.json` is tunable. Its keys are exactly those in `templates/tabular_sklearn/config_schema.json`; the tuner in Milestone 4 will be bounded by that schema.
- Charts: static matplotlib; palette constants in `plots.py` (series `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`, sequential blue ramp, diverging blue–neutral–orange); single-series charts have no legend, multi-series charts always do; text uses ink colours, never series colours; every figure goes through `present()` which saves to `plots/<name>.png`, displays only under IPython, and closes the figure.
- Windows dev machine: `pathlib`, `encoding="utf-8"`, `sys.executable` for subprocesses. Project conventions: `from __future__ import annotations`, `from collections.abc import Callable`.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2
  ```

## Rulings made while planning

- **One template family for tabular.** The spec lists `tabular_sklearn` and `tabular_torch`. Torch is not a dependency and gradient boosting with warm start gives genuine per-epoch curves, so Milestone 3 ships `tabular_sklearn` only; `tabular_torch` is deferred. Cost if wrong: a second template later.
- **Claude adapts configuration, not code.** The spec says the template is "adapted by Claude". Here Claude proposes `config.json` values within the schema and explains them; the code files are copied verbatim. The traceback-to-fix-diff loop is Milestone 4 work alongside tuning, where generated edits first become possible. Cost if wrong: less flexible first runs.
- **No tune stage yet.** Pipeline order is `intake -> data -> clean -> codegen -> train -> report`. Milestone 4 inserts `tune` between `train` and `report`. `TrainStage.is_complete` is "at least one run with status done" so a later `reset("train")` reruns training.
- **Cost gate skipped.** The spec says the gate is GPU-only; tabular runs are CPU. The train stage says so in one line. `train.py --dry-run` is implemented now (cheap) so Milestone 5 can time it.
- **Test split touched once.** `train.py` never evaluates on test. `train.py --eval-test` loads the checkpoint and writes `eval_test.json`; only `ReportStage` calls it, after a confirm.
- **`data_meta.json` contract is frozen** as the table in Task 2's Interfaces block and copied into `CLAUDE.md` in Task 10.

## File Structure

| File | Responsibility |
|---|---|
| `mlagent/config.py` (modify) | Add `CONFIG_FILE`, `RUNS_FILE`, `METRICS_FILE`, `REPORT_FILE` constants. |
| `mlagent/project.py` (modify) | Add `config_path`, `runs_path`, `metrics_path`, `report_path` properties. |
| `mlagent/runlog.py` (new) | `runs.jsonl` read/append, best run, markdown summary, `HIGHER_IS_BETTER`. |
| `mlagent/templates/tabular_sklearn/data.py` (new) | Standalone: load clean CSV via meta, encode categoricals, deterministic splits, validation. |
| `mlagent/templates/tabular_sklearn/model.py` (new) | Standalone: build HGB model from config; `grow()` adds rounds. |
| `mlagent/templates/tabular_sklearn/train.py` (new) | Standalone CLI: epoch loop, `metrics.json` per epoch, best checkpoint, `eval_val.json`, `--dry-run`, `--eval-test`. |
| `mlagent/templates/tabular_sklearn/config_schema.json` (new) | Tunable keys with type, default, min, max, description. |
| `mlagent/templates_io.py` (new) | Template lookup/copy, schema load, `default_config`, `validate_config`, `coerce_config`. |
| `mlagent/runner.py` (new) | Run a script in the project folder: stream stdout, poll `metrics.json`, timeout, `RunResult`. |
| `mlagent/plots.py` (modify) | `training_curves`, `confusion_matrix_plot`, `roc_pr_curves`, `per_class_bars`, `predicted_vs_actual`, `residual_plots`, `present_evaluation`. |
| `mlagent/stages/codegen.py` (new) | Validate data, copy template, LLM-proposed config via tool, summary, confirm/edit. |
| `mlagent/stages/train.py` (new) | Run training with live plot, log run, evaluation plots, narrative. |
| `mlagent/stages/report.py` (new) | Confirm, `--eval-test`, test plots, lessons narrative, `report.md`. |
| `mlagent/prompts/{codegen,train,report}.md` (new) | System prompts. |
| `mlagent/colab.py` (modify) | Wire the three stages; snapshot adds config and latest run. |
| `scripts/build_notebook.py`, `docs/colab-smoke.md`, `CLAUDE.md` (modify) | Retrain cell, M3 smoke checklist, architecture notes. |
| `tests/conftest.py` (modify) | `clean_project` / `regression_project` fixtures. |
| `tests/test_runlog.py`, `test_template_data.py`, `test_template_train.py`, `test_runner.py`, `test_plots_training.py`, `test_templates_io.py`, `test_codegen_stage.py`, `test_train_stage.py`, `test_report_stage.py` (new), `tests/test_pipeline_e2e.py` (modify) | Tests. |

---

### Task 1: Run log and project paths

**Files:**
- Modify: `mlagent/config.py`
- Modify: `mlagent/project.py`
- Create: `mlagent/runlog.py`
- Test: `tests/test_runlog.py`

**Interfaces:**
- Consumes: `Project` (root, `read_json`, `write_json`), `config` constants pattern.
- Produces: `config.CONFIG_FILE = "config.json"`, `config.RUNS_FILE = "runs.jsonl"`, `config.METRICS_FILE = "metrics.json"`, `config.REPORT_FILE = "report.md"`; `Project.config_path/runs_path/metrics_path/report_path -> Path`; `runlog.HIGHER_IS_BETTER: dict[str, bool]`, `read_runs(path) -> list[dict]`, `append_run(path, entry) -> dict`, `is_better(metric, candidate, incumbent) -> bool`, `best_run(runs, metric) -> dict | None`, `summarise(runs, metric) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runlog.py
from __future__ import annotations

from mlagent import runlog


def test_append_and_read_round_trip(tmp_path):
    path = tmp_path / "runs.jsonl"
    assert runlog.read_runs(path) == []
    first = runlog.append_run(path, {"status": "done", "best_val_metric": 0.8})
    second = runlog.append_run(path, {"status": "failed"})
    assert first["run_id"] == 1 and second["run_id"] == 2
    runs = runlog.read_runs(path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["best_val_metric"] == 0.8


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "runs.jsonl"
    runlog.append_run(path, {"status": "done"})
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    runlog.append_run(path, {"status": "done"})
    assert [r["run_id"] for r in runlog.read_runs(path)] == [1, 2]


def test_best_run_respects_metric_direction():
    runs = [
        {"run_id": 1, "status": "done", "best_val_metric": 0.7},
        {"run_id": 2, "status": "done", "best_val_metric": 0.9},
        {"run_id": 3, "status": "failed", "best_val_metric": 0.99},
    ]
    assert runlog.best_run(runs, "accuracy")["run_id"] == 2
    assert runlog.best_run(runs, "rmse")["run_id"] == 1
    assert runlog.best_run([], "accuracy") is None


def test_is_better():
    assert runlog.is_better("accuracy", 0.9, 0.8)
    assert not runlog.is_better("accuracy", 0.8, 0.9)
    assert runlog.is_better("rmse", 1.0, 2.0)
    assert runlog.is_better("rmse", 1.0, None)
    assert not runlog.is_better("rmse", None, 1.0)


def test_summarise_markdown_table():
    runs = [{"run_id": 1, "status": "done", "epochs_run": 5, "best_epoch": 4,
             "best_val_metric": 0.8123456, "final_train_loss": 0.3, "final_val_loss": 0.4,
             "seconds": 2.5}]
    table = runlog.summarise(runs, "accuracy")
    assert table.splitlines()[0].startswith("| run | status |")
    assert "val accuracy" in table
    assert "| 1 | done | 5 | 4 | 0.8123 |" in table
    assert runlog.summarise([], "accuracy") == "_No runs yet._"


def test_project_paths(project):
    assert project.config_path == project.root / "config.json"
    assert project.runs_path == project.root / "runs.jsonl"
    assert project.metrics_path == project.root / "metrics.json"
    assert project.report_path == project.root / "report.md"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runlog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.runlog'`.

- [ ] **Step 3: Add constants and project paths**

Append to `mlagent/config.py` after the existing `GLOSSARY_FILE` line:

```python
CONFIG_FILE = "config.json"
RUNS_FILE = "runs.jsonl"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"
```

Add to `class Project` in `mlagent/project.py`, directly after the `glossary_path` property:

```python
    @property
    def config_path(self) -> Path:
        return self.root / config.CONFIG_FILE

    @property
    def runs_path(self) -> Path:
        return self.root / config.RUNS_FILE

    @property
    def metrics_path(self) -> Path:
        return self.root / config.METRICS_FILE

    @property
    def report_path(self) -> Path:
        return self.root / config.REPORT_FILE
```

- [ ] **Step 4: Write `mlagent/runlog.py`**

```python
"""Append-only run history (runs.jsonl) and helpers to summarise it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Which direction is "better" for each metric the intake stage can choose.
HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "f1": True,
    "r2": True,
    "rmse": False,
    "mae": False,
}


def read_runs(path: Path) -> list[dict]:
    """Return every parseable JSON object line; a corrupt line is skipped, not fatal."""
    if not path.exists():
        return []
    runs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            runs.append(entry)
    return runs


def append_run(path: Path, entry: dict[str, Any]) -> dict:
    """Append one run. `run_id` defaults to the number of existing runs plus one."""
    runs = read_runs(path)
    entry = dict(entry)
    entry.setdefault("run_id", len(runs) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def is_better(metric: str, candidate: float | None, incumbent: float | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if HIGHER_IS_BETTER.get(metric, True):
        return candidate > incumbent
    return candidate < incumbent


def best_run(runs: list[dict], metric: str) -> dict | None:
    best: dict | None = None
    for run in runs:
        if run.get("status") != "done":
            continue
        if best is None or is_better(metric, run.get("best_val_metric"), best.get("best_val_metric")):
            best = run
    return best


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


_COLUMNS = (
    "run_id",
    "status",
    "epochs_run",
    "best_epoch",
    "best_val_metric",
    "final_train_loss",
    "final_val_loss",
    "seconds",
)


def summarise(runs: list[dict], metric: str) -> str:
    """Markdown table: run, status, epochs, best epoch, best val metric, final losses, seconds."""
    if not runs:
        return "_No runs yet._"
    header = (
        f"| run | status | epochs | best epoch | val {metric} | train loss | val loss | seconds |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = ["| " + " | ".join(_fmt(r.get(k)) for k in _COLUMNS) + " |" for r in runs]
    return header + "\n" + "\n".join(rows)
```

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass (139 existing + 6 new), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: run log (runs.jsonl) and project paths for training artifacts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 2: Template `data.py`, `model.py`, config schema, and test fixtures

**Files:**
- Create: `mlagent/templates/tabular_sklearn/data.py`
- Create: `mlagent/templates/tabular_sklearn/model.py`
- Create: `mlagent/templates/tabular_sklearn/config_schema.json`
- Modify: `tests/conftest.py`
- Test: `tests/test_template_data.py`

**Interfaces:**
- Consumes the frozen `data_meta.json` contract written by Milestone 2:

| key | type | written by | meaning |
|---|---|---|---|
| `target` | str | data | target column name |
| `task_type` | str | data | `tabular_classification` or `tabular_regression` |
| `source` | str | data | `synthetic`, `drive` or `huggingface` |
| `raw_path` | str | data | posix path relative to project root |
| `raw_n_rows`, `raw_n_cols` | int | data | raw shape |
| `clean_path` | str | clean | posix path relative to project root (`data/clean/data.csv`) |
| `clean_n_rows`, `clean_n_cols` | int | clean | clean shape |
| `dropped_columns` | list[str] | clean | columns removed |
| `feature_columns` | list[str] | clean | model inputs, in order |
| `categorical_columns` | list[str] | clean | subset of `feature_columns` |
| `splits` | `{"train": f, "val": f, "test": f}` | clean | fractions summing to 1 |
| `n_classes`, `class_labels` | int, list[str] | clean, classification only | sorted stringified labels |

- Produces (standalone, importable from the project folder): `data.load_data(project_dir, config) -> dict` with keys `X_train, X_val, X_test` (pandas DataFrames, float columns), `y_train, y_val, y_test` (numpy arrays; int class indices for classification, float for regression), `feature_names: list[str]`, `categorical_mask: list[bool]`, `classes: list[str] | None`, `task_type: str`; `data.validate(data) -> None` (raises `ValueError`); `model.build_model(config, task_type, categorical_mask)`; `model.grow(model, iters)`; `config_schema.json`; fixtures `clean_project` and `regression_project` in `tests/conftest.py`.

- [ ] **Step 1: Add fixtures to `tests/conftest.py`**

Append to the existing file (keep the `matplotlib.use("Agg")` lines first):

```python
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEATURES = [f"f{i}" for i in range(5)]


def write_clean_project(project: Project, task_type: str, n_rows: int = 240, seed: int = 0) -> Project:
    """Write data/clean/data.csv, data_meta.json and spec.json as Milestone 2 leaves them."""
    from sklearn.datasets import make_classification, make_regression

    rng = np.random.default_rng(seed)
    if task_type == "tabular_classification":
        X, y = make_classification(
            n_samples=n_rows, n_features=5, n_informative=3, n_redundant=0, random_state=seed
        )
        metric = "accuracy"
    else:
        X, y = make_regression(
            n_samples=n_rows, n_features=5, n_informative=3, noise=5.0, random_state=seed
        )
        metric = "rmse"
    df = pd.DataFrame(X, columns=FEATURES)
    df["colour"] = rng.choice(["red", "green", "blue"], size=n_rows)
    df["target"] = y
    project.ensure_dirs()
    df.to_csv(project.data_clean / "data.csv", index=False)
    meta = {
        "target": "target",
        "task_type": task_type,
        "source": "synthetic",
        "raw_path": "data/raw/data.csv",
        "raw_n_rows": n_rows,
        "raw_n_cols": 7,
        "clean_path": "data/clean/data.csv",
        "clean_n_rows": n_rows,
        "clean_n_cols": 7,
        "dropped_columns": [],
        "feature_columns": FEATURES + ["colour"],
        "categorical_columns": ["colour"],
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    if task_type == "tabular_classification":
        meta["n_classes"] = 2
        meta["class_labels"] = ["0", "1"]
    project.write_json("data_meta.json", meta)
    project.write_json(
        "spec.json",
        {
            "goal": "fixture project",
            "task_type": task_type,
            "metric": metric,
            "target_value": 0.8,
            "data_source": "synthetic",
            "minutes_per_run": 5,
            "max_rounds": 3,
            "gpu": "none",
            "notes": "",
        },
    )
    return project


@pytest.fixture
def clean_project(project: Project) -> Project:
    return write_clean_project(project, "tabular_classification")


@pytest.fixture
def regression_project(project: Project) -> Project:
    return write_clean_project(project, "tabular_regression")
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_template_data.py
"""The generated data.py/model.py are plain scripts; load them from the template folder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_load_data_classification_splits_and_encoding(clean_project):
    data = load_module("data")
    out = data.load_data(clean_project.root, {"seed": 1})
    n = 240
    assert len(out["X_train"]) + len(out["X_val"]) + len(out["X_test"]) == n
    assert abs(len(out["X_test"]) - 0.15 * n) <= 2
    assert out["feature_names"][-1] == "colour"
    assert out["categorical_mask"] == [False] * 5 + [True]
    assert out["classes"] == ["0", "1"]
    assert set(np.unique(out["y_train"])) == {0, 1}
    # categorical encoded as small non-negative codes (or NaN), all columns float
    codes = out["X_train"]["colour"].dropna().unique()
    assert set(codes) <= {0.0, 1.0, 2.0}
    assert all(dtype == float for dtype in out["X_train"].dtypes)
    # deterministic for a seed, different for another
    again = data.load_data(clean_project.root, {"seed": 1})
    pd.testing.assert_frame_equal(out["X_test"], again["X_test"])
    other = data.load_data(clean_project.root, {"seed": 2})
    assert not out["X_test"].index.equals(other["X_test"].index)


def test_load_data_regression(regression_project):
    data = load_module("data")
    out = data.load_data(regression_project.root, {"seed": 1})
    assert out["classes"] is None
    assert out["y_train"].dtype == float
    assert out["task_type"] == "tabular_regression"


def test_unseen_category_becomes_nan(clean_project):
    data = load_module("data")
    df = pd.DataFrame({"colour": ["red", "blue"]})
    encoded, cats, mask = data.encode_features(df, ["colour"], {"colour": ["blue"]})
    assert encoded["colour"].tolist()[1] == 0.0
    assert np.isnan(encoded["colour"].tolist()[0])
    assert mask == [True]


def test_validate_rejects_overlap_and_single_class():
    data = load_module("data")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    good = {"X_train": X.iloc[:2], "X_val": X.iloc[2:], "X_test": X.iloc[2:3],
            "y_train": np.array([0, 1]), "classes": ["0", "1"]}
    with pytest.raises(ValueError, match="overlap"):
        data.validate(good)
    bad = {"X_train": X.iloc[:2], "X_val": X.iloc[2:3], "X_test": X.iloc[1:2],
           "y_train": np.array([0, 0]), "classes": ["0", "1"]}
    with pytest.raises(ValueError):
        data.validate(bad)


def test_build_model_and_grow(clean_project):
    data = load_module("data")
    model = load_module("model")
    out = data.load_data(clean_project.root, {"seed": 1})
    est = model.build_model({"learning_rate": 0.2, "iters_per_epoch": 3, "seed": 1},
                            "tabular_classification", out["categorical_mask"])
    est.fit(out["X_train"], out["y_train"])
    assert est.n_iter_ == 3
    model.grow(est, 2)
    est.fit(out["X_train"], out["y_train"])
    assert est.n_iter_ == 5
    reg = model.build_model({}, "tabular_regression", [False] * 6)
    assert type(reg).__name__ == "HistGradientBoostingRegressor"
    with pytest.raises(ValueError):
        model.build_model({}, "image_classification", [])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_template_data.py -v`
Expected: FAIL (template files do not exist).

- [ ] **Step 4: Write `mlagent/templates/tabular_sklearn/data.py`**

```python
"""Load the cleaned dataset and split it. Generated by mlagent; safe to edit.

Reads data_meta.json (target, task type, feature and categorical columns, split
fractions) and data/clean/data.csv. Categorical columns are ordinal-encoded using
the categories seen in the training split; unseen values become NaN, which
HistGradientBoosting handles natively.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

META_FILE = "data_meta.json"
MAX_CATEGORY_CODES = 200  # above this a column is treated as plain numeric codes


def read_meta(project_dir: Path) -> dict:
    return json.loads((project_dir / META_FILE).read_text(encoding="utf-8"))


def _encode_column(series: pd.Series, categories: list[str]) -> pd.Series:
    mapping = {value: float(i) for i, value in enumerate(categories)}
    values = [mapping.get(str(v), np.nan) if pd.notna(v) else np.nan for v in series.tolist()]
    return pd.Series(values, index=series.index, dtype=float)


def encode_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    categories: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[bool]]:
    """Return (encoded frame, categories per column, categorical mask per column).

    Pass the `categories` returned for the training split when encoding validation
    and test splits so codes line up.
    """
    out = df.copy()
    cats: dict[str, list[str]] = dict(categories or {})
    for col in out.columns:
        if col in categorical_columns:
            if col not in cats:
                cats[col] = sorted(out[col].dropna().astype(str).unique().tolist())
            out[col] = _encode_column(out[col], cats[col])
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    mask = [
        col in categorical_columns and len(cats.get(col, [])) <= MAX_CATEGORY_CODES
        for col in out.columns
    ]
    return out, cats, mask


def _split(X: pd.DataFrame, y: pd.Series, fraction: float, seed: int, stratify: bool):
    strat = y if stratify and y.value_counts().min() >= 2 else None
    try:
        return train_test_split(X, y, test_size=fraction, random_state=seed, stratify=strat)
    except ValueError:
        return train_test_split(X, y, test_size=fraction, random_state=seed)


def load_data(project_dir: Path | str, config: dict) -> dict:
    project_dir = Path(project_dir)
    meta = read_meta(project_dir)
    df = pd.read_csv(project_dir / meta["clean_path"])
    target = meta["target"]
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in {meta['clean_path']}")
    df = df[df[target].notna()].reset_index(drop=True)
    wanted = meta.get("feature_columns") or [c for c in df.columns if c != target]
    features = [c for c in wanted if c in df.columns and c != target]
    if not features:
        raise ValueError("no feature columns available")
    categorical = [c for c in meta.get("categorical_columns", []) if c in features]
    task_type = meta["task_type"]

    X = df[features]
    y = df[target]
    classes: list[str] | None = None
    if task_type == "tabular_classification":
        classes = [str(c) for c in (meta.get("class_labels") or sorted(y.astype(str).unique()))]
        index = {label: i for i, label in enumerate(classes)}
        y = y.astype(str).map(index)
        if y.isna().any():
            raise ValueError("target has labels outside class_labels in data_meta.json")
        y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="raise").astype(float)

    splits = meta["splits"]
    seed = int(config.get("seed", 42))
    test_frac = float(splits["test"])
    val_frac = float(splits["val"]) / max(1e-9, 1.0 - test_frac)
    X_tv, X_test, y_tv, y_test = _split(X, y, test_frac, seed, classes is not None)
    X_train, X_val, y_train, y_val = _split(X_tv, y_tv, val_frac, seed, classes is not None)

    X_train_e, cats, mask = encode_features(X_train, categorical)
    X_val_e, _, _ = encode_features(X_val, categorical, cats)
    X_test_e, _, _ = encode_features(X_test, categorical, cats)

    data = {
        "X_train": X_train_e,
        "y_train": y_train.to_numpy(),
        "X_val": X_val_e,
        "y_val": y_val.to_numpy(),
        "X_test": X_test_e,
        "y_test": y_test.to_numpy(),
        "feature_names": features,
        "categorical_mask": mask,
        "classes": classes,
        "task_type": task_type,
    }
    validate(data)
    return data


def validate(data: dict) -> None:
    """Raise ValueError if the splits are unusable: empty, overlapping, or one class."""
    for name in ("X_train", "X_val", "X_test"):
        if len(data[name]) == 0:
            raise ValueError(f"{name} is empty; adjust the split fractions")
    idx = {name: set(data[name].index) for name in ("X_train", "X_val", "X_test")}
    if (idx["X_train"] & idx["X_val"]) or (idx["X_train"] & idx["X_test"]) or (
        idx["X_val"] & idx["X_test"]
    ):
        raise ValueError("splits overlap: the same row appears in two splits")
    if data.get("classes") is not None and len(np.unique(data["y_train"])) < 2:
        raise ValueError("training split contains fewer than 2 classes")
```

- [ ] **Step 5: Write `mlagent/templates/tabular_sklearn/model.py`**

```python
"""Build the model. Generated by mlagent; safe to edit.

HistGradientBoosting with warm_start=True: each "epoch" in train.py adds
`iters_per_epoch` boosting rounds, so train/validation loss can be tracked per epoch.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def build_model(config: dict, task_type: str, categorical_mask: list[bool]):
    params = {
        "learning_rate": float(config.get("learning_rate", 0.1)),
        "max_iter": int(config.get("iters_per_epoch", 10)),
        "max_leaf_nodes": int(config.get("max_leaf_nodes", 31)),
        "max_depth": config.get("max_depth"),
        "min_samples_leaf": int(config.get("min_samples_leaf", 20)),
        "l2_regularization": float(config.get("l2_regularization", 0.0)),
        "categorical_features": list(categorical_mask) if any(categorical_mask) else None,
        "warm_start": True,
        "early_stopping": False,
        "random_state": int(config.get("seed", 42)),
    }
    if task_type == "tabular_classification":
        return HistGradientBoostingClassifier(**params)
    if task_type == "tabular_regression":
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"unsupported task_type {task_type!r} for the tabular_sklearn template")


def grow(model, iters: int):
    """Add `iters` boosting rounds on the next fit() call (warm start keeps earlier rounds)."""
    model.set_params(max_iter=int(model.max_iter) + int(iters))
    return model
```

- [ ] **Step 6: Write `mlagent/templates/tabular_sklearn/config_schema.json`**

```json
{
  "learning_rate": {"type": "number", "default": 0.1, "min": 0.001, "max": 1.0,
    "description": "Shrinkage applied to each boosting round; lower is slower but steadier."},
  "epochs": {"type": "integer", "default": 10, "min": 1, "max": 100,
    "description": "Number of epochs; each adds iters_per_epoch boosting rounds."},
  "iters_per_epoch": {"type": "integer", "default": 10, "min": 1, "max": 100,
    "description": "Boosting rounds (trees) added per epoch."},
  "max_leaf_nodes": {"type": "integer", "default": 31, "min": 2, "max": 255,
    "description": "Maximum leaves per tree; more leaves fit more detail."},
  "max_depth": {"type": "integer", "default": null, "min": 1, "max": 32, "nullable": true,
    "description": "Maximum tree depth; null means unlimited."},
  "min_samples_leaf": {"type": "integer", "default": 20, "min": 1, "max": 200,
    "description": "Minimum rows per leaf; higher values regularise."},
  "l2_regularization": {"type": "number", "default": 0.0, "min": 0.0, "max": 10.0,
    "description": "L2 penalty on leaf values; higher values regularise."},
  "early_stopping_patience": {"type": "integer", "default": 5, "min": 0, "max": 50,
    "description": "Stop when validation has not improved for this many epochs; 0 disables."},
  "seed": {"type": "integer", "default": 42, "min": 0, "max": 1000000,
    "description": "Random seed for the split and the model."}
}
```

- [ ] **Step 7: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean (note ruff lints the template files; keep imports sorted).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: tabular_sklearn template data.py, model.py and config schema; clean-project fixtures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 3: Template `train.py`

**Files:**
- Create: `mlagent/templates/tabular_sklearn/train.py`
- Test: `tests/test_template_train.py`

**Interfaces:**
- Consumes: `data.load_data`, `model.build_model`, `model.grow` (Task 2); `config.json`, `spec.json` (`metric`), `data_meta.json` in the project folder.
- Produces, when run as `python train.py` with cwd = project folder:
  - `metrics.json`, rewritten atomically after every epoch: `{"status": "running"|"done"|"failed", "task_type", "metric", "higher_is_better", "config", "classes", "n_train", "n_val", "n_test", "epochs": [{"epoch", "train_loss", "val_loss", "train_metric", "val_metric", "seconds"}], "best_epoch", "best_val_metric", "stopped_early", "error", "seconds", "seconds_per_epoch"}`.
  - `checkpoints/best.joblib` (model snapshot at the best validation epoch).
  - `eval_val.json`: `{"split": "val", "task_type", "metric", "value", "loss", "classes", "y_true": [...], "y_pred": [...], "y_proba": [[...]] | null}`.
  - `python train.py --dry-run` trains one epoch of one round, writes nothing, prints one JSON line `{"seconds_per_epoch": float, "n_train": int}`.
  - `python train.py --eval-test` loads the checkpoint, evaluates on the test split, writes `eval_test.json` (same shape, `"split": "test"`).
  - Exit code 0 on success, 1 on failure (with `metrics.json` `status: "failed"` and `error` set when the failure happens inside training).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_template_train.py
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
CODE_FILES = ("data.py", "model.py", "train.py")


def install(project, config: dict) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "train.py", *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=120,
    )


SMALL = {"epochs": 4, "iters_per_epoch": 3, "learning_rate": 0.2, "seed": 1,
         "early_stopping_patience": 0}


def test_classification_run_writes_metrics_checkpoint_and_eval(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done"
    assert metrics["metric"] == "accuracy" and metrics["higher_is_better"] is True
    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2, 3, 4]
    assert all(e["train_loss"] > 0 and e["val_loss"] > 0 for e in metrics["epochs"])
    assert 1 <= metrics["best_epoch"] <= 4
    assert metrics["best_val_metric"] == max(e["val_metric"] for e in metrics["epochs"])
    assert metrics["seconds_per_epoch"] > 0
    assert (root / "checkpoints" / "best.joblib").exists()
    ev = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert ev["split"] == "val" and ev["task_type"] == "tabular_classification"
    assert len(ev["y_true"]) == len(ev["y_pred"]) == len(ev["y_proba"]) == metrics["n_val"]
    assert len(ev["y_proba"][0]) == 2
    assert "epoch 4/4" in proc.stdout
    assert not (root / "eval_test.json").exists()


def test_regression_run_and_eval_test(regression_project):
    root = install(regression_project, SMALL)
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric"] == "rmse" and metrics["higher_is_better"] is False
    assert metrics["best_val_metric"] == min(e["val_metric"] for e in metrics["epochs"])
    proc = run(root, "--eval-test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ev = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert ev["split"] == "test" and ev["y_proba"] is None
    assert len(ev["y_true"]) == metrics["n_test"]
    assert ev["value"] > 0


def test_early_stopping_stops_before_all_epochs(clean_project):
    # A huge learning rate on a tiny model plateaus quickly; patience 1 must stop early.
    root = install(clean_project, {**SMALL, "epochs": 30, "learning_rate": 1.0,
                                   "max_leaf_nodes": 2, "early_stopping_patience": 1})
    assert run(root).returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stopped_early"] is True
    assert len(metrics["epochs"]) < 30


def test_dry_run_prints_timing_and_writes_nothing(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = json.loads(proc.stdout.strip().splitlines()[-1])
    assert line["seconds_per_epoch"] >= 0 and line["n_train"] > 0
    assert not (root / "metrics.json").exists()
    assert not (root / "checkpoints" / "best.joblib").exists()


def test_failure_is_recorded_in_metrics(clean_project):
    root = install(clean_project, SMALL)
    meta = clean_project.read_json("data_meta.json")
    meta["target"] = "missing_column"
    clean_project.write_json("data_meta.json", meta)
    proc = run(root)
    assert proc.returncode == 1
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert "missing_column" in metrics["error"]


def test_eval_test_without_checkpoint_fails_clearly(clean_project):
    root = install(clean_project, SMALL)
    proc = run(root, "--eval-test")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_template_train.py -v`
Expected: FAIL (`train.py` missing, copy raises `FileNotFoundError`).

- [ ] **Step 3: Write `mlagent/templates/tabular_sklearn/train.py`**

```python
"""Train, evaluate per epoch, write metrics.json. Generated by mlagent; safe to edit.

Usage (run from the project folder):
  python train.py              full training run
  python train.py --dry-run    one tiny epoch; prints {"seconds_per_epoch": ..., "n_train": ...}
  python train.py --eval-test  evaluate checkpoints/best.joblib on the held-out test split
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from data import load_data
from model import build_model, grow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
METRICS_FILE = "metrics.json"
EVAL_VAL_FILE = "eval_val.json"
EVAL_TEST_FILE = "eval_test.json"
CHECKPOINT = Path("checkpoints") / "best.joblib"
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"tabular_classification": "accuracy", "tabular_regression": "rmse"}


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    """Atomic write so a reader polling the file never sees a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def compute_metric(name: str, y_true, y_pred) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if name == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"unknown metric {name!r}")


def full_proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with one column per class even if a class was absent from training."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


def evaluate(model, X, y, task_type: str, metric: str, n_classes: int | None) -> dict:
    y_pred = model.predict(X)
    if task_type == "tabular_classification":
        proba = full_proba(model, X, n_classes)
        loss = float(log_loss(y, proba, labels=list(range(n_classes))))
        y_proba = proba.tolist()
    else:
        loss = float(mean_squared_error(y, y_pred))
        y_proba = None
    return {
        "loss": loss,
        "value": compute_metric(metric, y, y_pred),
        "y_true": np.asarray(y).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": y_proba,
    }


def eval_record(split: str, task_type: str, metric: str, classes, ev: dict) -> dict:
    return {
        "split": split,
        "task_type": task_type,
        "metric": metric,
        "value": ev["value"],
        "loss": ev["loss"],
        "classes": classes,
        "y_true": ev["y_true"],
        "y_pred": ev["y_pred"],
        "y_proba": ev["y_proba"],
    }


def metric_for(project_dir: Path, task_type: str) -> str:
    spec = read_json(project_dir / SPEC_FILE, default={}) or {}
    return str(spec.get("metric") or DEFAULT_METRIC[task_type])


def train(project_dir: Path, dry_run: bool = False) -> dict:
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    task_type = data["task_type"]
    metric = metric_for(project_dir, task_type)
    higher = HIGHER_IS_BETTER[metric]
    n_classes = len(data["classes"]) if data["classes"] is not None else None
    epochs = int(config.get("epochs", 10))
    iters = int(config.get("iters_per_epoch", 10))
    patience = int(config.get("early_stopping_patience", 0))
    if dry_run:
        epochs, iters, patience = 1, 1, 0
    config = {**config, "iters_per_epoch": iters}
    metrics_path = None if dry_run else project_dir / METRICS_FILE

    metrics = {
        "status": "running",
        "task_type": task_type,
        "metric": metric,
        "higher_is_better": higher,
        "config": config,
        "classes": data["classes"],
        "n_train": int(len(data["X_train"])),
        "n_val": int(len(data["X_val"])),
        "n_test": int(len(data["X_test"])),
        "epochs": [],
        "best_epoch": None,
        "best_val_metric": None,
        "stopped_early": False,
        "error": None,
        "seconds": None,
        "seconds_per_epoch": None,
    }

    def save() -> None:
        if metrics_path is not None:
            write_json(metrics_path, metrics)

    save()
    model = build_model(config, task_type, data["categorical_mask"])
    best_model = None
    best_value = None
    since_best = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if epoch > 1:
            grow(model, iters)
        model.fit(data["X_train"], data["y_train"])
        tr = evaluate(model, data["X_train"], data["y_train"], task_type, metric, n_classes)
        va = evaluate(model, data["X_val"], data["y_val"], task_type, metric, n_classes)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_metric": tr["value"],
            "val_metric": va["value"],
            "seconds": round(time.time() - t0, 3),
        }
        metrics["epochs"].append(row)
        if not (math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"])):
            metrics["status"] = "failed"
            metrics["error"] = f"non-finite loss at epoch {epoch}"
            save()
            return metrics
        improved = best_value is None or (
            row["val_metric"] > best_value if higher else row["val_metric"] < best_value
        )
        if improved:
            best_value = row["val_metric"]
            metrics["best_epoch"] = epoch
            metrics["best_val_metric"] = best_value
            best_model = copy.deepcopy(model)
            since_best = 0
        else:
            since_best += 1
        save()
        print(
            f"epoch {epoch}/{epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_{metric}={row['val_metric']:.4f}",
            flush=True,
        )
        if patience and since_best >= patience:
            metrics["stopped_early"] = True
            break

    elapsed = time.time() - started
    metrics["seconds"] = round(elapsed, 3)
    metrics["seconds_per_epoch"] = round(elapsed / max(1, len(metrics["epochs"])), 4)
    if dry_run:
        print(json.dumps({"seconds_per_epoch": metrics["seconds_per_epoch"],
                          "n_train": metrics["n_train"]}), flush=True)
        return metrics

    (project_dir / CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, project_dir / CHECKPOINT)
    va = evaluate(best_model, data["X_val"], data["y_val"], task_type, metric, n_classes)
    write_json(project_dir / EVAL_VAL_FILE,
               eval_record("val", task_type, metric, data["classes"], va))
    metrics["status"] = "done"
    save()
    return metrics


def eval_test(project_dir: Path) -> dict:
    checkpoint = project_dir / CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}; run train.py first")
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    task_type = data["task_type"]
    metric = metric_for(project_dir, task_type)
    n_classes = len(data["classes"]) if data["classes"] is not None else None
    model = joblib.load(checkpoint)
    te = evaluate(model, data["X_test"], data["y_test"], task_type, metric, n_classes)
    record = eval_record("test", task_type, metric, data["classes"], te)
    write_json(project_dir / EVAL_TEST_FILE, record)
    print(f"test {metric}={te['value']:.4f} loss={te['loss']:.4f}", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="project folder (default: cwd)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
    try:
        if args.eval_test:
            eval_test(project_dir)
            return 0
        metrics = train(project_dir, dry_run=args.dry_run)
        return 0 if metrics["status"] != "failed" else 1
    except Exception as exc:  # noqa: BLE001 - record any failure for the runner
        traceback.print_exc()
        if not args.eval_test and not args.dry_run:
            existing = read_json(project_dir / METRICS_FILE, default=None) or {
                "status": "running", "epochs": []
            }
            existing["status"] = "failed"
            existing["error"] = f"{type(exc).__name__}: {exc}"
            write_json(project_dir / METRICS_FILE, existing)
        else:
            print(f"error: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_template_train.py -v`
Expected: PASS. `improved` uses strict comparison on purpose: an equal validation value does not reset the patience counter, which is what makes the early-stopping test stop.

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: tabular_sklearn train.py with per-epoch metrics, checkpoint, dry-run and eval-test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 4: Subprocess runner

**Files:**
- Create: `mlagent/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `project.read_json_file(path, default)`, `config.METRICS_FILE`.
- Produces: `RunResult(returncode: int, metrics: dict | None, log_tail: list[str], seconds: float, timed_out: bool, expects_metrics: bool)` with property `ok -> bool` (returncode 0, not timed out, and metrics status "done" when a metrics path was given); `run_script(project_root, args, *, on_line=None, on_metrics=None, metrics_path=None, poll_seconds=0.5, python=sys.executable, timeout=None) -> RunResult`; `run_training(project_root, **kwargs) -> RunResult` (runs `train.py`, polls `metrics.json`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runner.py
from __future__ import annotations

import textwrap

from mlagent import runner

EPOCH_SCRIPT = textwrap.dedent(
    """
    import json, time
    for epoch in range(1, 4):
        time.sleep(0.15)
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump({"status": "running" if epoch < 3 else "done",
                       "epochs": [{"epoch": e} for e in range(1, epoch + 1)]}, f)
        print(f"epoch {epoch}", flush=True)
    """
)


def test_streams_lines_and_polls_metrics(tmp_path):
    (tmp_path / "train.py").write_text(EPOCH_SCRIPT, encoding="utf-8")
    lines, updates = [], []
    result = runner.run_training(
        tmp_path, on_line=lines.append, on_metrics=updates.append, poll_seconds=0.05
    )
    assert result.returncode == 0 and result.ok and not result.timed_out
    assert [line.strip() for line in lines] == ["epoch 1", "epoch 2", "epoch 3"]
    assert result.log_tail[-1].strip() == "epoch 3"
    assert result.metrics["status"] == "done"
    assert len(updates) >= 2
    counts = [len(u["epochs"]) for u in updates]
    assert counts == sorted(counts)
    assert result.seconds > 0


def test_failing_script_reports_returncode_and_tail(tmp_path):
    (tmp_path / "bad.py").write_text("print('starting')\nraise SystemExit(3)\n", encoding="utf-8")
    result = runner.run_script(tmp_path, ["bad.py"])
    assert result.returncode == 3 and not result.ok
    assert result.metrics is None
    assert "starting" in "".join(result.log_tail)


def test_timeout_kills_process(tmp_path):
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    result = runner.run_script(tmp_path, ["slow.py"], timeout=0.5, poll_seconds=0.05)
    assert result.timed_out and not result.ok
    assert result.seconds < 10


def test_ok_requires_done_status_when_metrics_expected(tmp_path):
    (tmp_path / "train.py").write_text(
        "import json\n"
        "with open('metrics.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump({'status': 'failed', 'epochs': []}, f)\n",
        encoding="utf-8",
    )
    result = runner.run_training(tmp_path, poll_seconds=0.05)
    assert result.returncode == 0 and not result.ok
    assert result.metrics["status"] == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.runner'`.

- [ ] **Step 3: Write `mlagent/runner.py`**

```python
"""Run a generated script in the project folder, streaming output and polling metrics.json."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mlagent import config
from mlagent.project import read_json_file

LOG_TAIL_LINES = 60


@dataclass
class RunResult:
    returncode: int
    metrics: dict | None
    log_tail: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    expects_metrics: bool = False

    @property
    def ok(self) -> bool:
        if self.timed_out or self.returncode != 0:
            return False
        if self.expects_metrics:
            return bool(self.metrics) and self.metrics.get("status") == "done"
        return True


def _pump(stream, out: queue.Queue) -> None:
    for line in iter(stream.readline, ""):
        out.put(line)
    out.put(None)


def run_script(
    project_root: Path,
    args: list[str],
    *,
    on_line: Callable[[str], None] | None = None,
    on_metrics: Callable[[dict], None] | None = None,
    metrics_path: Path | None = None,
    poll_seconds: float = 0.5,
    python: str = sys.executable,
    timeout: float | None = None,
) -> RunResult:
    """Run `python <args>` with cwd=project_root.

    Every stdout/stderr line goes to `on_line`. If `metrics_path` is given it is re-read
    every `poll_seconds` and `on_metrics` is called whenever the number of recorded epochs
    changes. `timeout` (seconds) kills the process and marks the result timed out.
    """
    started = time.time()
    proc = subprocess.Popen(
        [python, *args],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True)
    reader.start()

    tail: list[str] = []
    seen_epochs = -1
    timed_out = False
    eof = False

    def poll_metrics() -> None:
        nonlocal seen_epochs
        if metrics_path is None:
            return
        data = read_json_file(metrics_path, default=None)
        if not isinstance(data, dict):
            return
        n = len(data.get("epochs") or [])
        if n != seen_epochs:
            seen_epochs = n
            if on_metrics is not None:
                on_metrics(data)

    while True:
        try:
            while True:
                item = lines.get_nowait()
                if item is None:
                    eof = True
                    break
                tail.append(item)
                del tail[:-LOG_TAIL_LINES]
                if on_line is not None:
                    on_line(item)
        except queue.Empty:
            pass
        poll_metrics()
        if eof and proc.poll() is not None:
            break
        if timeout is not None and time.time() - started > timeout:
            proc.kill()
            proc.wait()
            timed_out = True
            break
        time.sleep(poll_seconds)

    proc.wait()
    poll_metrics()
    metrics = read_json_file(metrics_path, default=None) if metrics_path is not None else None
    return RunResult(
        returncode=proc.returncode,
        metrics=metrics if isinstance(metrics, dict) else None,
        log_tail=tail,
        seconds=round(time.time() - started, 3),
        timed_out=timed_out,
        expects_metrics=metrics_path is not None,
    )


def run_training(project_root: Path, **kwargs) -> RunResult:
    """Run `train.py` in the project folder, polling `metrics.json`."""
    kwargs.setdefault("metrics_path", Path(project_root) / config.METRICS_FILE)
    return run_script(Path(project_root), ["train.py"], **kwargs)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: subprocess runner with stdout streaming, metrics polling and timeout

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 5: Training and evaluation figures

**Files:**
- Modify: `mlagent/plots.py` (append; do not change existing functions)
- Test: `tests/test_plots_training.py`

**Interfaces:**
- Consumes: existing palette constants `SERIES, SEQUENTIAL, INK, INK_2, MUTED, GRID, AXIS, SURFACE`, `present(fig, plots_dir, name) -> Path`, `save_figure`.
- Produces: `training_curves(epochs: list[dict], metric: str) -> Figure`; `confusion_matrix_plot(cm, labels: list[str]) -> Figure`; `roc_pr_curves(y_true, y_proba, labels) -> Figure`; `per_class_bars(y_true, y_pred, labels) -> Figure`; `predicted_vs_actual(y_true, y_pred) -> Figure`; `residual_plots(y_true, y_pred) -> Figure`; `present_evaluation(eval_data: dict, plots_dir: Path, prefix: str) -> list[Path]` (picks the figures by `eval_data["task_type"]`, saves `<prefix>_confusion.png`, `<prefix>_roc_pr.png`, `<prefix>_per_class.png` for classification or `<prefix>_pred_vs_actual.png`, `<prefix>_residuals.png` for regression).
- `eval_data` is the `eval_val.json` / `eval_test.json` record from Task 3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plots_training.py
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from mlagent import plots

EPOCHS = [
    {"epoch": 1, "train_loss": 0.9, "val_loss": 0.95, "train_metric": 0.6, "val_metric": 0.55},
    {"epoch": 2, "train_loss": 0.6, "val_loss": 0.7, "train_metric": 0.75, "val_metric": 0.7},
    {"epoch": 3, "train_loss": 0.4, "val_loss": 0.65, "train_metric": 0.85, "val_metric": 0.72},
]


def test_training_curves_two_panels_with_legends():
    fig = plots.training_curves(EPOCHS, "accuracy")
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_legend() is not None
        assert len(ax.lines) == 2
    assert "accuracy" in fig.axes[1].get_title()
    plots.plt.close(fig)


def test_training_curves_empty_does_not_crash():
    fig = plots.training_curves([], "rmse")
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_confusion_matrix_annotates_cells():
    fig = plots.confusion_matrix_plot([[5, 1], [2, 7]], ["no", "yes"])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert sorted(texts) == ["1", "2", "5", "7"]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["no", "yes"]
    plots.plt.close(fig)


def test_roc_pr_binary_has_no_legend_multiclass_has_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    p = rng.random((40, 2))
    p = p / p.sum(axis=1, keepdims=True)
    fig = plots.roc_pr_curves(y.tolist(), p.tolist(), ["a", "b"])
    assert len(fig.axes) == 2 and fig.axes[0].get_legend() is None
    plots.plt.close(fig)
    y3 = rng.integers(0, 3, size=60)
    p3 = rng.random((60, 3))
    fig = plots.roc_pr_curves(y3.tolist(), p3.tolist(), ["a", "b", "c"])
    assert fig.axes[0].get_legend() is not None
    assert len(fig.axes[0].lines) >= 3
    plots.plt.close(fig)


def test_per_class_bars_two_series_legend():
    fig = plots.per_class_bars([0, 1, 1, 0], [0, 1, 0, 0], ["a", "b"])
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.patches) == 4
    plots.plt.close(fig)


def test_regression_figures():
    y = [1.0, 2.0, 3.0, 4.0]
    p = [1.1, 1.9, 3.3, 3.8]
    fig = plots.predicted_vs_actual(y, p)
    assert len(fig.axes) == 1 and fig.axes[0].get_legend() is None
    plots.plt.close(fig)
    fig = plots.residual_plots(y, p)
    assert len(fig.axes) == 2
    plots.plt.close(fig)


def test_present_evaluation_classification_and_regression(tmp_path):
    cls = {"task_type": "tabular_classification", "classes": ["a", "b"],
           "y_true": [0, 1, 1, 0, 1], "y_pred": [0, 1, 0, 0, 1],
           "y_proba": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]]}
    saved = plots.present_evaluation(cls, tmp_path, "run1_val")
    assert [p.name for p in saved] == [
        "run1_val_confusion.png", "run1_val_roc_pr.png", "run1_val_per_class.png"
    ]
    assert all(p.exists() for p in saved)
    reg = {"task_type": "tabular_regression", "classes": None,
           "y_true": [1.0, 2.0, 3.0], "y_pred": [1.2, 1.8, 3.1], "y_proba": None}
    saved = plots.present_evaluation(reg, tmp_path, "test")
    assert [p.name for p in saved] == ["test_pred_vs_actual.png", "test_residuals.png"]
    assert plots.plt.get_fignums() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_plots_training.py -v`
Expected: FAIL with `AttributeError: module 'mlagent.plots' has no attribute 'training_curves'`.

- [ ] **Step 3: Append to `mlagent/plots.py`**

Add `from matplotlib.colors import LinearSegmentedColormap` to the imports (keep them sorted), then append:

```python
# --- Milestone 3: training and evaluation figures -------------------------------------


def _frame(ax, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)


def training_curves(epochs: list[dict], metric: str) -> Figure:
    """Loss (left) and the target metric (right) per epoch, train vs validation."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(9, 3.2))
    xs = [e.get("epoch") for e in epochs]
    panels = (
        (ax_loss, ("train_loss", "val_loss"), "Loss per epoch"),
        (ax_metric, ("train_metric", "val_metric"), f"{metric} per epoch"),
    )
    for ax, keys, title in panels:
        for key, colour, label in zip(keys, SERIES[:2], ("train", "validation"), strict=True):
            ax.plot(xs, [e.get(key) for e in epochs], color=colour, linewidth=2,
                    marker="o", markersize=4, label=label)
        _frame(ax, title)
        ax.set_xlabel("epoch", color=INK_2, fontsize=8)
        _legend(ax)
    if not epochs:
        ax_loss.text(0.5, 0.5, "no epochs yet", ha="center", va="center", color=MUTED,
                     transform=ax_loss.transAxes)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(cm, labels: list[str]) -> Figure:
    m = np.asarray(cm)
    n = len(labels)
    size = min(2.5 + 0.35 * n, 9)
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
    ax.imshow(m, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK_2, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=INK_2, fontsize=8)
    ax.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax.set_ylabel("actual", color=INK_2, fontsize=8)
    if n <= 20 and m.size:
        threshold = m.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color=SURFACE if m[i, j] > threshold else INK)
    ax.set_title("Confusion matrix", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def roc_pr_curves(y_true, y_proba, labels: list[str]) -> Figure:
    """ROC (left) and precision-recall (right). Binary: one curve for the positive class.
    Multiclass: one-vs-rest per class, at most len(SERIES) classes shown."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.6))
    n = p.shape[1] if p.ndim == 2 else 1
    if n == 2:
        curves = [(1, p[:, 1], labels[1])]
    else:
        curves = [(k, p[:, k], labels[k]) for k in range(min(n, len(SERIES)))]
    for (k, score, label), colour in zip(curves, SERIES, strict=False):
        positive = (y == k).astype(int)
        if positive.sum() in (0, len(positive)):
            continue
        fpr, tpr, _ = roc_curve(positive, score)
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, label=label)
        precision, recall, _ = precision_recall_curve(positive, score)
        ax_pr.plot(recall, precision, color=colour, linewidth=2, label=label)
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    suffix = f" (showing {len(curves)} of {n} classes)" if n > len(curves) else ""
    _frame(ax_roc, "ROC curve" + suffix)
    ax_roc.set_xlabel("false positive rate", color=INK_2, fontsize=8)
    ax_roc.set_ylabel("true positive rate", color=INK_2, fontsize=8)
    _frame(ax_pr, "Precision-recall curve")
    ax_pr.set_xlabel("recall", color=INK_2, fontsize=8)
    ax_pr.set_ylabel("precision", color=INK_2, fontsize=8)
    if len(curves) > 1:
        _legend(ax_roc)
        _legend(ax_pr)
    fig.tight_layout()
    return fig


def per_class_bars(y_true, y_pred, labels: list[str]) -> Figure:
    from sklearn.metrics import precision_recall_fscore_support

    n = len(labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * n + 2), 3.2))
    x = np.arange(n)
    width = 0.38
    ax.bar(x - width / 2, precision, width=width, color=SERIES[0], label="precision")
    ax.bar(x + width / 2, recall, width=width, color=SERIES[1], label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 6 else 0, ha="right" if n > 6 else "center",
                       color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.05)
    _frame(ax, "Precision and recall per class")
    _legend(ax)
    fig.tight_layout()
    return fig


def predicted_vs_actual(y_true, y_pred) -> Figure:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(y, p, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
    _frame(ax, "Predicted vs actual")
    ax.set_xlabel("actual", color=INK_2, fontsize=8)
    ax.set_ylabel("predicted", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def residual_plots(y_true, y_pred) -> Figure:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    residuals = y - p
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax_hist.hist(residuals, bins=min(30, max(5, len(residuals) // 5)), color=SERIES[0])
    _frame(ax_hist, "Residual distribution")
    ax_hist.set_xlabel("actual - predicted", color=INK_2, fontsize=8)
    ax_scatter.scatter(p, residuals, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    ax_scatter.axhline(0, color=AXIS, linestyle="--", linewidth=1)
    _frame(ax_scatter, "Residuals vs predicted")
    ax_scatter.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax_scatter.set_ylabel("residual", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def present_evaluation(eval_data: dict, plots_dir: Path, prefix: str) -> list[Path]:
    """Save and show the evaluation figures appropriate to the task; return saved paths."""
    y_true = list(eval_data.get("y_true") or [])
    y_pred = list(eval_data.get("y_pred") or [])
    saved: list[Path] = []
    if eval_data.get("task_type") == "tabular_classification":
        from sklearn.metrics import confusion_matrix

        classes = eval_data.get("classes") or sorted({*y_true, *y_pred})
        labels = [str(c) for c in classes]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        saved.append(present(confusion_matrix_plot(cm, labels), plots_dir, f"{prefix}_confusion"))
        if eval_data.get("y_proba"):
            fig = roc_pr_curves(y_true, eval_data["y_proba"], labels)
            saved.append(present(fig, plots_dir, f"{prefix}_roc_pr"))
        saved.append(present(per_class_bars(y_true, y_pred, labels), plots_dir,
                             f"{prefix}_per_class"))
    else:
        saved.append(present(predicted_vs_actual(y_true, y_pred), plots_dir,
                             f"{prefix}_pred_vs_actual"))
        saved.append(present(residual_plots(y_true, y_pred), plots_dir, f"{prefix}_residuals"))
    return saved
```

If `plots.py` already has helpers equivalent to `_frame`/`_legend`, use those instead and do not add duplicates. `plt` and `np` are already imported in `plots.py`; if `np` is not, add `import numpy as np`.

- [ ] **Step 4: Run the tests, then the deprecation-warning check**

Run: `python -m pytest tests/test_plots_training.py -v && python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_plots_training.py -q`
Expected: PASS both.

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: training curves and evaluation figures (confusion, ROC/PR, per-class, regression)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 6: Template I/O and config schema helpers

**Files:**
- Create: `mlagent/templates_io.py`
- Modify: `pyproject.toml` (package data)
- Test: `tests/test_templates_io.py`

**Interfaces:**
- Consumes: `mlagent/templates/tabular_sklearn/{data.py,model.py,train.py,config_schema.json}`.
- Produces: `TEMPLATES_DIR: Path`; `CODE_FILES = ("data.py", "model.py", "train.py")`; `TEMPLATE_FOR_TASK = {"tabular_classification": "tabular_sklearn", "tabular_regression": "tabular_sklearn"}`; `template_dir(name) -> Path`; `load_schema(name) -> dict`; `default_config(schema) -> dict`; `validate_config(config, schema) -> list[str]` (problems; empty means valid); `coerce_config(proposal, schema) -> tuple[dict, list[str]]` (defaults overlaid with valid proposed values, out-of-range values clamped, unknown keys dropped; second item is human-readable notes); `copy_template(name, project_root) -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_templates_io.py
from __future__ import annotations

import pytest

from mlagent import templates_io as tio


def test_template_dir_and_schema():
    d = tio.template_dir("tabular_sklearn")
    assert d.is_dir()
    for name in tio.CODE_FILES:
        assert (d / name).exists()
    schema = tio.load_schema("tabular_sklearn")
    assert set(schema) >= {"learning_rate", "epochs", "iters_per_epoch", "seed"}
    with pytest.raises(FileNotFoundError):
        tio.template_dir("no_such_template")


def test_default_config_matches_schema_defaults():
    schema = tio.load_schema("tabular_sklearn")
    cfg = tio.default_config(schema)
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10 and cfg["max_depth"] is None
    assert tio.validate_config(cfg, schema) == []


def test_validate_config_reports_problems():
    schema = tio.load_schema("tabular_sklearn")
    cfg = tio.default_config(schema)
    cfg["learning_rate"] = 5.0
    cfg["epochs"] = "ten"
    cfg["bogus"] = 1
    del cfg["seed"]
    problems = tio.validate_config(cfg, schema)
    assert any("learning_rate" in p for p in problems)
    assert any("epochs" in p for p in problems)
    assert any("bogus" in p for p in problems)
    assert any("seed" in p for p in problems)


def test_coerce_config_clamps_and_drops():
    schema = tio.load_schema("tabular_sklearn")
    cfg, notes = tio.coerce_config(
        {"learning_rate": 9, "epochs": 20.0, "max_depth": None, "bogus": 3, "seed": "7"}, schema
    )
    assert cfg["learning_rate"] == 1.0
    assert cfg["epochs"] == 20 and isinstance(cfg["epochs"], int)
    assert cfg["max_depth"] is None
    assert cfg["seed"] == 7
    assert "bogus" not in cfg
    assert cfg["iters_per_epoch"] == 10  # default kept
    assert any("learning_rate" in n for n in notes) and any("bogus" in n for n in notes)
    assert tio.validate_config(cfg, schema) == []


def test_copy_template(project):
    paths = tio.copy_template("tabular_sklearn", project.root)
    assert [p.name for p in paths] == list(tio.CODE_FILES)
    assert all(p.exists() for p in paths)
    assert "load_data" in (project.root / "data.py").read_text(encoding="utf-8")
    assert not (project.root / "config_schema.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_templates_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.templates_io'`.

- [ ] **Step 3: Write `mlagent/templates_io.py`**

```python
"""Locate, copy and validate the reference training templates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
CODE_FILES = ("data.py", "model.py", "train.py")
SCHEMA_FILE = "config_schema.json"
TEMPLATE_FOR_TASK = {
    "tabular_classification": "tabular_sklearn",
    "tabular_regression": "tabular_sklearn",
}


def template_dir(name: str) -> Path:
    path = TEMPLATES_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"no template named {name!r} under {TEMPLATES_DIR}")
    return path


def load_schema(name: str) -> dict:
    return json.loads((template_dir(name) / SCHEMA_FILE).read_text(encoding="utf-8"))


def default_config(schema: dict) -> dict:
    return {key: rule.get("default") for key, rule in schema.items()}


def _check_value(key: str, value, rule: dict) -> str | None:
    if value is None:
        return None if rule.get("nullable") else f"{key}: must not be null"
    if isinstance(value, bool):
        return f"{key}: expected a number, got a boolean"
    if rule.get("type") == "integer":
        if not isinstance(value, int) and not (isinstance(value, float) and value.is_integer()):
            return f"{key}: expected an integer, got {value!r}"
    elif not isinstance(value, int | float):
        return f"{key}: expected a number, got {value!r}"
    if rule.get("min") is not None and value < rule["min"]:
        return f"{key}: {value!r} is below the minimum {rule['min']}"
    if rule.get("max") is not None and value > rule["max"]:
        return f"{key}: {value!r} is above the maximum {rule['max']}"
    return None


def validate_config(config: dict, schema: dict) -> list[str]:
    """Return a list of problems; empty means the config is valid against the schema."""
    problems: list[str] = []
    for key in config:
        if key not in schema:
            problems.append(f"{key}: not a tunable key")
    for key, rule in schema.items():
        if key not in config:
            problems.append(f"{key}: missing")
            continue
        problem = _check_value(key, config[key], rule)
        if problem:
            problems.append(problem)
    return problems


def _cast(value, rule: dict):
    if value is None:
        return None
    if isinstance(value, str):
        value = float(value)
    return int(round(value)) if rule.get("type") == "integer" else float(value)


def coerce_config(proposal: dict, schema: dict) -> tuple[dict, list[str]]:
    """Overlay a proposal on the defaults, clamping out-of-range values and dropping
    unknown keys. Notes describe every adjustment in plain words."""
    config = default_config(schema)
    notes: list[str] = []
    for key, value in (proposal or {}).items():
        if key not in schema:
            notes.append(f"Ignored unknown key {key!r}.")
            continue
        rule = schema[key]
        try:
            cast = _cast(value, rule)
        except (TypeError, ValueError):
            notes.append(f"Ignored {key}={value!r}: not a number; kept {config[key]!r}.")
            continue
        if cast is None:
            if rule.get("nullable"):
                config[key] = None
            else:
                notes.append(f"Ignored null for {key}; kept {config[key]!r}.")
            continue
        low, high = rule.get("min"), rule.get("max")
        if low is not None and cast < low:
            notes.append(f"Raised {key} from {cast!r} to the minimum {low!r}.")
            cast = _cast(low, rule)
        elif high is not None and cast > high:
            notes.append(f"Lowered {key} from {cast!r} to the maximum {high!r}.")
            cast = _cast(high, rule)
        config[key] = cast
    return config, notes


def copy_template(name: str, project_root: Path) -> list[Path]:
    """Copy the template's code files into the project folder, overwriting; return the paths."""
    src = template_dir(name)
    written: list[Path] = []
    for filename in CODE_FILES:
        target = Path(project_root) / filename
        shutil.copyfile(src / filename, target)
        written.append(target)
    return written
```

- [ ] **Step 4: Package data**

In `pyproject.toml`, make sure the template files ship with the package. If a `[tool.setuptools.package-data]` table exists, extend its `mlagent` list; otherwise add:

```toml
[tool.setuptools.package-data]
mlagent = ["prompts/*.md", "templates/*/*"]
```

Also confirm `[tool.setuptools.packages.find]` (or the equivalent) does not exclude `mlagent.templates.*`; the template folders have no `__init__.py` and are found as package data, not as packages. Verify with:

Run: `python -c "from mlagent import templates_io as t; print(t.template_dir('tabular_sklearn'))"`

- [ ] **Step 5: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: template lookup/copy and config schema validation helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 7: Codegen stage

**Files:**
- Create: `mlagent/stages/codegen.py`
- Create: `mlagent/prompts/codegen.md`
- Test: `tests/test_codegen_stage.py`

**Interfaces:**
- Consumes: `StageContext` (`project`, `llm`, `questioner`, `display`, `spec()`), `ToolSpec`, `LLMError`, `load_prompt`, `templates_io.*`, `stages.data.META_FILE`, `config.CONFIG_FILE`.
- Produces: `CodegenStage` with `name = "codegen"`, `is_complete(ctx)` (all `CODE_FILES` exist and `config.json` validates), `run(ctx)`; module functions `check_data(meta: dict, project_root: Path) -> list[str]`, `meta_summary(meta) -> dict`, `config_table(config, schema) -> str` (markdown). Writes `data.py`, `model.py`, `train.py`, `config.json`.
- The LLM tool is `propose_config` with input `{"config": {<schema key>: value}, "rationale": str}`.

- [ ] **Step 1: Write `mlagent/prompts/codegen.md`**

```markdown
You are the code-generation stage of an ML training assistant that runs inside Google Colab. The training code is a fixed, tested template (gradient-boosted trees via scikit-learn's HistGradientBoosting with warm start, so each epoch adds boosting rounds). Your job is to choose sensible starting hyperparameters for this dataset and explain them to a learner.

You receive JSON with the project spec (goal, task type, metric, target value, minutes per run), a data summary (rows, columns, categorical columns, classes, split fractions), and the config schema (every tunable key with its type, default, min, max and description).

Call the `propose_config` tool exactly once with:
- `config`: only keys from the schema. Omit keys you would leave at their default. Keep values inside the min/max bounds. Prefer defaults unless the data suggests otherwise (for example: small datasets want a lower learning rate and larger min_samples_leaf; many classes or rows can afford more epochs; a short minutes-per-run budget wants fewer epochs).
- `rationale`: under 120 words, plain-spoken, for someone learning ML. Say what you changed from the defaults and why, or say you kept the defaults and why they fit.

Wrap technical terms in double square brackets like [[learning rate]] or [[overfitting]] so the user can click them. Do not write code and do not restate the raw JSON.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_codegen_stage.py
from __future__ import annotations

import json

from mlagent.llm import FakeLLM
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage, check_data, config_table
from mlagent.templates_io import CODE_FILES, load_schema
from mlagent.ui.questions import ScriptedQuestioner


def make_ctx(project, llm, answers):
    shown = []
    ctx = StageContext(project=project, llm=llm, questioner=ScriptedQuestioner(answers),
                       explainer=None, display=shown.append)
    return ctx, shown


def test_llm_proposal_is_coerced_and_written(clean_project):
    llm = FakeLLM([
        [("tool", "propose_config", {"config": {"learning_rate": 0.05, "epochs": 500,
                                                 "bogus": 1},
                                      "rationale": "Small data, so a lower [[learning rate]]."})],
        [("text", "done")],
    ])
    ctx, shown = make_ctx(clean_project, llm, ["y"])  # confirm: happy with config
    stage = CodegenStage()
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    for name in CODE_FILES:
        assert (clean_project.root / name).exists()
    cfg = clean_project.read_json("config.json")
    assert cfg["learning_rate"] == 0.05
    assert cfg["epochs"] == 100  # clamped to the schema maximum
    assert "bogus" not in cfg
    assert cfg["seed"] == 42
    text = "\n".join(shown)
    assert "[[learning rate]]" in text
    assert "Lowered epochs" in text
    assert llm.calls and llm.calls[0]["tools"][0].name == "propose_config"
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "config_schema" in prompt or "learning_rate" in prompt


def test_llm_failure_falls_back_to_defaults(clean_project):
    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["y"])
    CodegenStage().run(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["learning_rate"] == 0.1 and cfg["epochs"] == 10
    assert any("defaults" in s for s in shown)


def test_user_edits_config_values(clean_project):
    answers = [
        "n",                         # not happy with the config
        "epochs = 10",               # pick the key
        "3",                         # new value
        "max_depth = None",          # pick nullable key
        "4",                         # new value
        "Done",
    ]
    ctx, _ = make_ctx(clean_project, FakeLLM([]), answers)
    CodegenStage().run(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["epochs"] == 3 and cfg["max_depth"] == 4


def test_bad_data_stops_stage_without_writing(clean_project):
    meta = clean_project.read_json("data_meta.json")
    meta["target"] = "nope"
    clean_project.write_json("data_meta.json", meta)
    ctx, shown = make_ctx(clean_project, FakeLLM([]), [])
    stage = CodegenStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert not (clean_project.root / "config.json").exists()
    assert any("nope" in s for s in shown)


def test_unsupported_task_type_is_reported(clean_project):
    spec = clean_project.read_json("spec.json")
    spec["task_type"] = "image_classification"
    clean_project.write_json("spec.json", spec)
    ctx, shown = make_ctx(clean_project, FakeLLM([]), [])
    stage = CodegenStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert any("image_classification" in s for s in shown)


def test_check_data_and_config_table(clean_project):
    meta = clean_project.read_json("data_meta.json")
    assert check_data(meta, clean_project.root) == []
    meta["splits"] = {"train": 0.5, "val": 0.1, "test": 0.1}
    meta["feature_columns"] = []
    problems = check_data(meta, clean_project.root)
    assert any("split" in p for p in problems) and any("feature" in p for p in problems)
    schema = load_schema("tabular_sklearn")
    table = config_table({"learning_rate": 0.1, "max_depth": None}, schema)
    assert "| learning_rate | 0.1 |" in table and "| max_depth | none |" in table


def test_is_complete_requires_valid_config(clean_project):
    ctx, _ = make_ctx(clean_project, FakeLLM([]), ["y"])
    stage = CodegenStage()
    stage.run(ctx)
    cfg = clean_project.read_json("config.json")
    cfg["learning_rate"] = 99
    clean_project.write_json("config.json", cfg)
    assert not stage.is_complete(ctx)
    assert json.loads(clean_project.config_path.read_text(encoding="utf-8"))["learning_rate"] == 99
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_codegen_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.codegen'`.

- [ ] **Step 4: Write `mlagent/stages/codegen.py`**

```python
"""Codegen stage: copy the training template and choose a starting config."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlagent import config as cfg
from mlagent.llm import LLMError, ToolSpec
from mlagent.prompts_io import load_prompt
from mlagent.spec import Spec
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE
from mlagent.templates_io import (
    CODE_FILES,
    TEMPLATE_FOR_TASK,
    coerce_config,
    copy_template,
    load_schema,
    validate_config,
)

MAX_LISTED = 20
PROPOSE_TOOL = ToolSpec(
    name="propose_config",
    description=(
        "Propose starting hyperparameters. `config` may only contain keys from the schema; "
        "omit keys left at their default. `rationale` explains the choices to a learner."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "schema key -> value"},
            "rationale": {"type": "string"},
        },
        "required": ["config", "rationale"],
    },
    handler=lambda inp: "recorded",
)


def check_data(meta: dict, project_root: Path) -> list[str]:
    """Problems that would make training impossible; empty list means go ahead."""
    problems: list[str] = []
    clean_path = project_root / str(meta.get("clean_path") or "data/clean/data.csv")
    if not clean_path.exists():
        return [f"clean data file {clean_path.name} is missing; rerun the clean stage"]
    columns = list(pd.read_csv(clean_path, nrows=0).columns)
    target = meta.get("target")
    if not target or target not in columns:
        problems.append(f"target column {target!r} is not in the clean data")
    features = [c for c in meta.get("feature_columns") or [] if c in columns and c != target]
    if not features:
        problems.append("no feature columns are available for training")
    splits = meta.get("splits") or {}
    total = sum(float(splits.get(k, 0)) for k in ("train", "val", "test"))
    if abs(total - 1.0) > 0.01 or any(float(splits.get(k, 0)) <= 0 for k in ("train", "val", "test")):
        problems.append(f"split fractions {splits} must be positive and sum to 1")
    if meta.get("task_type") == "tabular_classification" and (meta.get("n_classes") or 0) < 2:
        problems.append("classification needs at least 2 classes in the target")
    return problems


def meta_summary(meta: dict) -> dict:
    features = list(meta.get("feature_columns") or [])
    labels = list(meta.get("class_labels") or [])
    return {
        "task_type": meta.get("task_type"),
        "target": meta.get("target"),
        "n_rows": meta.get("clean_n_rows"),
        "n_features": len(features),
        "feature_columns": features[:MAX_LISTED],
        "categorical_columns": list(meta.get("categorical_columns") or [])[:MAX_LISTED],
        "n_classes": meta.get("n_classes"),
        "class_labels": labels[:MAX_LISTED],
        "splits": meta.get("splits"),
    }


def config_table(config: dict, schema: dict) -> str:
    rows = ["| key | value | what it does |", "|---|---|---|"]
    for key, value in config.items():
        desc = schema.get(key, {}).get("description", "")
        shown = "none" if value is None else f"{value:g}" if isinstance(value, float) else str(value)
        rows.append(f"| {key} | {shown} | {desc} |")
    return "\n".join(rows)


class CodegenStage:
    name = "codegen"

    def is_complete(self, ctx: StageContext) -> bool:
        root = ctx.project.root
        if not all((root / f).exists() for f in CODE_FILES):
            return False
        config = ctx.project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict):
            return False
        try:
            spec = ctx.spec()
            schema = load_schema(TEMPLATE_FOR_TASK[spec.task_type])
        except Exception:  # noqa: BLE001 - missing spec or unsupported task: not complete
            return False
        return validate_config(config, schema) == []

    def run(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        template = TEMPLATE_FOR_TASK.get(spec.task_type)
        if template is None:
            ctx.display(
                f"No training template for task type `{spec.task_type}` yet; "
                "this milestone covers tabular tasks only."
            )
            return
        meta = ctx.project.read_json(META_FILE) or {}
        problems = check_data(meta, ctx.project.root)
        if problems:
            ctx.display("The data is not ready for training:\n- " + "\n- ".join(problems))
            return

        schema = load_schema(template)
        proposal, rationale = self._propose(ctx, spec, meta, schema)
        config, notes = coerce_config(proposal, schema)
        written = copy_template(template, ctx.project.root)
        ctx.project.write_json(cfg.CONFIG_FILE, config)

        files = ", ".join(f"`{p.name}`" for p in written) + ", `config.json`"
        message = [
            f"I wrote the training project into the project folder: {files}.",
            "`train.py` trains [[gradient boosting]] trees; each [[epoch]] adds boosting rounds "
            "and records train and validation [[loss]] so we can watch for [[overfitting]].",
            "",
            rationale,
            "",
            config_table(config, schema),
        ]
        if notes:
            message += ["", "Adjustments to keep values inside the schema:", *[f"- {n}" for n in notes]]
        ctx.display("\n".join(message))

        if not ctx.questioner.confirm("Happy with this configuration? (No lets you change values)"):
            config = self._edit_config(ctx, config, schema)
            ctx.project.write_json(cfg.CONFIG_FILE, config)
            ctx.display("Updated configuration:\n\n" + config_table(config, schema))

    def _propose(self, ctx: StageContext, spec: Spec, meta: dict, schema: dict) -> tuple[dict, str]:
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["config"] = inp.get("config") or {}
            captured["rationale"] = str(inp.get("rationale") or "")
            return "recorded"

        tool = ToolSpec(
            name=PROPOSE_TOOL.name,
            description=PROPOSE_TOOL.description,
            input_schema=PROPOSE_TOOL.input_schema,
            handler=handler,
        )
        prompt = json.dumps(
            {"spec": spec.to_dict(), "data": meta_summary(meta), "config_schema": schema},
            indent=2,
            default=str,
        )
        try:
            result = ctx.llm.run(
                load_prompt("codegen"), [{"role": "user", "content": prompt}], [tool]
            )
        except LLMError as exc:
            return {}, f"Using the template defaults (the assistant was unavailable: {exc})."
        rationale = captured.get("rationale") or result.text or "Using the template defaults."
        return dict(captured.get("config") or {}), rationale

    def _edit_config(self, ctx: StageContext, config: dict, schema: dict) -> dict:
        config = dict(config)
        while True:
            options = [f"{k} = {v}" for k, v in config.items()] + ["Done"]
            pick = ctx.questioner.choice(
                "Which value do you want to change?", options, allow_other=False
            )
            if pick == "Done":
                return config
            key = pick.split(" = ", 1)[0]
            rule = schema.get(key)
            if rule is None:
                continue
            nullable = bool(rule.get("nullable"))
            hint = " (0 means no limit)" if nullable else ""
            current = config.get(key)
            value = ctx.questioner.number(
                f"New value for {key}{hint}: {rule.get('description', '')}",
                default=0 if current is None else current,
                minimum=0 if nullable else rule.get("min"),
                maximum=rule.get("max"),
            )
            if nullable and value == 0:
                config[key] = None
            elif rule.get("type") == "integer":
                config[key] = int(round(value))
            else:
                config[key] = float(value)
            problems = validate_config(config, schema)
            if problems:
                ctx.display("That value is not allowed: " + "; ".join(problems))
                config[key] = current
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_codegen_stage.py -v`
Expected: PASS. If `ScriptedQuestioner.choice` returns the option text verbatim (it does for scripted answers), `"epochs = 10"` and `"max_depth = None"` match the generated option strings; if a test fails on the option string, print `options` and match the exact format.

- [ ] **Step 6: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: codegen stage writes the training project and an LLM-proposed config within the schema

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 8: Train stage with live plot

**Files:**
- Create: `mlagent/stages/train.py`
- Create: `mlagent/prompts/train.md`
- Test: `tests/test_train_stage.py`

**Interfaces:**
- Consumes: `runner.run_training`, `RunResult`; `runlog.append_run/read_runs/summarise`; `plots.training_curves/present/present_evaluation`; `ask_text`, `load_prompt`; `config.CONFIG_FILE/METRICS_FILE`; `templates_io.CODE_FILES`.
- Produces: `TrainStage(runner=run_training, python=sys.executable, timeout=None, display_fig=None, poll_seconds=1.0)` with `name = "train"`, `is_complete(ctx)` (any run in `runs.jsonl` with status `done`), `run(ctx)`; `LivePlotter(metric, display_fig=None)` with `.update(metrics)` and `.updates: int`; `build_run_entry(config, result, started_at) -> dict`; constant `EVAL_VAL_FILE = "eval_val.json"`.
- `runs.jsonl` entry: `{"run_id", "started_at" (ISO), "status": "done"|"failed", "config", "epochs_run", "best_epoch", "best_val_metric", "final_train_loss", "final_val_loss", "seconds", "error", "applied_diff": None}`.

- [ ] **Step 1: Write `mlagent/prompts/train.md`**

```markdown
You are the training stage of an ML training assistant that runs inside Google Colab. A training run has just finished. You receive JSON with the project spec (task type, metric, target value), the run's per-epoch train and validation loss and metric, the best epoch, and the validation evaluation summary.

Write a short debrief for a learner (under 180 words):

- One sentence on the result: the best validation value versus the target value, and whether the target was met.
- One or two sentences reading the curves: is validation loss still falling (underfitting, more epochs may help), flat (plateau), or rising while training loss falls ([[overfitting]])? Mention if training stopped early.
- One sentence on what the evaluation plots show for this task (for example the confusion matrix or the residuals).
- End with one plain suggestion for the next run; the tuning stage will handle the details.

Wrap technical terms in double square brackets like [[validation loss]] so the user can click them. Use the numbers given; do not invent any. Do not restate the raw JSON.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_train_stage.py
from __future__ import annotations

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.runner import RunResult
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import LivePlotter, TrainStage, build_run_entry
from mlagent.ui.questions import ScriptedQuestioner


def prepared(project):
    """Run codegen with defaults so the training project exists."""
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner(["y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().run(ctx)
    cfg = project.read_json("config.json")
    cfg.update({"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", cfg)
    return project


def make_ctx(project, llm=None, answers=()):
    shown = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append)
    return ctx, shown


def test_real_training_run_logs_and_plots(clean_project):
    project = prepared(clean_project)
    llm = FakeLLM([[("text", "Best [[validation accuracy]] beat the target.")]])
    ctx, shown = make_ctx(project, llm)
    figs = []
    stage = TrainStage(display_fig=figs.append, poll_seconds=0.05)
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["run_id"] == 1
    assert runs[0]["epochs_run"] == 3 and runs[0]["best_val_metric"] is not None
    assert runs[0]["config"]["epochs"] == 3
    assert project.metrics_path.exists() and project.exists("eval_val.json")
    names = sorted(p.name for p in project.plots_dir.glob("run1_*.png"))
    assert names == ["run1_training.png", "run1_val_confusion.png", "run1_val_per_class.png",
                     "run1_val_roc_pr.png"]
    assert len(figs) >= 1  # live plot redrawn at least once
    text = "\n".join(shown)
    assert "[[validation accuracy]]" in text
    assert "cost" in text.lower() and "cpu" in text.lower()
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "best_epoch" in prompt


def test_failed_run_is_logged_and_stage_incomplete(clean_project):
    project = prepared(clean_project)

    def fake_runner(root, **kwargs):
        return RunResult(returncode=1, metrics={"status": "failed", "epochs": [],
                                                "error": "ValueError: boom"},
                         log_tail=["Traceback\n", "ValueError: boom\n"], seconds=0.2,
                         expects_metrics=True)

    ctx, shown = make_ctx(project)
    stage = TrainStage(runner=fake_runner)
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert runs[0]["status"] == "failed" and "boom" in runs[0]["error"]
    assert any("boom" in s for s in shown)


def test_llm_failure_still_completes(clean_project):
    project = prepared(clean_project)
    ctx, shown = make_ctx(project, FakeLLM([]))
    TrainStage(poll_seconds=0.05).run(ctx)
    assert TrainStage().is_complete(ctx)
    assert any("best" in s.lower() for s in shown)


def test_missing_config_raises(clean_project):
    ctx, _ = make_ctx(clean_project)
    try:
        TrainStage().run(ctx)
    except RuntimeError as exc:
        assert "codegen" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_live_plotter_and_run_entry():
    figs = []
    plotter = LivePlotter("accuracy", display_fig=figs.append)
    plotter.update({"epochs": []})
    plotter.update({"epochs": [{"epoch": 1, "train_loss": 1, "val_loss": 1,
                                "train_metric": 0.5, "val_metric": 0.5}]})
    assert plotter.updates == 2 and len(figs) == 1
    result = RunResult(returncode=0, seconds=1.5, expects_metrics=True, metrics={
        "status": "done", "best_epoch": 2, "best_val_metric": 0.9,
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
                   {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4}]})
    entry = build_run_entry({"epochs": 2}, result, "2026-09-06T10:00:00")
    assert entry["status"] == "done" and entry["epochs_run"] == 2
    assert entry["final_val_loss"] == 0.4 and entry["seconds"] == 1.5
    assert entry["applied_diff"] is None and entry["error"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_train_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.train'`.

- [ ] **Step 4: Write `mlagent/stages/train.py`**

```python
"""Train stage: run train.py in a subprocess with a live plot; log and evaluate the run."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mlagent import config as cfg
from mlagent.llm import LLMError, ask_text
from mlagent.plots import present, present_evaluation, training_curves
from mlagent.prompts_io import load_prompt
from mlagent.runlog import append_run, read_runs
from mlagent.runner import RunResult, run_training
from mlagent.stages.base import StageContext
from mlagent.templates_io import CODE_FILES

EVAL_VAL_FILE = "eval_val.json"
LOG_TAIL_SHOWN = 15


def _ipython_display(fig: Figure) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import clear_output, display
    except ImportError:
        return
    if get_ipython() is None:
        return
    clear_output(wait=True)
    display(fig)


class LivePlotter:
    """Redraws the training curves each time metrics.json gains an epoch."""

    def __init__(self, metric: str, display_fig: Callable[[Figure], None] | None = None):
        self.metric = metric
        self.updates = 0
        self._display = display_fig or _ipython_display

    def update(self, metrics: dict) -> None:
        self.updates += 1
        epochs = metrics.get("epochs") or []
        if not epochs:
            return
        fig = training_curves(epochs, self.metric)
        try:
            self._display(fig)
        finally:
            plt.close(fig)


def build_run_entry(config: dict, result: RunResult, started_at: str) -> dict:
    metrics = result.metrics or {}
    epochs = metrics.get("epochs") or []
    last = epochs[-1] if epochs else {}
    error = metrics.get("error")
    if error is None and result.timed_out:
        error = "timed out"
    elif error is None and result.returncode != 0:
        error = f"exit code {result.returncode}"
    return {
        "started_at": started_at,
        "status": "done" if result.ok else "failed",
        "config": dict(config),
        "epochs_run": len(epochs),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_metric": metrics.get("best_val_metric"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "seconds": result.seconds,
        "error": error,
        "applied_diff": None,
    }


class TrainStage:
    name = "train"

    def __init__(
        self,
        runner: Callable[..., RunResult] = run_training,
        python: str = sys.executable,
        timeout: float | None = None,
        display_fig: Callable[[Figure], None] | None = None,
        poll_seconds: float = 1.0,
    ):
        self.runner = runner
        self.python = python
        self.timeout = timeout
        self.display_fig = display_fig
        self.poll_seconds = poll_seconds

    def is_complete(self, ctx: StageContext) -> bool:
        return any(r.get("status") == "done" for r in read_runs(ctx.project.runs_path))

    def run(self, ctx: StageContext) -> None:
        project = ctx.project
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not all((project.root / f).exists() for f in CODE_FILES):
            raise RuntimeError("training project not found; run the codegen stage first")
        spec = ctx.spec()
        ctx.display(
            "Training runs on the [[CPU]] for tabular data, so there is no [[compute unit]] "
            "cost gate for this run. Watch the curves update each [[epoch]]."
        )
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        plotter = LivePlotter(spec.metric, display_fig=self.display_fig)
        result = self.runner(
            project.root,
            on_line=lambda line: None,
            on_metrics=plotter.update,
            python=self.python,
            timeout=self.timeout,
            poll_seconds=self.poll_seconds,
        )
        entry = append_run(project.runs_path, build_run_entry(config, result, started_at))
        run_id = entry["run_id"]
        if not result.ok:
            tail = "".join(result.log_tail[-LOG_TAIL_SHOWN:]).rstrip()
            ctx.display(
                f"Run {run_id} failed ({entry['error']}). Last lines of output:\n\n"
                f"```\n{tail}\n```\n\nFix the cause and rerun this stage."
            )
            return

        metrics = result.metrics or {}
        present(training_curves(metrics.get("epochs") or [], spec.metric), project.plots_dir,
                f"run{run_id}_training")
        eval_data = project.read_json(EVAL_VAL_FILE) or {}
        figure_paths = present_evaluation(eval_data, project.plots_dir, f"run{run_id}_val")
        ctx.display(self._debrief(ctx, spec, entry, metrics, eval_data, figure_paths))

    def _debrief(self, ctx: StageContext, spec, entry: dict, metrics: dict, eval_data: dict,
                 figure_paths: list[Path]) -> str:
        summary = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "run_id": entry["run_id"],
            "epochs": metrics.get("epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "stopped_early": metrics.get("stopped_early"),
            "validation": {"metric": eval_data.get("metric"), "value": eval_data.get("value"),
                           "loss": eval_data.get("loss")},
            "figures": [p.name for p in figure_paths],
        }
        headline = (
            f"Run {entry['run_id']} finished: best validation {spec.metric} "
            f"{entry['best_val_metric']:.4g} at epoch {entry['best_epoch']} "
            f"(target {spec.target_value:g})."
        )
        try:
            narrative = ask_text(ctx.llm, load_prompt("train"), json.dumps(summary, default=str))
        except LLMError:
            narrative = "Look at the [[loss]] curves: if validation loss rises while training " \
                        "loss keeps falling, the model is [[overfitting]]."
        return headline + "\n\n" + narrative
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_train_stage.py -v`
Expected: PASS (the real-run test takes a few seconds).

- [ ] **Step 6: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: train stage runs train.py with a live plot, logs the run and shows evaluation figures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 9: Report stage

**Files:**
- Create: `mlagent/stages/report.py`
- Create: `mlagent/prompts/report.md`
- Test: `tests/test_report_stage.py`

**Interfaces:**
- Consumes: `runner.run_script`, `RunResult`; `runlog.read_runs/best_run/summarise`; `plots.present_evaluation`; `ask_text`, `load_prompt`; `config.REPORT_FILE`; `Project.report_path/plots_dir`.
- Produces: `ReportStage(runner=run_script, python=sys.executable, timeout=None, poll_seconds=0.5)` with `name = "report"`, `is_complete(ctx)` (`report.md` exists), `run(ctx)`; `render_report(project_name, spec, runs, best, eval_test, lessons, figures: list[Path]) -> str`; constant `EVAL_TEST_FILE = "eval_test.json"`.

- [ ] **Step 1: Write `mlagent/prompts/report.md`**

```markdown
You are the report stage of an ML training assistant that runs inside Google Colab. The user has finished experimenting. You receive JSON with the project spec (goal, task type, metric, target value), the run history (each run's config, best validation value, final losses, status), the best run, and the single held-out test evaluation.

Write the "What we learned" section of the final report (under 200 words) for a learner:

- One sentence comparing the test result with the best validation result and the target; say plainly whether the goal was met and, if test is noticeably worse than validation, name that gap as [[generalisation]] error.
- Two or three sentences on what the run history shows: which configuration changes helped, which did not, and what the loss curves suggested (for example [[overfitting]] or a [[plateau]]).
- One or two concrete next steps if the user wants to improve further (more data, a different feature, longer training, regularisation).

Wrap technical terms in double square brackets like [[test set]] so the user can click them. Use only the numbers given. Do not restate the raw JSON and do not write a table; the report already has one.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_report_stage.py
from __future__ import annotations

from pathlib import Path

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.runner import RunResult
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.report import ReportStage, render_report
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner


def trained(project):
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner(["y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().run(ctx)
    cfg = project.read_json("config.json")
    cfg.update({"epochs": 2, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", cfg)
    TrainStage(poll_seconds=0.05).run(ctx)
    assert TrainStage().is_complete(ctx)
    return project


def make_ctx(project, llm=None, answers=("y",)):
    shown = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append)
    return ctx, shown


def test_report_evaluates_test_once_and_writes_markdown(clean_project):
    project = trained(clean_project)
    llm = FakeLLM([[("text", "The [[test set]] score was close to validation.")]])
    ctx, shown = make_ctx(project, llm)
    stage = ReportStage(poll_seconds=0.05)
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    assert project.exists("eval_test.json")
    report = project.report_path.read_text(encoding="utf-8")
    assert report.startswith("# ")
    assert "| run | status |" in report
    assert "## Held-out test result" in report and "accuracy" in report
    assert "## Best configuration" in report and "learning_rate" in report
    assert "## What we learned" in report and "[[test set]]" in report
    assert "![" in report and "plots/test_confusion.png" in report
    assert "plots/run1_training.png" in report
    assert sorted(p.name for p in project.plots_dir.glob("test_*.png")) == [
        "test_confusion.png", "test_per_class.png", "test_roc_pr.png"
    ]
    assert any("[[test set]]" in s for s in shown)


def test_declining_the_confirm_leaves_stage_incomplete(clean_project):
    project = trained(clean_project)
    ctx, _ = make_ctx(project, answers=["n"])
    stage = ReportStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert not project.exists("eval_test.json")


def test_no_successful_run_stops_early(clean_project):
    ctx, shown = make_ctx(clean_project)
    stage = ReportStage()
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert any("train" in s.lower() for s in shown)


def test_eval_failure_is_shown(clean_project):
    project = trained(clean_project)

    def fake_runner(root, args, **kwargs):
        return RunResult(returncode=1, metrics=None, log_tail=["KeyError: 'x'\n"], seconds=0.1)

    ctx, shown = make_ctx(project)
    stage = ReportStage(runner=fake_runner)
    stage.run(ctx)
    assert not stage.is_complete(ctx)
    assert any("KeyError" in s for s in shown)


def test_render_report_structure():
    runs = [{"run_id": 1, "status": "done", "epochs_run": 2, "best_epoch": 2,
             "best_val_metric": 0.8, "final_train_loss": 0.3, "final_val_loss": 0.5,
             "seconds": 1.0, "config": {"learning_rate": 0.1}}]
    spec = {"goal": "Predict churn", "task_type": "tabular_classification",
            "metric": "accuracy", "target_value": 0.9}
    eval_test = {"metric": "accuracy", "value": 0.78, "loss": 0.55}
    text = render_report("demo", spec, runs, runs[0], eval_test, "Lessons here.",
                         [Path("plots/test_confusion.png")])
    assert text.splitlines()[0] == "# demo: training report"
    assert "Predict churn" in text and "0.78" in text and "0.9" in text
    assert "Lessons here." in text and "![test_confusion](plots/test_confusion.png)" in text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_report_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.report'`.

- [ ] **Step 4: Write `mlagent/stages/report.py`**

```python
"""Report stage: one held-out test evaluation, then report.md with figures."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from mlagent import config as cfg
from mlagent.llm import LLMError, ask_text
from mlagent.plots import present_evaluation
from mlagent.prompts_io import load_prompt
from mlagent.runlog import best_run, read_runs, summarise
from mlagent.runner import RunResult, run_script
from mlagent.stages.base import StageContext

EVAL_TEST_FILE = "eval_test.json"
LOG_TAIL_SHOWN = 15


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(project_name: str, spec: dict, runs: list[dict], best: dict | None,
                  eval_test: dict, lessons: str, figures: list[Path]) -> str:
    metric = spec.get("metric", "")
    best_cfg = (best or {}).get("config") or {}
    lines = [
        f"# {project_name}: training report",
        "",
        f"**Goal:** {spec.get('goal', '')}",
        "",
        f"**Task:** {spec.get('task_type', '')} | **Metric:** {metric} | "
        f"**Target:** {_fmt(spec.get('target_value'))}",
        "",
        "## Run history",
        "",
        summarise(runs, metric),
        "",
        "## Best configuration",
        "",
        f"Run {(best or {}).get('run_id', '-')} with validation {metric} "
        f"{_fmt((best or {}).get('best_val_metric'))}:",
        "",
        "```json",
        json.dumps(best_cfg, indent=2, sort_keys=True),
        "```",
        "",
        "## Held-out test result",
        "",
        f"Test {eval_test.get('metric', metric)}: **{_fmt(eval_test.get('value'))}** "
        f"(loss {_fmt(eval_test.get('loss'))}). Evaluated once on the test split.",
        "",
        "## What we learned",
        "",
        lessons.strip(),
        "",
        "## Figures",
        "",
    ]
    for path in figures:
        rel = Path("plots") / path.name
        lines.append(f"![{path.stem}]({rel.as_posix()})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class ReportStage:
    name = "report"

    def __init__(
        self,
        runner: Callable[..., RunResult] = run_script,
        python: str = sys.executable,
        timeout: float | None = None,
        poll_seconds: float = 0.5,
    ):
        self.runner = runner
        self.python = python
        self.timeout = timeout
        self.poll_seconds = poll_seconds

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.report_path.exists()

    def run(self, ctx: StageContext) -> None:
        project = ctx.project
        spec = ctx.spec()
        runs = read_runs(project.runs_path)
        best = best_run(runs, spec.metric)
        if best is None:
            ctx.display("No successful training run yet; run the train stage first.")
            return
        ctx.display(
            f"The best run so far is run {best['run_id']} with validation {spec.metric} "
            f"{_fmt(best.get('best_val_metric'))}. The [[test set]] has been untouched until now; "
            "evaluating on it once gives an honest estimate of real-world performance."
        )
        if not ctx.questioner.confirm(
            "Evaluate the best model on the held-out test set now and write the report?",
            default=True,
        ):
            ctx.display("Skipped. Rerun this stage when you have finished tuning.")
            return

        result = self.runner(
            project.root,
            ["train.py", "--eval-test"],
            python=self.python,
            timeout=self.timeout,
            poll_seconds=self.poll_seconds,
        )
        if not result.ok:
            tail = "".join(result.log_tail[-LOG_TAIL_SHOWN:]).rstrip()
            ctx.display(f"Test evaluation failed. Last lines of output:\n\n```\n{tail}\n```")
            return
        eval_test = project.read_json(EVAL_TEST_FILE) or {}
        test_figures = present_evaluation(eval_test, project.plots_dir, "test")
        run_figures = sorted(project.plots_dir.glob("run*_training.png"))

        lessons = self._lessons(ctx, spec, runs, best, eval_test)
        report = render_report(
            project.name, spec.to_dict(), runs, best,
            eval_test, lessons, run_figures + test_figures,
        )
        project.report_path.write_text(report, encoding="utf-8")
        ctx.display(
            f"Test {spec.metric}: **{_fmt(eval_test.get('value'))}** "
            f"(target {spec.target_value:g}).\n\n{lessons}\n\n"
            f"Report written to `{cfg.REPORT_FILE}` in the project folder."
        )

    def _lessons(self, ctx: StageContext, spec, runs: list[dict], best: dict,
                 eval_test: dict) -> str:
        summary = {
            "spec": spec.to_dict(),
            "runs": [
                {k: r.get(k) for k in ("run_id", "status", "config", "best_epoch",
                                        "best_val_metric", "final_train_loss",
                                        "final_val_loss", "error")}
                for r in runs
            ],
            "best_run_id": best.get("run_id"),
            "test": {"metric": eval_test.get("metric"), "value": eval_test.get("value"),
                     "loss": eval_test.get("loss")},
        }
        try:
            return ask_text(ctx.llm, load_prompt("report"), json.dumps(summary, default=str))
        except LLMError:
            return (
                f"Best validation {spec.metric} was {_fmt(best.get('best_val_metric'))}; "
                f"the [[test set]] gave {_fmt(eval_test.get('value'))}. A large gap between "
                "them means the model does not [[generalise]] well."
            )
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_report_stage.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: report stage evaluates once on the test split and writes report.md with figures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 10: Wire into Colab, notebook, docs, end-to-end test

**Files:**
- Modify: `mlagent/colab.py`
- Modify: `scripts/build_notebook.py` and regenerate `notebooks/ML_Training_Agent.ipynb`
- Modify: `docs/colab-smoke.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_pipeline_e2e.py`
- Test: `tests/test_colab.py` (extend)

**Interfaces:**
- Consumes: `CodegenStage`, `TrainStage`, `ReportStage`; `runlog.read_runs`; `config.CONFIG_FILE`.
- Produces: `colab.start()` runs six stages; `_context_snapshot` adds `"config"` and `"latest_run"`.

- [ ] **Step 1: Extend the end-to-end test**

Replace the imports, `make_orchestrator`, and the two final blocks of `tests/test_pipeline_e2e.py` so the file becomes:

```python
"""End-to-end run of intake -> data -> clean -> codegen -> train -> report, no network."""

from __future__ import annotations

import pandas as pd

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.runlog import read_runs
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CLEAN_FILE, CLEAN_PY, CleanStage
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.data import RAW_FILE, DataStage
from mlagent.stages.intake import IntakeStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner

# Answers for every non-confirm question asked across the pipeline, in order.
# codegen, train and report ask only confirm() questions, which AutoApproveQuestioner answers.
ANSWERS = [
    "Predict churn from account data",  # intake: goal
    "Tabular classification",           # intake: task type label
    "accuracy",                         # intake: metric
    "0.9",                              # intake: target value
    "Synthetic data",                   # intake: data source label
    "10",                               # intake: minutes per run
    "5",                                # intake: max rounds
    "No GPU (CPU only)",                # intake: gpu label
    "300",                              # data: n rows
    "6",                                # data: n features
    "2",                                # data: n classes
    "0.6",                              # data: class balance
    "0.1",                              # data: noise
    "",                                 # clean: extra columns to drop (none)
    "0.7",                              # clean: train fraction
    "0.15",                             # clean: validation fraction
]

ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Like ScriptedQuestioner, but every confirm() is approved without consuming
    a scripted answer (the number of audit fixes varies with the data)."""

    def confirm(self, question: str, default: bool = True) -> bool:
        self.asked.append(question)
        return True


def make_orchestrator(project, answers):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback text
        questioner=AutoApproveQuestioner(answers),
        explainer=None,
        display=lambda s: None,
    )
    stages = [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
              TrainStage(poll_seconds=0.05), ReportStage(poll_seconds=0.05)]
    return Orchestrator(ctx, stages)


def test_full_pipeline_runs_and_is_reproducible_and_resumable(project):
    orch = make_orchestrator(project, list(ANSWERS))
    ran = orch.run()
    assert ran == ALL_STAGES

    # Artifacts from every stage exist.
    assert project.exists("spec.json")
    assert project.exists("draft_spec.json")
    assert (project.data_raw / RAW_FILE).exists()
    assert project.exists("data_meta.json")
    assert project.exists("profile_raw.json")
    assert (project.data_clean / CLEAN_FILE).exists()
    assert project.exists(AUDIT_FILE)
    assert project.exists("profile_clean.json")
    assert (project.root / CLEAN_PY).exists()
    for name in ("data.py", "model.py", "train.py", "config.json", "metrics.json",
                 "eval_val.json", "eval_test.json", "runs.jsonl", "report.md"):
        assert project.exists(name), name
    assert (project.checkpoints_dir / "best.joblib").exists()
    assert (project.plots_dir / "run1_training.png").exists()
    assert (project.plots_dir / "test_confusion.png").exists()
    assert project.exists("state.json")
    runs = read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done"

    # clean.py reproduces data/clean/data.csv exactly when run on data/raw/data.csv.
    raw_df = pd.read_csv(project.data_raw / RAW_FILE)
    clean_df = pd.read_csv(project.data_clean / CLEAN_FILE)
    namespace: dict = {}
    code = compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec")
    exec(code, namespace)
    reproduced = namespace["clean"](raw_df).reset_index(drop=True)
    pd.testing.assert_frame_equal(reproduced, clean_df, check_dtype=False)

    # Deleting state.json: stages are already complete via artifact detection, nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []

    # Resetting "train" reruns training and the report; a second run is logged.
    orch.reset("train")
    orch.ctx.questioner = AutoApproveQuestioner([])
    assert orch.run() == ["train", "report"]
    assert len(read_runs(project.runs_path)) == 2
    assert (project.plots_dir / "run2_training.png").exists()
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_pipeline_e2e.py -v`
Expected: PASS already if Tasks 7 to 9 are correct (this test exercises them through the orchestrator). If it fails, fix the stage, not the test.

- [ ] **Step 3: Wire `mlagent/colab.py`**

Add imports:

```python
from mlagent.runlog import read_runs
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
```

Replace `_context_snapshot` with:

```python
def _context_snapshot(project: Project, stage_name: str = "") -> dict:
    runs = read_runs(project.runs_path)
    return {
        "project": project.name,
        "stage": stage_name,
        "spec": project.read_json(config.SPEC_FILE),
        "state": project.read_json(config.STATE_FILE),
        "data_meta": project.read_json(META_FILE),
        "audit_issue_kinds": [
            i.get("kind") for i in (project.read_json(AUDIT_FILE) or {}).get("issues", [])
        ],
        "config": project.read_json(config.CONFIG_FILE),
        "latest_run": runs[-1] if runs else None,
    }
```

Replace the stage list in `start()` with:

```python
    return Orchestrator(
        ctx,
        [IntakeStage(), DataStage(), CleanStage(), CodegenStage(), TrainStage(), ReportStage()],
    )
```

Add to `tests/test_colab.py`:

```python
def test_start_wires_six_stages(tmp_path):
    from mlagent import colab
    from mlagent.llm import FakeLLM

    orch = colab.start("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert [s.name for s in orch.stages] == ["intake", "data", "clean", "codegen", "train",
                                             "report"]
    snap = colab._context_snapshot(orch.ctx.project, "train")
    assert snap["config"] is None and snap["latest_run"] is None
```

If `test_colab.py` already has a test asserting the three-stage list, update it to the six-stage list instead of adding a duplicate.

- [ ] **Step 4: Notebook**

In `scripts/build_notebook.py`, after the cell that runs `orch.run()`, add a markdown cell and a code cell:

```python
    nbf.v4.new_markdown_cell(
        "## Train again\n\n"
        "Edit `config.json` in the project folder (or let the assistant propose changes in a "
        "later milestone), then rerun training and the report from here. `reset(\"train\")` "
        "forgets the train and report stages; earlier stages are kept."
    ),
    nbf.v4.new_code_cell('orch.reset("train")\norch.run()'),
```

Match the variable names and cell-construction style already used in the script. Then regenerate:

Run: `python scripts/build_notebook.py`

- [ ] **Step 5: Smoke checklist and CLAUDE.md**

Append to `docs/colab-smoke.md`:

```markdown
## Milestone 3 (Tabular training end-to-end)

Prerequisite: a project that has completed the Milestone 2 checklist (clean data and splits).

1. Run the start cell. The codegen stage lists `data.py`, `model.py`, `train.py`, `config.json`
   in the project folder on Drive and shows a config table with a rationale. Click one
   `[[term]]` and confirm an explanation appears.
2. Answer "y" to the configuration question. The train stage prints the CPU/no-cost-gate note,
   then a loss/metric figure redraws in the cell as epochs complete.
3. When training ends: `runs.jsonl` has one line; `plots/` contains `run1_training.png` and the
   validation evaluation figures; the debrief mentions the best epoch.
4. The report stage asks before touching the test set. Answer "y". `eval_test.json` and
   `report.md` appear in the project folder; open `report.md` in Drive and check the figures render.
5. Run the "Train again" cell. A second run is logged as run 2 and the report is rewritten.
6. Disconnect and reconnect the runtime, rerun the setup and start cells: the orchestrator reports
   nothing to do.
```

Update `CLAUDE.md`:

- In "What this is", the pipeline sentence should read `intake -> data -> clean -> codegen -> train -> report`.
- In Architecture, replace the pipeline bullet with:

```markdown
- Pipeline: `intake` -> `data` -> `clean` -> `codegen` -> `train` -> `report` (`mlagent/stages/`). `data` writes `data/raw/data.csv`, `profile_raw.json`, `data_meta.json`. `clean` writes `data/clean/data.csv`, `clean.py`, `audit.json`, `profile_clean.json` and completes `data_meta.json`. `codegen` copies `mlagent/templates/<family>/{data,model,train}.py` into the project (they import only numpy/pandas/scikit-learn/joblib, never `mlagent`) and writes `config.json`, whose keys are exactly those in the template's `config_schema.json`. `train` runs `train.py` in a subprocess (`runner.py`), polls `metrics.json` for the live plot, appends to `runs.jsonl` (`runlog.py`), and shows evaluation figures from `eval_val.json`. `report` runs `train.py --eval-test` once and writes `report.md`. Raw data is never modified; the test split is evaluated only by the report stage.
- `data_meta.json` contract (frozen in the Milestone 3 plan): `target`, `task_type`, `source`, `raw_path`, `raw_n_rows`, `raw_n_cols` (data stage); `clean_path`, `clean_n_rows`, `clean_n_cols`, `dropped_columns`, `feature_columns`, `categorical_columns`, `splits` `{train, val, test}` fractions, and for classification `n_classes`, `class_labels` (clean stage). Generated `data.py` reads exactly these keys.
```

- Add to Commands: `python -m pytest tests/test_pipeline_e2e.py -v   # full pipeline including a real CPU training run`.

- [ ] **Step 6: Run the full suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: wire codegen, train and report stages into the Colab entry point; notebook, smoke checklist, docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

## Deferred to Milestone 4 or later (recorded so nothing is lost)

- `tune` stage: diagnosis, config-diff proposals bounded by `config_schema.json`, run comparison and overlaid loss curve plots, `applied_diff` on `runs.jsonl` entries (the field already exists, always `None` here).
- Traceback-to-fix-diff loop for failing generated scripts (max 3 attempts) and NaN-loss feedback into tuning; `train.py` already exits with `status: failed` and an `error` message for the runner.
- `tabular_torch` template (torch is not a dependency).
- Cost gate wiring (`train.py --dry-run` exists; `cost.py` and the gate are Milestone 5).
- A test that exercises `CleanStage`'s readable-error path when a cleaning step fails (carried from the Milestone 2 review).
