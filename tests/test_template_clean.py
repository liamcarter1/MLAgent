from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlagent import captions
from mlagent.cleaning import OPS as LIBRARY_OPS
from mlagent.cleaning import apply_steps, render_clean_py

TEMPLATE = Path("mlagent/templates/common").resolve()

STEPS = [
    {"op": "drop_columns", "params": {"columns": ["row_id"]}},
    {"op": "drop_duplicates", "params": {}},
    {"op": "fill_missing", "params": {"column": "f1", "strategy": "median"}},
    {"op": "normalise_categories", "params": {"column": "cat"}},
    {"op": "clip_outliers", "params": {"column": "f2", "lower": -3.0, "upper": 3.0}},
    {"op": "coerce_numeric", "params": {"column": "f2"}},
    {"op": "drop_rows_missing_target", "params": {"target": "target"}},
]


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_common_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_common_{name}"] = module
    spec.loader.exec_module(module)
    return module


def messy(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n) * 4,
        "cat": rng.choice(["A ", " b", "C"], size=n),
        "target": rng.integers(0, 2, size=n).astype(float),
    })
    df.loc[:4, "f1"] = np.nan
    df.loc[5, "target"] = np.nan
    return pd.concat([df, df.iloc[:3]], ignore_index=True)


def prepare(project) -> pd.DataFrame:
    df = messy()
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / "data.csv", index=False)
    project.write_json("data_meta.json", {
        "target": "target", "task_type": "tabular_classification",
        "raw_path": "data/raw/data.csv",
    })
    (project.root / "clean.py").write_text(render_clean_py(STEPS), encoding="utf-8")
    return df


def run_clean(project):
    result = subprocess.run(
        [sys.executable, "clean.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_rendered_clean_py_matches_apply_steps_and_writes_artifacts(project):
    raw = prepare(project)
    run_clean(project)

    produced = pd.read_csv(project.data_clean / "data.csv")
    expected = apply_steps(raw, STEPS).reset_index(drop=True)
    expected["target"] = expected["target"].astype("int64")
    pd.testing.assert_frame_equal(produced, expected, check_dtype=False)

    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["steps"] == STEPS
    assert written["figures"] == ["clean_before_after_missing.png"]
    assert written["before"]["n_rows"] == len(raw)
    assert written["after"]["n_rows"] == len(produced)
    assert {c["name"] for c in written["after"]["columns"]} == set(produced.columns)
    assert (project.plots_dir / "clean_before_after_missing.png").exists()


def test_rendered_clean_py_is_importable_and_standalone(project):
    prepare(project)
    source = (project.root / "clean.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    namespace: dict = {}
    exec(compile(source, "clean.py", "exec"), namespace)
    assert namespace["STEPS"] == STEPS
    out = namespace["clean"](messy())
    pd.testing.assert_frame_equal(out, apply_steps(messy(), STEPS), check_dtype=False)


def test_template_ops_table_matches_the_cleaning_library():
    module = load_module("clean")
    assert set(module.OPS) == set(LIBRARY_OPS)


def test_template_captions_match_mlagent_captions():
    module = load_module("clean")
    assert module.CAPTIONS == {
        "clean_before_after_missing": captions.CAPTIONS["clean_before_after_missing"]
    }


def test_empty_steps_render_and_run(project):
    df = messy()
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / "data.csv", index=False)
    project.write_json("data_meta.json", {"target": "target",
                                          "task_type": "tabular_regression",
                                          "raw_path": "data/raw/data.csv"})
    (project.root / "clean.py").write_text(render_clean_py([]), encoding="utf-8")
    run_clean(project)
    produced = pd.read_csv(project.data_clean / "data.csv")
    assert len(produced) == len(df)


def test_cli_argv_ignores_kernel_launchers_but_parses_run_and_script_argv(monkeypatch):
    module = load_module("clean")

    monkeypatch.setattr(sys, "argv", ["/x/ipykernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["/x/colab_kernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []

    monkeypatch.setattr(sys, "argv", ["clean.py", "--project", "foo"])
    assert module.cli_argv() == ["--project", "foo"]

    monkeypatch.setattr(sys, "argv", ["/some/dir/clean.py", "--project", "foo"])
    assert module.cli_argv() == ["--project", "foo"]

    monkeypatch.setattr(sys, "argv", ["clean.py"])
    assert module.cli_argv() == []
