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
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Report card [[missing values]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
    )
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
    assert (
        meta["splits"] == {"train": 0.7, "val": 0.15, "test": 0.15}
        and meta["dropped_columns"] == []
    )
    assert project.read_json(PROFILE_CLEAN_FILE)["n_rows"] == len(cleaned)
    assert (project.plots_dir / "clean_before_after_missing.png").exists()
    assert any("[[missing values]]" in s for s in shown)
    assert any("Cleaning summary" in s for s in shown)
    namespace: dict = {}
    code = compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec")
    exec(code, namespace)
    pd.testing.assert_frame_equal(
        namespace["clean"](df).reset_index(drop=True), cleaned, check_dtype=False
    )


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
    df = pd.DataFrame({
        "a": rng.normal(size=100), "b": rng.normal(size=100), "target": rng.integers(0, 2, 100),
    })
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
