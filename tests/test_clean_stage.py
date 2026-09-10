import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlagent.audit import audit_tabular
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.clean import (
    AUDIT_FILE,
    CLEAN_FILE,
    CLEAN_PY,
    PROFILE_CLEAN_FILE,
    SPLIT_SEED,
    CleanStage,
)
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


def prepare(project, df, task_type="tabular_classification"):
    project.write_json("spec.json", {**SPEC, "task_type": task_type})
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / RAW_FILE, index=False)
    project.write_json(META_FILE, {"source": "synthetic", "target": "target",
                                   "task_type": task_type,
                                   "raw_path": "data/raw/data.csv"})


def make_ctx(project, answers, llm=None):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Report card [[missing values]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return ctx, shown, figures


def run_clean_script(project):
    result = subprocess.run(
        [sys.executable, "clean.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def user_id_collision_df() -> pd.DataFrame:
    """Unique-on-raw-values user_id (triggers id_column) with case collisions once
    normalised (triggers inconsistent_categories on the same column)."""
    n = 40
    values = [f"User{i}" for i in range(n)]
    values[0] = "user1"  # collides with "User1" once lower-cased
    values[2] = "user3"  # collides with "User3" once lower-cased
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "user_id": values,
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })


def test_approved_fixes_are_written_to_clean_py_and_debriefed(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    assert n_fixable >= 4
    ctx, shown, figures = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"])
    stage = CleanStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(
        stage="clean", commands=[["clean.py"]],
        outputs=["data/clean/data.csv", "profile_clean.json"],
    )
    audit = project.read_json(AUDIT_FILE)
    assert len(audit["decisions"]) == len(audit["issues"]) and len(audit["steps"]) == n_fixable
    assert all(d["approved"] for d in audit["decisions"] if d["fix"])
    meta = project.read_json(META_FILE)
    assert meta["splits"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert meta["dropped_columns"] == [] and meta["split_seed"] == SPLIT_SEED
    assert "feature_columns" not in meta  # only known after clean.py has run
    assert (project.root / CLEAN_PY).exists()
    assert not stage.is_complete(ctx)

    run_clean_script(project)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert "row_id" not in cleaned.columns and "const" not in cleaned.columns
    assert cleaned.duplicated().sum() == 0 and cleaned["f1"].isna().sum() == 0
    assert set(cleaned["cat"].unique()) == {"a", "b", "c"}
    meta = project.read_json(META_FILE)
    assert meta["clean_path"] == "data/clean/data.csv"
    assert meta["clean_n_rows"] == len(cleaned) and meta["clean_n_cols"] == cleaned.shape[1]
    assert meta["feature_columns"] == [c for c in cleaned.columns if c != "target"]
    assert meta["categorical_columns"] == ["cat"]
    assert meta["n_classes"] == 2 and meta["class_labels"] == ["0", "1"]
    assert project.read_json(PROFILE_CLEAN_FILE)["after"]["n_rows"] == len(cleaned)
    assert [Path(p).name for p, _c in figures] == ["clean_before_after_missing.png"]
    assert "missing" in figures[0][1]
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
    ctx, _shown, _figures = make_ctx(project, ["n"] * n_fixable + ["f2, nope", "0.8", "0.1"])
    CleanStage().prepare(ctx)
    audit = project.read_json(AUDIT_FILE)
    assert audit["steps"] == [{"op": "drop_columns", "params": {"columns": ["f2"]}}]
    assert project.read_json(META_FILE)["dropped_columns"] == ["f2"]


def test_clean_data_has_no_issues_and_splits_are_adjusted(project):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "a": rng.normal(size=100), "b": rng.normal(size=100), "target": rng.integers(0, 2, 100),
    })
    prepare(project, df)
    ctx, shown, _figures = make_ctx(project, ["", "0.9", "0.3"])
    CleanStage().prepare(ctx)
    assert any("no problems" in s for s in shown)
    splits = project.read_json(META_FILE)["splits"]
    assert splits["test"] >= 0.05 and abs(sum(splits.values()) - 1.0) < 1e-6
    assert json.loads((project.root / AUDIT_FILE).read_text(encoding="utf-8"))["issues"] == []


def test_approved_drop_and_percolumn_fix_on_same_column_does_not_crash(project):
    df = user_id_collision_df()
    prepare(project, df)
    issues = audit_tabular(df, "target")
    by_kind = {i.kind: i for i in issues if i.column == "user_id"}
    assert "id_column" in by_kind and "inconsistent_categories" in by_kind
    n_fixable = sum(1 for i in issues if i.fix)
    ctx, _shown, _figures = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"])
    CleanStage().prepare(ctx)  # must not raise
    run_clean_script(project)
    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert "user_id" not in cleaned.columns
    audit = project.read_json(AUDIT_FILE)
    skipped = next(d for d in audit["decisions"] if d["kind"] == "inconsistent_categories")
    assert skipped["approved"] is True
    assert skipped["applied"] is False
    assert skipped["reason"] == "column dropped"
    dropped = next(d for d in audit["decisions"] if d["kind"] == "id_column")
    assert dropped["approved"] is True and dropped["applied"] is True


def test_target_becomes_float_via_nan_drop_is_saved_as_int(project):
    rng = np.random.default_rng(9)
    n = 100
    target = rng.integers(0, 2, size=n).astype(float)
    target[:5] = np.nan
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "target": target,
    })
    prepare(project, df)
    issues = audit_tabular(df, "target")
    n_fixable = sum(1 for i in issues if i.fix)
    ctx, _shown, _figures = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"])
    CleanStage().prepare(ctx)
    run_clean_script(project)
    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert cleaned["target"].isna().sum() == 0
    assert pd.api.types.is_integer_dtype(cleaned["target"].dtype)


def test_learning_level_reaches_the_debrief_prompt(project):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "a": rng.normal(size=100), "b": rng.normal(size=100), "target": rng.integers(0, 2, 100),
    })
    prepare(project, df)
    project.write_json("spec.json", {**SPEC, "learning_level": "beginner"})
    llm = FakeLLM([
        [("text", "Getting your data ready to clean.")],  # prepare()'s preamble
        [("text", "done")],  # the post-fix clean_debrief call
    ])
    ctx, _shown, _figures = make_ctx(project, ["", "0.9", "0.3"], llm=llm)
    stage = CleanStage()
    stage.prepare(ctx)
    run_clean_script(project)
    stage.debrief(ctx)
    system = llm.calls[-1]["system"]
    assert "new to machine learning" in system
    assert "The cleaning steps have now run" in system  # unique to clean_debrief.md


def test_llm_failure_does_not_block_cleaning(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    ctx, shown, _figures = make_ctx(
        project, ["y"] * n_fixable + ["", "0.7", "0.15"], llm=FakeLLM([])
    )
    CleanStage().prepare(ctx)
    assert any("Couldn't reach Claude" in s for s in shown)
    assert (project.root / CLEAN_PY).exists()


def test_debrief_without_the_clean_csv_says_so(project):
    prepare(project, messy_df())
    n_fixable = sum(1 for i in audit_tabular(messy_df(), "target") if i.fix)
    ctx, shown, _figures = make_ctx(project, ["n"] * n_fixable + ["", "0.7", "0.15"])
    stage = CleanStage()
    stage.prepare(ctx)
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("clean.py" in s for s in shown)


def image_ctx(project, answers=(), form=None):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    shown: list[str] = []
    scripted = ScriptedQuestioner(list(answers))
    questioner = FormQuestioner(form or {}, scripted) if form is not None else scripted
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=questioner,
                       explainer=None, display=shown.append,
                       display_figure=lambda path, caption="": None)
    return ctx, shown


def test_image_clean_prepare_writes_the_audit_and_an_image_clean_py(project):
    from mlagent.imageset import write_pair
    from mlagent.stages.clean import CleanStage
    from mlagent.synth.images import SynthImageConfig, generate

    imageset = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=3,
                                         duplicate_fraction=0.2, blank_fraction=0.1))
    write_pair(imageset, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "source": "synthetic", "raw_path": "data/raw/data.npz", "raw_n_rows": 40,
        "raw_n_cols": 32 * 32 * 3, "image_size": 32, "n_channels": 3,
        "class_labels": list(imageset.class_names), "n_classes": 2, "skipped_files": [],
    })
    project.write_json("spec.json", {
        "goal": "shapes", "task_type": "image_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 3, "gpu": "none", "notes": "",
    })
    ctx, _shown = image_ctx(project, answers=["y"] * 10,
                            form={"clean.train_fraction": 0.7, "clean.val_fraction": 0.15})
    handoff = CleanStage().prepare(ctx)

    assert handoff.commands == [["clean.py"]]
    assert handoff.outputs == ["data/clean/data.npz", "profile_clean.json"]
    audit = project.read_json("audit.json")
    assert {i["kind"] for i in audit["issues"]} & {"duplicate_images", "blank_images"}
    assert all(s["op"] == "drop_indices" for s in audit["steps"])
    source = (project.root / "clean.py").read_text(encoding="utf-8")
    assert "Apply the image drops you approved" in source
    meta = project.read_json("data_meta.json")
    assert meta["dropped_columns"] == [] and meta["split_seed"] == 42


def test_image_clean_report_card_payload_uses_image_shaped_counts(project):
    from mlagent.imageset import write_pair
    from mlagent.stages.clean import CleanStage
    from mlagent.synth.images import SynthImageConfig, generate

    imageset = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=3,
                                         duplicate_fraction=0.2, blank_fraction=0.1))
    write_pair(imageset, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "source": "synthetic", "raw_path": "data/raw/data.npz", "raw_n_rows": 40,
        "raw_n_cols": 32 * 32 * 3, "image_size": 32, "n_channels": 3,
        "class_labels": list(imageset.class_names), "n_classes": 2, "skipped_files": [],
    })
    project.write_json("spec.json", {
        "goal": "shapes", "task_type": "image_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 3, "gpu": "none", "notes": "",
    })
    ctx, _shown = image_ctx(project, answers=["y"] * 10,
                            form={"clean.train_fraction": 0.7, "clean.val_fraction": 0.15})
    CleanStage().prepare(ctx)

    payload = json.loads(ctx.llm.calls[-1]["messages"][0]["content"])
    summary = payload["profile_summary"]
    assert summary["n_rows"] == 40
    assert summary["n_cols"] == 32 * 32 * 3
    assert summary["n_classes"] == 2
    assert summary["image_size"] == 32


def test_image_clean_debrief_completes_the_meta(clean_image_project):
    from mlagent.stages.clean import CleanStage

    project = clean_image_project
    project.write_json("profile_clean.json", {
        "before": {"n_images": 60, "class_counts": {"circle": 20, "cross": 20, "square": 20}},
        "after": {"n_images": 58, "class_counts": {"circle": 19, "cross": 20, "square": 19}},
        "steps": [], "figures": ["clean_before_after_classes.png"],
    })
    ctx, shown = image_ctx(project)
    CleanStage().debrief(ctx)
    meta = project.read_json("data_meta.json")
    assert meta["clean_path"] == "data/clean/data.npz"
    assert meta["clean_n_rows"] == 60
    assert meta["clean_n_cols"] == 32 * 32 * 3
    assert meta["feature_columns"] == [] and meta["categorical_columns"] == []
    assert meta["n_classes"] == 3
    assert meta["class_labels"] == ["circle", "square", "triangle"]
    assert any("58" in text or "Cleaning summary" in text for text in shown)


def test_image_clean_debrief_reports_an_emptied_class_as_an_error(clean_image_project):
    from mlagent.imageset import read_pair, write_pair
    from mlagent.stages.clean import CleanStage

    project = clean_image_project
    imageset = read_pair(project.data_clean)
    keep = [i for i, label in enumerate(imageset.labels.tolist()) if label != 0]
    write_pair(imageset.take(keep), project.data_clean)
    ctx, shown = image_ctx(project)
    CleanStage().debrief(ctx)
    joined = "\n".join(shown)
    assert "no images left" in joined
    assert imageset.class_names[0] in joined
