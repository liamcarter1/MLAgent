from pathlib import Path

import pandas as pd
import pytest

from mlagent.datasources.hf import HFDataset
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.data import META_FILE, PROFILE_RAW_FILE, RAW_FILE, DataStage
from mlagent.templates_io import COMMON_FILES
from mlagent.ui.questions import ScriptedQuestioner

SPEC = {
    "goal": "Predict churn", "task_type": "tabular_classification", "metric": "accuracy",
    "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5, "max_rounds": 2,
    "gpu": "none", "notes": "",
}


def make_ctx(project, answers, source="synthetic", task="tabular_classification", llm=None):
    project.write_json("spec.json", {**SPEC, "data_source": source, "task_type": task})
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Narrative [[class balance]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return ctx, shown, figures


def run_profile_script(project):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "profile.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_synthetic_classification_prepares_then_debriefs(project):
    ctx, shown, figures = make_ctx(project, ["300", "5", "2", "0.6", "0.1", "y"])
    stage = DataStage()
    assert not stage.is_complete(ctx)
    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="data", commands=[["profile.py"]],
                              outputs=["profile_raw.json"])
    assert (project.root / "profile.py").exists()
    for name in COMMON_FILES:
        assert (project.root / name).exists()
    df = pd.read_csv(project.data_raw / RAW_FILE)
    assert len(df) > 300 and "row_id" in df.columns and "target" in df.columns
    meta = project.read_json(META_FILE)
    assert meta["source"] == "synthetic" and meta["target"] == "target"
    assert meta["synth_config"]["n_samples"] == 300 and meta["synth_config"]["class_balance"] == 0.6
    assert meta["raw_n_rows"] == len(df) and meta["raw_n_cols"] == df.shape[1]
    assert meta["task_type"] == "tabular_classification"
    assert meta["raw_path"] == "data/raw/data.csv"
    assert not stage.is_complete(ctx)  # the profile has not been produced yet
    assert not stage.outputs_ready(ctx, handoff)

    run_profile_script(project)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert project.read_json(PROFILE_RAW_FILE)["target"]["kind"] == "categorical"
    shown_figures = {Path(p).name for p, _c in figures}
    assert shown_figures == {"raw_histograms.png", "raw_missing.png",
                             "raw_class_balance.png", "raw_correlation.png"}
    captions = {Path(p).name: c for p, c in figures}
    assert "histogram" in captions["raw_histograms.png"]
    assert any("[[class balance]]" in s for s in shown)
    assert any("Data profile" in s for s in shown)


def test_synthetic_regression_without_quirks(project):
    ctx, _shown, _figures = make_ctx(
        project, ["200", "4", "0.2", "n"], task="tabular_regression"
    )
    DataStage().prepare(ctx)
    df = pd.read_csv(project.data_raw / RAW_FILE)
    assert df.shape == (200, 5)


def test_drive_source_lists_files_and_asks_target(project, tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    csv = root / "customers.csv"
    csv.write_text("age,income,churned\n30,100,0\n40,200,1\n50,300,0\n", encoding="utf-8")
    ctx, _shown, _figures = make_ctx(project, [str(csv), "churned"], source="drive")
    DataStage(search_roots=[root]).prepare(ctx)
    meta = project.read_json(META_FILE)
    assert meta == {
        **meta, "source": "drive", "target": "churned", "source_path": str(csv),
        "raw_n_rows": 3,
    }
    assert "Which column is the target" in ctx.questioner.asked[-1]


def test_drive_source_missing_file_raises(project, tmp_path):
    ctx, _shown, _figures = make_ctx(project, [str(tmp_path / "nope.csv")], source="drive")
    with pytest.raises(FileNotFoundError):
        DataStage(search_roots=[tmp_path]).prepare(ctx)


def test_huggingface_source_searches_picks_and_loads(project):
    searches: list[str] = []

    def fake_search(query, limit=8):
        searches.append(query)
        return [] if len(searches) == 1 else [HFDataset("org/churn", 10, 1, "customer churn")]

    def fake_load(dataset_id):
        assert dataset_id == "org/churn"
        return pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    ctx, shown, _figures = make_ctx(
        project, ["nothing", "churn", "org/churn — 10 downloads — customer churn", "y"],
        source="huggingface",
    )
    DataStage(hf_search=fake_search, hf_load=fake_load).prepare(ctx)
    assert searches == ["nothing", "churn"]
    assert project.read_json(META_FILE)["hf_id"] == "org/churn"
    assert any("No datasets found" in s for s in shown)


def test_image_task_is_not_supported_yet(project):
    ctx, _shown, _figures = make_ctx(project, [], task="image_classification")
    with pytest.raises(NotImplementedError):
        DataStage().prepare(ctx)


def test_llm_failure_still_completes(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"],
                                    llm=FakeLLM([]))
    stage = DataStage()
    stage.prepare(ctx)
    run_profile_script(project)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("Couldn't reach Claude" in s for s in shown)


def test_missing_profile_debrief_says_so_and_stays_incomplete(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"])
    stage = DataStage()
    stage.prepare(ctx)
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("profile.py" in s for s in shown)
