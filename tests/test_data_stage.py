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
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Narrative [[class balance]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
    )
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
    assert meta == {
        **meta, "source": "drive", "target": "churned", "source_path": str(csv), "n_rows": 3,
    }
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

    ctx, shown = make_ctx(
        project, ["nothing", "churn", "org/churn — 10 downloads — customer churn", "y"],
        source="huggingface",
    )
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
