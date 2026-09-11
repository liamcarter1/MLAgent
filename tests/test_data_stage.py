from pathlib import Path

import pandas as pd
import pytest

from mlagent.datasources.hf import HFDataset
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.data import META_FILE, PROFILE_RAW_FILE, RAW_FILE, DataStage, guess_target
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


def test_guess_target_matches_a_named_column():
    df = pd.DataFrame({"age": [1], "income": [2], "target": [0]})
    assert guess_target(df) == "target"


def test_guess_target_matches_a_keyword_suffix():
    df = pd.DataFrame({"age": [1], "customer_churn": [0]})
    assert guess_target(df) == "customer_churn"


def test_guess_target_prefers_an_earlier_keyword():
    df = pd.DataFrame({"class": [1], "label": [0]})
    assert guess_target(df) == "label"


def test_guess_target_falls_back_to_the_last_column():
    df = pd.DataFrame({"age": [1], "income": [2], "notes": [3]})
    assert guess_target(df) == "notes"


def test_guess_target_none_for_an_empty_frame():
    assert guess_target(pd.DataFrame()) is None


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
    ctx, shown, _figures = make_ctx(project, [str(csv), "churned"], source="drive")
    DataStage(search_roots=[root]).prepare(ctx)
    meta = project.read_json(META_FILE)
    assert meta == {
        **meta, "source": "drive", "target": "churned", "source_path": str(csv),
        "raw_n_rows": 3,
    }
    assert "Which column is the target" in ctx.questioner.asked[-1]
    assert any("My guess is `churned`" in s for s in shown)


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


def test_huggingface_blank_query_form_value_asks_instead_of_searching_empty(project):
    """A blank `data.hf_query` form value must ask (Task 4): the query question has no
    default, so a blank value is not silently accepted as an empty search."""
    from mlagent.ui.questions import FormQuestioner

    searches: list[str] = []

    def fake_search(query, limit=8):
        searches.append(query)
        return [HFDataset("org/churn", 10, 1, "customer churn")]

    def fake_load(dataset_id):
        return pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    ctx, _shown, _figures = make_ctx(
        project, ["churn", "org/churn — 10 downloads — customer churn", "y"],
        source="huggingface",
    )
    fallback = ctx.questioner
    ctx.questioner = FormQuestioner({"data.hf_query": ""}, fallback=fallback)
    DataStage(hf_search=fake_search, hf_load=fake_load).prepare(ctx)
    assert searches == ["churn"]
    assert project.read_json(META_FILE)["hf_id"] == "org/churn"


def test_llm_failure_still_completes(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"],
                                    llm=FakeLLM([]))
    stage = DataStage()
    stage.prepare(ctx)
    run_profile_script(project)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("Data saved" in s for s in shown)  # the fixed fallback, LLM unreachable


def test_learning_level_reaches_the_debrief_prompt(project):
    llm = FakeLLM([
        [("text", "Getting your data ready.")],  # prepare()'s preamble
        [("tool", "write_debrief", {"narrative": "n", "figure_notes": {}})],
        [("text", "done")],
    ])
    ctx, _shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"], llm=llm)
    project.write_json("spec.json", {**SPEC, "learning_level": "beginner"})
    stage = DataStage()
    stage.prepare(ctx)
    run_profile_script(project)
    stage.debrief(ctx)
    system = llm.calls[-1]["system"]
    assert "new to machine learning" in system
    assert "You are the data stage of an ML training assistant" in system


def test_missing_profile_debrief_says_so_and_stays_incomplete(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"])
    stage = DataStage()
    stage.prepare(ctx)
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("profile.py" in s for s in shown)


def make_image_ctx(project, answers):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    shown: list[str] = []
    project.write_json("spec.json", {
        "goal": "classify shapes", "task_type": "image_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 3, "gpu": "T4", "notes": "",
    })
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner(answers, ScriptedQuestioner([])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    return ctx, shown


def test_image_synthetic_writes_the_npz_pair_and_the_image_meta(project):
    from mlagent.imageset import read_pair
    from mlagent.stages.data import DataStage

    ctx, _shown = make_image_ctx(project, {
        "data.n_images": 40, "data.image_size": 32, "data.n_classes": 3,
        "data.noise": 0.1, "data.inject_quirks": True,
    })
    handoff = DataStage().prepare(ctx)

    assert handoff.commands == [["profile.py"]]
    assert handoff.outputs == ["profile_raw.json"]
    assert (project.root / "profile.py").exists()
    imageset = read_pair(project.data_raw)
    assert imageset.n_images == 40 and imageset.image_size == 32

    meta = project.read_json("data_meta.json")
    assert meta["target"] == "label"
    assert meta["modality"] == "image"
    assert meta["task_type"] == "image_classification"
    assert meta["source"] == "synthetic"
    assert meta["raw_path"] == "data/raw/data.npz"
    assert meta["raw_n_rows"] == 40
    assert meta["raw_n_cols"] == 32 * 32 * 3
    assert meta["image_size"] == 32
    assert meta["n_channels"] == 3
    assert meta["class_labels"] == imageset.class_names
    assert meta["skipped_files"] == []


def test_the_image_profile_template_is_the_one_copied(project):
    from mlagent.stages.data import DataStage

    ctx, _shown = make_image_ctx(project, {
        "data.n_images": 20, "data.image_size": 32, "data.n_classes": 2,
        "data.noise": 0.0, "data.inject_quirks": False,
    })
    DataStage().prepare(ctx)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "Profile an image dataset" in source


def test_image_drive_source_records_the_skipped_files(project, tmp_path):
    from PIL import Image

    from mlagent.stages.data import DataStage

    folder = tmp_path / "pics"
    for cls in ("cats", "dogs"):
        for i in range(6):
            path = folder / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (10 * i, 60, 90)).save(path)
    (folder / "stray.png").write_bytes(b"")

    ctx, _shown = make_image_ctx(project, {
        "data.drive_folder": str(folder), "data.image_size": 32, "data.max_images": 0,
    })
    ctx.project.write_json("spec.json", {**ctx.project.read_json("spec.json"),
                                         "data_source": "drive"})
    DataStage().prepare(ctx)
    meta = project.read_json("data_meta.json")
    assert meta["source"] == "drive"
    assert meta["source_path"] == str(folder)
    assert meta["raw_n_rows"] == 12
    assert any(reason == "no class folder" for _path, reason in meta["skipped_files"])


def test_image_huggingface_source_uses_the_injected_loader(project):
    from mlagent.stages.data import DataStage
    from mlagent.synth.images import SynthImageConfig, generate

    fake = generate(SynthImageConfig(n_images=18, image_size=32, n_classes=2, seed=2))
    calls = []

    def loader(dataset_id, image_size, split="train", max_images=None, **kwargs):
        calls.append((dataset_id, image_size, max_images))
        return fake

    ctx, _shown = make_image_ctx(project, {
        "data.hf_dataset": "acme/shapes", "data.image_size": 32, "data.max_images": 500,
    })
    ctx.project.write_json("spec.json", {**ctx.project.read_json("spec.json"),
                                         "data_source": "huggingface"})
    DataStage(hf_image_load=loader).prepare(ctx)
    assert calls == [("acme/shapes", 32, 500)]
    meta = project.read_json("data_meta.json")
    assert meta["source"] == "huggingface" and meta["hf_id"] == "acme/shapes"
    assert meta["raw_n_rows"] == 18


def test_image_debrief_reads_the_image_shaped_profile_not_the_tabular_one(project):
    """`profile_images`' JSON has no `n_rows`/`columns` keys, so the tabular
    `profile_markdown` (which the debrief used unconditionally) would raise a KeyError for
    an image project. Regression test for the bug the image pipeline e2e test caught."""
    from mlagent.stages.data import DataStage

    ctx, shown = make_image_ctx(project, {
        "data.n_images": 20, "data.image_size": 32, "data.n_classes": 2,
        "data.noise": 0.0, "data.inject_quirks": False,
    })
    DataStage().prepare(ctx)
    project.write_json("profile_raw.json", {
        "n_images": 20, "image_size": 32, "n_channels": 3, "n_classes": 2,
        "class_counts": {"circle": 10, "square": 10}, "duplicate_images": 0,
        "blank_images": 0, "figures": [],
    })
    DataStage().debrief(ctx)
    assert any("20 images" in s for s in shown)
