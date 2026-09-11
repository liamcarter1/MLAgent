"""End-to-end image run of intake -> data -> clean -> codegen -> train -> tune -> report."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import partial

import pytest

from mlagent.imageset import read_pair
from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.runlog import read_runs
from mlagent.stages import cost_gate
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CLEAN_PY, CleanStage
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.data import DataStage
from mlagent.stages.intake import IntakeStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
from mlagent.stages.tune import STOP_LABEL, TuneStage, apply_label
from mlagent.ui.questions import ScriptedQuestioner

pytest.importorskip("torch")

E2E_GATE = partial(cost_gate.gate, probes=())

FORM_ANSWERS = {
    "intake.goal": "Sort pictures of shapes into their kind",
    "intake.learning_level": "Beginner - explain everything as we go",
    "intake.task_type": "Image classification",
    "intake.metric": "accuracy",
    "intake.target_value": 0.95,
    "intake.data_source": "Synthetic data",
    "intake.minutes_per_run": 5,
    "intake.max_rounds": 3,
    "intake.gpu": "No GPU (CPU only)",
    "data.n_images": 60,
    "data.image_size": "32",
    "data.n_classes": 3,
    "data.noise": 0.05,
    "data.inject_quirks": True,
    "clean.train_fraction": 0.7,
    "clean.val_fraction": 0.15,
    "codegen.model_type": "Tiny CNN",
    "train.price_per_unit": 0.0999,
    "train.currency": "$",
    "train.cost_decision": "Run it",
    "tune.action": STOP_LABEL,
}
ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "tune", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Every confirm() is approved without consuming a scripted answer (the number of
    audit fixes varies with the data)."""

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        self.asked.append(question)
        return True


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The generated scripts run as subprocesses; keep them off the dev machine's GPU."""
    # "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def make_orchestrator(project, questioner=None, display=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=questioner or AutoApproveQuestioner([]),
        explainer=None,
        display=display or (lambda s: None),
        display_figure=lambda path, caption="": None,
    )
    return Orchestrator(ctx, [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                              TrainStage(gate=E2E_GATE), TuneStage(gate=E2E_GATE),
                              ReportStage()])


def small_config(project) -> None:
    """Keep the real training run inside a few seconds."""
    project.write_json("config.json", {**project.read_json("config.json"), "epochs": 2,
                                       "batch_size": 16})


def advance_to_codegen(orch, project, answers) -> list[str]:
    """Like the `advance` fixture, but stops the moment codegen completes, so the test can
    call `small_config` before the real `train.py` subprocess starts."""
    ran: list[str] = []
    for _ in range(12):
        ran += orch.run(until="codegen", answers=answers)
        if "codegen" in ran:
            return ran
        handoff = orch.waiting()
        assert handoff is not None, "pipeline stalled before reaching codegen"
        for command in handoff.commands:
            result = subprocess.run(
                [sys.executable, *command], cwd=str(project.root), capture_output=True,
                text=True, encoding="utf-8", timeout=300,
            )
            assert result.returncode == 0, result.stdout + result.stderr
    raise AssertionError("pipeline did not reach codegen")


def test_the_image_pipeline_runs_through_every_handoff(project, advance):
    shown: list[str] = []
    orch = make_orchestrator(project, display=shown.append)
    ran = advance_to_codegen(orch, project, FORM_ANSWERS)
    small_config(project)
    ran += advance(orch, project, answers=FORM_ANSWERS)
    assert ran == ALL_STAGES

    assert project.read_json("spec.json")["task_type"] == "image_classification"
    meta = project.read_json("data_meta.json")
    assert meta["modality"] == "image" and meta["target"] == "label"
    assert meta["image_size"] == 32 and meta["n_channels"] == 3
    assert meta["raw_path"] == "data/raw/data.npz"
    assert meta["clean_path"] == "data/clean/data.npz"
    assert meta["feature_columns"] == [] and meta["dropped_columns"] == []
    assert meta["n_classes"] == 3 and len(meta["class_labels"]) == 3

    for name in ("profile.py", "profile_raw.json", AUDIT_FILE, CLEAN_PY,
                 "profile_clean.json", "data.py", "model.py", "train.py", "evaluate.py",
                 "config.json", "metrics.json", "eval_val.json", "eval_test.json",
                 "runs.jsonl", "report.md", "report_meta.json"):
        assert project.exists(name), name
    assert (project.data_raw / "data.npz").exists()
    assert (project.data_raw / "manifest.csv").exists()
    assert (project.data_clean / "manifest.csv").exists()
    assert read_pair(project.data_clean).n_images <= 60
    assert (project.checkpoints_dir / "best.pt").exists()
    assert (project.checkpoints_dir / "run1.pt").exists()
    for figure in ("raw_thumbnails.png", "raw_class_means.png", "raw_intensity.png",
                   "clean_before_after_classes.png", "run1_training.png",
                   "run1_val_misclassified.png", "test_confusion.png"):
        assert (project.plots_dir / figure).exists(), figure

    runs = read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done"
    assert runs[0]["checkpoint"] == "checkpoints/run1.pt"
    assert project.read_json("metrics.json")["device"] == "cpu"
    report = project.report_path.read_text(encoding="utf-8")
    assert "image_classification" in report
    assert "32" in report and "image" in report.lower()

    text = "\n".join(shown)
    assert "Timing a short dry run..." in text
    assert "This runtime has no [[GPU]]." in text
    assert "Estimated" in text
    stored = project.read_json("cost.json")["last_estimate"]
    assert stored["basis"] == "dry_run" and stored["device"] == "cpu"

    # Deleting state.json: every stage is complete via its artifacts, so nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []


def test_one_guided_tuning_round_on_an_image_run_then_stop(project, advance):
    answers = {k: v for k, v in FORM_ANSWERS.items() if k != "tune.action"}
    answers["intake.target_value"] = 1.5   # unreachable: the loop must not end on target_met
    orch = make_orchestrator(
        project, questioner=AutoApproveQuestioner([apply_label(1), STOP_LABEL])
    )
    ran = advance_to_codegen(orch, project, answers)
    small_config(project)
    ran += advance(orch, project, answers=answers)
    assert ran == ALL_STAGES

    runs = read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["applied_diff"] is None and runs[1]["applied_diff"]
    assert set(runs[1]["applied_diff"]) <= set(project.read_json("config.json"))
    assert runs[1]["checkpoint"] == "checkpoints/run2.pt"
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()
    state = project.read_json("tune_state.json")
    assert state["decision"] == "stopped" and state["round"] == 1
    best = max(runs, key=lambda r: r["best_val_metric"])
    assert project.read_json("report_meta.json")["best_run"] == best["run_id"]
    assert project.read_json("eval_test.json")["run_id"] == best["run_id"]


def test_the_expert_level_image_run_still_produces_every_figure(project, advance):
    answers = {**FORM_ANSWERS, "intake.learning_level": "Expert - just the numbers"}
    orch = make_orchestrator(project)
    ran = advance_to_codegen(orch, project, answers)
    small_config(project)
    ran += advance(orch, project, answers=answers)
    assert ran == ALL_STAGES
    assert project.read_json("spec.json")["learning_level"] == "expert"
    for figure in ("raw_thumbnails.png", "raw_class_balance.png", "raw_intensity.png",
                   "raw_class_means.png"):
        assert (project.plots_dir / figure).exists(), figure
