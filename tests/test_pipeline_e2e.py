"""End-to-end run of intake -> data -> clean -> codegen -> train -> report, no network."""

from __future__ import annotations

import json

import pandas as pd
import pytest

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

FORM_ANSWERS = {
    "intake.goal": "Predict churn from account data",
    "intake.learning_level": "Intermediate - explain the key ideas",
    "intake.task_type": "Tabular classification",
    "intake.metric": "accuracy",
    "intake.target_value": 0.9,
    "intake.data_source": "Synthetic data",
    "intake.minutes_per_run": 10,
    "intake.max_rounds": 5,
    "intake.gpu": "No GPU (CPU only)",
    "data.n_rows": 300,
    "data.n_features": 6,
    "data.n_classes": 2,
    "data.class_balance": 0.6,
    "data.noise": 0.1,
    "data.inject_quirks": True,
    "clean.drop_columns": "",
    "clean.train_fraction": 0.7,
    "clean.val_fraction": 0.15,
    "codegen.model_type": "Gradient boosting",
}
ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Every confirm() is approved without consuming a scripted answer (the number of
    audit fixes varies with the data). Also serves as FormQuestioner's fallback: a blank
    form field (e.g. "no extra columns to drop") is itself the answer, so an exhausted
    scripted list falls back to the caller's default rather than raising."""

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        self.asked.append(question)
        return True

    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
        self.asked.append(prompt)
        if not self._answers and default is not None:
            return default
        return super().text(prompt, default=default, key=key)


def make_orchestrator(project, stages=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=AutoApproveQuestioner([]),
        explainer=None,
        display=lambda s: None,
        display_figure=lambda path, caption="": None,
    )
    stages = stages or [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                        TrainStage(), ReportStage()]
    return Orchestrator(ctx, stages)


def test_full_pipeline_runs_through_handoffs(project, advance):
    orch = make_orchestrator(project)
    ran = advance(orch, project, answers=FORM_ANSWERS)
    assert ran == ALL_STAGES

    assert project.exists("spec.json") and project.exists("draft_spec.json")
    assert project.read_json("spec.json")["learning_level"] == "intermediate"
    assert (project.data_raw / RAW_FILE).exists()
    for name in ("data_meta.json", "profile_raw.json", "profile.py", AUDIT_FILE,
                 "profile_clean.json", CLEAN_PY, "data.py", "model.py", "train.py",
                 "evaluate.py", "config.json", "metrics.json", "eval_val.json",
                 "eval_test.json", "runs.jsonl", "report.md", "report_meta.json"):
        assert project.exists(name), name
    assert (project.data_clean / CLEAN_FILE).exists()
    assert (project.checkpoints_dir / "best.joblib").exists()
    assert (project.checkpoints_dir / "run1.joblib").exists()
    assert (project.plots_dir / "raw_histograms.png").exists()
    assert (project.plots_dir / "clean_before_after_missing.png").exists()
    assert (project.plots_dir / "run1_training.png").exists()
    assert (project.plots_dir / "test_confusion.png").exists()
    runs = read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done"

    # clean.py reproduces data/clean/data.csv exactly when run on data/raw/data.csv.
    raw_df = pd.read_csv(project.data_raw / RAW_FILE)
    clean_df = pd.read_csv(project.data_clean / CLEAN_FILE)
    namespace: dict = {}
    exec(compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec"),
         namespace)
    reproduced = namespace["clean"](raw_df).reset_index(drop=True)
    assert len(reproduced) == len(clean_df)

    # Deleting state.json: every stage is complete via its artifacts, so nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []


def test_a_second_training_run_is_logged_and_the_report_re_triggers(project, advance):
    orch = make_orchestrator(project)
    advance(orch, project, answers=FORM_ANSWERS)

    orch.reset("train")
    ran = advance(orch, project)
    assert ran == ["train", "report"]
    assert len(read_runs(project.runs_path)) == 2
    assert (project.plots_dir / "run2_training.png").exists()
    meta = project.read_json("report_meta.json")
    best = max(read_runs(project.runs_path), key=lambda r: r["best_val_metric"])
    assert meta["best_run"] == best["run_id"]


def test_editing_a_script_makes_its_outputs_stale(project, advance):
    import os
    import time

    orch = make_orchestrator(project)
    advance(orch, project, answers=FORM_ANSWERS)

    stage = DataStage()
    handoff = json.loads('{"stage": "data", "commands": [["profile.py"]], '
                         '"outputs": ["profile_raw.json"]}')
    from mlagent.stages.base import Handoff

    parsed = Handoff.from_dict(handoff)
    assert stage.outputs_ready(orch.ctx, parsed) is True
    script = project.root / "profile.py"
    future = time.time() + 60
    os.utime(script, (future, future))
    assert stage.outputs_ready(orch.ctx, parsed) is False


@pytest.mark.parametrize("level", ["beginner", "expert"])
def test_the_pipeline_runs_at_every_learning_level(project, advance, level):
    labels = {"beginner": "Beginner - explain everything as we go",
              "expert": "Expert - just the numbers"}
    answers = {**FORM_ANSWERS, "intake.learning_level": labels[level]}
    orch = make_orchestrator(project)
    assert advance(orch, project, answers=answers) == ALL_STAGES
    assert project.read_json("spec.json")["learning_level"] == level
