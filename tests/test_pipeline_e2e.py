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
from mlagent.stages.tune import STOP_LABEL, TuneStage, apply_label
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
    "tune.action": STOP_LABEL,
}
ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "tune", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Every confirm() is approved without consuming a scripted answer (the number of
    audit fixes varies with the data). Any other question still requires a scripted
    answer, so a stage that starts asking something new still fails the test as before."""

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        self.asked.append(question)
        return True


def make_orchestrator(project, stages=None, questioner=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=questioner or AutoApproveQuestioner([]),
        explainer=None,
        display=lambda s: None,
        display_figure=lambda path, caption="": None,
    )
    stages = stages or [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                        TrainStage(), TuneStage(), ReportStage()]
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
    tune_state = project.read_json("tune_state.json")
    assert tune_state["decision"] in ("stopped", "target_met") and tune_state["history"] == []

    # clean.py reproduces data/clean/data.csv exactly when run on data/raw/data.csv.
    raw_df = pd.read_csv(project.data_raw / RAW_FILE)
    clean_df = pd.read_csv(project.data_clean / CLEAN_FILE)
    namespace: dict = {}
    exec(compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec"),
         namespace)
    reproduced = namespace["clean"](raw_df).reset_index(drop=True)
    pd.testing.assert_frame_equal(reproduced, clean_df, check_dtype=False)

    # Deleting state.json: every stage is complete via its artifacts, so nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []


def test_a_second_training_run_is_logged_and_the_report_re_triggers(project, advance):
    orch = make_orchestrator(project)
    advance(orch, project, answers=FORM_ANSWERS)

    orch.reset("train")
    assert not project.exists("tune_state.json") or project.read_json(
        "tune_state.json")["history"] == []
    ran = advance(orch, project, answers=FORM_ANSWERS)
    assert ran == ["train", "tune", "report"]
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


def test_one_guided_round_then_stop_and_the_report_scores_the_new_best(project, advance):
    answers = {k: v for k, v in FORM_ANSWERS.items() if k != "tune.action"}
    answers["intake.target_value"] = 1.5   # unreachable: the loop must not end on target_met
    orch = make_orchestrator(
        project, questioner=AutoApproveQuestioner([apply_label(1), STOP_LABEL]))
    ran = advance(orch, project, answers=answers)
    assert ran == ALL_STAGES
    runs = read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["applied_diff"] is None and runs[1]["applied_diff"]
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()
    state = project.read_json("tune_state.json")
    assert state["decision"] == "stopped" and state["round"] == 1
    assert [h["run_id"] for h in state["history"]] == [2]
    best = max(runs, key=lambda r: r["best_val_metric"])
    assert project.read_json("report_meta.json")["best_run"] == best["run_id"]
    assert project.read_json("eval_test.json")["run_id"] == best["run_id"]
