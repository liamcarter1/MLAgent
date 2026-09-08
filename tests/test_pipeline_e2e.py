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
    "Intermediate - explain the key ideas",  # intake: learning level
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
