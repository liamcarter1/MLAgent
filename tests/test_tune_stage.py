from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mlagent import runlog
from mlagent.diagnose import Diagnosis
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import TrainStage
from mlagent.stages.tune import (
    EDIT_LABEL,
    STOP_LABEL,
    TUNE_COMMANDS,
    TUNE_OUTPUTS,
    TuneStage,
    apply_label,
    describe,
    load_tune_state,
)
from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

FAILED_METRICS = {
    "status": "failed", "started_at": "2026-09-10T09:00:00.000000+00:00",
    "model_type": "gradient_boosting", "epochs": [], "best_epoch": None,
    "best_val_metric": None, "stopped_early": False,
    "error": "ValueError: non-finite loss at epoch 1", "seconds": 0.1,
}

SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


def make_ctx(project, llm=None, answers=()):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append,
                       display_figure=lambda path, caption="": figures.append((path, caption)))
    return ctx, shown, figures


def run_cells(project, handoff):
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command], cwd=str(project.root),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def with_run_one(project, target=2.0):
    """Codegen with gradient boosting, a small config, and one real logged run.

    `target` defaults to an unreachable accuracy (2.0) so a perfect validation score on
    this small fixture cannot flip a test into the target_met path by accident.
    """
    ctx, _s, _f = make_ctx(project, answers=["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(SMALL)
    project.write_json("config.json", cfg)
    spec = project.read_json("spec.json")
    spec["target_value"] = target
    project.write_json("spec.json", spec)
    ctx, _s, _f = make_ctx(project)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert len(runlog.read_runs(project.runs_path)) == 1
    return project


def one_proposal(changes, reason="Slower steps.", expected="steadier"):
    """Two FakeLLM turns: the `propose_diffs` tool call, then an empty turn.

    FakeLLM.run() does not return as soon as it dispatches a tool call -- it loops and
    pops the next script turn too, mirroring the real tool-use loop where the model is
    asked again after seeing the tool result. A trailing empty turn is what lets that
    loop end (see the codegen stage's tests for the same tool-then-terminator pairing).
    Spread this at the call site: `*one_proposal(...)`.
    """
    return [
        [("tool", "propose_diffs", {"proposals": [
            {"rank": 1, "changes": changes, "reason": reason, "expected": expected}]})],
        [],
    ]


def test_labels():
    assert apply_label(2) == "Apply proposal 2"
    assert TUNE_COMMANDS == [["train.py"], ["evaluate.py"]]
    assert TUNE_OUTPUTS == ["metrics.json", "eval_val.json"]


def test_a_guided_round_applies_the_proposal_and_logs_run_two(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([
        [("text", "Tuning is about to start.")],           # round-1 preamble
        *one_proposal({"learning_rate": 0.05}),              # propose_diffs
        [("tool", "write_debrief", {"narrative": "Run 2 was [[steadier]].",
                                    "figure_notes": {"compare_curves.png": "Lower line."}})],
        [],
    ])
    ctx, shown, figures = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="tune", commands=TUNE_COMMANDS, outputs=TUNE_OUTPUTS)
    text = "\n".join(shown)
    assert "round 1 of 3" in text.lower()
    assert "Proposal 1" in text and "Slower steps." in text
    assert "| learning_rate | 0.1 | 0.05 |" in text
    assert project.read_json("config.json")["learning_rate"] == 0.05
    state = project.read_json("tune_state.json")
    assert state["round"] == 0 and state["decision"] == "continue"
    assert state["pending"]["round"] == 1
    assert state["pending"]["applied_diff"] == {"learning_rate": {"from": 0.1, "to": 0.05}}
    assert state["pending"]["diagnosis"] in ("overfitting", "underfitting",
                                              "learning_rate_too_high", "plateau")
    assert not project.exists("metrics.json") and not project.exists("eval_val.json")
    assert not stage.outputs_ready(ctx, handoff)
    prompt = llm.calls[1]["messages"][0]["content"]
    assert '"label"' in prompt and '"schema"' in prompt and '"model_types"' in prompt

    run_cells(project, handoff)
    assert stage.outputs_ready(ctx, handoff)
    assert stage.debrief(ctx) is True
    runs = runlog.read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[1]["applied_diff"] == {"learning_rate": {"from": 0.1, "to": 0.05}}
    assert runs[1]["config"]["learning_rate"] == 0.05
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()
    names = [Path(p).name for p, _c in figures]
    assert names[:2] == ["compare_curves.png", "compare_runs.png"]
    assert "run2_training.png" in names
    assert all(caption for _p, caption in figures)
    assert "Lower line." in dict((Path(p).name, c) for p, c in figures)["compare_curves.png"]
    assert "Run 2 was [[steadier]]." in "\n".join(shown)
    assert "Run 2 finished" in "\n".join(shown)
    state = project.read_json("tune_state.json")
    assert state["round"] == 1 and state["pending"] is None
    assert state["decision"] == "continue"
    assert state["history"] == [{"round": 1, "diagnosis": state["history"][0]["diagnosis"],
                                 "applied_diff": {"learning_rate": {"from": 0.1, "to": 0.05}},
                                 "run_id": 2, "improved": state["history"][0]["improved"]}]
    assert not stage.is_complete(ctx)
    # A second prepare in the same loop does not repeat the preamble or the primer.
    llm2 = FakeLLM([*one_proposal({"epochs": 4})])
    ctx2, shown2, _f = make_ctx(project, llm2, answers=[STOP_LABEL])
    assert stage.prepare(ctx2) is None
    assert "round 2 of 3" in "\n".join(shown2).lower()
    assert "Tuning is about to start." not in "\n".join(shown2)


def test_stop_ends_the_loop_with_a_heuristic_when_the_llm_is_down(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _f = make_ctx(project, answers=[STOP_LABEL])  # FakeLLM([]) -> LLMError
    stage = TuneStage()
    assert stage.prepare(ctx) is None
    text = "\n".join(shown)
    assert "Proposal 1" in text  # the heuristic still produced one
    assert project.read_json("tune_state.json")["decision"] == "stopped"
    assert stage.is_complete(ctx)
    assert project.read_json("config.json")["learning_rate"] == 0.1  # nothing applied
    # Once decided, prepare only explains why and asks nothing.
    ctx2, shown2, _f = make_ctx(project)
    assert stage.prepare(ctx2) is None
    assert "stopped" in "\n".join(shown2).lower()


def test_heuristic_proposal_is_applied_when_the_llm_is_down(clean_project):
    project = with_run_one(clean_project)
    before = project.read_json("config.json")
    ctx, _shown, _f = make_ctx(project, answers=[apply_label(1)])
    handoff = TuneStage().prepare(ctx)
    assert handoff is not None
    after = project.read_json("config.json")
    assert after != before and after["model_type"] == "gradient_boosting"
    assert project.read_json("tune_state.json")["pending"]["applied_diff"]


def test_target_met_ends_the_loop_without_asking(clean_project):
    project = with_run_one(clean_project, target=0.0)  # any accuracy meets 0.0
    ctx, shown, _f = make_ctx(project)  # no scripted answers: a question would raise
    stage = TuneStage()
    assert stage.prepare(ctx) is None
    assert project.read_json("tune_state.json")["decision"] == "target_met"
    assert "target" in "\n".join(shown).lower()
    assert stage.is_complete(ctx)


def test_rounds_exhausted_after_the_last_allowed_round(clean_project):
    project = with_run_one(clean_project)
    spec = project.read_json("spec.json")
    spec["max_rounds"] = 1
    project.write_json("spec.json", spec)
    llm = FakeLLM([[("text", "pre")], *one_proposal({"epochs": 4})])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    assert stage.debrief(ctx) is None  # debrief narrative falls back (script exhausted)
    state = project.read_json("tune_state.json")
    assert state["round"] == 1 and state["decision"] == "rounds_exhausted"
    assert stage.is_complete(ctx)
    assert "rounds" in "\n".join(shown).lower()


def test_edit_path_changes_the_proposal_before_applying(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")], *one_proposal({"learning_rate": 0.05})])
    ctx, shown, _f = make_ctx(project, llm,
                              answers=[EDIT_LABEL, "epochs = 3", "2", "Done"])
    handoff = TuneStage().prepare(ctx)
    assert handoff is not None
    config = project.read_json("config.json")
    assert config["learning_rate"] == 0.05 and config["epochs"] == 2
    diff = project.read_json("tune_state.json")["pending"]["applied_diff"]
    assert diff == {"epochs": {"from": 3, "to": 2}, "learning_rate": {"from": 0.1, "to": 0.05}}
    assert "edited" in project.read_json("tune_state.json")["pending"]["reason"].lower()


def test_three_proposals_are_ranked_and_the_second_can_be_chosen(clean_project):
    project = with_run_one(clean_project)
    # "second" is deliberately avoided as a reason word: the run table's "seconds" column
    # header would match it first and break the ordering check below.
    llm = FakeLLM([[("text", "pre")], [("tool", "propose_diffs", {"proposals": [
        {"rank": 2, "changes": {"epochs": 4}, "reason": "rank-two", "expected": "better"},
        {"rank": 1, "changes": {"learning_rate": 0.05}, "reason": "rank-one",
         "expected": "steadier"},
        {"rank": 3, "changes": {"learning_rate": 0.1}, "reason": "no-op", "expected": "faster"},
        {"rank": 4, "changes": {"min_samples_leaf": 40}, "reason": "rank-four",
         "expected": "steadier"},
    ]})], []])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(2)])
    TuneStage().prepare(ctx)
    text = "\n".join(shown)
    assert text.index("rank-one") < text.index("rank-two") < text.index("rank-four")
    assert "no-op" not in text  # coerced to no change -> dropped
    assert project.read_json("config.json")["epochs"] == 4


def test_family_switch_produces_a_valid_config_and_a_run(clean_project):
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")],
                   *one_proposal({"model_type": "random_forest"}, reason="Forest time.")])
    ctx, shown, _f = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    config = project.read_json("config.json")
    assert config["model_type"] == "random_forest" and "trees_per_epoch" in config
    assert "learning_rate" not in config and config["epochs"] == 3
    run_cells(project, handoff)
    assert stage.debrief(ctx) is True
    runs = runlog.read_runs(project.runs_path)
    assert runs[1]["config"]["model_type"] == "random_forest"
    assert runs[1]["applied_diff"]["model_type"] == {"from": "gradient_boosting",
                                                     "to": "random_forest"}


def test_debrief_without_a_pending_round_does_nothing(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _f = make_ctx(project)
    assert TuneStage().debrief(ctx) is None
    assert shown == [] and not project.exists("tune_state.json")


def test_debrief_before_the_cells_ran_explains_and_waits(clean_project):
    project = with_run_one(clean_project)
    ctx, _s, _f = make_ctx(project, answers=[apply_label(1)])
    stage = TuneStage()
    stage.prepare(ctx)  # unlinks metrics.json
    ctx2, shown, _f = make_ctx(project)
    assert stage.debrief(ctx2) is None
    assert "train.py" in "\n".join(shown)
    assert project.read_json("tune_state.json")["pending"] is not None


def test_on_reset_forgets_the_loop(clean_project):
    project = with_run_one(clean_project)
    project.write_json("tune_state.json", {"round": 2, "max_rounds": 3, "decision": "stopped",
                                           "pending": None, "history": []})
    ctx, _s, _f = make_ctx(project)
    TuneStage().on_reset(ctx)
    assert not project.exists("tune_state.json")


def test_expert_level_skips_preamble_and_primer(clean_project):
    project = with_run_one(clean_project)
    spec = project.read_json("spec.json")
    spec["learning_level"] = "expert"
    project.write_json("spec.json", spec)
    llm = FakeLLM([*one_proposal({"epochs": 4})])  # no preamble turn
    ctx, shown, _f = make_ctx(project, llm, answers=[STOP_LABEL])
    TuneStage().prepare(ctx)
    assert "What each family's knobs do" not in "\n".join(shown)
    assert len(llm.calls) == 1


def test_debriefing_the_same_round_twice_logs_one_history_entry(clean_project):
    """debrief must be idempotent per run (ruling 3). This replays the `pending` dict
    `prepare` originally wrote back into `tune_state.json` after a full, successful
    round -- simulating a kernel death between the history append and the write that
    clears `pending` -- so the second `debrief` call reaches the "a history entry for
    this run_id already exists" guard instead of bailing out early on "no pending
    round" the way a fresh tune_state.json would (that path is already covered by
    test_debrief_without_a_pending_round_does_nothing)."""
    project = with_run_one(clean_project)
    llm = FakeLLM([
        [("text", "pre")],
        *one_proposal({"learning_rate": 0.05}),
        [("tool", "write_debrief", {"narrative": "Run 2 was steadier.", "figure_notes": {}})],
        [],
    ])
    ctx, _shown, _f = make_ctx(project, llm, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    pending = project.read_json("tune_state.json")["pending"]
    assert pending is not None

    run_cells(project, handoff)
    first_result = stage.debrief(ctx)
    assert first_result is True
    state_after_first = project.read_json("tune_state.json")
    assert state_after_first["pending"] is None
    assert len(state_after_first["history"]) == 1
    assert state_after_first["history"][0]["run_id"] == 2
    assert len(runlog.read_runs(project.runs_path)) == 2

    # Replay the original pending round: the run is already logged and in history, but
    # `pending` looks as if it was never cleared.
    project.write_json("tune_state.json", dict(state_after_first, pending=pending))

    ctx2, _s2, _f2 = make_ctx(project)
    second_result = stage.debrief(ctx2)
    state_after_second = project.read_json("tune_state.json")
    assert len(state_after_second["history"]) == 1
    assert len(runlog.read_runs(project.runs_path)) == 2
    assert state_after_second["pending"] is None
    assert second_result == first_result


def test_a_stale_form_answer_does_not_block_stopping_after_a_no_op_edit(clean_project):
    """A Colab form's `tune.action` answer is fixed for the whole cell run. If a no-op
    edit (the user edits the proposed change back to its original value) sends the
    "what shall we do?" question round again, that re-ask must fall through to the
    fallback (console/scripted) questioner instead of returning the same form answer
    forever, or the user could never reach Stop."""
    project = with_run_one(clean_project)
    llm = FakeLLM([[("text", "pre")], *one_proposal({"learning_rate": 0.05})])
    fallback = ScriptedQuestioner([
        "learning_rate = 0.05",  # edit_config: pick the key the proposal changed
        "0.1",                    # ...and set it back to its original value: a no-op
        "Done",                   # finish editing
        STOP_LABEL,                # the re-ask must reach here, not repeat EDIT_LABEL
    ])
    questioner = FormQuestioner({"tune.action": EDIT_LABEL}, fallback=fallback)
    ctx, shown, _f = make_ctx(project, llm)
    ctx.questioner = questioner
    assert TuneStage().prepare(ctx) is None
    assert project.read_json("tune_state.json")["decision"] == "stopped"
    assert project.read_json("config.json")["learning_rate"] == 0.1  # nothing applied


def test_prepare_backfills_a_legacy_run_archive(clean_project):
    project = with_run_one(clean_project)
    (project.runs_dir / "run1_metrics.json").unlink()
    ctx, shown, _f = make_ctx(project, answers=[STOP_LABEL])
    TuneStage().prepare(ctx)
    assert (project.runs_dir / "run1_metrics.json").exists()
    assert project.read_json("runs/run1_metrics.json") == project.read_json("metrics.json")
    assert "no per-epoch curve" not in "\n".join(shown)


def test_describe_is_honest_without_epoch_data():
    d = Diagnosis("plateau", {"no_epoch_data": True}, 1, 1, False)
    text = describe(d)
    assert "no per-epoch curve" in text
    assert "flattened" not in text


def test_unknown_decision_is_treated_as_continue(clean_project):
    project = with_run_one(clean_project)
    project.write_json("tune_state.json", {"round": 1, "max_rounds": 3, "decision": "bogus",
                                           "pending": None, "history": []})
    ctx, _s, _f = make_ctx(project)
    assert TuneStage().is_complete(ctx) is False
    assert load_tune_state(project, 3)["decision"] == "continue"


def test_a_failed_run_is_logged_with_its_diff_and_the_next_round_proposes_a_gentler_config(
    clean_project,
):
    project = with_run_one(clean_project)
    ctx, _s, _f = make_ctx(project, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    assert handoff is not None
    config = project.read_json("config.json")
    project.write_json("metrics.json", dict(FAILED_METRICS, config=config))

    ctx2, shown2, _f2 = make_ctx(project, FakeLLM([]))
    assert stage.debrief(ctx2) is True
    text2 = "\n".join(shown2)
    assert "Run 2 failed" in text2 and "non-finite" in text2
    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 2
    assert runs[1]["status"] == "failed"
    assert runs[1]["applied_diff"] is not None
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()

    ctx3, shown3, _f3 = make_ctx(project, FakeLLM([]), answers=[STOP_LABEL])
    stage.prepare(ctx3)
    text3 = "\n".join(shown3)
    assert "the last run failed" in text3.lower()
    assert "| learning_rate |" in text3


def test_failed_run_headline_says_last_round_when_no_rounds_remain(clean_project):
    project = with_run_one(clean_project)
    spec = project.read_json("spec.json")
    spec["max_rounds"] = 1
    project.write_json("spec.json", spec)
    ctx, _s, _f = make_ctx(project, answers=[apply_label(1)])
    stage = TuneStage()
    handoff = stage.prepare(ctx)
    assert handoff is not None
    config = project.read_json("config.json")
    project.write_json("metrics.json", dict(FAILED_METRICS, config=config))

    ctx2, shown2, _f2 = make_ctx(project, FakeLLM([]))
    assert stage.debrief(ctx2) is None
    text2 = "\n".join(shown2)
    assert "last allowed round" in text2
    assert "gentler" not in text2
    assert project.read_json("tune_state.json")["decision"] == "rounds_exhausted"


def test_debrief_with_a_finished_round_and_no_pending_asks_to_re_prepare(clean_project):
    project = with_run_one(clean_project)
    project.write_json("tune_state.json", {
        "round": 1, "max_rounds": 3, "decision": "continue", "pending": None,
        "history": [{"round": 1, "diagnosis": "plateau", "applied_diff": {}, "run_id": 2,
                     "improved": False}],
    })
    ctx, shown, _f = make_ctx(project)
    assert TuneStage().debrief(ctx) is True
    assert shown == []
