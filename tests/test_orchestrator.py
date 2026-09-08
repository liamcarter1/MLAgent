import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.stages.base import Handoff, StageContext, outputs_ready, waiting_message
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ScriptedQuestioner


class RecordingStage:
    """A legacy single-phase stage: only `run` and `is_complete`."""

    def __init__(self, name: str, fail: bool = False, write: bool = True):
        self.name = name
        self.fail = fail
        self.write = write
        self.runs = 0

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        if self.fail:
            raise RuntimeError("boom")
        if self.write:
            ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


class ScriptStage:
    """A two-phase stage: prepare writes a script, the 'user' runs it, debrief reads it."""

    def __init__(self, name: str):
        self.name = name
        self.prepared = 0
        self.debriefed = 0

    def prepare(self, ctx: StageContext) -> Handoff:
        self.prepared += 1
        (ctx.project.root / f"{self.name}.py").write_text("print('hi')\n", encoding="utf-8")
        return Handoff(stage=self.name, commands=[[f"{self.name}.py"]],
                       outputs=[f"{self.name}_out.json"])

    def debrief(self, ctx: StageContext) -> None:
        self.debriefed += 1
        ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


def user_runs(project, stage_name: str) -> None:
    """Stand in for the user running the handoff cell."""
    project.write_json(f"{stage_name}_out.json", {"done": True})


class ContextRecordingStage:
    """Records the explainer's context snapshot as seen while this stage is running."""

    name = "ctxstage"

    def __init__(self):
        self.captured: dict | None = None

    def run(self, ctx: StageContext) -> None:
        self.captured = ctx.explainer.context_provider()
        ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


def make_ctx(project, display=None):
    return StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner([]),
                        explainer=None, display=display or (lambda s: None))


def test_runs_stages_in_order_and_checkpoints(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["a", "b"]
    assert orch.completed() == ["a", "b"]
    assert project.read_json("state.json")["completed"] == ["a", "b"]
    assert project.read_json("state.json")["current"] is None


def test_resumes_after_restart(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    Orchestrator(make_ctx(project), [a, b]).run(until="a")
    a2, b2 = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a2, b2])
    assert orch.run() == ["b"]
    assert a2.runs == 0 and b2.runs == 1


def test_failure_leaves_current_and_does_not_mark_complete(project):
    a, b = RecordingStage("a"), RecordingStage("b", fail=True)
    orch = Orchestrator(make_ctx(project), [a, b])
    with pytest.raises(RuntimeError):
        orch.run()
    state = project.read_json("state.json")
    assert state["completed"] == ["a"]
    assert state["current"] == "b"


def test_reset_drops_later_stages(project):
    a, b, c = RecordingStage("a"), RecordingStage("b"), RecordingStage("c")
    orch = Orchestrator(make_ctx(project), [a, b, c])
    orch.run()
    orch.reset("b")
    assert orch.completed() == ["a"]
    assert orch.run() == ["b", "c"]


def test_stage_with_artifact_but_no_state_is_skipped(project):
    project.write_json("a.json", {"ok": True})
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["b"]
    assert a.runs == 0


def test_non_dict_state_json_treated_as_empty_state(project):
    project.write_json("state.json", [])
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["a", "b"]
    assert orch.completed() == ["a", "b"]


def test_stage_that_does_not_write_its_artifact_is_not_marked_complete(project):
    shown: list[str] = []
    a, b = RecordingStage("a", write=False), RecordingStage("b")
    orch = Orchestrator(make_ctx(project, display=shown.append), [a, b])
    ran = orch.run()
    assert ran == []
    assert "a" not in orch.completed()
    state = project.read_json("state.json")
    assert state["current"] == "a"
    assert b.runs == 0
    assert any("did not finish" in s and "a" in s for s in shown)


def test_nothing_to_do_message_when_all_stages_already_complete(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    Orchestrator(make_ctx(project), [a, b]).run()
    shown: list[str] = []
    a2, b2 = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project, display=shown.append), [a2, b2])
    assert orch.run() == []
    assert shown == ["Nothing to do: all stages complete."]


def test_ctx_stage_is_set_to_the_running_stage_name(project):
    ctx = make_ctx(project)
    ctx.explainer = Explainer(
        FakeLLM([]), Glossary(project.glossary_path),
        context_provider=lambda: {"stage": ctx.stage}, display=lambda s: None,
    )
    stage = ContextRecordingStage()
    orch = Orchestrator(ctx, [stage])
    orch.run()
    assert stage.captured == {"stage": "ctxstage"}


def test_ctx_spec_reads_spec_or_raises(project):
    from mlagent.spec import Spec, SpecError

    ctx = make_ctx(project)
    with pytest.raises(SpecError):
        ctx.spec()
    project.write_json("spec.json", {
        "goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 2, "gpu": "none", "notes": "",
    })
    assert isinstance(ctx.spec(), Spec)
    assert ctx.spec().metric == "accuracy"


def test_handoff_round_trips_through_state_json():
    h = Handoff(stage="train", commands=[["train.py"], ["evaluate.py"]],
                outputs=["metrics.json"])
    assert Handoff.from_dict(h.to_dict()) == h
    assert Handoff.from_dict(None) is None
    assert Handoff.from_dict({"stage": "x"}) == Handoff(stage="x", commands=[], outputs=[])


def test_outputs_ready_checks_existence_and_freshness(tmp_path):
    import os
    import time

    script = tmp_path / "s.py"
    script.write_text("x = 1\n", encoding="utf-8")
    h = Handoff(stage="s", commands=[["s.py"]], outputs=["out.json"])
    assert outputs_ready(tmp_path, h) is False
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")
    assert outputs_ready(tmp_path, h) is True
    # Editing the script after the output was produced makes the output stale.
    future = time.time() + 60
    os.utime(script, (future, future))
    assert outputs_ready(tmp_path, h) is False


def test_waiting_message_names_every_cell():
    h = Handoff(stage="train", commands=[["train.py"], ["evaluate.py", "--split", "val"]],
                outputs=[])
    text = waiting_message(h)
    assert "`train.py`" in text and "`evaluate.py --split val`" in text
    assert "run this cell again" in text


def test_script_stage_pauses_until_outputs_exist(project):
    stage = ScriptStage("s")
    shown: list[str] = []
    orch = Orchestrator(make_ctx(project, display=shown.append), [stage])
    assert orch.run() == []
    assert stage.prepared == 1 and stage.debriefed == 0
    assert orch.waiting() == Handoff(stage="s", commands=[["s.py"]], outputs=["s_out.json"])
    assert any("run this cell again" in s for s in shown)
    state = project.read_json("state.json")
    assert state["prepared"] == ["s"] and state["handoff"]["stage"] == "s"

    # Running again without the outputs does not re-prepare.
    assert orch.run() == []
    assert stage.prepared == 1

    user_runs(project, "s")
    assert orch.run() == ["s"]
    assert stage.prepared == 1 and stage.debriefed == 1
    state = project.read_json("state.json")
    assert state["completed"] == ["s"] and state["handoff"] is None and state["prepared"] == []


def test_prepared_survives_a_fresh_orchestrator(project):
    first = ScriptStage("s")
    Orchestrator(make_ctx(project), [first]).run()
    user_runs(project, "s")
    second = ScriptStage("s")
    assert Orchestrator(make_ctx(project), [second]).run() == ["s"]
    assert second.prepared == 0 and second.debriefed == 1


def test_debrief_by_name_forces_a_stage_whose_mtimes_lie(project):
    stage = ScriptStage("s")
    orch = Orchestrator(make_ctx(project), [stage])
    orch.run()
    orch.debrief("s")  # outputs missing, but the user says they ran it
    assert stage.debriefed == 1
    assert orch.completed() == ["s"]
    with pytest.raises(ValueError):
        orch.debrief("nope")


def test_reset_forgets_prepared_and_handoff(project):
    a, b = ScriptStage("a"), ScriptStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    orch.run()
    user_runs(project, "a")
    orch.run()
    assert orch.completed() == ["a"]
    orch.reset("a")
    state = project.read_json("state.json")
    assert state["prepared"] == [] and state["handoff"] is None
    assert state["forced"] == ["a", "b"]


def test_answers_wrap_the_questioner_for_one_call_only(project):
    class AskingStage:
        name = "ask"

        def __init__(self):
            self.seen: list[str] = []

        def prepare(self, ctx):
            self.seen.append(ctx.questioner.text("Goal?", key="intake.goal"))
            ctx.project.write_json("ask.json", {"ok": True})
            return None

        def debrief(self, ctx):
            return None

        def is_complete(self, ctx):
            return ctx.project.exists("ask.json")

    stage = AskingStage()
    ctx = make_ctx(project)
    ctx.questioner = ScriptedQuestioner(["typed"])
    orch = Orchestrator(ctx, [stage])
    assert orch.run(answers={"intake.goal": "from the form"}) == ["ask"]
    assert stage.seen == ["from the form"]
    assert isinstance(orch.ctx.questioner, ScriptedQuestioner)


def test_legacy_single_phase_stages_still_run(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["a", "b"]
    assert orch.waiting() is None
