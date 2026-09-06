import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.stages.base import StageContext
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ScriptedQuestioner


class RecordingStage:
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
