import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.stages.base import StageContext
from mlagent.ui.questions import ScriptedQuestioner


class RecordingStage:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.runs = 0

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        if self.fail:
            raise RuntimeError("boom")
        ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


def make_ctx(project):
    return StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner([]),
                        explainer=None, display=lambda s: None)


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
