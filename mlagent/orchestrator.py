"""Run stages in order, checkpointing to state.json so a Colab reset can resume."""

from __future__ import annotations

from mlagent import config
from mlagent.stages.base import (
    Handoff,
    Stage,
    StageContext,
    stage_debrief,
    stage_outputs_ready,
    stage_prepare,
    waiting_message,
)
from mlagent.ui.questions import FormQuestioner

EMPTY_STATE = {"completed": [], "current": None, "forced": [], "prepared": [], "handoff": None}


class Orchestrator:
    def __init__(self, ctx: StageContext, stages: list[Stage]):
        self.ctx = ctx
        self.stages = list(stages)

    def _state(self) -> dict:
        state = self.ctx.project.read_json(config.STATE_FILE, default=None)
        if not isinstance(state, dict) or not isinstance(state.get("completed"), list):
            return dict(EMPTY_STATE, completed=[], forced=[], prepared=[])
        for key, default in EMPTY_STATE.items():
            state.setdefault(key, [] if isinstance(default, list) else default)
        if not isinstance(state.get("prepared"), list):
            state["prepared"] = []
        return state

    def _save(self, state: dict) -> None:
        self.ctx.project.write_json(config.STATE_FILE, state)

    def completed(self) -> list[str]:
        return list(self._state()["completed"])

    def waiting(self) -> Handoff | None:
        """The handoff the user still has to run, or None."""
        return Handoff.from_dict(self._state().get("handoff"))

    def mark_complete(self, name: str) -> None:
        state = self._state()
        if name not in state["completed"]:
            state["completed"].append(name)
        state["forced"] = [n for n in state["forced"] if n != name]
        state["prepared"] = [n for n in state["prepared"] if n != name]
        state["current"] = None
        state["handoff"] = None
        self._save(state)

    def reset(self, stage_name: str) -> None:
        """Forget this stage and every later one; they rerun even if their artifacts exist."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            raise ValueError(f"unknown stage {stage_name!r}; known: {names}")
        idx = names.index(stage_name)
        state = self._state()
        state["completed"] = [n for n in state["completed"] if n in names[:idx]]
        state["prepared"] = [n for n in state["prepared"] if n in names[:idx]]
        state["forced"] = names[idx:]
        state["current"] = None
        state["handoff"] = None
        self._save(state)

    def _stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"unknown stage {name!r}; known: {[s.name for s in self.stages]}")

    def debrief(self, name: str) -> None:
        """Force a stage's second phase, skipping the freshness check.

        The escape hatch for when Drive's mtimes lie about outputs the user really did make.
        """
        stage = self._stage(name)
        previous = self.ctx.stage
        self.ctx.stage = name
        try:
            stage_debrief(stage, self.ctx)
            if stage.is_complete(self.ctx):
                self.mark_complete(name)
        finally:
            self.ctx.stage = previous

    def run(
        self, until: str | None = None, answers: dict[str, object] | None = None
    ) -> list[str]:
        """Advance the pipeline. `answers` supplies Colab form values for this call only."""
        if answers is None:
            return self._run(until)
        original = self.ctx.questioner
        self.ctx.questioner = FormQuestioner(
            answers, fallback=original, note=self.ctx.display
        )
        try:
            return self._run(until)
        finally:
            self.ctx.questioner = original

    def _run(self, until: str | None) -> list[str]:
        ran: list[str] = []
        for stage in self.stages:
            state = self._state()
            done = stage.name in state["completed"]
            forced = stage.name in state["forced"]
            prepared = stage.name in state["prepared"]
            # Only take this shortcut for a stage we have not started a handoff for in this
            # session (e.g. artifacts already on disk from an earlier run). A stage with a
            # pending handoff must still go through debrief() below even once its outputs
            # exist on disk, so the user sees the debrief and the stage is recorded as run.
            if not done and not forced and not prepared and stage.is_complete(self.ctx):
                self.mark_complete(stage.name)
                done = True
            if not done:
                self.ctx.stage = stage.name
                state = self._state()
                state["current"] = stage.name
                self._save(state)
                if stage.name in state["prepared"]:
                    handoff = Handoff.from_dict(state.get("handoff"))
                else:
                    self.ctx.display(f"**Stage: {stage.name}**")
                    handoff = stage_prepare(stage, self.ctx)
                    state = self._state()
                    state["handoff"] = handoff.to_dict() if handoff is not None else None
                    if handoff is not None:
                        state["prepared"] = [*state["prepared"], stage.name]
                    self._save(state)
                if handoff is not None and not stage_outputs_ready(stage, self.ctx, handoff):
                    self.ctx.display(waiting_message(handoff))
                    return ran
                stage_debrief(stage, self.ctx)
                if stage.is_complete(self.ctx):
                    self.mark_complete(stage.name)
                    ran.append(stage.name)
                else:
                    self.ctx.display(
                        f"Stage {stage.name} did not finish; rerun orch.run() to continue."
                    )
                    return ran
            if until is not None and stage.name == until:
                break
        if not ran:
            state = self._state()
            if all(s.name in state["completed"] for s in self.stages):
                self.ctx.display("Nothing to do: all stages complete.")
        return ran
