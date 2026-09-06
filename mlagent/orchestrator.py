"""Run stages in order, checkpointing to state.json so a Colab reset can resume."""

from __future__ import annotations

from mlagent import config
from mlagent.stages.base import Stage, StageContext


class Orchestrator:
    def __init__(self, ctx: StageContext, stages: list[Stage]):
        self.ctx = ctx
        self.stages = list(stages)

    def _state(self) -> dict:
        state = self.ctx.project.read_json(config.STATE_FILE, default=None)
        if state is None:
            return {"completed": [], "current": None, "forced": []}
        state.setdefault("forced", [])
        return state

    def _save(self, state: dict) -> None:
        self.ctx.project.write_json(config.STATE_FILE, state)

    def completed(self) -> list[str]:
        return list(self._state()["completed"])

    def mark_complete(self, name: str) -> None:
        state = self._state()
        if name not in state["completed"]:
            state["completed"].append(name)
        state["forced"] = [n for n in state["forced"] if n != name]
        state["current"] = None
        self._save(state)

    def reset(self, stage_name: str) -> None:
        """Forget this stage and every later one; they rerun even if their artifacts exist."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            raise ValueError(f"unknown stage {stage_name!r}; known: {names}")
        idx = names.index(stage_name)
        state = self._state()
        state["completed"] = [n for n in state["completed"] if n in names[:idx]]
        state["forced"] = names[idx:]
        state["current"] = None
        self._save(state)

    def run(self, until: str | None = None) -> list[str]:
        ran: list[str] = []
        for stage in self.stages:
            state = self._state()
            done = stage.name in state["completed"]
            forced = stage.name in state["forced"]
            if not done and not forced and stage.is_complete(self.ctx):
                self.mark_complete(stage.name)
                done = True
            if not done:
                state = self._state()
                state["current"] = stage.name
                self._save(state)
                self.ctx.display(f"**Stage: {stage.name}**")
                stage.run(self.ctx)
                self.mark_complete(stage.name)
                ran.append(stage.name)
            if until is not None and stage.name == until:
                break
        return ran
