"""Shared types for pipeline stages.

A stage runs in two phases. `prepare` interviews the user, writes the scripts they will
run, and returns a `Handoff` naming the notebook cells to run (or `None` when the stage
needs no cells). `debrief` reads whatever those cells produced, shows it, and writes the
stage's completion artifact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mlagent import config
from mlagent.llm import LLM
from mlagent.project import Project
from mlagent.spec import DEFAULT_LEARNING_LEVEL, Spec, SpecError
from mlagent.teaching import Teaching
from mlagent.ui.explain import Explainer
from mlagent.ui.questions import Questioner

# Drive's mtimes can lag behind the write that produced them; allow a second of slack
# before declaring an output stale relative to the script that made it.
MTIME_TOLERANCE = 1.0


@dataclass
class Handoff:
    """Cells the user must run before the stage can be debriefed."""

    stage: str
    commands: list[list[str]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "commands": [list(c) for c in self.commands],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Handoff | None:
        if not isinstance(d, dict) or not d.get("stage"):
            return None
        return cls(
            stage=str(d["stage"]),
            commands=[[str(part) for part in c] for c in (d.get("commands") or [])],
            outputs=[str(o) for o in (d.get("outputs") or [])],
        )

    def scripts(self) -> list[str]:
        return [c[0] for c in self.commands if c]


def outputs_ready(root: Path, handoff: Handoff) -> bool:
    """True when every output exists and none is older than the scripts that make it."""
    root = Path(root)
    out_paths = [root / name for name in handoff.outputs]
    if not out_paths or not all(p.exists() for p in out_paths):
        return False
    script_paths = [root / name for name in handoff.scripts()]
    existing_scripts = [p for p in script_paths if p.exists()]
    if not existing_scripts:
        return True
    newest_script = max(p.stat().st_mtime for p in existing_scripts)
    oldest_output = min(p.stat().st_mtime for p in out_paths)
    return oldest_output + MTIME_TOLERANCE >= newest_script


def waiting_message(handoff: Handoff) -> str:
    cells = ", ".join("`" + " ".join(c) + "`" for c in handoff.commands)
    return (
        f"Run the next cell(s) ({cells}) to see and run the code, then run this cell again "
        "to continue."
    )


def _default_display_figure(path: Path, caption: str = "") -> None:
    """Looked up lazily so importing this module never requires `render.display_figure`
    to exist yet; a later milestone task adds it to `mlagent.ui.render`."""
    from mlagent.ui.render import display_figure

    display_figure(path, caption)


@dataclass
class StageContext:
    project: Project
    llm: LLM
    questioner: Questioner
    explainer: Explainer | None
    display: Callable[[str], None]
    display_figure: Callable[[Path, str], None] = field(
        default_factory=lambda: _default_display_figure
    )
    stage: str = ""

    def spec(self) -> Spec:
        data = self.project.read_json(config.SPEC_FILE)
        if not data:
            raise SpecError("spec.json not found; run the intake stage first")
        return Spec.from_dict(data)

    def learning_level(self) -> str:
        """The user's chosen level, or the default before intake has run."""
        try:
            return self.spec().learning_level
        except (SpecError, TypeError, ValueError):
            return DEFAULT_LEARNING_LEVEL

    def teaching(self) -> Teaching:
        """The explainer for this stage, at the user's chosen level."""
        return Teaching(
            level=self.learning_level(),
            llm=self.llm,
            display=self.display,
            display_figure=self.display_figure,
        )


class Stage(Protocol):
    name: str

    def prepare(self, ctx: StageContext) -> Handoff | None: ...

    def outputs_ready(self, ctx: StageContext, handoff: Handoff) -> bool: ...

    def debrief(self, ctx: StageContext) -> None: ...

    def is_complete(self, ctx: StageContext) -> bool: ...


class ScriptStageBase:
    """Mixin for stages whose handoff is 'run these scripts'."""

    name = ""

    def outputs_ready(self, ctx: StageContext, handoff: Handoff) -> bool:
        return outputs_ready(ctx.project.root, handoff)


def stage_prepare(stage, ctx: StageContext) -> Handoff | None:
    """Run a stage's first phase. Single-phase stages that only define `run` still work."""
    prepare = getattr(stage, "prepare", None)
    if prepare is not None:
        return prepare(ctx)
    stage.run(ctx)
    return None


def stage_debrief(stage, ctx: StageContext) -> None:
    debrief = getattr(stage, "debrief", None)
    if debrief is not None:
        debrief(ctx)


def stage_outputs_ready(stage, ctx: StageContext, handoff: Handoff) -> bool:
    ready = getattr(stage, "outputs_ready", None)
    if ready is not None:
        return ready(ctx, handoff)
    return outputs_ready(ctx.project.root, handoff)
