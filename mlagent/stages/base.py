"""Shared types for pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from mlagent import config
from mlagent.llm import LLM
from mlagent.project import Project
from mlagent.spec import Spec, SpecError
from mlagent.ui.explain import Explainer
from mlagent.ui.questions import Questioner


@dataclass
class StageContext:
    project: Project
    llm: LLM
    questioner: Questioner
    explainer: Explainer | None
    display: Callable[[str], None]
    stage: str = ""

    def spec(self) -> Spec:
        data = self.project.read_json(config.SPEC_FILE)
        if not data:
            raise SpecError("spec.json not found; run the intake stage first")
        return Spec.from_dict(data)


class Stage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None: ...

    def is_complete(self, ctx: StageContext) -> bool: ...
