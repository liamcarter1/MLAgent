"""Shared types for pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from mlagent.llm import LLM
from mlagent.project import Project
from mlagent.ui.explain import Explainer
from mlagent.ui.questions import Questioner


@dataclass
class StageContext:
    project: Project
    llm: LLM
    questioner: Questioner
    explainer: Explainer | None
    display: Callable[[str], None]


class Stage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None: ...

    def is_complete(self, ctx: StageContext) -> bool: ...
