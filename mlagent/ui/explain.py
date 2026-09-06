"""Click-to-explain: cached, project-contextual explanations of technical terms."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from mlagent.llm import LLM, ask_text
from mlagent.prompts_io import load_prompt
from mlagent.ui.render import display_message


class Glossary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, term: str) -> str | None:
        return self._data.get(term.strip().lower())

    def set(self, term: str, text: str) -> None:
        self._data[term.strip().lower()] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def terms(self) -> list[str]:
        return sorted(self._data)


class Explainer:
    CALLBACK_NAME = "mlagent.explain"

    def __init__(
        self,
        llm: LLM,
        glossary: Glossary,
        context_provider: Callable[[], dict],
        display: Callable[[str], None] = display_message,
    ):
        self.llm = llm
        self.glossary = glossary
        self.context_provider = context_provider
        self.display = display

    def explain(self, term: str, refresh: bool = False) -> str:
        if not refresh:
            cached = self.glossary.get(term)
            if cached:
                return cached
        context = json.dumps(self.context_provider(), indent=2, sort_keys=True, default=str)
        system = load_prompt("explain").format(term=term, context=context)
        text = ask_text(self.llm, system, f"Explain: {term}\n\nContext:\n{context}")
        self.glossary.set(term, text)
        return text

    def show(self, term: str, refresh: bool = False) -> None:
        text = self.explain(term, refresh=refresh)
        self.display(f"### {term}\n\n{text}")

    def _on_click(self, term: str) -> None:
        self.show(term)

    def register_colab_callback(self) -> bool:
        try:
            from google.colab import output  # type: ignore
        except ImportError:
            return False
        output.register_callback(self.CALLBACK_NAME, self._on_click)
        return True
