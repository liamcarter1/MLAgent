"""How much to explain, and the LLM calls that do the explaining.

One object per stage run. Everything it does degrades to fixed text on `LLMError`, so a
missing API key costs the user detail, never a stage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mlagent.captions import caption_for
from mlagent.codewalk import render_walkthrough, split_sections
from mlagent.llm import LLM, LLMError, ToolSpec
from mlagent.prompts_io import audience, load_prompt

LEVEL_BLOCK_RE = re.compile(r"<!--level:([a-z, ]+)-->(.*?)<!--/level-->\n?", re.DOTALL)
EXPERT = "expert"
BEGINNER = "beginner"

WRITE_WALKTHROUGH_TOOL = ToolSpec(
    name="write_walkthrough",
    description=(
        "Explain the code you were shown. `explanations` maps each key you were given "
        "(a section title, or a filename) to your explanation of it."
    ),
    input_schema={
        "type": "object",
        "properties": {"explanations": {"type": "object"}},
        "required": ["explanations"],
    },
    handler=lambda inp: "recorded",
)

WRITE_DEBRIEF_TOOL = ToolSpec(
    name="write_debrief",
    description=(
        "Record the debrief. `narrative` is the prose shown to the user; `figure_notes` "
        "maps a figure filename to one sentence about what this data shows in it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
            "figure_notes": {"type": "object"},
        },
        "required": ["narrative"],
    },
    handler=lambda inp: "recorded",
)


def trim_levels(text: str, level: str) -> str:
    """Drop the level-fenced blocks that are not for this reader; keep everything else."""

    def keep(match: re.Match) -> str:
        levels = {part.strip() for part in match.group(1).split(",") if part.strip()}
        return match.group(2).strip() + "\n" if level in levels else ""

    return LEVEL_BLOCK_RE.sub(keep, text).strip() + "\n"


def material(name: str, level: str) -> str:
    """A teaching document from `prompts/teaching/`, trimmed to this level."""
    return trim_levels(load_prompt(f"teaching/{name}"), level)


def _capture(store: dict, key: str) -> Callable[[dict], str]:
    def handler(inp: dict) -> str:
        value = inp.get(key)
        store[key] = value if isinstance(value, dict) else {}
        if "narrative" in inp:
            store["narrative"] = str(inp.get("narrative") or "")
        return "recorded"

    return handler


@dataclass
class Teaching:
    level: str
    llm: LLM
    display: Callable[[str], None]
    display_figure: Callable[[Path, str], None]

    def _system(self, name: str) -> str:
        return load_prompt(name, audience=audience(self.level))

    # --- before a stage ---
    def preamble(self, stage: str, payload: dict) -> None:
        if self.level == EXPERT:
            return
        prompt = json.dumps({"stage": stage, "context": payload}, indent=2, default=str)
        try:
            result = self.llm.run(
                self._system("preamble"), [{"role": "user", "content": prompt}], []
            )
        except LLMError:
            return
        if result.text.strip():
            self.display(result.text.strip())

    # --- explaining generated code ---
    def walkthrough(self, paths: list[Path]) -> None:
        paths = [Path(p) for p in paths]
        if not paths:
            return
        if self.level == EXPERT:
            for path in paths:
                titles = ", ".join(
                    title for title, _code in split_sections(path.read_text(encoding="utf-8"))
                )
                self.display(f"`{path.name}` — sections: {titles}")
            return
        if self.level == BEGINNER:
            for path in paths:
                sections = split_sections(path.read_text(encoding="utf-8"))
                explanations = self._ask_walkthrough(
                    {
                        "file": path.name,
                        "sections": [
                            {"title": title, "source": code} for title, code in sections
                        ],
                    }
                )
                self.display(f"### `{path.name}`")
                self.display(render_walkthrough(sections, explanations))
            return
        files = [
            {"file": path.name, "source": path.read_text(encoding="utf-8")} for path in paths
        ]
        explanations = self._ask_walkthrough({"files": files})
        lines = ["### The generated code", ""]
        for path in paths:
            note = (explanations.get(path.name) or "").strip()
            titles = ", ".join(
                title for title, _code in split_sections(path.read_text(encoding="utf-8"))
            )
            lines.append(f"**`{path.name}`** — sections: {titles}")
            if note:
                lines.append("")
                lines.append(note)
            lines.append("")
        self.display("\n".join(lines).strip())

    def _ask_walkthrough(self, payload: dict) -> dict[str, str]:
        store: dict = {}
        tool = ToolSpec(
            name=WRITE_WALKTHROUGH_TOOL.name,
            description=WRITE_WALKTHROUGH_TOOL.description,
            input_schema=WRITE_WALKTHROUGH_TOOL.input_schema,
            handler=_capture(store, "explanations"),
        )
        try:
            self.llm.run(
                self._system("walkthrough"),
                [{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            return {}
        return {str(k): str(v) for k, v in (store.get("explanations") or {}).items()}

    # --- after a stage ---
    def show_figures(self, paths: list[Path], notes: dict[str, str]) -> None:
        for path in paths:
            path = Path(path)
            caption = caption_for(path)
            note = (notes.get(path.name) or "").strip()
            if note:
                caption = f"{caption} {note}".strip()
            self.display_figure(path, caption)

    def debrief(
        self,
        prompt_name: str,
        payload: dict,
        figures: list[Path],
        fallback: str = "",
    ) -> str:
        figures = [Path(p) for p in figures]
        body = dict(payload)
        if self.level != EXPERT and figures:
            body["figures"] = [p.name for p in figures]
        store: dict = {}
        tool = ToolSpec(
            name=WRITE_DEBRIEF_TOOL.name,
            description=WRITE_DEBRIEF_TOOL.description,
            input_schema=WRITE_DEBRIEF_TOOL.input_schema,
            handler=_capture(store, "figure_notes"),
        )
        narrative = fallback
        notes: dict[str, str] = {}
        try:
            result = self.llm.run(
                self._system(prompt_name),
                [{"role": "user", "content": json.dumps(body, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            pass
        else:
            narrative = store.get("narrative") or result.text.strip() or fallback
            notes = {str(k): str(v) for k, v in (store.get("figure_notes") or {}).items()}
        self.show_figures(figures, notes)
        return narrative
