"""Load system prompts from mlagent/prompts/**.md."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
LEVELS_DIRNAME = "levels"
DEFAULT_LEVEL = "intermediate"


def load_prompt(name: str, **params: object) -> str:
    """Read `prompts/<name>.md`. With no params the text is returned verbatim, so
    prompts that contain literal braces are safe until someone actually formats them."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if not params:
        return text
    return text.format(**params)


def audience(level: str) -> str:
    """The level guidance injected into every stage prompt as `{audience}`."""
    path = PROMPTS_DIR / LEVELS_DIRNAME / f"{level}.md"
    if not path.exists():
        path = PROMPTS_DIR / LEVELS_DIRNAME / f"{DEFAULT_LEVEL}.md"
    return path.read_text(encoding="utf-8").strip()
