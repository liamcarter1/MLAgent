"""Split a generated script into the sections the walkthrough explains.

Every template marks its parts with a line of exactly `# --- Title ---` at column zero.
Splitting on those markers gives short, quotable blocks: short enough to explain one at a
time, long enough to be worth reading.
"""

from __future__ import annotations

import re

SECTION_RE = re.compile(r"^# --- (.+?) ---[ \t]*$", re.MULTILINE)
HEADER_TITLE = "Header"


def split_sections(source: str) -> list[tuple[str, str]]:
    """`(title, code)` per section, with anything before the first marker as `Header`."""
    matches = list(SECTION_RE.finditer(source))
    if not matches:
        stripped = source.strip()
        return [(HEADER_TITLE, stripped)] if stripped else []
    sections: list[tuple[str, str]] = []
    head = source[: matches[0].start()].strip()
    if head:
        sections.append((HEADER_TITLE, head))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[match.end() : end].strip("\n").rstrip()
        sections.append((match.group(1).strip(), body))
    return sections


def render_walkthrough(sections: list[tuple[str, str]], explanations: dict[str, str]) -> str:
    """Markdown: each section's code in a fenced block, with its explanation beneath."""
    parts: list[str] = []
    for title, code in sections:
        parts.append(f"#### {title}")
        parts.append("")
        parts.append("```python")
        parts.append(code)
        parts.append("```")
        explanation = (explanations.get(title) or "").strip()
        if explanation:
            parts.append("")
            parts.append(explanation)
        parts.append("")
    return "\n".join(parts).strip()
