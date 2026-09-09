from __future__ import annotations

from mlagent import templates_io
from mlagent.codewalk import HEADER_TITLE, render_walkthrough, split_sections

SOURCE = '''"""A script."""

import json

# --- settings ---
NAME = "demo"

# --- doing the work ---
def work():
    return 1
'''


def test_split_sections_keeps_the_header_and_each_marked_block():
    sections = split_sections(SOURCE)
    assert [title for title, _code in sections] == [HEADER_TITLE, "settings", "doing the work"]
    assert sections[0][1].startswith('"""A script."""')
    assert "import json" in sections[0][1]
    assert sections[1][1] == 'NAME = "demo"'
    assert sections[2][1].splitlines()[0] == "def work():"


def test_a_file_without_markers_is_one_header_section():
    assert split_sections("x = 1\n") == [(HEADER_TITLE, "x = 1")]
    assert split_sections("   \n") == []


def test_a_marker_like_line_that_is_not_at_column_zero_is_not_a_section():
    source = "def f():\n    # --- not a section ---\n    return 1\n"
    assert [t for t, _c in split_sections(source)] == [HEADER_TITLE]


def test_render_walkthrough_fences_the_code_and_adds_explanations():
    sections = split_sections(SOURCE)
    md = render_walkthrough(sections, {"settings": "Constants you can change."})
    assert "#### settings" in md
    assert "```python" in md and "```" in md
    assert "Constants you can change." in md
    assert "#### doing the work" in md
    # Sections with no explanation still show their code.
    assert "def work():" in md


def test_render_walkthrough_with_no_sections():
    assert render_walkthrough([], {}) == ""


def test_every_generated_script_has_at_least_two_sections_including_settings():
    template_dir = templates_io.template_dir("tabular_sklearn")
    paths = [template_dir / name for name in templates_io.CODE_FILES]
    paths += [templates_io.COMMON_DIR / "profile.py", templates_io.COMMON_DIR / "clean.py"]
    for path in paths:
        sections = split_sections(path.read_text(encoding="utf-8"))
        titles = [title for title, _code in sections]
        assert len(sections) >= 2, f"{path.name} has too few sections: {titles}"
        assert "settings" in titles, f"{path.name} has no settings section: {titles}"
