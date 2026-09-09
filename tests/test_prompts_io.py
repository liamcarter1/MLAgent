from __future__ import annotations

import pytest

from mlagent.prompts_io import PROMPTS_DIR, audience, load_prompt

STAGE_PROMPTS = ["intake", "data", "clean", "codegen", "train", "report"]


def test_load_prompt_without_params_does_not_format():
    text = (PROMPTS_DIR / "data.md").read_text(encoding="utf-8")
    assert load_prompt("data") == text
    assert "{audience}" in text


def test_load_prompt_with_params_formats():
    text = load_prompt("data", audience="Write for an expert.")
    assert "{audience}" not in text
    assert "Write for an expert." in text


def test_every_stage_prompt_has_an_audience_line():
    for name in STAGE_PROMPTS:
        assert "Audience: {audience}" in (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def test_audience_returns_level_text_and_falls_back():
    for level in ("beginner", "intermediate", "expert"):
        assert audience(level).strip()
    assert audience("guru") == audience("intermediate")


def test_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")
