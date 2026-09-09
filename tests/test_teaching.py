from __future__ import annotations

from pathlib import Path

from mlagent.llm import FakeLLM
from mlagent.teaching import Teaching, material, trim_levels

TEXT = """Always shown.

<!--level:beginner-->
Only beginners.
<!--/level-->

<!--level:beginner,intermediate-->
Beginners and intermediates.
<!--/level-->

The end.
"""


def make(level, llm):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    teaching = Teaching(
        level=level, llm=llm, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return teaching, shown, figures


def test_trim_levels_keeps_matching_blocks_only():
    beginner = trim_levels(TEXT, "beginner")
    assert "Only beginners." in beginner and "Beginners and intermediates." in beginner
    intermediate = trim_levels(TEXT, "intermediate")
    assert "Only beginners." not in intermediate
    assert "Beginners and intermediates." in intermediate
    expert = trim_levels(TEXT, "expert")
    assert "Only beginners." not in expert and "Beginners and intermediates." not in expert
    for level in ("beginner", "intermediate", "expert"):
        text = trim_levels(TEXT, level)
        assert "Always shown." in text and "The end." in text
        assert "<!--" not in text


def test_material_loads_and_trims_a_teaching_file():
    expert = material("model_choices", "expert")
    beginner = material("model_choices", "beginner")
    assert "Random forest" in expert and "Random forest" in beginner
    assert len(beginner) > len(expert)
    assert "<!--" not in beginner


def test_preamble_is_silent_for_experts_and_costs_one_call_otherwise():
    llm = FakeLLM([])
    teaching, shown, _figures = make("expert", llm)
    teaching.preamble("data", {"rows": 100})
    assert shown == [] and llm.calls == []

    llm = FakeLLM([[("text", "Next we look at the shape of your data.")]])
    teaching, shown, _figures = make("beginner", llm)
    teaching.preamble("data", {"rows": 100})
    assert shown == ["Next we look at the shape of your data."]
    assert len(llm.calls) == 1
    assert "data" in llm.calls[0]["messages"][0]["content"]


def test_preamble_is_silent_when_the_llm_fails():
    teaching, shown, _figures = make("beginner", FakeLLM([]))
    teaching.preamble("clean", {})
    assert shown == []


def test_walkthrough_costs_no_call_for_experts(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("# --- settings ---\nX = 1\n# --- loop ---\nY = 2\n", encoding="utf-8")
    llm = FakeLLM([])
    teaching, shown, _figures = make("expert", llm)
    teaching.walkthrough([script])
    assert llm.calls == []
    text = "\n".join(shown)
    assert "train.py" in text and "settings" in text and "loop" in text
    assert "```python" not in text  # experts get the file list, not the source


def test_walkthrough_beginner_explains_every_section(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("# --- settings ---\nX = 1\n# --- loop ---\nY = 2\n", encoding="utf-8")
    llm = FakeLLM([
        [("tool", "write_walkthrough",
          {"explanations": {"settings": "Knobs.", "loop": "The epochs."}})],
        [("text", "done")],
    ])
    teaching, shown, _figures = make("beginner", llm)
    teaching.walkthrough([script])
    text = "\n".join(shown)
    assert "```python" in text and "Knobs." in text and "The epochs." in text
    assert len(llm.calls) == 1  # one call per file


def test_walkthrough_intermediate_uses_one_call_for_all_files(tmp_path):
    a = tmp_path / "data.py"
    b = tmp_path / "model.py"
    a.write_text("# --- load ---\nA = 1\n", encoding="utf-8")
    b.write_text("# --- build ---\nB = 2\n", encoding="utf-8")
    llm = FakeLLM([
        [("tool", "write_walkthrough",
          {"explanations": {"data.py": "Loads and splits.", "model.py": "Builds the model."}})],
        [("text", "done")],
    ])
    teaching, shown, _figures = make("intermediate", llm)
    teaching.walkthrough([a, b])
    text = "\n".join(shown)
    assert "Loads and splits." in text and "Builds the model." in text
    assert len(llm.calls) == 1


def test_debrief_returns_the_narrative_and_annotates_figures(tmp_path):
    fig = tmp_path / "val_confusion.png"
    fig.write_bytes(b"x")
    llm = FakeLLM([
        [("tool", "write_debrief",
          {"narrative": "Validation held up.",
           "figure_notes": {"val_confusion.png": "Class 1 is the weak one here."}})],
        [("text", "done")],
    ])
    teaching, shown, figures = make("intermediate", llm)
    text = teaching.debrief("train", {"run_id": 1}, [fig], fallback="fallback")
    assert text == "Validation held up."
    assert shown == []  # the caller displays the narrative
    assert len(figures) == 1
    path, caption = figures[0]
    assert path == fig
    assert "Rows are the true label" in caption  # the fixed caption
    assert "Class 1 is the weak one here." in caption


def test_debrief_falls_back_to_fixed_captions_when_the_llm_fails(tmp_path):
    fig = tmp_path / "val_confusion.png"
    fig.write_bytes(b"x")
    teaching, _shown, figures = make("beginner", FakeLLM([]))
    text = teaching.debrief("train", {}, [fig], fallback="fallback text")
    assert text == "fallback text"
    assert len(figures) == 1
    assert "Rows are the true label" in figures[0][1]


def test_expert_debrief_asks_for_no_figure_notes(tmp_path):
    fig = tmp_path / "test_residuals.png"
    fig.write_bytes(b"x")
    llm = FakeLLM([
        [("tool", "write_debrief", {"narrative": "rmse 0.4 vs target 0.5.",
                                     "figure_notes": {}})],
        [("text", "done")],
    ])
    teaching, _shown, figures = make("expert", llm)
    assert teaching.debrief("report", {}, [fig]) == "rmse 0.4 vs target 0.5."
    payload = llm.calls[0]["messages"][0]["content"]
    assert "figure_notes" not in payload or "figures" not in payload
    assert figures[0][1].startswith("Left: the spread of actual minus predicted")
