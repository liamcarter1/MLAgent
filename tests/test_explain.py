import json
import sys
import types

import pytest

from mlagent.llm import FakeLLM, LLMError
from mlagent.prompts_io import load_prompt
from mlagent.ui import render
from mlagent.ui.explain import Explainer, Glossary


def test_load_prompt_reads_markdown():
    text = load_prompt("explain")
    assert "{term}" in text and "{context}" in text


def test_glossary_persists(tmp_path):
    g = Glossary(tmp_path / "glossary.json")
    assert g.get("epoch") is None
    g.set("epoch", "one pass over the data")
    assert Glossary(tmp_path / "glossary.json").get("epoch") == "one pass over the data"
    assert g.terms() == ["epoch"]
    assert json.loads((tmp_path / "glossary.json").read_text(encoding="utf-8"))["epoch"]


def test_explainer_caches_and_uses_context(tmp_path):
    llm = FakeLLM(script=[[("text", "An [[epoch]] is one pass.")]])
    g = Glossary(tmp_path / "glossary.json")
    shown: list[str] = []
    ex = Explainer(
        llm, g,
        context_provider=lambda: {"stage": "intake", "task_type": "tabular_classification"},
        display=shown.append,
    )
    assert ex.explain("epoch") == "An [[epoch]] is one pass."
    assert ex.explain("epoch") == "An [[epoch]] is one pass."  # cached, no second call
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "epoch" in prompt and "tabular_classification" in prompt
    ex.show("epoch")
    assert shown and "epoch" in shown[0]


def test_explainer_refresh_reasks(tmp_path):
    llm = FakeLLM(script=[[("text", "v1")], [("text", "v2")]])
    ex = Explainer(
        llm, Glossary(tmp_path / "g.json"), context_provider=dict, display=lambda s: None
    )
    assert ex.explain("loss") == "v1"
    assert ex.explain("loss", refresh=True) == "v2"


def test_register_colab_callback_without_colab_returns_false(tmp_path):
    ex = Explainer(
        FakeLLM([]), Glossary(tmp_path / "g.json"), context_provider=dict, display=lambda s: None
    )
    assert ex.register_colab_callback() is False


def test_glossary_tolerates_corrupt_json_and_warns(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text('{"epoch": "one pass"', encoding="utf-8")
    with pytest.warns(UserWarning, match="glossary.json"):
        g = Glossary(path)
    assert g.terms() == []
    assert g.get("epoch") is None


def test_on_click_surfaces_llm_error_instead_of_raising(tmp_path):
    shown: list[str] = []
    ex = Explainer(
        FakeLLM([]), Glossary(tmp_path / "g.json"), context_provider=dict, display=shown.append
    )
    ex._on_click("epoch")
    assert len(shown) == 1
    assert "epoch" in shown[0]
    assert "Couldn't explain" in shown[0]


def test_explain_raises_llm_error_when_script_exhausted(tmp_path):
    ex = Explainer(
        FakeLLM([]), Glossary(tmp_path / "g.json"), context_provider=dict, display=lambda s: None
    )
    with pytest.raises(LLMError):
        ex.explain("epoch")


def test_register_colab_callback_wires_click_to_explain(tmp_path):
    registered: dict = {}

    def register_callback(name, fn):
        registered["name"] = name
        registered["fn"] = fn

    google_mod = types.ModuleType("google")
    colab_mod = types.ModuleType("google.colab")
    output_mod = types.ModuleType("google.colab.output")
    output_mod.register_callback = register_callback
    colab_mod.output = output_mod
    google_mod.colab = colab_mod

    sys.modules["google"] = google_mod
    sys.modules["google.colab"] = colab_mod
    sys.modules["google.colab.output"] = output_mod
    try:
        llm = FakeLLM(script=[[("text", "An [[epoch]] is one pass.")]])
        shown: list[str] = []
        ex = Explainer(
            llm, Glossary(tmp_path / "g.json"), context_provider=dict, display=shown.append
        )

        assert ex.register_colab_callback() is True
        assert registered["name"] == render.EXPLAIN_CALLBACK_NAME

        registered["fn"]("epoch")
        assert shown and "epoch" in shown[0]
    finally:
        del sys.modules["google"]
        del sys.modules["google.colab"]
        del sys.modules["google.colab.output"]
