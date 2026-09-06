from pathlib import Path

from mlagent import colab
from mlagent.llm import FakeLLM
from mlagent.ui.questions import ScriptedQuestioner


def test_setup_without_colab_adds_path_and_makes_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    projects = colab.setup(drive_root=str(tmp_path), mount=False)
    assert projects == tmp_path / "projects"
    assert projects.is_dir()
    import sys

    assert str(tmp_path) in sys.path


def test_setup_prints_actionable_message_when_api_key_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    colab.setup(drive_root=str(tmp_path), mount=False)
    out = capsys.readouterr().out
    assert "No ANTHROPIC_API_KEY found" in out
    assert "Secrets panel" in out


def test_setup_silent_when_api_key_present(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    colab.setup(drive_root=str(tmp_path), mount=False)
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" not in out


def test_make_context_and_start(tmp_path):
    ctx = colab.make_context("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert ctx.project.root == tmp_path / "projects" / "demo"
    assert ctx.project.root.is_dir()
    assert ctx.explainer is not None
    orch = colab.start("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert [s.name for s in orch.stages] == ["intake"]


def test_explain_uses_last_context(tmp_path, monkeypatch):
    llm = FakeLLM(script=[[("text", "One pass over the data.")]])
    shown: list[str] = []
    orch = colab.start("demo", drive_root=str(tmp_path), llm=llm)
    orch.ctx.questioner = ScriptedQuestioner([])
    orch.ctx.explainer.display = shown.append
    colab.explain("epoch")
    assert shown and "One pass" in shown[0]
    assert Path(tmp_path, "projects", "demo", "glossary.json").exists()
