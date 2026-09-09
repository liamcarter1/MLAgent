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


def test_make_context_and_start(tmp_path, monkeypatch):
    ctx = colab.make_context("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert ctx.project.root == tmp_path / "projects" / "demo"
    assert ctx.project.root.is_dir()
    assert ctx.explainer is not None
    # start() chdirs into the project folder; restore the real cwd once the test ends so
    # later tests that resolve relative paths are not affected.
    monkeypatch.chdir(Path.cwd())
    orch = colab.start("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert [s.name for s in orch.stages] == ["intake", "data", "clean", "codegen", "train",
                                             "report"]
    snap = colab._context_snapshot(orch.ctx.project, "train")
    assert snap["config"] is None and snap["latest_run"] is None


def test_context_snapshot_includes_data_meta_and_audit(tmp_path):
    ctx = colab.make_context("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    ctx.project.write_json("data_meta.json", {"target": "y"})
    ctx.project.write_json(
        "audit.json", {"issues": [{"kind": "missing_values"}], "decisions": [], "steps": []}
    )
    snap = ctx.explainer.context_provider()
    assert snap["data_meta"] == {"target": "y"}
    assert snap["audit_issue_kinds"] == ["missing_values"]


def test_explain_uses_last_context(tmp_path, monkeypatch):
    llm = FakeLLM(script=[[("text", "One pass over the data.")]])
    shown: list[str] = []
    # start() chdirs into the project folder; restore the real cwd once the test ends so
    # later tests that resolve relative paths are not affected.
    monkeypatch.chdir(Path.cwd())
    orch = colab.start("demo", drive_root=str(tmp_path), llm=llm)
    orch.ctx.questioner = ScriptedQuestioner([])
    orch.ctx.explainer.display = shown.append
    colab.explain("epoch")
    assert shown and "One pass" in shown[0]
    assert Path(tmp_path, "projects", "demo", "glossary.json").exists()


def test_cell_source_and_script_cells():
    from mlagent import colab

    assert colab.cell_source(["profile.py"]) == "%load profile.py"
    assert colab.cell_source(["evaluate.py", "--split", "test"]) == "%run evaluate.py --split test"
    assert colab.script_cells("train") == ["%load train.py", "%load evaluate.py"]
    assert colab.script_cells("intake") == []
    assert colab.script_cells("nope") == []
    assert set(colab.HANDOFF_COMMANDS) == {"intake", "data", "clean", "codegen", "train",
                                           "report"}


def test_handoff_commands_match_what_the_stages_return(tmp_path):
    from mlagent import colab
    from mlagent.stages.clean import CLEAN_PY
    from mlagent.stages.report import EVAL_TEST_COMMAND

    assert colab.HANDOFF_COMMANDS["clean"] == [[CLEAN_PY]]
    assert colab.HANDOFF_COMMANDS["report"] == [list(EVAL_TEST_COMMAND)]


def test_start_changes_into_the_project_folder(tmp_path, monkeypatch):
    import os

    from mlagent import colab
    from mlagent.llm import FakeLLM

    monkeypatch.chdir(tmp_path)
    colab.start("demo", drive_root=str(tmp_path / "drive"), llm=FakeLLM([]))
    assert Path(os.getcwd()).resolve() == (
        tmp_path / "drive" / "projects" / "demo"
    ).resolve()


def test_purge_modules_forgets_generated_scripts(monkeypatch):
    import sys

    from mlagent import colab

    sentinel = object()
    monkeypatch.setitem(sys.modules, "model", sentinel)
    monkeypatch.setitem(sys.modules, "evaluate", sentinel)
    colab._purge_modules()
    assert "model" not in sys.modules and "evaluate" not in sys.modules
