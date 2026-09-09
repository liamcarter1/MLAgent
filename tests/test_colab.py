import re
from pathlib import Path

from mlagent import colab
from mlagent.llm import FakeLLM
from mlagent.ui.questions import ScriptedQuestioner

BUILD_NOTEBOOK = Path(__file__).resolve().parents[1] / "scripts" / "build_notebook.py"


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


def _load_notebook_cells():
    """Execute scripts/build_notebook.py and return its `cells` list of nbformat cells."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_notebook_module", BUILD_NOTEBOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cells


def test_every_param_field_has_a_hint_naming_it():
    cells = _load_notebook_cells()
    checked = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        lines = source.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=.*#@param", line)
            if not m:
                continue
            var_name = m.group(1)
            checked += 1
            preceding = [j for j in range(i) if lines[j].lstrip().startswith("#@markdown")]
            assert preceding, f"no #@markdown hint found before '{line}' in cell:\n{source}"
            hint_line = lines[preceding[-1]]
            label_match = re.search(r"\*\*(.+?)\*\*", hint_line)
            assert label_match, f"hint line has no **label**: {hint_line!r}"
            assert label_match.group(1).lower() == var_name.lower(), (
                f"hint label {label_match.group(1)!r} does not name field {var_name!r}"
            )
    assert checked == 23  # every #@param field across the four form cells was checked


def test_every_form_cell_has_a_purpose_line_after_its_title():
    cells = _load_notebook_cells()
    form_cells = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        lines = cell["source"].splitlines()
        if not lines or "#@title" not in lines[0]:
            continue
        if "display-mode: 'form'" not in lines[0]:
            continue
        form_cells += 1
        assert len(lines) > 1 and lines[1].lstrip().startswith("#@markdown"), (
            f"form cell has no purpose line directly after its title: {lines[:2]}"
        )
    assert form_cells == 4  # the four #@param form cells (project/interview, data, clean, model)


def test_purge_modules_forgets_generated_scripts(monkeypatch):
    import sys

    from mlagent import colab

    sentinel = object()
    monkeypatch.setitem(sys.modules, "model", sentinel)
    monkeypatch.setitem(sys.modules, "evaluate", sentinel)
    colab._purge_modules()
    assert "model" not in sys.modules and "evaluate" not in sys.modules
