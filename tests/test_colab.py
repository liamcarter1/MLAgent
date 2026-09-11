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
                                             "tune", "report"]
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
                                           "tune", "report"}


def test_handoff_commands_match_what_the_stages_return(tmp_path):
    from mlagent import colab
    from mlagent.stages.clean import CLEAN_PY
    from mlagent.stages.report import EVAL_TEST_COMMAND
    from mlagent.stages.tune import TUNE_COMMANDS

    assert colab.HANDOFF_COMMANDS["clean"] == [[CLEAN_PY]]
    assert colab.HANDOFF_COMMANDS["train"] == [["train.py"], ["evaluate.py"]]
    assert colab.HANDOFF_COMMANDS["tune"] == TUNE_COMMANDS
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
    assert checked == 30  # every #@param field across the four form cells was checked


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


def test_notebook_has_a_tune_cell_between_evaluate_and_report():
    sources = [c["source"] for c in _load_notebook_cells() if c.get("cell_type") == "code"]
    tune = next(i for i, s in enumerate(sources) if "orch.run(until='tune')" in s)
    assert sources[tune - 1] == "%load evaluate.py"
    assert "orch.run(until='report')" in sources[tune + 1]
    assert "#@param" not in sources[tune]
    assert sources[tune].startswith("#@title 6. Tune")
    assert "#@title 7. Report" in sources[tune + 1]


NOTEBOOK = Path("notebooks/ML_Training_Agent.ipynb")


def notebook_sources() -> list[str]:
    import json

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def data_cell() -> str:
    return next(s for s in notebook_sources() if "#@title 2. Data" in s)


def test_the_task_field_offers_image_classification():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    assert "'Image classification'" in intake
    assert "TASK_LABELS" not in intake  # the labels live in the intake stage, not the cell


def test_every_task_label_in_the_notebook_is_one_the_intake_stage_knows():
    import re

    from mlagent.stages.intake import TASK_LABELS

    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    line = next(ln for ln in intake.splitlines() if ln.startswith("TASK = "))
    assert set(re.findall(r"'([^']+)'", line)) <= set(TASK_LABELS) | {
        "Tabular classification"
    }


def test_the_data_cell_has_the_image_fields_with_hints():
    source = data_cell()
    for field in ("N_IMAGES", "IMAGE_SIZE", "DRIVE_FOLDER", "HF_DATASET", "MAX_IMAGES"):
        assert f"{field} = " in source, field
        assert f"**{field}**" in source, field
    assert "IMAGE_SIZE = '64'" in source
    assert "['32', '64', '128']" in source


def test_the_max_images_default_is_2000():
    source = data_cell()
    assert "MAX_IMAGES = 2000" in source


def test_the_tabular_only_hints_say_so():
    source = data_cell()
    for line in source.splitlines():
        if "**N_FEATURES**" in line or "**TARGET_COLUMN**" in line:
            assert "ignored for image tasks" in line


def test_the_n_classes_hint_covers_images_too():
    source = data_cell()
    line = next(ln for ln in source.splitlines() if "**N_CLASSES**" in ln)
    assert "for images, 2 to 5" in line


def model_cell() -> str:
    return next(s for s in notebook_sources() if "#@title 4. Model" in s)


def test_the_model_cell_s_image_choices_match_the_image_family_labels():
    """A label drift between the notebook and codegen.py must never silently fall back to
    a console prompt in Colab (input() blocks a running cell)."""
    import re

    from mlagent.stages.codegen import labels_for

    source = model_cell()
    line = next(ln for ln in source.splitlines() if ln.startswith("MODEL = "))
    choices = re.findall(r"'([^']+)'", line.split("#@param", 1)[1])
    tabular_and_ask = set(labels_for("tabular_sklearn")) | {"Ask me after the explanation"}
    image_choices = [c for c in choices if c not in tabular_and_ask]
    assert set(image_choices) == set(labels_for("image_torch"))


def test_the_data_cell_passes_every_image_answer_key():
    source = data_cell()
    for key in ("data.n_images", "data.image_size", "data.drive_folder", "data.hf_dataset",
                "data.max_images"):
        assert f"'{key}'" in source, key


def test_the_answer_keys_the_data_cell_sends_are_ones_the_data_stage_reads():
    import re

    source = (Path("mlagent/stages/data.py")).read_text(encoding="utf-8")
    used = set(re.findall(r'key="(data\.[a-z_]+)"', source))
    sent = set(re.findall(r"'(data\.[a-z_]+)'", data_cell()))
    assert sent <= used, sent - used


def test_the_pip_line_installs_torch():
    pip = next(s for s in notebook_sources() if s.startswith("%pip"))
    assert "torch" in pip and "torchvision" in pip
    assert "pillow" in pip


def test_the_intro_mentions_image_tasks():
    intro = notebook_sources()[0]
    assert "image" in intro.lower()


def test_the_gpu_hint_no_longer_defers_images_to_a_later_milestone():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    assert "later milestone" not in intake


def test_the_model_cell_has_the_price_and_currency_fields_with_hints():
    source = model_cell()
    for field in ("PRICE_PER_UNIT", "CURRENCY"):
        assert f"{field} = " in source, field
        assert f"**{field}**" in source, field
    assert "Resources panel" in source
    assert "next to the cost estimate" in source


def test_the_model_cell_defaults_match_rates_json():
    from mlagent.cost import load_rates

    rates = load_rates()
    source = model_cell()
    assert f"PRICE_PER_UNIT = {rates['default_price_per_unit']}  #@param" in source
    assert f"CURRENCY = '{rates['default_currency']}'  #@param" in source


def test_the_model_cell_passes_every_cost_answer_key():
    source = model_cell()
    for key in ("codegen.model_type", "train.price_per_unit", "train.currency"):
        assert f"'{key}'" in source, key


def test_the_train_answer_keys_the_model_cell_sends_are_ones_the_gate_reads():
    import re

    source = Path("mlagent/stages/cost_gate.py").read_text(encoding="utf-8")
    used = set(re.findall(r'key="(train\.[a-z_]+)"', source))
    sent = set(re.findall(r"'(train\.[a-z_]+)'", model_cell()))
    assert sent <= used, sent - used


def test_the_tune_cell_gains_no_form_fields():
    tune = next(s for s in notebook_sources() if "#@title 6. Tune" in s)
    assert "#@param" not in tune
    assert "PRICE_PER_UNIT" not in tune


def test_the_minutes_per_run_hint_mentions_the_cost_gate():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    line = next(ln for ln in intake.splitlines() if "**MINUTES_PER_RUN**" in ln)
    assert "cost gate" in line


def test_the_notebook_is_still_twenty_one_cells():
    assert len(notebook_sources()) == 21


def test_the_notebook_is_up_to_date_with_the_builder():
    import subprocess
    import sys

    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, "scripts/build_notebook.py"], check=True,
                   capture_output=True, text=True)
    assert NOTEBOOK.read_bytes() == before
