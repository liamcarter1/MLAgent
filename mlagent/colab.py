"""Entry points used by the notebook. Safe to import outside Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mlagent import config
from mlagent.llm import LLM, AnthropicLLM
from mlagent.orchestrator import Orchestrator
from mlagent.project import Project
from mlagent.runlog import read_runs
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CleanStage
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.data import META_FILE, DataStage
from mlagent.stages.intake import IntakeStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
from mlagent.stages.tune import TuneStage
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ConsoleQuestioner
from mlagent.ui.render import display_message

_LAST_CTX: StageContext | None = None

# The single place the "run a script in its own cell" mechanism is defined. Swapping
# %load for %run here changes every generated cell and the notebook builder at once.
SCRIPT_CELL_FORMATS = {"load": "%load {script}", "run": "%run {script} {args}"}

# What each stage's handoff asks the user to run. Kept in step with the stages by
# tests/test_colab.py, and shared with scripts/build_notebook.py.
HANDOFF_COMMANDS: dict[str, list[list[str]]] = {
    "intake": [],
    "data": [["profile.py"]],
    "clean": [["clean.py"]],
    "codegen": [],
    "train": [["train.py"], ["evaluate.py"]],
    "tune": [["train.py"], ["evaluate.py"]],
    "report": [["evaluate.py", "--split", "test"]],
}

# Scripts the user runs in their own cells. Python caches them after the first import,
# so a regenerated file would be ignored without this purge.
PURGED_MODULES = ("clean", "data", "evaluate", "model", "profile", "train")


def cell_source(command: list[str]) -> str:
    """The cell text for one handoff command."""
    script, *args = command
    if args:
        return SCRIPT_CELL_FORMATS["run"].format(script=script, args=" ".join(args))
    return SCRIPT_CELL_FORMATS["load"].format(script=script)


def script_cells(stage_name: str) -> list[str]:
    return [cell_source(command) for command in HANDOFF_COMMANDS.get(stage_name, [])]


def _purge_modules() -> None:
    for name in PURGED_MODULES:
        sys.modules.pop(name, None)


def _register_purge_hook() -> bool:
    """Forget the generated modules before every cell, so an edited script is re-read."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    shell.events.register("pre_run_cell", lambda *_args, **_kwargs: _purge_modules())
    return True


def _try_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:  # noqa: BLE001 - not running in Colab, nothing to mount
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def _load_api_key_from_secrets() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 - not running in Colab or secret unavailable
        return
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key


def setup(drive_root: str = config.DRIVE_ROOT, mount: bool = True) -> Path:
    if mount:
        _try_mount_drive()
    root = Path(drive_root)
    projects = root / config.PROJECTS_DIRNAME
    projects.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _load_api_key_from_secrets()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "No ANTHROPIC_API_KEY found. In Colab, open the Secrets panel (key icon), "
            "add ANTHROPIC_API_KEY, and turn on notebook access; then rerun this cell."
        )
    return projects


def _context_snapshot(project: Project, stage_name: str = "") -> dict:
    runs = read_runs(project.runs_path)
    return {
        "project": project.name,
        "stage": stage_name,
        "spec": project.read_json(config.SPEC_FILE),
        "state": project.read_json(config.STATE_FILE),
        "data_meta": project.read_json(META_FILE),
        "audit_issue_kinds": [
            i.get("kind") for i in (project.read_json(AUDIT_FILE) or {}).get("issues", [])
        ],
        "config": project.read_json(config.CONFIG_FILE),
        "latest_run": runs[-1] if runs else None,
    }


def make_context(
    project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None
) -> StageContext:
    global _LAST_CTX
    projects = setup(drive_root=drive_root, mount=False)
    project = Project(projects / project_name)
    project.ensure_dirs()
    llm = llm or AnthropicLLM()
    ctx = StageContext(project=project, llm=llm, questioner=ConsoleQuestioner(),
                       explainer=None, display=display_message)
    ctx.explainer = Explainer(
        llm=llm,
        glossary=Glossary(project.glossary_path),
        # Reads ctx.stage at call time, not at construction time, so it reflects
        # whichever stage the orchestrator is currently running.
        context_provider=lambda: _context_snapshot(project, ctx.stage),
        display=display_message,
    )
    _LAST_CTX = ctx
    return ctx


def start(
    project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None
) -> Orchestrator:
    ctx = make_context(project_name, drive_root=drive_root, llm=llm)
    ctx.explainer.register_colab_callback()
    # The generated scripts read and write project-relative paths, and the user runs them
    # from their own cells, so the notebook's working directory must be the project.
    os.chdir(ctx.project.root)
    _register_purge_hook()
    return Orchestrator(
        ctx,
        [IntakeStage(), DataStage(), CleanStage(), CodegenStage(), TrainStage(), TuneStage(),
         ReportStage()],
    )


def explain(term: str, refresh: bool = False) -> None:
    if _LAST_CTX is None or _LAST_CTX.explainer is None:
        raise RuntimeError("call start(project_name) first")
    _LAST_CTX.explainer.show(term, refresh=refresh)
