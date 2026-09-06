"""Entry points used by the notebook. Safe to import outside Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mlagent import config
from mlagent.llm import LLM, AnthropicLLM
from mlagent.orchestrator import Orchestrator
from mlagent.project import Project
from mlagent.stages.base import StageContext
from mlagent.stages.intake import IntakeStage
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ConsoleQuestioner
from mlagent.ui.render import display_message

_LAST_CTX: StageContext | None = None


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
    return {
        "project": project.name,
        "stage": stage_name,
        "spec": project.read_json(config.SPEC_FILE),
        "state": project.read_json(config.STATE_FILE),
    }


def make_context(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> StageContext:
    global _LAST_CTX
    projects = setup(drive_root=drive_root, mount=False)
    project = Project(projects / project_name)
    project.ensure_dirs()
    llm = llm or AnthropicLLM()
    explainer = Explainer(
        llm=llm,
        glossary=Glossary(project.glossary_path),
        context_provider=lambda: _context_snapshot(project),
        display=display_message,
    )
    ctx = StageContext(project=project, llm=llm, questioner=ConsoleQuestioner(),
                       explainer=explainer, display=display_message)
    _LAST_CTX = ctx
    return ctx


def start(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> Orchestrator:
    ctx = make_context(project_name, drive_root=drive_root, llm=llm)
    ctx.explainer.register_colab_callback()
    return Orchestrator(ctx, [IntakeStage()])


def explain(term: str, refresh: bool = False) -> None:
    if _LAST_CTX is None or _LAST_CTX.explainer is None:
        raise RuntimeError("call start(project_name) first")
    _LAST_CTX.explainer.show(term, refresh=refresh)
