from pathlib import Path

import pytest

from mlagent.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project(tmp_path / "projects" / "demo")
    p.ensure_dirs()
    return p
