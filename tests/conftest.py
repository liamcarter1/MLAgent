import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from mlagent.project import Project  # noqa: E402


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project(tmp_path / "projects" / "demo")
    p.ensure_dirs()
    return p
