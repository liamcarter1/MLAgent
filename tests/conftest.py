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


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEATURES = [f"f{i}" for i in range(5)]


def write_clean_project(
    project: Project, task_type: str, n_rows: int = 240, seed: int = 0
) -> Project:
    """Write data/clean/data.csv, data_meta.json and spec.json as Milestone 2 leaves them."""
    from sklearn.datasets import make_classification, make_regression

    rng = np.random.default_rng(seed)
    if task_type == "tabular_classification":
        X, y = make_classification(
            n_samples=n_rows, n_features=5, n_informative=3, n_redundant=0, random_state=seed
        )
        metric = "accuracy"
    else:
        X, y = make_regression(
            n_samples=n_rows, n_features=5, n_informative=3, noise=5.0, random_state=seed
        )
        metric = "rmse"
    df = pd.DataFrame(X, columns=FEATURES)
    df["colour"] = rng.choice(["red", "green", "blue"], size=n_rows)
    df["target"] = y
    project.ensure_dirs()
    df.to_csv(project.data_clean / "data.csv", index=False)
    meta = {
        "target": "target",
        "task_type": task_type,
        "source": "synthetic",
        "raw_path": "data/raw/data.csv",
        "raw_n_rows": n_rows,
        "raw_n_cols": 7,
        "clean_path": "data/clean/data.csv",
        "clean_n_rows": n_rows,
        "clean_n_cols": 7,
        "dropped_columns": [],
        "feature_columns": FEATURES + ["colour"],
        "categorical_columns": ["colour"],
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "split_seed": 42,
    }
    if task_type == "tabular_classification":
        meta["n_classes"] = 2
        meta["class_labels"] = ["0", "1"]
    project.write_json("data_meta.json", meta)
    project.write_json(
        "spec.json",
        {
            "goal": "fixture project",
            "task_type": task_type,
            "metric": metric,
            "target_value": 0.8,
            "data_source": "synthetic",
            "minutes_per_run": 5,
            "max_rounds": 3,
            "gpu": "none",
            "notes": "",
        },
    )
    return project


@pytest.fixture
def clean_project(project: Project) -> Project:
    return write_clean_project(project, "tabular_classification")


@pytest.fixture
def regression_project(project: Project) -> Project:
    return write_clean_project(project, "tabular_regression")


import subprocess  # noqa: E402
import sys  # noqa: E402


def run_handoff(project, handoff) -> None:
    """Run every cell of a handoff the way the user would, from the project folder."""
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=str(project.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
        assert result.returncode == 0, (
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )


@pytest.fixture
def advance():
    """Drive an orchestrator to a standstill, running each handoff's cells in between."""

    def _advance(orch, project, answers=None, limit=12) -> list[str]:
        ran: list[str] = []
        for _ in range(limit):
            ran += orch.run(answers=answers)
            handoff = orch.waiting()
            if handoff is None:
                return ran
            run_handoff(project, handoff)
        raise AssertionError(f"pipeline did not settle after {limit} rounds")

    return _advance
