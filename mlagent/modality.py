"""What a stage needs to know to treat one kind of data uniformly.

`DataStage`, `CleanStage` and `CodegenStage` never import `synth.tabular`,
`datasources.drive`, `datasources.hf`, `audit` or `cleaning` directly: they look up a
`Modality` record by task type and call its callables. Adding a modality is adding a
record here plus the modules it points at, never an `if task_type == ...` branch
scattered through the stages.

`TEMPLATE_FOR_TASK` stays a literal dict in `templates_io.py` (this module imports
`cleaning`, which imports `templates_io`, so building it here would be circular);
`tests/test_modality.py` asserts the two agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlagent import audit, cleaning
from mlagent.datasources import drive, hf
from mlagent.synth import tabular

TABULAR_RAW_FILE = "data.csv"


@dataclass(frozen=True)
class Modality:
    name: str                          # "tabular" | "image"
    task_types: tuple[str, ...]
    data_file: str                     # "data.csv" | "data.npz"
    profile_template: str              # source path under mlagent/templates
    clean_template: str
    template_family: str               # "tabular_sklearn" | "image_torch"
    teaching_material: str             # prompts/teaching/<name>.md shown before the model pick
    profile_figures: tuple[str, ...]   # ordered figure kinds the profile draws
    generate: Callable[..., object]
    load_drive: Callable[..., object]
    load_hf: Callable[..., object]
    audit: Callable[..., object]
    apply_steps: Callable[..., object]
    render_clean_py: Callable[..., str]
    write_raw: Callable[..., object]
    read: Callable[[Path], object]


def _write_table(df: pd.DataFrame, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABULAR_RAW_FILE
    df.to_csv(path, index=False)
    return path


def _read_table(directory: Path) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / TABULAR_RAW_FILE)


TABULAR = Modality(
    name="tabular",
    task_types=("tabular_classification", "tabular_regression"),
    data_file=TABULAR_RAW_FILE,
    profile_template="common/profile.py",
    clean_template="common/clean.py",
    template_family="tabular_sklearn",
    teaching_material="model_choices",
    profile_figures=("histograms", "missing", "class_balance", "correlation"),
    generate=tabular.generate,
    load_drive=drive.load_table,
    load_hf=hf.load_tabular,
    audit=audit.audit_tabular,
    apply_steps=cleaning.apply_steps,
    render_clean_py=cleaning.render_clean_py,
    write_raw=_write_table,
    read=_read_table,
)

MODALITIES: tuple[Modality, ...] = (TABULAR,)


def modality_for(task_type: str) -> Modality:
    """The record for a task type; the only lookup function."""
    for modality in MODALITIES:
        if task_type in modality.task_types:
            return modality
    valid = sorted(t for m in MODALITIES for t in m.task_types)
    raise ValueError(f"unknown task type {task_type!r}; expected one of {valid}")
