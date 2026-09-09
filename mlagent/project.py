"""A project folder on disk (Drive in Colab, tmp dir in tests)."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from mlagent import config


def read_json_file(path: Path, default: Any = None) -> Any:
    """Read JSON from `path`, tolerating a missing or corrupt file.

    Returns `default` (and warns, naming the file) if the file exists but is not
    valid JSON, instead of raising.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.warn(f"could not parse JSON from {path}; using default", stacklevel=2)
        return default


def write_json_file(path: Path, obj: Any) -> Path:
    """Write `obj` as JSON to `path` atomically: write to a temp file in the same
    directory, then `os.replace` onto the target so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True))
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


class Project:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def spec_path(self) -> Path:
        return self.root / config.SPEC_FILE

    @property
    def state_path(self) -> Path:
        return self.root / config.STATE_FILE

    @property
    def glossary_path(self) -> Path:
        return self.root / config.GLOSSARY_FILE

    @property
    def config_path(self) -> Path:
        return self.root / config.CONFIG_FILE

    @property
    def runs_path(self) -> Path:
        return self.root / config.RUNS_FILE

    @property
    def metrics_path(self) -> Path:
        return self.root / config.METRICS_FILE

    @property
    def report_path(self) -> Path:
        return self.root / config.REPORT_FILE

    @property
    def report_meta_path(self) -> Path:
        return self.root / config.REPORT_META_FILE

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_clean(self) -> Path:
        return self.root / "data" / "clean"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def runs_dir(self) -> Path:
        """Per-run archives (`run{N}_metrics.json`); runs.jsonl itself is `runs_path`."""
        return self.root / config.RUNS_DIRNAME

    def ensure_dirs(self) -> None:
        for d in (self.root, self.data_raw, self.data_clean, self.plots_dir,
                  self.checkpoints_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        return (self.root / filename).exists()

    def read_json(self, filename: str, default: Any = None) -> Any:
        return read_json_file(self.root / filename, default)

    def write_json(self, filename: str, obj: Any) -> Path:
        return write_json_file(self.root / filename, obj)
