"""A project folder on disk (Drive in Colab, tmp dir in tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlagent import config


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

    def ensure_dirs(self) -> None:
        for d in (self.root, self.data_raw, self.data_clean, self.plots_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        return (self.root / filename).exists()

    def read_json(self, filename: str, default: Any = None) -> Any:
        path = self.root / filename
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, filename: str, obj: Any) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / filename
        path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        return path
