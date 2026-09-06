"""Find and load tabular files on the mounted Drive (or any directory)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def list_candidates(
    roots, exts=TABLE_EXTS, max_depth: int = 3, limit: int = 50
) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - base
            if depth >= max_depth:
                dirnames[:] = []
            else:
                dirnames[:] = sorted(
                    d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
                )
            for name in sorted(filenames):
                if Path(name).suffix.lower() in exts:
                    found.append(Path(dirpath) / name)
                    if len(found) >= limit:
                        return found
    return found


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(
        f"unsupported file type {suffix!r}; use CSV, TSV, Parquet, or Excel"
    )
