"""Append-only run history (runs.jsonl) and helpers to summarise it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Which direction is "better" for each metric the intake stage can choose.
HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "f1": True,
    "r2": True,
    "rmse": False,
    "mae": False,
}


def read_runs(path: Path) -> list[dict]:
    """Return every parseable JSON object line; a corrupt line is skipped, not fatal."""
    if not path.exists():
        return []
    runs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            runs.append(entry)
    return runs


def append_run(path: Path, entry: dict[str, Any]) -> dict:
    """Append one run. `run_id` defaults to the number of existing runs plus one."""
    runs = read_runs(path)
    entry = dict(entry)
    entry.setdefault("run_id", len(runs) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def is_better(metric: str, candidate: float | None, incumbent: float | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if HIGHER_IS_BETTER.get(metric, True):
        return candidate > incumbent
    return candidate < incumbent


def best_run(runs: list[dict], metric: str) -> dict | None:
    best: dict | None = None
    for run in runs:
        if run.get("status") != "done":
            continue
        run_metric = run.get("best_val_metric")
        best_metric = best.get("best_val_metric") if best else None
        if best is None or is_better(metric, run_metric, best_metric):
            best = run
    return best


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


_COLUMNS = (
    "run_id",
    "status",
    "epochs_run",
    "best_epoch",
    "best_val_metric",
    "final_train_loss",
    "final_val_loss",
    "seconds",
)


def summarise(runs: list[dict], metric: str) -> str:
    """Markdown table: run, status, epochs, best epoch, best val metric, final losses, seconds."""
    if not runs:
        return "_No runs yet._"
    header = (
        f"| run | status | epochs | best epoch | val {metric} | train loss | val loss | seconds |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = ["| " + " | ".join(_fmt(r.get(k)) for k in _COLUMNS) + " |" for r in runs]
    return header + "\n" + "\n".join(rows)
