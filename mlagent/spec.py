"""The project specification produced by the intake stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

TASK_TYPES = ("tabular_classification", "tabular_regression", "image_classification")
DATA_SOURCES = ("synthetic", "drive", "huggingface")
GPU_CHOICES = ("none", "T4", "any")
METRICS_FOR_TASK: dict[str, list[str]] = {
    "tabular_classification": ["accuracy", "f1"],
    "tabular_regression": ["rmse", "mae", "r2"],
    "image_classification": ["accuracy", "f1"],
}


class SpecError(ValueError):
    pass


@dataclass
class Spec:
    goal: str
    task_type: str
    metric: str
    target_value: float
    data_source: str
    minutes_per_run: int
    max_rounds: int
    gpu: str
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.goal.strip():
            problems.append("goal must not be empty")
        if self.task_type not in TASK_TYPES:
            problems.append(f"task_type must be one of {TASK_TYPES}")
        allowed = METRICS_FOR_TASK.get(self.task_type, [])
        if self.task_type in TASK_TYPES and self.metric not in allowed:
            problems.append(f"metric must be one of {allowed} for {self.task_type}")
        elif self.task_type not in TASK_TYPES and self.metric not in {
            m for ms in METRICS_FOR_TASK.values() for m in ms
        }:
            problems.append("metric is not a known metric")
        if self.data_source not in DATA_SOURCES:
            problems.append(f"data_source must be one of {DATA_SOURCES}")
        if self.gpu not in GPU_CHOICES:
            problems.append(f"gpu must be one of {GPU_CHOICES}")
        if self.minutes_per_run < 1:
            problems.append("minutes_per_run must be >= 1")
        if self.max_rounds < 1:
            problems.append("max_rounds must be >= 1")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Spec:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise SpecError(f"unknown spec keys: {sorted(unknown)}")
        missing = known - set(d) - {"notes"}
        if missing:
            raise SpecError(f"missing spec keys: {sorted(missing)}")
        return cls(
            goal=str(d["goal"]),
            task_type=str(d["task_type"]),
            metric=str(d["metric"]),
            target_value=float(d["target_value"]),
            data_source=str(d["data_source"]),
            minutes_per_run=int(d["minutes_per_run"]),
            max_rounds=int(d["max_rounds"]),
            gpu=str(d["gpu"]),
            notes=str(d.get("notes", "")),
        )
