"""Estimate what a training run costs, before it starts.

Pure functions only: no stage, no LLM, no user interaction. `mlagent/stages/cost_gate.py`
is the only caller that asks questions or writes project files; everything here takes its
inputs as arguments so it can be unit-tested without a project on disk.

Rates live in `rates.json` next to this module and are read the same way
`prompts_io.load_prompt` reads `prompts/*.md`: Colab's per-GPU compute-unit rates and the
plan price change over time, so they are data, not constants baked into Python.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

RATES_PATH = Path(__file__).parent / "rates.json"
DRY_RUN_FILE = "dry_run.json"
SAFETY_FACTOR = 1.2
DRY_RUN_TIMEOUT = 600
STDERR_TAIL_LINES = 20


def load_rates() -> dict:
    """The packaged rate table: units per hour per GPU, plus the price defaults."""
    return json.loads(RATES_PATH.read_text(encoding="utf-8"))


# --- runtime detection ---
@dataclass(frozen=True)
class RuntimeInfo:
    device: str                 # "cuda" | "cpu"
    gpu_name: str | None        # what the probe reported, verbatim
    gpu_type: str | None        # a rates key, "unknown", or None when there is no GPU
    source: str                 # "torch" | "nvidia-smi" | "none"


def torch_gpu_name() -> str | None:
    """The CUDA device's name, or None. `torch` is imported lazily exactly as
    `mlagent/templates/image_torch/data.py::pick_device` does -- nothing in `mlagent/`
    imports torch at module scope, so a tabular-only install still works."""
    import torch

    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def nvidia_smi_gpu_name(runner=subprocess.run) -> str | None:
    """The first GPU `nvidia-smi` reports, or None when it is absent or fails."""
    command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    try:
        result = runner(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first = (result.stdout or "").strip().splitlines()
    return first[0].strip() if first and first[0].strip() else None


DEFAULT_PROBES: tuple[tuple[str, Callable[[], str | None]], ...] = (
    ("torch", torch_gpu_name),
    ("nvidia-smi", nvidia_smi_gpu_name),
)


def gpu_type_for(gpu_name: str | None, rates: dict) -> str | None:
    """A rates key matched case-insensitively inside the name, else "unknown", else None."""
    if not gpu_name:
        return None
    lowered = gpu_name.lower()
    for key in (rates.get("units_per_hour") or {}):
        if key.lower() in lowered:
            return key
    return "unknown"


def detect_runtime(
    probes: Sequence[tuple[str, Callable[[], str | None]]] | None = None
) -> RuntimeInfo:
    """Which accelerator this runtime has, without ever raising.

    `probes` is a sequence of `(source, callable)` pairs; the callable returns a GPU name
    or None. Passing anything other than None *replaces* the built-in probes, so a test
    can assert the no-GPU path on a machine that has a GPU (`detect_runtime(())`).
    """
    rates = load_rates()
    for source, probe in (DEFAULT_PROBES if probes is None else probes):
        try:
            name = probe()
        except Exception:  # noqa: BLE001 - a missing driver must not break the gate
            continue
        if name:
            return RuntimeInfo(device="cuda", gpu_name=name,
                               gpu_type=gpu_type_for(name, rates), source=source)
    return RuntimeInfo(device="cpu", gpu_name=None, gpu_type=None, source="none")


def rate_for(runtime: RuntimeInfo, rates: dict) -> tuple[float, str | None]:
    """(compute units per hour, a note to show the user) for this runtime."""
    if runtime.device != "cuda":
        return 0.0, None
    table = rates.get("units_per_hour") or {}
    if runtime.gpu_type in table:
        return float(table[runtime.gpu_type]), None
    fallback = float(rates.get("unknown_gpu_rate", 2.0))
    return fallback, (
        f"'{runtime.gpu_name}' is not one of the known [[GPU]] types; using the T4 rate "
        "as an estimate."
    )


# --- the dry run's record ---
@dataclass(frozen=True)
class DryRunResult:
    device: str = "cpu"
    gpu_name: str | None = None
    batches_per_epoch: int = 0
    seconds_per_batch: float = 0.0
    seconds_per_epoch: float = 0.0
    n_train: int = 0
    script: str = "train.py"
    error: str | None = None


def read_dry_run(path: Path) -> DryRunResult:
    """Parse `dry_run.json`; a missing or unreadable file comes back zeroed with `error`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return DryRunResult(error=f"could not read {path.name}: {exc}")
    if not isinstance(data, dict):
        return DryRunResult(error=f"could not read {path.name}: not a JSON object")
    return DryRunResult(
        device=str(data.get("device") or "cpu"),
        gpu_name=str(data["gpu_name"]) if data.get("gpu_name") else None,
        batches_per_epoch=int(data.get("batches_per_epoch") or 0),
        seconds_per_batch=float(data.get("seconds_per_batch") or 0.0),
        seconds_per_epoch=float(data.get("seconds_per_epoch") or 0.0),
        n_train=int(data.get("n_train") or 0),
        script=str(data.get("script") or "train.py"),
        error=None,
    )
