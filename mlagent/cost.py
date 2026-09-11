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
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mlagent import config

RATES_PATH = Path(__file__).parent / "rates.json"
DRY_RUN_FILE = config.DRY_RUN_FILE
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


# --- the estimate ---
@dataclass(frozen=True)
class Estimate:
    seconds_per_epoch: float
    epochs: int
    minutes: float
    units: float
    cost: float
    price_per_unit: float
    currency: str
    rate_units_per_hour: float
    rounds_remaining: int
    minutes_all_rounds: float
    units_all_rounds: float
    cost_all_rounds: float
    basis: str                  # "dry_run" | "history"
    runtime: RuntimeInfo
    note: str | None = None

    def to_dict(self) -> dict:
        """A plain JSON-ready dict for `cost.json`'s `last_estimate`."""
        return {
            "seconds_per_epoch": self.seconds_per_epoch,
            "epochs": self.epochs,
            "minutes": self.minutes,
            "units": self.units,
            "cost": self.cost,
            "price_per_unit": self.price_per_unit,
            "currency": self.currency,
            "rate_units_per_hour": self.rate_units_per_hour,
            "rounds_remaining": self.rounds_remaining,
            "minutes_all_rounds": self.minutes_all_rounds,
            "units_all_rounds": self.units_all_rounds,
            "cost_all_rounds": self.cost_all_rounds,
            "basis": self.basis,
            "device": self.runtime.device,
            "gpu_name": self.runtime.gpu_name,
            "gpu_type": self.runtime.gpu_type,
            "note": self.note,
        }


def estimate_run(seconds_per_epoch: float, epochs: int, runtime: RuntimeInfo, rates: dict,
                 price_per_unit: float, currency: str, rounds_remaining: int,
                 basis: str) -> Estimate:
    """The 2026-09-06 design spec's formula, in per-epoch terms.

        hours = seconds_per_epoch * epochs * 1.2 / 3600
        units = hours * rate_units_per_hour
        cost  = units * price_per_unit

    `seconds_per_epoch * epochs` is the same quantity as the original
    `batches_per_epoch * epochs * sec_per_batch`; the dry run already reduces to a
    per-epoch time. The all-rounds figures are this run times `1 + rounds_remaining`.
    """
    rate, note = rate_for(runtime, rates)
    rounds = max(0, int(rounds_remaining))
    hours = float(seconds_per_epoch) * int(epochs) * SAFETY_FACTOR / 3600.0
    units = hours * rate
    money_cost = units * float(price_per_unit)
    factor = 1 + rounds
    return Estimate(
        seconds_per_epoch=float(seconds_per_epoch),
        epochs=int(epochs),
        minutes=round(hours * 60.0, 3),
        units=round(units, 4),
        cost=round(money_cost, 4),
        price_per_unit=float(price_per_unit),
        currency=str(currency),
        rate_units_per_hour=rate,
        rounds_remaining=rounds,
        minutes_all_rounds=round(hours * 60.0 * factor, 3),
        units_all_rounds=round(units * factor, 4),
        cost_all_rounds=round(money_cost * factor, 4),
        basis=str(basis),
        runtime=runtime,
        note=note,
    )


def estimate_from_dry_run(dry: DryRunResult, epochs: int, runtime: RuntimeInfo, rates: dict,
                          price_per_unit: float, currency: str,
                          rounds_remaining: int) -> Estimate:
    return estimate_run(dry.seconds_per_epoch, epochs, runtime, rates, price_per_unit,
                        currency, rounds_remaining, "dry_run")


def estimate_from_history(seconds_per_epoch: float, epochs: int, runtime: RuntimeInfo,
                          rates: dict, price_per_unit: float, currency: str,
                          rounds_remaining: int) -> Estimate:
    """The caller supplies `seconds_per_epoch` out of an archived `run{N}_metrics.json`;
    this function does no file I/O, so it unit-tests like `estimate_from_dry_run`."""
    return estimate_run(seconds_per_epoch, epochs, runtime, rates, price_per_unit,
                        currency, rounds_remaining, "history")


# --- the budget check ---
@dataclass(frozen=True)
class BudgetAdvice:
    over: bool
    minutes_over: float
    suggestions: list[tuple[str, object, object, str]]


def budget_check(estimate: Estimate, minutes_per_run: float, config: dict, schema: dict,
                 modality) -> BudgetAdvice:
    """Is this run over the intake budget, and which config keys would bring it back?

    `schema` is a flat `templates_io.schema_for` result: a suggestion is only ever made
    for a key that is actually in it, so a family that drops a key never gets a
    suggestion naming it. `image_size` is fixed at ingest, not a config key, so it is
    never suggested -- `render_estimate` mentions re-ingesting instead.
    """
    budget = float(minutes_per_run)
    if estimate.minutes <= budget:
        return BudgetAdvice(over=False, minutes_over=0.0, suggestions=[])
    suggestions: list[tuple[str, object, object, str]] = []
    if "epochs" in schema and estimate.minutes > 0:
        current = int(config.get("epochs", estimate.epochs))
        proposed = int(math.floor(current * budget / estimate.minutes))
        low = schema["epochs"].get("min")
        proposed = max(1 if low is None else int(low), min(proposed, current))
        if proposed < current:
            suggestions.append((
                "epochs", current, proposed,
                f"minutes scale with epochs, so {proposed} fits the "
                f"{budget:.0f}-minute budget",
            ))
    if getattr(modality, "name", "") == "image" and "batch_size" in schema:
        current = int(config.get("batch_size", schema["batch_size"].get("default", 32)))
        proposed = min(int(schema["batch_size"].get("max", current)), current * 2)
        if proposed > current:
            suggestions.append((
                "batch_size", current, proposed,
                "bigger batches cut the per-batch overhead, so each epoch is quicker",
            ))
    return BudgetAdvice(over=True, minutes_over=round(estimate.minutes - budget, 3),
                        suggestions=suggestions)


# --- rendering ---
def money(value: float, currency: str) -> str:
    """`$1.50` for a one-character symbol, `1.50 GBP` for a code."""
    currency = (currency or "").strip()
    if len(currency) == 1:
        return f"{currency}{value:.2f}"
    return f"{value:.2f} {currency}".strip()


def _mismatch_line(spec_gpu: str, runtime: RuntimeInfo) -> str | None:
    if spec_gpu in ("T4", "any") and runtime.device == "cpu":
        return ("You asked for a [[GPU]] at intake, but this runtime has none. Switch with "
                "Runtime -> Change runtime type, then run this cell again.")
    if spec_gpu == "none" and runtime.device == "cuda":
        return ("You asked for no [[GPU]] at intake, but this runtime has one and spends "
                "[[compute unit]]s. Switch with Runtime -> Change runtime type.")
    return None


def render_estimate(estimate: Estimate, advice: BudgetAdvice, spec_gpu: str, modality,
                    basis_run_id: int | None = None) -> str:
    """Fixed markdown -- never an LLM call, identical at every learning level.

    `[[term]]` markup throughout, matching every other stage-authored sentence.
    """
    runtime = estimate.runtime
    lines: list[str] = []
    if runtime.device == "cuda":
        lines.append(f"This runtime has a {runtime.gpu_name} [[GPU]].")
    else:
        lines.append("This runtime has no [[GPU]].")
    mismatch = _mismatch_line(spec_gpu, runtime)
    if mismatch:
        lines.append(mismatch)

    lines += ["", "| | minutes | [[compute units]] | cost |", "|---|---|---|---|"]
    lines.append(f"| this run | {estimate.minutes:.1f} | {estimate.units:.2f} | "
                 f"{money(estimate.cost, estimate.currency)} |")
    if estimate.rounds_remaining > 0:
        rounds = estimate.rounds_remaining
        label = "1 remaining round" if rounds == 1 else f"{rounds} remaining rounds"
        lines.append(f"| this run plus {label} | {estimate.minutes_all_rounds:.1f} | "
                     f"{estimate.units_all_rounds:.2f} | "
                     f"{money(estimate.cost_all_rounds, estimate.currency)} |")
    lines.append("")

    if estimate.basis == "history" and basis_run_id is not None:
        lines.append(f"From run {basis_run_id}'s measured time.")
    elif estimate.basis == "history":
        lines.append("From a previous run's measured time.")
    else:
        lines.append("Timed with a 3-batch dry run.")
    if estimate.note:
        lines.append(estimate.note)
    if runtime.device == "cpu":
        lines.append("A [[CPU]] runtime uses no [[compute unit]]s, so this run is free.")
    elif getattr(modality, "name", "") == "tabular":
        lines.append("This tabular model runs on the [[CPU]], but a [[GPU]] runtime still "
                     "spends [[compute unit]]s; switch to a CPU runtime to train for free.")

    if advice.over:
        budget = estimate.minutes - advice.minutes_over
        lines += ["", f"**Over budget:** about {estimate.minutes:.1f} minutes against your "
                      f"{budget:.0f}-minute limit. Cuts that would fit:", ""]
        for key, current, proposed, reason in advice.suggestions:
            verb = "lower" if _is_lower(proposed, current) else "raise"
            lines.append(f"- {verb} `{key}` from {current} to {proposed} -- {reason}")
        if getattr(modality, "name", "") == "image":
            lines += ["", "The other lever is re-ingesting the data at a smaller "
                          "`IMAGE_SIZE` from the *2. Data* cell: the [[image size]] is "
                          "fixed at ingest, not a config key."]
    return "\n".join(lines)


def _is_lower(proposed, current) -> bool:
    return (isinstance(proposed, int | float) and isinstance(current, int | float)
            and proposed < current)


# --- running the dry run ---
def _stderr_tail(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return "\n".join(lines[-STDERR_TAIL_LINES:])


def run_dry_run(project_dir, python=sys.executable, timeout: int = DRY_RUN_TIMEOUT,
                env: dict | None = None) -> DryRunResult:
    """`python train.py --dry-run` in the project folder, then read `dry_run.json`.

    Never raises: a non-zero exit, a timeout or a missing record all come back as a
    `DryRunResult` with `error` set to a one-line summary plus the last ~20 stderr lines.
    `env` is passed straight to `subprocess.run`, so a caller that wants a CPU-only dry
    run sets `CUDA_VISIBLE_DEVICES` itself.
    """
    project_dir = Path(project_dir)
    record = project_dir / DRY_RUN_FILE
    record.unlink(missing_ok=True)
    command = [str(python), "train.py", "--dry-run"]
    try:
        result = subprocess.run(command, cwd=str(project_dir), capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return DryRunResult(error=f"The dry run did not finish within {timeout} s.")
    except OSError as exc:
        return DryRunResult(error=f"The dry run could not start: {exc}")
    tail = _stderr_tail(result.stderr)
    if result.returncode != 0:
        return DryRunResult(
            error=f"The dry run failed (exit {result.returncode})."
                  + (f"\n\n```\n{tail}\n```" if tail else ""))
    if not record.exists():
        return DryRunResult(
            error=f"The dry run finished but wrote no {DRY_RUN_FILE}."
                  + (f"\n\n```\n{tail}\n```" if tail else ""))
    return read_dry_run(record)
