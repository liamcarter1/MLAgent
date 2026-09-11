# Milestone 6b: The Colab Cost Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before every training run in Colab — the train stage's first run and each tune round — the user sees how many minutes, compute units and how much money the run should cost against the `minutes_per_run` budget they gave at intake, and confirms, edits the config, or stops; afterwards the debrief compares the estimate with the actual time.

**Architecture:** A pure `mlagent/cost.py` holds the whole calculation — runtime detection through injectable probes, `rates.json` lookup, the `hours = seconds_per_epoch * epochs * 1.2 / 3600` formula, a budget check that names concrete config cuts, and a fixed-text renderer — with no stage, no LLM and no file I/O beyond reading `dry_run.json`. Both `train.py` templates gain a `--dry-run` flag that times a warm-up plus three real training batches and writes `dry_run.json`; `mlagent/stages/cost_gate.py::gate` runs that dry run as a subprocess inside `prepare` (or reads a finished run's archived `seconds_per_epoch` instead, once the project has history), renders the estimate, asks Run / Edit / Stop, and persists the answer and the estimate to `cost.json`. `TrainStage.prepare` and `TuneStage.prepare` call the gate just before building their `Handoff` and return `None` on a stop, which the orchestrator already treats as "no cells to wait for" once one line stops calling `debrief` after a `None` handoff.

**Tech Stack:** Python 3.10+, stdlib only for `cost.py` (`subprocess`, `dataclasses`, `pathlib`, `json`, `math`); numpy / pandas / scikit-learn / matplotlib / joblib / torch in the generated templates; pytest, ruff. Tests never hit the network and never touch a real GPU: `FakeLLM` for Claude, injected `probes` for `detect_runtime`, an injected `dry_runner` for the gate, and `CUDA_VISIBLE_DEVICES="-1"` for every torch subprocess.

**Spec:** docs/superpowers/specs/2026-09-11-milestone-6b-cost-gate-design.md

## Global Constraints

- Pipeline order is unchanged: `intake -> data -> clean -> codegen -> train -> tune -> report`. No new stages.
- `mlagent/colab.py`'s `HANDOFF_COMMANDS` (`mlagent/colab.py:34-42`) is unchanged: `dry_run.json` is never a `Handoff.outputs` entry and `--dry-run` is never a handoff command.
- Generated scripts (`mlagent/templates/**`) import only numpy / pandas / scikit-learn / matplotlib / joblib / torch (with `torchvision` imported lazily *inside* `model.py`'s ResNet builder) and never `mlagent`; IPython is imported only inside `try/except`.
- Every generated script keeps its `# --- settings ---` block containing `SCRIPT_NAME`, its `# --- Title ---` section markers that `codewalk.split_sections` splits (both train templates gain a `# --- Dry run ---` section), its `cli_argv()` that returns `[]` unless `Path(sys.argv[0]).name.lower() == SCRIPT_NAME.lower()`, and a `__main__` guard that never calls `sys.exit(0)`.
- A failed `--dry-run` prints its error to stderr and exits non-zero — the only permitted non-zero exit added by this milestone. `sys.exit(0)` stays forbidden everywhere.
- `--dry-run` never writes `metrics.json`, a checkpoint or a figure. It writes exactly one file, `dry_run.json`, and prints exactly one human-readable line to stdout.
- Rates live in `mlagent/rates.json` and are loaded the way `mlagent/prompts_io.py:7` loads prompts — `Path(__file__).parent / "rates.json"`, read directly, no `importlib.resources`.
- The rendered estimate is fixed f-string markdown built by `cost.render_estimate`; it is never an LLM call and never goes through `mlagent/teaching.py`. Every sentence uses `[[term]]` markup.
- Every subprocess test that runs an `image_torch` script sets `CUDA_VISIBLE_DEVICES="-1"` in the child environment (not `""`, which unsets the variable on Windows).
- `detect_runtime` in tests always uses injected `probes`, never hardware; the gate in tests always uses an injected `probes` sequence and (outside the two e2e tests) an injected `dry_runner`.
- Tests never hit the network: `FakeLLM` for Claude, `pretrained=none` for every ResNet test run.
- Tabular and image pipelines behave exactly as before apart from the gate and the two new `runs.jsonl` keys (`estimated_minutes`, `estimated_units`).
- The notebook stays at 21 cells: only the "4. Model" and "1. Project and interview" cell bodies change.
- `python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py tests/test_template_profile_images.py tests/test_template_evaluate_images.py` must stay clean.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere.
- `ruff check .` clean (line-length 100); `python -m pytest` green.
- Commit messages end with exactly:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c
  ```

## Rulings made while planning (deviations from the spec's letter)

1. **`detect_runtime(probes=None)` takes `(source, callable)` pairs, and a non-`None` `probes` replaces the built-ins rather than preceding them.** The spec (lines 71-73) describes "zero-argument callables ... tried in order before the two built-in probes". Two problems: (a) `RuntimeInfo.source` must stay one of `"torch" | "nvidia-smi" | "none"` (spec line 63), and a bare callable carries no source; (b) if injected probes only *precede* the built-ins, a test asserting the no-GPU path would fall through to the real `torch.cuda.is_available()` and fail on any dev machine with a GPU — breaking the Global Constraint that tests never touch hardware. So `probes: Sequence[tuple[str, Callable[[], str | None]]] | None`; `None` means `DEFAULT_PROBES`, and `()` means "no GPU, definitively".
2. **`render_estimate` takes two more parameters than the spec's one-line signature: `render_estimate(estimate, advice, spec_gpu, modality, basis_run_id=None)`.** The spec's own body requires both — line 202 keys the tabular-on-GPU line off `modality.name`, and line 196 says the basis line's run id is "threaded through from the caller in Section 3". The signature in the heading simply omits what the body demands.
3. **`gate` returns `str` (`"run"` / `"stop"`), not a `GateDecision` dataclass.** The spec's Section 3 heading says `-> str`; the estimate itself is persisted to `cost.json` under `last_estimate`, which is where `runs.log_finished_run` reads it, so a richer return type would have no consumer.
4. **An over-budget suggestion bullet reads "lower" or "raise" depending on direction.** The spec's line 205 template is `"lower {key} from {current} to {proposed} — {reason}"`, but the image `batch_size` suggestion (spec line 174) proposes a *larger* value. The verb is derived: `"lower"` when `proposed < current`, `"raise"` otherwise.
5. **`gate` is injectable through the stage constructor, mirroring `DataStage`.** `TrainStage(gate=cost_gate.gate)` and `TuneStage(gate=cost_gate.gate)` take the callable the same way `DataStage.__init__` takes `hf_load` (`mlagent/stages/data.py:71-80`), and `gate` itself takes `probes=None, dry_runner=cost.run_dry_run`. Without this, `tests/test_train_stage.py` and `tests/test_tune_stage.py` would shell out to a real dry run and read real hardware on every existing test.
6. **A failed dry run clears any stale `last_estimate` from `cost.json`.** The spec (lines 437-438) says the estimate keys stay `None` "when `cost.json` or `last_estimate` is absent", but an earlier successful run leaves one behind; without clearing it the debrief would compare this run against a previous run's estimate.
7. **A cost-gate stop in `TuneStage` leaves `config.json` and `pending` exactly as they were before this `prepare` call — it does not re-offer the identical proposal.** The gate runs *before* `config.json` is written and before `state["pending"]` is set (immediately after `chosen` is unpacked, ahead of the old plan's after-the-write placement), so a stop is a plain early return: nothing was mutated, so there is nothing to restore or clear. The spec (lines 356-359) says "`tune_state.json`'s `pending` key stays set so the next `prepare` call re-offers the same proposal rather than asking Claude again", which would require a resume-a-pending-proposal branch that `TuneStage.prepare` does not have today (`mlagent/stages/tune.py:233-313` re-diagnoses and re-proposes from scratch on every call), and adding one is a tuning-loop change out of this milestone's scope. The observable requirement that matters here — no `decision` is set, the stage stays incomplete, the next `orch.run()` diagnoses and proposes again, and the refused proposal is never left applied to `config.json` — holds exactly as written.
8. **The gate is asked at most four times per `prepare`: three that offer "Edit the config first", then one that offers only Run / Stop.** The spec (lines 317-319) says "at most 3 times; on the 3rd repeat the question drops the option", which is self-contradictory read strictly. `MAX_EDITS = 3` edits, then a final non-editable ask, matches the intent ("a user cannot loop forever") and is what the test asserts.
9. **After the first edit, the `train.cost_decision` questioner key is dropped (passed as `None`).** A Colab form answer is fixed for a whole cell run, so a form answer of "Edit the config first" would be returned forever. This is the same fix `TuneStage._choose` already applies to `tune.action` (`mlagent/stages/tune.py:387-390, 413`).
10. **`tabular_sklearn/train.py::train` loses its `dry_run` parameter.** Today `main` routes the dry run through `train(project_dir, dry_run=True)`, which returns a fake metrics dict (`mlagent/templates/tabular_sklearn/train.py:184-186, 300`). The new `dry_run.json` contract has nothing to do with a metrics dict, so `main` calls `dry_run_timing(project_dir)` directly and `train(project_dir) -> dict` only ever does a real run. Nothing outside the script calls `train`.
11. **`COST_FILE` and `DRY_RUN_FILE` are added to `mlagent/config.py`** alongside the other file-name constants, replacing the removed `DEFAULT_RATES` / `PRICE_PER_100_UNITS_USD`, so `runs.py` can read `cost.json` without importing `cost.py`.
12. **After a failed dry run, the gate asks Run / Stop without the Edit option.** `mlagent/stages/cost_gate.py::gate` calls `_ask(ctx, allow_edit=False, key=DECISION_KEY)` on the `dry.error` branch: there is no measured `seconds_per_epoch` to re-estimate an edited config against, so offering "Edit the config first" would have nothing to recompute from. The spec's instruction to "skip straight to the confirm" (Section 3) is implemented exactly this way.

## File Structure

| File | Responsibility |
|---|---|
| `mlagent/rates.json` (new) | Colab units/hour per GPU, the unknown-GPU fallback rate, the default price per unit and currency |
| `mlagent/cost.py` (new) | `RuntimeInfo`, probes, `detect_runtime`, `load_rates`, `rate_for`, `DryRunResult`, `read_dry_run`, `run_dry_run`, `Estimate`, `estimate_run` / `estimate_from_dry_run` / `estimate_from_history`, `BudgetAdvice`, `budget_check`, `render_estimate` |
| `mlagent/stages/cost_gate.py` (new) | `gate(ctx, *, rounds_remaining, probes=None, dry_runner=...)` — price/currency, basis choice, budget, display, Run/Edit/Stop, `cost.json` persistence |
| `mlagent/config.py` | `DEFAULT_RATES` / `PRICE_PER_100_UNITS_USD` removed; `COST_FILE`, `DRY_RUN_FILE` added |
| `mlagent/templates/tabular_sklearn/train.py` | `# --- Dry run ---` section: warm-up + 3 timed `fit_epoch` chunks, writes `dry_run.json` |
| `mlagent/templates/image_torch/train.py` | `# --- Dry run ---` section and a `--dry-run` argparse flag: warm-up + 3 timed batches, writes `dry_run.json` |
| `mlagent/orchestrator.py` | One-line change: a `None` handoff skips the `debrief` call |
| `mlagent/runs.py` | `build_run_entry` gains `estimated_minutes` / `estimated_units`; `log_finished_run` reads them from `cost.json`; new `estimate_vs_actual(entry)` |
| `mlagent/stages/train.py` | `__init__(gate=...)`, `prepare -> Handoff \| None` calling the gate, the 6a placeholder sentence removed, estimate-vs-actual in the debrief |
| `mlagent/stages/tune.py` | `__init__(gate=...)`, the gate before the round's `Handoff`, estimate-vs-actual in the debrief |
| `scripts/build_notebook.py`, `notebooks/ML_Training_Agent.ipynb` | "4. Model" gains `PRICE_PER_UNIT` / `CURRENCY` with hints and answer keys; the `MINUTES_PER_RUN` hint mentions the gate; still 21 cells |
| `docs/colab-smoke.md`, `CLAUDE.md` | Milestone 6b smoke checklist (seven items); architecture, module and test-command notes |
| `tests/test_cost.py` (new) | Every pure function in `cost.py` |
| `tests/test_template_dry_run.py` (new) | Both templates run for real with `--dry-run` as CPU subprocesses |
| `tests/test_cost_gate.py` (new) | The gate's paths with injected probes and dry runner |
| `tests/test_config.py`, `tests/test_orchestrator.py`, `tests/test_runs.py`, `tests/test_train_stage.py`, `tests/test_tune_stage.py`, `tests/test_template_train.py`, `tests/test_pipeline_e2e.py`, `tests/test_pipeline_e2e_images.py`, `tests/test_colab.py`, `tests/test_report_stage.py` | Modified |

---

### Task 1: `rates.json`, runtime detection and the dry-run record

**Files:**
- Create: `mlagent/rates.json`, `mlagent/cost.py`, `tests/test_cost.py`
- Modify: `mlagent/config.py:24-27`, `tests/test_config.py:1-10`
- Test: `tests/test_cost.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from this milestone. Mirrors `mlagent/prompts_io.py:7` (`PROMPTS_DIR = Path(__file__).parent / "prompts"`) for locating a packaged data file.
- Produces (used by Tasks 2, 5, 6):
  - `cost.RATES_PATH: Path`, `cost.load_rates() -> dict`
  - `@dataclass(frozen=True) cost.RuntimeInfo` with fields `device: str`, `gpu_name: str | None`, `gpu_type: str | None`, `source: str`
  - `cost.torch_gpu_name() -> str | None`, `cost.nvidia_smi_gpu_name(runner=subprocess.run) -> str | None`
  - `cost.DEFAULT_PROBES: tuple[tuple[str, Callable[[], str | None]], ...]`
  - `cost.gpu_type_for(gpu_name: str | None, rates: dict) -> str | None`
  - `cost.detect_runtime(probes=None) -> RuntimeInfo`
  - `cost.rate_for(runtime: RuntimeInfo, rates: dict) -> tuple[float, str | None]`
  - `@dataclass(frozen=True) cost.DryRunResult` with fields `device: str`, `gpu_name: str | None`, `batches_per_epoch: int`, `seconds_per_batch: float`, `seconds_per_epoch: float`, `n_train: int`, `script: str`, `error: str | None`
  - `cost.read_dry_run(path: Path) -> DryRunResult`
  - `config.COST_FILE = "cost.json"`, `config.DRY_RUN_FILE = "dry_run.json"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cost.py`:

```python
"""The pure cost module: rates, runtime detection and the dry-run record."""

from __future__ import annotations

import json
import subprocess

import pytest

from mlagent import config as cfg
from mlagent import cost


def test_rates_json_holds_the_three_known_gpus_and_the_colab_pro_price():
    rates = cost.load_rates()
    assert rates["units_per_hour"] == {"T4": 2.0, "L4": 4.8, "A100": 13.0}
    assert rates["unknown_gpu_rate"] == 2.0
    assert rates["default_price_per_unit"] == pytest.approx(0.0999)
    assert rates["default_currency"] == "$"


def test_the_rates_file_sits_next_to_the_package_like_the_prompts_do():
    assert cost.RATES_PATH.name == "rates.json"
    assert cost.RATES_PATH.parent.name == "mlagent"
    assert json.loads(cost.RATES_PATH.read_text(encoding="utf-8")) == cost.load_rates()


def test_config_no_longer_carries_the_rates():
    assert not hasattr(cfg, "DEFAULT_RATES")
    assert not hasattr(cfg, "PRICE_PER_100_UNITS_USD")
    assert cfg.COST_FILE == "cost.json"
    assert cfg.DRY_RUN_FILE == "dry_run.json"


def test_a_torch_probe_that_finds_a_t4_gives_a_cuda_runtime():
    runtime = cost.detect_runtime([("torch", lambda: "Tesla T4")])
    assert runtime == cost.RuntimeInfo(device="cuda", gpu_name="Tesla T4",
                                       gpu_type="T4", source="torch")


def test_the_nvidia_smi_probe_is_tried_when_the_torch_probe_finds_nothing():
    runtime = cost.detect_runtime([("torch", lambda: None),
                                   ("nvidia-smi", lambda: "NVIDIA L4")])
    assert runtime.device == "cuda" and runtime.gpu_type == "L4"
    assert runtime.source == "nvidia-smi" and runtime.gpu_name == "NVIDIA L4"


def test_no_probe_finding_anything_is_a_cpu_runtime():
    runtime = cost.detect_runtime(())
    assert runtime == cost.RuntimeInfo(device="cpu", gpu_name=None, gpu_type=None,
                                       source="none")


def test_an_unrecognised_gpu_name_is_typed_unknown():
    runtime = cost.detect_runtime([("torch", lambda: "NVIDIA H200")])
    assert runtime.device == "cuda" and runtime.gpu_type == "unknown"


def test_gpu_types_match_case_insensitively_as_a_substring():
    rates = cost.load_rates()
    assert cost.gpu_type_for("NVIDIA A100-SXM4-40GB", rates) == "A100"
    assert cost.gpu_type_for("nvidia a100", rates) == "A100"
    assert cost.gpu_type_for(None, rates) is None
    assert cost.gpu_type_for("Radeon Pro", rates) == "unknown"


def test_a_probe_that_raises_is_skipped_not_fatal():
    def boom() -> str:
        raise RuntimeError("no driver")

    runtime = cost.detect_runtime([("torch", boom), ("nvidia-smi", lambda: "Tesla T4")])
    assert runtime.gpu_type == "T4" and runtime.source == "nvidia-smi"


def test_the_nvidia_smi_probe_parses_the_first_line_of_a_fake_subprocess():
    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(command, 0, stdout="Tesla T4\nTesla T4\n",
                                           stderr="")

    assert cost.nvidia_smi_gpu_name(runner) == "Tesla T4"


def test_the_nvidia_smi_probe_returns_none_when_the_binary_is_absent():
    def runner(command, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert cost.nvidia_smi_gpu_name(runner) is None


def test_the_nvidia_smi_probe_returns_none_on_a_non_zero_exit():
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom")

    assert cost.nvidia_smi_gpu_name(runner) is None


def test_rate_for_a_cpu_runtime_is_zero_with_no_note():
    rates = cost.load_rates()
    units, note = cost.rate_for(cost.detect_runtime(()), rates)
    assert units == 0.0 and note is None


def test_rate_for_a_known_gpu_is_the_table_value_with_no_note():
    rates = cost.load_rates()
    units, note = cost.rate_for(cost.detect_runtime([("torch", lambda: "Tesla T4")]), rates)
    assert units == 2.0 and note is None


def test_rate_for_an_unknown_gpu_falls_back_and_says_so():
    rates = cost.load_rates()
    runtime = cost.detect_runtime([("torch", lambda: "NVIDIA H200")])
    units, note = cost.rate_for(runtime, rates)
    assert units == 2.0
    assert note is not None and "NVIDIA H200" in note and "T4" in note


def test_read_dry_run_parses_every_key(tmp_path):
    (tmp_path / "dry_run.json").write_text(json.dumps({
        "device": "cuda", "gpu_name": "Tesla T4", "batches_per_epoch": 38,
        "seconds_per_batch": 0.14, "seconds_per_epoch": 5.32, "n_train": 1200,
        "script": "train.py",
    }), encoding="utf-8")
    dry = cost.read_dry_run(tmp_path / "dry_run.json")
    assert dry == cost.DryRunResult(device="cuda", gpu_name="Tesla T4", batches_per_epoch=38,
                                    seconds_per_batch=0.14, seconds_per_epoch=5.32,
                                    n_train=1200, script="train.py", error=None)


def test_read_dry_run_of_a_missing_file_is_zeroed_with_an_error(tmp_path):
    dry = cost.read_dry_run(tmp_path / "nope.json")
    assert dry.seconds_per_epoch == 0.0 and dry.batches_per_epoch == 0
    assert dry.error is not None and "nope.json" in dry.error


def test_read_dry_run_of_broken_json_is_zeroed_with_an_error(tmp_path):
    path = tmp_path / "dry_run.json"
    path.write_text("{not json", encoding="utf-8")
    dry = cost.read_dry_run(path)
    assert dry.seconds_per_epoch == 0.0
    assert dry.error is not None and "dry_run.json" in dry.error
```

`tests/test_config.py` is 9 lines today, in one test named `test_defaults_are_sane`; only its
last two assertions test the constants this task removes. Keep the file's name and its first
three assertions unchanged, drop the `DEFAULT_RATES` / `PRICE_PER_100_UNITS_USD` assertions,
and add the two new ones:

```python
from mlagent import config


def test_defaults_are_sane():
    assert config.MODEL_ID.startswith("claude-")
    assert config.MAX_TOKENS >= 4096
    assert config.EFFORT in {"low", "medium", "high", "xhigh", "max"}
    assert config.COST_FILE == "cost.json"
    assert config.DRY_RUN_FILE == "dry_run.json"
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_cost.py tests/test_config.py -q
```
Expected: collection of `tests/test_cost.py` aborts with `ModuleNotFoundError: No module named 'mlagent.cost'`, and `tests/test_config.py::test_defaults_are_sane` fails with `AttributeError: module 'mlagent.config' has no attribute 'COST_FILE'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/rates.json`:

```json
{
  "units_per_hour": {"T4": 2.0, "L4": 4.8, "A100": 13.0},
  "unknown_gpu_rate": 2.0,
  "default_price_per_unit": 0.0999,
  "default_currency": "$"
}
```

In `mlagent/config.py`, replace lines 24-27 (the `DEFAULT_RATES` comment block and both constants) with:

```python
COST_FILE = "cost.json"
DRY_RUN_FILE = "dry_run.json"
```

Create `mlagent/cost.py`:

```python
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
```

Task 1 does not import `sys`: nothing above uses it. Task 2 adds `import sys` when it adds
`run_dry_run(project_dir, python=sys.executable, ...)` to the same module.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_cost.py tests/test_config.py -q
ruff check .
```
Expected: `19 passed` (18 in `tests/test_cost.py`, 1 in `tests/test_config.py`), ruff clean.

Then confirm nothing else referenced the removed constants:

```
python -m pytest -q
```
Expected: the whole suite still passes (no module outside `tests/test_config.py` referenced `DEFAULT_RATES` or `PRICE_PER_100_UNITS_USD`).

- [ ] **Step 5: Commit**

```bash
git add mlagent/rates.json mlagent/cost.py mlagent/config.py tests/test_cost.py tests/test_config.py
git commit -m "feat: rates.json, runtime detection and the dry-run record

Colab's per-GPU compute-unit rates and the Colab Pro price move out of
config.py into a packaged rates.json, loaded the way prompts are. cost.py
gains RuntimeInfo, injectable GPU probes, detect_runtime, rate_for and the
dry_run.json reader.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 2: The estimate, the budget check, the renderer and the dry-run runner

**Files:**
- Modify: `mlagent/cost.py` (append after `read_dry_run`)
- Test: `tests/test_cost.py` (append)

**Interfaces:**
- Consumes (Task 1): `cost.RuntimeInfo`, `cost.DryRunResult`, `cost.read_dry_run(path) -> DryRunResult`, `cost.rate_for(runtime, rates) -> tuple[float, str | None]`, `cost.load_rates() -> dict`, `cost.SAFETY_FACTOR = 1.2`, `cost.DRY_RUN_FILE = "dry_run.json"`, `cost.DRY_RUN_TIMEOUT = 600`, `cost.STDERR_TAIL_LINES = 20`. Existing repo symbols: `templates_io.load_schema(name) -> dict` (`mlagent/templates_io.py:36`), `templates_io.schema_for(schema, model_type) -> dict` (`mlagent/templates_io.py:44`), `modality_for(task_type) -> Modality` with a `.name` of `"tabular"` or `"image"` (`mlagent/modality.py:110`).
- Produces (used by Tasks 5, 6, 7):
  - `@dataclass(frozen=True) cost.Estimate` with fields `seconds_per_epoch: float`, `epochs: int`, `minutes: float`, `units: float`, `cost: float`, `price_per_unit: float`, `currency: str`, `rate_units_per_hour: float`, `rounds_remaining: int`, `minutes_all_rounds: float`, `units_all_rounds: float`, `cost_all_rounds: float`, `basis: str`, `runtime: RuntimeInfo`, `note: str | None = None`, and a method `to_dict() -> dict`
  - `cost.estimate_run(seconds_per_epoch: float, epochs: int, runtime: RuntimeInfo, rates: dict, price_per_unit: float, currency: str, rounds_remaining: int, basis: str) -> Estimate`
  - `cost.estimate_from_dry_run(dry: DryRunResult, epochs: int, runtime: RuntimeInfo, rates: dict, price_per_unit: float, currency: str, rounds_remaining: int) -> Estimate`
  - `cost.estimate_from_history(seconds_per_epoch: float, epochs: int, runtime: RuntimeInfo, rates: dict, price_per_unit: float, currency: str, rounds_remaining: int) -> Estimate`
  - `@dataclass(frozen=True) cost.BudgetAdvice` with fields `over: bool`, `minutes_over: float`, `suggestions: list[tuple[str, object, object, str]]`
  - `cost.budget_check(estimate: Estimate, minutes_per_run: float, config: dict, schema: dict, modality) -> BudgetAdvice`
  - `cost.money(value: float, currency: str) -> str`
  - `cost.render_estimate(estimate: Estimate, advice: BudgetAdvice, spec_gpu: str, modality, basis_run_id: int | None = None) -> str`
  - `cost.run_dry_run(project_dir, python=sys.executable, timeout=DRY_RUN_TIMEOUT, env=None) -> DryRunResult`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cost.py`:

```python
import sys  # noqa: E402

from mlagent.modality import modality_for  # noqa: E402
from mlagent.templates_io import load_schema, schema_for  # noqa: E402

TABULAR = modality_for("tabular_classification")
IMAGE = modality_for("image_classification")
GB_SCHEMA = schema_for(load_schema("tabular_sklearn"), "gradient_boosting")
CNN_SCHEMA = schema_for(load_schema("image_torch"), "small_cnn")

T4 = cost.RuntimeInfo(device="cuda", gpu_name="Tesla T4", gpu_type="T4", source="torch")
CPU = cost.RuntimeInfo(device="cpu", gpu_name=None, gpu_type=None, source="none")
H200 = cost.RuntimeInfo(device="cuda", gpu_name="NVIDIA H200", gpu_type="unknown",
                        source="torch")


def make_estimate(seconds_per_epoch=6.0, epochs=10, runtime=T4, rounds_remaining=3,
                  basis="dry_run", price=0.0999, currency="$"):
    return cost.estimate_run(seconds_per_epoch, epochs, runtime, cost.load_rates(),
                             price, currency, rounds_remaining, basis)


def test_the_formula_matches_the_2026_09_06_spec_by_hand():
    # hours = 6 * 10 * 1.2 / 3600 = 0.02; units = 0.02 * 2.0 = 0.04;
    # cost = 0.04 * 0.0999 = 0.003996
    est = make_estimate()
    assert est.minutes == pytest.approx(1.2)
    assert est.units == pytest.approx(0.04)
    assert est.cost == pytest.approx(0.004, abs=5e-4)
    assert est.rate_units_per_hour == 2.0
    assert est.basis == "dry_run" and est.runtime is T4
    assert est.seconds_per_epoch == 6.0 and est.epochs == 10


def test_all_rounds_is_this_run_times_one_plus_the_remaining_rounds():
    est = make_estimate(rounds_remaining=3)
    assert est.minutes_all_rounds == pytest.approx(4.8)
    assert est.units_all_rounds == pytest.approx(0.16)
    assert est.cost_all_rounds == pytest.approx(est.cost * 4, rel=1e-3)


def test_no_remaining_rounds_makes_the_totals_equal_this_run():
    est = make_estimate(rounds_remaining=0)
    assert est.minutes_all_rounds == est.minutes
    assert est.units_all_rounds == est.units


def test_a_cpu_run_spends_no_units_and_no_money():
    est = make_estimate(runtime=CPU)
    assert est.rate_units_per_hour == 0.0
    assert est.units == 0.0 and est.cost == 0.0
    assert est.minutes == pytest.approx(1.2)


def test_an_unknown_gpu_carries_the_note_onto_the_estimate():
    est = make_estimate(runtime=H200)
    assert est.rate_units_per_hour == 2.0
    assert est.note is not None and "NVIDIA H200" in est.note


def test_estimate_from_dry_run_uses_the_measured_seconds_per_epoch():
    dry = cost.DryRunResult(device="cuda", gpu_name="Tesla T4", batches_per_epoch=38,
                            seconds_per_batch=0.2, seconds_per_epoch=7.6, n_train=1200)
    est = cost.estimate_from_dry_run(dry, 5, T4, cost.load_rates(), 0.0999, "$", 0)
    assert est.basis == "dry_run" and est.seconds_per_epoch == 7.6
    assert est.minutes == pytest.approx(7.6 * 5 * 1.2 / 60)


def test_estimate_from_history_is_the_same_arithmetic_with_a_history_basis():
    est = cost.estimate_from_history(7.6, 5, T4, cost.load_rates(), 0.0999, "$", 0)
    same = cost.estimate_from_dry_run(
        cost.DryRunResult(seconds_per_epoch=7.6), 5, T4, cost.load_rates(), 0.0999, "$", 0)
    assert est.basis == "history" and same.basis == "dry_run"
    assert est.minutes == same.minutes and est.units == same.units


def test_estimate_to_dict_is_json_ready_and_flattens_the_runtime():
    payload = make_estimate().to_dict()
    assert json.loads(json.dumps(payload))["basis"] == "dry_run"
    assert payload["device"] == "cuda" and payload["gpu_type"] == "T4"
    assert payload["minutes"] == pytest.approx(1.2)
    assert payload["units"] == pytest.approx(0.04)


def test_under_budget_means_no_advice_and_no_suggestions():
    est = make_estimate(seconds_per_epoch=6.0, epochs=10)      # 1.2 minutes
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    assert advice.over is False and advice.suggestions == []
    assert advice.minutes_over == 0.0


def test_over_budget_names_the_largest_epochs_that_fits():
    # 60 s/epoch * 20 epochs * 1.2 = 1440 s = 24 minutes against a 5 minute budget.
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    config = {"epochs": 20, "model_type": "gradient_boosting"}
    advice = cost.budget_check(est, 5, config, GB_SCHEMA, TABULAR)
    assert advice.over is True
    assert advice.minutes_over == pytest.approx(19.0)
    assert advice.suggestions[0][0] == "epochs"
    assert advice.suggestions[0][1] == 20 and advice.suggestions[0][2] == 4
    # The proposal really does fit: 60 * 4 * 1.2 / 60 = 4.8 minutes.
    assert 60.0 * 4 * cost.SAFETY_FACTOR / 60 <= 5


def test_the_epochs_proposal_never_drops_below_one():
    est = make_estimate(seconds_per_epoch=600.0, epochs=2, runtime=CPU)   # 24 minutes
    advice = cost.budget_check(est, 1, {"epochs": 2, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    assert advice.suggestions[0][2] == 1


def test_an_image_run_over_budget_also_offers_a_bigger_batch_size():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    config = {"epochs": 20, "batch_size": 32, "model_type": "small_cnn"}
    advice = cost.budget_check(est, 5, config, CNN_SCHEMA, IMAGE)
    keys = [s[0] for s in advice.suggestions]
    assert keys == ["epochs", "batch_size"]
    batch = next(s for s in advice.suggestions if s[0] == "batch_size")
    assert batch[1] == 32 and batch[2] == 64
    assert "image_size" not in keys       # fixed at ingest, never a config key


def test_a_batch_size_already_at_the_schema_maximum_is_not_suggested():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    config = {"epochs": 20, "batch_size": 256, "model_type": "small_cnn"}
    advice = cost.budget_check(est, 5, config, CNN_SCHEMA, IMAGE)
    assert [s[0] for s in advice.suggestions] == ["epochs"]


def test_a_suggestion_is_never_made_for_a_key_the_schema_does_not_have():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    stripped = {k: v for k, v in GB_SCHEMA.items() if k != "epochs"}
    advice = cost.budget_check(est, 5, {"epochs": 20, "model_type": "gradient_boosting"},
                               stripped, TABULAR)
    assert advice.over is True and advice.suggestions == []


def test_money_puts_a_one_character_symbol_in_front_and_a_code_behind():
    assert cost.money(1.5, "$") == "$1.50"
    assert cost.money(1.5, "GBP") == "1.50 GBP"


def test_render_names_the_gpu_the_table_and_the_dry_run_basis():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "T4", TABULAR)
    assert "This runtime has a Tesla T4 [[GPU]]." in text
    assert "| minutes | [[compute units]] | cost |" in text
    assert "| this run | 1.2 |" in text
    assert "this run plus 3 remaining rounds" in text
    assert "Timed with a 3-batch dry run." in text
    assert "Change runtime type" not in text       # spec asked for T4 and got one


def test_render_names_the_source_run_for_a_history_estimate():
    est = make_estimate(runtime=CPU, basis="history")
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR, basis_run_id=2)
    assert "From run 2's measured time." in text
    assert "dry run" not in text


def test_render_says_a_cpu_run_is_free():
    est = make_estimate(runtime=CPU)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR)
    assert "This runtime has no [[GPU]]." in text
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text


def test_render_warns_when_intake_asked_for_a_gpu_and_there_is_none():
    est = make_estimate(runtime=CPU)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "T4", TABULAR)
    assert "Runtime -> Change runtime type" in text
    assert "asked for a [[GPU]] at intake" in text


def test_render_warns_when_intake_asked_for_no_gpu_and_there_is_one():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "small_cnn"},
                               CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "none", IMAGE)
    assert "asked for no [[GPU]] at intake" in text


def test_render_tells_a_tabular_run_on_a_gpu_it_is_paying_for_nothing():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "any", TABULAR)
    assert "switch to a CPU runtime to train for free" in text


def test_render_lists_the_cuts_when_over_budget():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    advice = cost.budget_check(est, 5, {"epochs": 20, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR)
    assert "**Over budget:**" in text
    assert "5-minute limit" in text
    assert "- lower `epochs` from 20 to 4 --" in text
    assert "IMAGE_SIZE" not in text


def test_render_adds_the_reingest_sentence_for_an_over_budget_image_run():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    advice = cost.budget_check(est, 5, {"epochs": 20, "batch_size": 32,
                                        "model_type": "small_cnn"}, CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "T4", IMAGE)
    assert "- raise `batch_size` from 32 to 64 --" in text
    assert "re-ingesting the data at a smaller `IMAGE_SIZE`" in text


def test_render_shows_the_unknown_gpu_note():
    est = make_estimate(runtime=H200)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "small_cnn"},
                               CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "any", IMAGE)
    assert "not one of the known [[GPU]] types" in text


def write_fake_train(root, body: str) -> None:
    (root / "train.py").write_text(body, encoding="utf-8")


def test_run_dry_run_reads_back_what_the_script_wrote(tmp_path):
    write_fake_train(tmp_path, (
        "import json, pathlib, sys\n"
        "assert sys.argv[1] == '--dry-run'\n"
        "pathlib.Path('dry_run.json').write_text(json.dumps({\n"
        "    'device': 'cpu', 'gpu_name': None, 'batches_per_epoch': 4,\n"
        "    'seconds_per_batch': 0.25, 'seconds_per_epoch': 1.0, 'n_train': 80,\n"
        "    'script': 'train.py'}), encoding='utf-8')\n"
    ))
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is None
    assert dry.batches_per_epoch == 4 and dry.seconds_per_epoch == 1.0


def test_run_dry_run_reports_a_non_zero_exit_with_the_stderr_tail(tmp_path):
    write_fake_train(tmp_path, (
        "import sys\n"
        "print('ValueError: the training split is empty', file=sys.stderr)\n"
        "sys.exit(3)\n"
    ))
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None
    assert "exit 3" in dry.error
    assert "the training split is empty" in dry.error
    assert dry.seconds_per_epoch == 0.0


def test_run_dry_run_reports_a_script_that_wrote_nothing(tmp_path):
    write_fake_train(tmp_path, "print('nothing to see')\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None and "dry_run.json" in dry.error


def test_run_dry_run_reports_a_timeout_instead_of_raising(tmp_path):
    write_fake_train(tmp_path, "import time\ntime.sleep(30)\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=1)
    assert dry.error is not None and "did not finish" in dry.error


def test_run_dry_run_deletes_a_stale_record_before_running(tmp_path):
    (tmp_path / "dry_run.json").write_text('{"seconds_per_epoch": 99.0}', encoding="utf-8")
    write_fake_train(tmp_path, "import sys\nsys.exit(1)\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None
    assert not (tmp_path / "dry_run.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_cost.py -q
```
Expected: `29 failed, 18 passed`. Collection succeeds — the appended module-level constants (`T4`, `CPU`, `H200`, the two schemas) only use symbols Task 1 already defined — and each new test fails with `AttributeError: module 'mlagent.cost' has no attribute 'estimate_run'` (or `estimate_from_dry_run`, `estimate_from_history`, `budget_check`, `money`, `render_estimate`, `run_dry_run`). Task 1's 18 tests still pass.

- [ ] **Step 3: Write minimal implementation**

Add `import math` to `mlagent/cost.py`'s imports (alphabetically after `json`, before `subprocess`)
and `import sys` (alphabetically after `subprocess`, at the end of the stdlib imports) — this is
where `run_dry_run` first uses `sys.executable` as its default `python` argument — then append to
the end of the module:

```python
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
                                text=True, encoding="utf-8", timeout=timeout, env=env)
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
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_cost.py -q
ruff check .
```
Expected: `47 passed` (18 from Task 1 plus 29 added here), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/cost.py tests/test_cost.py
git commit -m "feat: the cost estimate, the budget check, the renderer and the dry-run runner

estimate_run implements the design spec's hours = seconds_per_epoch * epochs
* 1.2 / 3600 directly; budget_check names the largest epochs (and, for
images, the larger batch_size) that fits minutes_per_run; render_estimate
builds the whole message as fixed [[term]]-marked markdown with no LLM call;
run_dry_run never raises.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 3: `--dry-run` in `tabular_sklearn/train.py`

**Files:**
- Create: `tests/test_template_dry_run.py`
- Modify: `mlagent/templates/tabular_sklearn/train.py:1-8` (docstring), `:39-44` (settings), `:164-186` (the `# --- training ---` header and `dry_run_timing`/`train`), `:289-312` (`main`); `tests/test_template_train.py:86-116` (delete the two old dry-run tests)
- Test: `tests/test_template_dry_run.py`, `tests/test_template_train.py`

**Interfaces:**
- Consumes: `cost.read_dry_run` (Task 1) reads exactly the keys this task writes. Existing template symbols: `read_json` (`mlagent/templates/tabular_sklearn/train.py:59`), `write_json` (`:65`), `load_data` (imported at `:26`), `build_model` (`:37`), `SCRIPT_NAME = "train.py"` (`:40`).
- Produces (used by Tasks 6, 7, 8): `dry_run.json` in the project folder with keys `device` (always `"cpu"`), `gpu_name` (always `null`), `batches_per_epoch` (always `1`), `seconds_per_batch` (float), `seconds_per_epoch` (float, `= batches_per_epoch * seconds_per_batch`), `n_train` (int), `script` (`"train.py"`); exit code `0`; one stdout line; `train(project_dir) -> dict` (no `dry_run` parameter, ruling 10).

- [ ] **Step 1: Write the failing test**

Create `tests/test_template_dry_run.py` (the image half arrives in Task 4):

```python
"""`python train.py --dry-run` in both template families, run for real as a subprocess.

Nothing here touches a GPU: CUDA_VISIBLE_DEVICES="-1" (not "", which unsets the variable
on Windows) is in every child environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TABULAR_TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
IMAGE_TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
CPU_ONLY = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
DRY_RUN_KEYS = {"device", "gpu_name", "batches_per_epoch", "seconds_per_batch",
                "seconds_per_epoch", "n_train", "script"}

GB = {"model_type": "gradient_boosting", "epochs": 4, "iters_per_epoch": 3,
      "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
      "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
      "l2_regularization": 0.0}
RF = {"model_type": "random_forest", "epochs": 2, "trees_per_epoch": 5, "max_depth": 4,
      "min_samples_leaf": 1, "max_features": 0.8, "seed": 1, "early_stopping_patience": 0}
LINEAR = {"model_type": "linear", "epochs": 3, "learning_rate": 0.05, "alpha": 0.0001,
          "seed": 1, "early_stopping_patience": 0}


def install(project, template: Path, config: dict) -> Path:
    for name in CODE_FILES:
        shutil.copy(template / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "train.py", *args], cwd=str(root), capture_output=True, text=True,
        encoding="utf-8", timeout=600, env=CPU_ONLY,
    )


def record(root: Path) -> dict:
    return json.loads((root / "dry_run.json").read_text(encoding="utf-8"))


def assert_wrote_nothing_else(root: Path) -> None:
    assert not (root / "metrics.json").exists()
    assert not (root / "checkpoints" / "best.joblib").exists()
    assert not (root / "checkpoints" / "best.pt").exists()
    assert not (root / "plots" / "training_curves.png").exists()


@pytest.mark.parametrize("config", [GB, RF, LINEAR], ids=["gb", "rf", "linear"])
def test_the_tabular_dry_run_writes_the_whole_contract(clean_project, config):
    root = install(clean_project, TABULAR_TEMPLATE, config)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = record(root)
    assert set(data) == DRY_RUN_KEYS
    assert data["device"] == "cpu" and data["gpu_name"] is None
    assert data["batches_per_epoch"] == 1
    assert data["seconds_per_batch"] >= 0          # a 4dp round can legitimately be 0.0
    assert data["seconds_per_epoch"] == pytest.approx(
        data["seconds_per_batch"] * data["batches_per_epoch"])
    assert data["n_train"] > 0
    assert data["script"] == "train.py"
    assert_wrote_nothing_else(root)


def test_the_tabular_dry_run_prints_a_human_line(clean_project):
    root = install(clean_project, TABULAR_TEMPLATE, GB)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    # Asserting on the last non-blank line, not that it is the only line: a stricter
    # exactly-one-line assertion is brittle against any incidental print (warnings,
    # library banners) that lands on stdout ahead of it.
    assert lines[-1].startswith("Dry run: 3 batches in ")
    assert "s per batch" in lines[-1] and "batches per epoch" in lines[-1]


def test_a_broken_tabular_dry_run_exits_non_zero_with_the_error_on_stderr(clean_project):
    root = install(clean_project, TABULAR_TEMPLATE, GB)
    (root / "data" / "clean" / "data.csv").unlink()
    proc = run(root, "--dry-run")
    assert proc.returncode != 0
    assert "error:" in proc.stderr.lower() or "Error" in proc.stderr
    assert not (root / "dry_run.json").exists()
    assert_wrote_nothing_else(root)


def test_the_tabular_dry_run_section_is_in_the_walkthrough():
    from mlagent.codewalk import split_sections

    source = (TABULAR_TEMPLATE / "train.py").read_text(encoding="utf-8")
    titles = [title for title, _body in split_sections(source)]
    assert "Dry run" in titles
```

Delete `tests/test_template_train.py:86-113` — `test_dry_run_prints_timing_and_writes_nothing` (86-93) and the parametrized `test_dry_run_works_for_every_family` (96-113), both of which assert the old stdout-JSON contract. (Line 116 begins the next test, `test_linear_run_config_has_no_iters_per_epoch_key`, and must stay.) Their replacement is `tests/test_template_dry_run.py`'s parametrized test above, which covers the same three families.

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_template_dry_run.py -q
```
Expected: `5 failed` — the three parametrized cases and `test_the_tabular_dry_run_prints_a_human_line` fail at `record(root)` / the stdout check with `FileNotFoundError: ... dry_run.json` (today's `dry_run_timing` prints JSON and writes nothing), and `test_the_tabular_dry_run_section_is_in_the_walkthrough` fails with `AssertionError: assert 'Dry run' in ['settings', 'captions', 'small helpers', ...]`. `test_a_broken_tabular_dry_run_exits_non_zero_with_the_error_on_stderr` already passes.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/templates/tabular_sklearn/train.py`, replace line 5 of the docstring:

```python
  python train.py --dry-run      times three training chunks, writes dry_run.json only
```

Add to the `# --- settings ---` block, after `CURVES_FIGURE` (line 44):

```python
DRY_RUN_FILE = "dry_run.json"
DRY_RUN_BATCHES = 3
```

Replace lines 164-188 (the `# --- training ---` marker, `dry_run_timing`, `train`'s
`if dry_run: return dry_run_timing(project_dir)` and its blank line, and the
`config = read_json(...)` line that follows) with a `# --- Dry run ---` section followed by
the unchanged `# --- training ---` section — the replacement block below ends with that same
`config = read_json(...)` line so `train`'s body is not left duplicated or missing it:

```python
# --- Dry run ---
def dry_run_timing(project_dir: Path) -> dict:
    """Time three real training chunks so the agent can estimate the whole run.

    One `fit_epoch` call is this family's "batch": gradient boosting adds
    `iters_per_epoch` trees, the forest adds `trees_per_epoch`, the linear model takes one
    gradient pass. The first call is a warm-up and is not timed, so one-off costs
    (sklearn's first-call imports, the data cache warming) stay out of the number.
    Writes `dry_run.json` and nothing else: no metrics.json, no checkpoint, no figure.
    """
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    model = build_model(config, data["task_type"], data["categorical_mask"])
    n_train = int(len(data["X_train"]))
    model.fit_epoch(data["X_train"], data["y_train"])          # warm-up, not timed
    started = time.time()
    for _ in range(DRY_RUN_BATCHES):
        model.fit_epoch(data["X_train"], data["y_train"])
    elapsed = time.time() - started
    seconds_per_batch = round(elapsed / DRY_RUN_BATCHES, 4)
    batches_per_epoch = 1
    record = {
        "device": "cpu",
        "gpu_name": None,
        "batches_per_epoch": batches_per_epoch,
        "seconds_per_batch": seconds_per_batch,
        "seconds_per_epoch": round(seconds_per_batch * batches_per_epoch, 4),
        "n_train": n_train,
        "script": SCRIPT_NAME,
    }
    write_json(project_dir / DRY_RUN_FILE, record)
    print(
        f"Dry run: {DRY_RUN_BATCHES} batches in {elapsed:.2f} s on cpu; about "
        f"{seconds_per_batch:.3f} s per batch, {batches_per_epoch} batches per epoch",
        flush=True,
    )
    return record


# --- training ---
def train(project_dir: Path) -> dict:
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
```

(the rest of `train`, from `data = load_data(project_dir, config)` at line 189 onward, is
unchanged).

Replace `main` (lines 289-312) with:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the model and record every epoch.")
    parser.add_argument("--project", default=".", help="project folder (default: cwd)")
    parser.add_argument("--dry-run", action="store_true",
                        help="time three training chunks and write dry_run.json only")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
    if args.dry_run:
        try:
            dry_run_timing(project_dir)
        except Exception as exc:  # noqa: BLE001 - the agent reads the exit code and stderr
            traceback.print_exc()
            print(f"error: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    try:
        # Fresh placeholders before training starts, so a failure before the first
        # save() cannot leave a previous run's epochs/best metric on disk.
        write_json(project_dir / METRICS_FILE, empty_metrics())
        metrics = train(project_dir)
        return 0 if metrics["status"] != "failed" else 1
    except Exception as exc:  # noqa: BLE001 - record any failure for the debrief
        traceback.print_exc()
        read_metrics = read_json(project_dir / METRICS_FILE, default=None) or {}
        existing = {**empty_metrics(), **read_metrics}
        existing["status"] = "failed"
        existing["error"] = f"{type(exc).__name__}: {exc}"
        write_json(project_dir / METRICS_FILE, existing)
        return 1
```

`copy` and `math` stay imported (still used by `train`); nothing is removed from the
import block.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_template_dry_run.py tests/test_template_train.py -q
ruff check .
```
Expected: `tests/test_template_dry_run.py` reports `6 passed` (3 parametrized + 3), `tests/test_template_train.py` passes with its two dry-run tests gone, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates/tabular_sklearn/train.py tests/test_template_dry_run.py tests/test_template_train.py
git commit -m "feat: the tabular train.py --dry-run writes dry_run.json

A warm-up chunk plus three timed fit_epoch chunks, recorded in dry_run.json
with device/gpu_name/batches_per_epoch/seconds_per_batch/seconds_per_epoch/
n_train/script instead of the old stdout JSON. Still writes no metrics.json,
no checkpoint and no figure; a failure exits non-zero with the error on stderr.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 4: `--dry-run` in `image_torch/train.py`

**Files:**
- Modify: `mlagent/templates/image_torch/train.py:1-10` (docstring), `:41-47` (settings), insert a `# --- Dry run ---` section before `# --- training ---` at `:184`, `:319-339` (`main`)
- Test: `tests/test_template_dry_run.py` (append the image half)

**Interfaces:**
- Consumes: `pick_device()` (imported at `mlagent/templates/image_torch/train.py:27` from `data.py:170`), `make_loader(pair, batch_size, shuffle, seed=0) -> DataLoader` (`mlagent/templates/image_torch/data.py:130`), `augment_batch`, `load_data(project_dir, config) -> dict` with keys `train`, `n_train`, `classes`, `image_size` (`data.py:99-126`), `build_model(config, n_classes, image_size)` (`model.py`), `read_json` / `write_json` (`train.py:78`, `:84`), `SCRIPT_NAME = "train.py"` (`:42`).
- Produces: the same `dry_run.json` contract as Task 3, with `device` from `pick_device()`, `gpu_name` from `torch.cuda.get_device_name(0)` when the device is `cuda` (else `null`), and `batches_per_epoch = len(train_loader)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_template_dry_run.py`:

```python
torch = pytest.importorskip("torch")

TINY_CNN = {"model_type": "tiny_cnn", "epochs": 2, "batch_size": 16, "learning_rate": 0.01,
            "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1,
            "augment": "basic", "dropout": 0.3}


def test_the_image_dry_run_writes_the_whole_contract(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = record(root)
    assert set(data) == DRY_RUN_KEYS
    assert data["device"] == "cpu" and data["gpu_name"] is None   # CUDA_VISIBLE_DEVICES=-1
    # 60 images, 70% train split, batch_size 16 -> 3 batches per epoch.
    assert data["batches_per_epoch"] == 3
    assert data["seconds_per_batch"] > 0
    assert data["seconds_per_epoch"] == pytest.approx(
        data["seconds_per_batch"] * data["batches_per_epoch"])
    assert data["n_train"] == 42
    assert data["script"] == "train.py"
    assert_wrote_nothing_else(root)


def test_the_image_dry_run_prints_a_human_line(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    proc = run(root, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    # Last non-blank line, not the only line -- see the tabular test's comment above.
    assert lines[-1].startswith("Dry run: 3 batches in ")
    assert "on cpu;" in lines[-1] and "3 batches per epoch" in lines[-1]


def test_the_image_dry_run_leaves_the_model_untrained_on_disk(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    assert run(root, "--dry-run").returncode == 0
    assert not (root / "checkpoints").exists() or not any(
        (root / "checkpoints").glob("*.pt"))


def test_a_broken_image_dry_run_exits_non_zero_with_the_error_on_stderr(clean_image_project):
    root = install(clean_image_project, IMAGE_TEMPLATE, TINY_CNN)
    (root / "data" / "clean" / "data.npz").unlink()
    proc = run(root, "--dry-run")
    assert proc.returncode != 0
    assert proc.stderr.strip()
    assert not (root / "dry_run.json").exists()
    assert_wrote_nothing_else(root)


def test_a_plain_image_run_still_ignores_stray_kernel_flags():
    source = (IMAGE_TEMPLATE / "train.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--dry-run", action="store_true"' in source
    assert "def cli_argv()" in source


def test_the_image_dry_run_section_is_in_the_walkthrough():
    from mlagent.codewalk import split_sections

    source = (IMAGE_TEMPLATE / "train.py").read_text(encoding="utf-8")
    titles = [title for title, _body in split_sections(source)]
    assert "Dry run" in titles
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_template_dry_run.py -q -k image
```
Expected: `5 failed, 1 passed` — the four subprocess tests fail because `train.py` has no
`--dry-run` flag, so `argparse` exits 2 with `error: unrecognized arguments: --dry-run`
(`test_a_broken_image_dry_run_exits_non_zero_with_the_error_on_stderr` passes for the
wrong reason today and must be re-run after Step 3); `test_a_plain_image_run_still_ignores_stray_kernel_flags`
and `test_the_image_dry_run_section_is_in_the_walkthrough` fail on the missing flag and
the missing section marker.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/templates/image_torch/train.py`, add to the docstring's usage block (after line 5):

```python
  python train.py --dry-run      times three training batches, writes dry_run.json only
```

Add to the `# --- settings ---` block after `CURVES_FIGURE` (line 47):

```python
DRY_RUN_FILE = "dry_run.json"
DRY_RUN_BATCHES = 3
```

Insert a new section immediately before `# --- training ---` (line 184):

```python
# --- Dry run ---
def _dry_batches(loader, count: int) -> list:
    """`count` batches from the loader, cycling it when the training split is shorter."""
    batches: list = []
    while len(batches) < count:
        before = len(batches)
        for batch in loader:
            batches.append(batch)
            if len(batches) == count:
                return batches
        if len(batches) == before:
            break
    return batches


def dry_run_timing(project_dir: Path) -> dict:
    """Time three real training batches so the agent can estimate the whole run.

    The first batch is a warm-up and is not timed, so one-off costs (the first CUDA
    kernel launch, cuDNN's algorithm search, the first reads off the tensor cache) stay
    out of the number. Writes `dry_run.json` and nothing else: no metrics.json, no
    checkpoint, no figure.
    """
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    batch_size = int(config.get("batch_size", 32))
    seed = int(config.get("seed", 42))
    device = pick_device()
    torch.manual_seed(seed)
    model = build_model(config, len(data["classes"]), data["image_size"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    loader = make_loader(data["train"], batch_size, shuffle=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    augment = str(config.get("augment", "basic")) != "none"
    batches_per_epoch = len(loader)

    def step(batch) -> None:
        batch_x, batch_y = batch
        if augment:
            batch_x = augment_batch(batch_x, generator)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimiser.step()

    batches = _dry_batches(loader, DRY_RUN_BATCHES + 1)
    if not batches:
        raise ValueError("the training split has no batches to time")
    model.train()
    step(batches[0])                                   # warm-up, not timed
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.time()
    for batch in batches[1:]:
        step(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    timed = max(1, len(batches) - 1)
    seconds_per_batch = round(elapsed / timed, 4)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    record = {
        "device": device,
        "gpu_name": gpu_name,
        "batches_per_epoch": batches_per_epoch,
        "seconds_per_batch": seconds_per_batch,
        "seconds_per_epoch": round(seconds_per_batch * batches_per_epoch, 4),
        "n_train": int(data["n_train"]),
        "script": SCRIPT_NAME,
    }
    write_json(project_dir / DRY_RUN_FILE, record)
    where = f"{device} ({gpu_name})" if gpu_name else device
    print(
        f"Dry run: {timed} batches in {elapsed:.2f} s on {where}; about "
        f"{seconds_per_batch:.3f} s per batch, {batches_per_epoch} batches per epoch",
        flush=True,
    )
    return record
```

Replace `main` (lines 319-339) with:

```python
def main(argv: list[str] | None = None) -> int:
    prefer_stdlib_modules()
    parser = argparse.ArgumentParser(description="Train the network and record every epoch.")
    parser.add_argument("--project", default=".", help="project folder (default: cwd)")
    parser.add_argument("--dry-run", action="store_true",
                        help="time three training batches and write dry_run.json only")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
    if args.dry_run:
        try:
            dry_run_timing(project_dir)
        except Exception as exc:  # noqa: BLE001 - the agent reads the exit code and stderr
            traceback.print_exc()
            print(f"error: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    try:
        # Fresh placeholders before training starts, so a failure before the first save()
        # cannot leave a previous run's epochs or best metric on disk.
        write_json(project_dir / METRICS_FILE, empty_metrics())
        metrics = train(project_dir)
        return 0 if metrics["status"] != "failed" else 1
    except Exception as exc:  # noqa: BLE001 - record any failure for the debrief
        traceback.print_exc()
        existing = {**empty_metrics(),
                    **(read_json(project_dir / METRICS_FILE, default=None) or {})}
        existing["status"] = "failed"
        existing["error"] = f"{type(exc).__name__}: {exc}"
        existing["checkpoint"] = None
        write_json(project_dir / METRICS_FILE, existing)
        return 1
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_template_dry_run.py -q
ruff check .
```
Expected: `12 passed` (Task 3's 6 plus the 6 added here), ruff clean.

```
python -m pytest tests/test_template_evaluate_images.py tests/test_template_model_images.py -q
```
Expected: unchanged, all pass — nothing in the real-run path moved.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates/image_torch/train.py tests/test_template_dry_run.py
git commit -m "feat: the image train.py --dry-run writes dry_run.json

A warm-up batch plus three timed batches on the real train loader, with the
device from pick_device() and the GPU name from torch when it is cuda. The
flag joins --project on the existing argparse; cli_argv() is unchanged, so a
bare kernel cell still passes no arguments.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 5: A `None` handoff skips the debrief, and runs carry their estimate

**Files:**
- Modify: `mlagent/orchestrator.py:150`, `mlagent/runs.py:30-50` (`build_run_entry`), `:128-151` (`log_finished_run`), plus a new `estimate_vs_actual` after `build_run_entry` — no import changes: `cfg` and `Project` are already imported at `runs.py:14-15`
- Test: `tests/test_orchestrator.py` (append), `tests/test_runs.py` (append)

**Interfaces:**
- Consumes (Task 1): `config.COST_FILE = "cost.json"`. Existing repo symbols: `Project.read_json(filename, default=None)` (`mlagent/project.py:114`), `stage_debrief` / `stage_prepare` / `stage_outputs_ready` (`mlagent/stages/base.py`), `append_run` (`mlagent/runlog.py`).
- Produces (used by Tasks 6, 7):
  - `runs.build_run_entry(metrics: dict, checkpoint: str | None = None, applied_diff: dict | None = None, estimated_minutes: float | None = None, estimated_units: float | None = None) -> dict` — the returned dict gains `"estimated_minutes"` and `"estimated_units"`
  - `runs.estimate_vs_actual(entry: dict) -> str | None`
  - `Orchestrator._run` never calls `stage_debrief` for a round whose `prepare` returned `None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
class GatedStage:
    """A stage whose prepare() refuses to hand off, the way a cost-gate stop does."""

    name = "gated"

    def __init__(self):
        self.prepared = 0
        self.debriefed = 0

    def prepare(self, ctx):
        self.prepared += 1
        ctx.display("Stop: change the config and run this cell again.")
        return None

    def debrief(self, ctx):
        self.debriefed += 1
        ctx.display("I can't see a finished run in metrics.json yet.")
        return None

    def is_complete(self, ctx):
        return False


def test_a_prepare_that_returns_no_handoff_never_reaches_debrief(project):
    shown: list[str] = []
    ctx = make_ctx(project, display=shown.append)
    stage = GatedStage()
    orch = Orchestrator(ctx, [stage])

    assert orch.run() == []
    assert stage.prepared == 1
    assert stage.debriefed == 0
    assert orch.waiting() is None
    assert not any("metrics.json" in s for s in shown)
    assert any("Stop: change the config" in s for s in shown)


def test_the_stage_stays_current_and_unprepared_after_a_no_handoff_round(project):
    ctx = make_ctx(project)
    stage = GatedStage()
    orch = Orchestrator(ctx, [stage])
    orch.run()
    state = project.read_json("state.json")
    assert state["current"] == "gated"
    assert state["completed"] == [] and state["prepared"] == []
    assert state["handoff"] is None

    # The next run() prepares it again rather than resuming a handoff that never existed.
    orch.run()
    assert stage.prepared == 2 and stage.debriefed == 0
```

Append to `tests/test_runs.py`:

```python
def test_build_run_entry_defaults_the_estimate_keys_to_none():
    entry = runs.build_run_entry({"status": "done", "started_at": "t", "epochs": []})
    assert entry["estimated_minutes"] is None
    assert entry["estimated_units"] is None


def test_build_run_entry_carries_the_estimate_through():
    entry = runs.build_run_entry({"status": "done", "started_at": "t", "epochs": []},
                                 estimated_minutes=3.1, estimated_units=0.21)
    assert entry["estimated_minutes"] == 3.1 and entry["estimated_units"] == 0.21


def test_logging_a_run_reads_the_estimate_out_of_cost_json(project):
    write_fake_run(project)
    project.write_json("cost.json", {
        "price_per_unit": 0.0999, "currency": "$",
        "last_estimate": {"minutes": 3.1, "units": 0.21, "basis": "dry_run"},
    })
    logged = runs.log_finished_run(project)
    assert logged is not None
    assert logged.entry["estimated_minutes"] == 3.1
    assert logged.entry["estimated_units"] == 0.21


def test_logging_a_run_without_cost_json_leaves_the_estimate_none(project):
    write_fake_run(project)
    logged = runs.log_finished_run(project)
    assert logged is not None
    assert logged.entry["estimated_minutes"] is None
    assert logged.entry["estimated_units"] is None


def test_estimate_vs_actual_reports_the_percentage_under():
    entry = {"estimated_minutes": 3.1, "seconds": 162.0}   # 2.7 minutes
    assert runs.estimate_vs_actual(entry) == (
        "Estimated 3.1 min, actual 2.7 min (13% under).")


def test_estimate_vs_actual_reports_the_percentage_over():
    entry = {"estimated_minutes": 2.0, "seconds": 150.0}   # 2.5 minutes
    assert runs.estimate_vs_actual(entry) == (
        "Estimated 2.0 min, actual 2.5 min (25% over).")


def test_estimate_vs_actual_is_omitted_when_there_is_no_estimate():
    assert runs.estimate_vs_actual({"estimated_minutes": None, "seconds": 60.0}) is None
    assert runs.estimate_vs_actual({"seconds": 60.0}) is None
    assert runs.estimate_vs_actual({"estimated_minutes": 3.0, "seconds": None}) is None
    assert runs.estimate_vs_actual({"estimated_minutes": 0, "seconds": 60.0}) is None
```

`write_fake_run(project)` is the existing helper at the top of `tests/test_runs.py`; it
writes a matching `metrics.json` and `eval_val.json` so `run_problem` returns `None`.

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_orchestrator.py tests/test_runs.py -q
```
Expected: eight failures (1 orchestrator + 4 estimate + 3 `estimate_vs_actual`).
`test_a_prepare_that_returns_no_handoff_never_reaches_debrief` fails on
`assert stage.debriefed == 0` (it is `1` — `mlagent/orchestrator.py:150` calls
`stage_debrief` unconditionally). The four estimate tests fail with
`KeyError: 'estimated_minutes'`, and the three `estimate_vs_actual` tests fail with
`AttributeError: module 'mlagent.runs' has no attribute 'estimate_vs_actual'`.
`test_the_stage_stays_current_and_unprepared_after_a_no_handoff_round` passes already.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/orchestrator.py`, replace line 150:

```python
                    if not stage_debrief(stage, self.ctx):
```

with:

```python
                    # No handoff means the round produced nothing for a debrief to read:
                    # `TrainStage`/`TuneStage.debrief` would call `runs.run_problem` on a
                    # stale metrics.json and print "I can't see a finished run" under the
                    # cost gate's own stop message. Behaviour-preserving for the stages
                    # that already return None here (intake and codegen debrief to None).
                    if handoff is None or not stage_debrief(stage, self.ctx):
```

In `mlagent/runs.py`, replace `build_run_entry` (lines 30-50) with:

```python
def build_run_entry(
    metrics: dict, checkpoint: str | None = None, applied_diff: dict | None = None,
    estimated_minutes: float | None = None, estimated_units: float | None = None,
) -> dict:
    """One `runs.jsonl` line, derived from the metrics.json that train.py wrote.

    `estimated_minutes` / `estimated_units` come from the cost gate's `cost.json`
    `last_estimate`; both stay None when no estimate was made (the dry run failed).
    """
    epochs = metrics.get("epochs") or []
    last = epochs[-1] if epochs else {}
    ok = metrics.get("status") == "done"
    return {
        "started_at": metrics.get("started_at"),
        "status": "done" if ok else "failed",
        "config": dict(metrics.get("config") or {}),
        "epochs_run": len(epochs),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_metric": metrics.get("best_val_metric"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "seconds": metrics.get("seconds"),
        "error": metrics.get("error"),
        "applied_diff": dict(applied_diff) if applied_diff else None,
        "checkpoint": checkpoint if ok else None,
        "estimated_minutes": estimated_minutes,
        "estimated_units": estimated_units,
    }


def last_estimate(project: Project) -> dict:
    """The cost gate's most recent estimate for this project, or an empty dict."""
    record = project.read_json(cfg.COST_FILE)
    estimate = record.get("last_estimate") if isinstance(record, dict) else None
    return estimate if isinstance(estimate, dict) else {}


def estimate_vs_actual(entry: dict) -> str | None:
    """`"Estimated 3.1 min, actual 2.7 min (13% under)."`, or None with no estimate.

    Fixed text, built without an LLM call, and omitted entirely rather than showing a
    placeholder when the gate had no estimate to make.
    """
    estimated = entry.get("estimated_minutes")
    seconds = entry.get("seconds")
    if not isinstance(estimated, int | float) or not estimated:
        return None
    if not isinstance(seconds, int | float):
        return None
    actual = float(seconds) / 60.0
    percent = round(abs(actual - estimated) / float(estimated) * 100)
    direction = "over" if actual > estimated else "under"
    return (f"Estimated {float(estimated):.1f} min, actual {actual:.1f} min "
            f"({percent}% {direction}).")
```

In `log_finished_run`, replace line 150 with:

```python
    estimate = last_estimate(project)
    entry = append_run(project.runs_path, build_run_entry(
        metrics, checkpoint, applied_diff,
        estimated_minutes=estimate.get("minutes"),
        estimated_units=estimate.get("units"),
    ))
```

`cfg` is already imported at `mlagent/runs.py:14` (`from mlagent import config as cfg`) and
`Project` at `:15`; no new imports.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_orchestrator.py tests/test_runs.py tests/test_train_stage.py tests/test_tune_stage.py -q
ruff check .
```
Expected: all pass. The nine new tests pass; the existing `runs.jsonl` assertions in
`tests/test_train_stage.py` and `tests/test_tune_stage.py` keep passing because they read
named keys, never the whole dict.

```
python -m pytest -q
```
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add mlagent/orchestrator.py mlagent/runs.py tests/test_orchestrator.py tests/test_runs.py
git commit -m "fix: a prepare that returns no handoff skips the debrief; runs carry their estimate

A stage that stops inside prepare has produced nothing for a debrief to read,
so the orchestrator no longer calls one -- which keeps runs.run_problem's
'I can't see a finished run' message from appearing under a cost-gate stop.
runs.jsonl entries gain estimated_minutes/estimated_units from cost.json, and
runs.estimate_vs_actual formats the debrief's comparison line.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 6: `mlagent/stages/cost_gate.py`

**Files:**
- Create: `mlagent/stages/cost_gate.py`, `tests/test_cost_gate.py`
- Test: `tests/test_cost_gate.py`

**Interfaces:**
- Consumes (Tasks 1-2): `cost.load_rates()`, `cost.detect_runtime(probes)`, `cost.run_dry_run(project_dir, python=sys.executable, timeout=600, env=None)`, `cost.DryRunResult`, `cost.estimate_from_dry_run(dry, epochs, runtime, rates, price, currency, rounds)`, `cost.estimate_from_history(seconds_per_epoch, epochs, runtime, rates, price, currency, rounds)`, `cost.budget_check(estimate, minutes_per_run, config, schema, modality)`, `cost.render_estimate(estimate, advice, spec_gpu, modality, basis_run_id=None)`. Existing repo symbols: `config.CONFIG_FILE`/`COST_FILE`, `modality_for(task_type)` (`mlagent/modality.py:110`), `load_schema(name)` / `schema_for(schema, model_type)` / `edit_config(questioner, config, schema, display)` (`mlagent/templates_io.py:36`, `:44`, `:220`), `read_runs(path)` (`mlagent/runlog.py:19`), `read_run_metrics(project, runs)` (`mlagent/runs.py:87`), `StageContext.spec()` / `.project` / `.questioner` / `.display` (`mlagent/stages/base.py:89-105`), `Questioner.number(prompt, default, minimum, maximum, key)` / `.text(prompt, default, key)` / `.choice(question, options, allow_other, key, default)` (`mlagent/ui/questions.py:9-27`).
- Produces (used by Task 7):
  - `cost_gate.gate(ctx, *, rounds_remaining: int, probes=None, dry_runner=cost.run_dry_run) -> str` returning `"run"` or `"stop"`
  - `cost_gate.RUN_LABEL = "Run it"`, `EDIT_LABEL = "Edit the config first"`, `STOP_LABEL = "Stop"`, `MAX_EDITS = 3`, `STOP_TEXT`, `NO_ESTIMATE_TEXT`, `TIMING_TEXT`
  - `cost.json` shape: `{"price_per_unit": float, "currency": str, "last_estimate": {...}}`
  - questioner keys `train.price_per_unit`, `train.currency`, `train.cost_decision`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cost_gate.py`:

```python
"""The cost gate: price, basis, budget, the question, and what lands in cost.json."""

from __future__ import annotations

from functools import partial

import pytest

from mlagent import cost
from mlagent.llm import FakeLLM
from mlagent.stages import cost_gate
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

T4_PROBES = [("torch", lambda: "Tesla T4")]
CPU_PROBES = ()
SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


class Runner:
    """A stand-in for cost.run_dry_run that counts how often it is asked to shell out."""

    def __init__(self, seconds_per_epoch=6.0, error=None, batches_per_epoch=1):
        self.calls = 0
        self.result = cost.DryRunResult(
            device="cpu", gpu_name=None, batches_per_epoch=batches_per_epoch,
            seconds_per_batch=seconds_per_epoch / max(1, batches_per_epoch),
            seconds_per_epoch=seconds_per_epoch, n_train=168, script="train.py",
            error=error,
        )

    def __call__(self, project_dir, **kwargs):
        self.calls += 1
        return self.result


def make_ctx(project, answers=(), form=None):
    shown: list[str] = []
    questioner = ScriptedQuestioner(list(answers))
    if form is not None:
        questioner = FormQuestioner(form, fallback=questioner, note=shown.append)
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=questioner,
                       explainer=None, display=shown.append,
                       display_figure=lambda path, caption="": None)
    return ctx, shown


def codegen(project, model_label="Gradient boosting", config_updates=None):
    """A project with data.py/model.py/train.py/evaluate.py and a config.json."""
    ctx, _shown = make_ctx(project, answers=[model_label, "y"])
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(config_updates or SMALL)
    project.write_json("config.json", cfg)
    return project


def priced(project, price=0.0999, currency="$"):
    project.write_json("cost.json", {"price_per_unit": price, "currency": currency})
    return project


def test_a_cpu_run_under_budget_never_asks_and_never_stops(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner(seconds_per_epoch=6.0)     # 3 epochs -> 0.36 minutes, budget is 5
    ctx, shown = make_ctx(project)             # ScriptedQuestioner with no answers at all
    decision = cost_gate.gate(ctx, rounds_remaining=3, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"
    assert runner.calls == 1
    text = "\n".join(shown)
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text
    assert "Timed with a 3-batch dry run." in text
    assert ctx.questioner.asked == []           # a beginner on CPU is never interrupted


def test_a_gpu_run_asks_and_run_it_means_run(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=2, probes=T4_PROBES, dry_runner=runner)
    assert decision == "run"
    text = "\n".join(shown)
    assert "This runtime has a Tesla T4 [[GPU]]." in text
    assert "| minutes | [[compute units]] | cost |" in text
    assert "this run plus 2 remaining rounds" in text
    assert ctx.questioner.asked == ["Run it?"]


def test_stop_returns_stop_and_says_what_to_do_next(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.STOP_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES, dry_runner=runner)
    assert decision == "stop"
    assert cost_gate.STOP_TEXT in shown


def test_over_budget_shows_the_cuts_and_still_asks(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)    # 20 epochs -> 24 minutes against 5
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"                   # the gate warns, it never refuses
    text = "\n".join(shown)
    assert "**Over budget:**" in text
    assert "- lower `epochs` from 20 to 4 --" in text
    assert ctx.questioner.asked == ["Run it?"]  # asked even though the runtime is CPU


def test_editing_the_config_reuses_the_measured_timing_and_re_estimates(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)
    ctx, shown = make_ctx(project, answers=[
        cost_gate.EDIT_LABEL,        # "Run it?"
        "epochs = 20",               # edit_config: which value
        "4",                         # edit_config: new value
        "Done",                      # edit_config: finished
        cost_gate.RUN_LABEL,         # "Run it?" again
    ])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"
    assert runner.calls == 1                   # no second dry run, ever
    assert project.read_json("config.json")["epochs"] == 4
    text = "\n".join(shown)
    assert text.count("Timed with a 3-batch dry run.") == 2
    assert "**Over budget:**" in text          # the first estimate was
    assert text.rstrip().endswith("free.")     # the second one is not


def test_the_edit_option_disappears_after_three_edits(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)
    seen: list[list[str]] = []

    class Watcher(ScriptedQuestioner):
        def choice(self, question, options, allow_other=True, key=None, default=None):
            if question == "Run it?":
                seen.append(list(options))
            return super().choice(question, options, allow_other, key, default)

    ctx, shown = make_ctx(project)
    ctx.questioner = Watcher([
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.RUN_LABEL,
    ])
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    assert len(seen) == 4
    assert all(cost_gate.EDIT_LABEL in options for options in seen[:3])
    assert seen[3] == [cost_gate.RUN_LABEL, cost_gate.STOP_LABEL]


def test_the_second_run_estimates_from_history_and_never_shells_out(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("runs/run1_metrics.json", {"seconds_per_epoch": 4.0})
    project.runs_path.write_text(
        '{"run_id": 1, "status": "done", "seconds_per_epoch": 4.0, '
        '"config": {"model_type": "gradient_boosting"}}\n', encoding="utf-8")
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    assert cost_gate.gate(ctx, rounds_remaining=1, probes=T4_PROBES,
                          dry_runner=runner) == "run"
    assert runner.calls == 0
    assert "From run 1's measured time." in "\n".join(shown)


def test_a_run_of_another_family_is_not_used_as_history(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("runs/run1_metrics.json", {"seconds_per_epoch": 4.0})
    project.runs_path.write_text(
        '{"run_id": 1, "status": "done", "config": {"model_type": "random_forest"}}\n',
        encoding="utf-8")
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES, dry_runner=runner)
    assert runner.calls == 1
    assert "Timed with a 3-batch dry run." in "\n".join(shown)


def test_a_failed_dry_run_says_so_and_still_reaches_the_question(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner(error="The dry run failed (exit 1).\n\n```\nValueError: boom\n```")
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    text = "\n".join(shown)
    assert "ValueError: boom" in text
    assert cost_gate.NO_ESTIMATE_TEXT in text
    assert "| minutes |" not in text            # nothing to tabulate
    assert ctx.questioner.asked == ["Run it?"]


def test_a_failed_dry_run_clears_a_stale_estimate(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("cost.json", {"price_per_unit": 0.0999, "currency": "$",
                                     "last_estimate": {"minutes": 9.9, "units": 1.0}})
    runner = Runner(error="The dry run failed (exit 1).")
    ctx, _shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert project.read_json("cost.json").get("last_estimate") is None


def test_the_price_and_currency_are_asked_once_then_reused(clean_project):
    project = codegen(clean_project)            # no cost.json yet
    runner = Runner()
    ctx, shown = make_ctx(project, form={"train.price_per_unit": 0.05,
                                         "train.currency": "GBP"})
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    record = project.read_json("cost.json")
    assert record["price_per_unit"] == pytest.approx(0.05)
    assert record["currency"] == "GBP"
    assert ctx.questioner.used == ["train.price_per_unit", "train.currency"]

    # A second gate call on the same project reads cost.json and asks nothing.
    ctx2, _shown2 = make_ctx(project)            # no answers of any kind
    assert cost_gate.gate(ctx2, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=Runner()) == "run"
    assert ctx2.questioner.asked == []


def test_the_default_price_comes_from_rates_json(clean_project):
    project = codegen(clean_project)
    ctx, _shown = make_ctx(project, answers=["", ""])   # blank -> take the defaults
    cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=Runner())
    record = project.read_json("cost.json")
    rates = cost.load_rates()
    assert record["price_per_unit"] == pytest.approx(rates["default_price_per_unit"])
    assert record["currency"] == rates["default_currency"]


def test_the_estimate_is_persisted_for_the_debrief(clean_project):
    project = priced(codegen(clean_project))
    ctx, _shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES,
                   dry_runner=Runner(seconds_per_epoch=6.0))
    stored = project.read_json("cost.json")["last_estimate"]
    assert stored["basis"] == "dry_run" and stored["device"] == "cuda"
    assert stored["minutes"] == pytest.approx(0.36)       # 6 * 3 * 1.2 / 60
    assert stored["units"] == pytest.approx(0.012, abs=5e-4)
    assert stored["gpu_type"] == "T4"


def test_gate_is_usable_as_a_partial_the_way_the_stages_take_it(clean_project):
    project = priced(codegen(clean_project))
    bound = partial(cost_gate.gate, probes=CPU_PROBES, dry_runner=Runner())
    ctx, _shown = make_ctx(project)
    assert bound(ctx, rounds_remaining=0) == "run"
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_cost_gate.py -q
```
Expected: collection aborts with `ModuleNotFoundError: No module named 'mlagent.stages.cost_gate'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/stages/cost_gate.py`:

```python
"""Show what a training run will cost, then ask: run it, edit the config, or stop.

Called from `TrainStage.prepare` and `TuneStage.prepare` just before each builds its
`Handoff`. Everything numeric lives in `mlagent/cost.py`; this module only gathers the
inputs, asks the questions and persists the answers to `cost.json`.

`probes` and `dry_runner` are injectable for the same reason `DataStage` takes `hf_load`:
tests must never read real hardware or shell out to a real training script.
"""

from __future__ import annotations

from mlagent import config as cfg
from mlagent import cost
from mlagent.modality import modality_for
from mlagent.runlog import read_runs
from mlagent.runs import read_run_metrics
from mlagent.templates_io import edit_config, load_schema, schema_for

RUN_LABEL = "Run it"
EDIT_LABEL = "Edit the config first"
STOP_LABEL = "Stop"
DECISION_KEY = "train.cost_decision"
MAX_EDITS = 3
TIMING_TEXT = "Timing a short dry run..."
NO_ESTIMATE_TEXT = "No estimate is available; the run can still proceed."
STOP_TEXT = ("Change the config and run this cell again, or switch runtime, then run this "
             "cell again.")


def price_and_currency(ctx, rates: dict) -> tuple[float, str]:
    """From `cost.json` when it is there, otherwise asked once and written there.

    The notebook's "4. Model" form supplies both keys through `FormQuestioner`, so a
    beginner never meets a console `input()` for them; the console and scripted
    questioners are only reached outside the form flow and in tests.
    """
    stored = ctx.project.read_json(cfg.COST_FILE)
    stored = dict(stored) if isinstance(stored, dict) else {}
    if isinstance(stored.get("price_per_unit"), int | float) and stored.get("currency"):
        return float(stored["price_per_unit"]), str(stored["currency"])
    default_price = float(rates["default_price_per_unit"])
    default_currency = str(rates["default_currency"])
    price = float(ctx.questioner.number(
        "Price per compute unit?", default=default_price, minimum=0,
        key="train.price_per_unit"))
    currency = str(ctx.questioner.text(
        "Currency symbol or code?", default=default_currency,
        key="train.currency")).strip() or default_currency
    stored.update({"price_per_unit": price, "currency": currency})
    ctx.project.write_json(cfg.COST_FILE, stored)
    return price, currency


def history_basis(project, model_type: str) -> tuple[float, int] | None:
    """`(seconds_per_epoch, run_id)` of the latest finished run of the same family.

    A tune proposal that switched family leaves no matching entry, so the caller falls
    back to a fresh dry run rather than estimating from unrelated timing.
    """
    runs = read_runs(project.runs_path)
    metrics_by_run = read_run_metrics(project, runs)
    for run in reversed(runs):
        if run.get("status") != "done":
            continue
        if str((run.get("config") or {}).get("model_type")) != model_type:
            continue
        seconds = (metrics_by_run.get(run.get("run_id")) or {}).get("seconds_per_epoch")
        if isinstance(seconds, int | float) and seconds > 0:
            return float(seconds), int(run["run_id"])
    return None


def _write_cost(project, **changes) -> None:
    record = project.read_json(cfg.COST_FILE)
    record = dict(record) if isinstance(record, dict) else {}
    record.update(changes)
    project.write_json(cfg.COST_FILE, record)


def _ask(ctx, *, allow_edit: bool, key: str | None) -> str:
    options = [RUN_LABEL, EDIT_LABEL, STOP_LABEL] if allow_edit else [RUN_LABEL, STOP_LABEL]
    answer = ctx.questioner.choice("Run it?", options, allow_other=False, key=key,
                                   default=RUN_LABEL)
    if answer == STOP_LABEL:
        ctx.display(STOP_TEXT)
        return "stop"
    return "edit" if answer == EDIT_LABEL else "run"


def gate(ctx, *, rounds_remaining: int, probes=None, dry_runner=cost.run_dry_run) -> str:
    """`"run"` or `"stop"`. Displays the estimate and persists it to `cost.json`."""
    project = ctx.project
    config = project.read_json(cfg.CONFIG_FILE)
    if not isinstance(config, dict) or not config.get("model_type"):
        raise RuntimeError("config.json not found; run the codegen stage first")
    spec = ctx.spec()
    modality = modality_for(spec.task_type)
    flat_schema = schema_for(load_schema(modality.template_family),
                             str(config["model_type"]))
    rates = cost.load_rates()
    price, currency = price_and_currency(ctx, rates)
    runtime = cost.detect_runtime(probes)
    rounds = max(0, int(rounds_remaining))
    epochs = int(config.get("epochs", 10))

    found = history_basis(project, str(config["model_type"]))
    dry: cost.DryRunResult | None = None
    basis_run_id: int | None = None
    if found is not None:
        seconds_per_epoch, basis_run_id = found
        estimate = cost.estimate_from_history(seconds_per_epoch, epochs, runtime, rates,
                                              price, currency, rounds)
    else:
        ctx.display(TIMING_TEXT)
        dry = dry_runner(project.root)
        if dry.error:
            ctx.display(dry.error)
            ctx.display(NO_ESTIMATE_TEXT)
            _write_cost(project, last_estimate=None)
            return _ask(ctx, allow_edit=False, key=DECISION_KEY)
        estimate = cost.estimate_from_dry_run(dry, epochs, runtime, rates, price, currency,
                                              rounds)

    # A Colab form answer for `train.cost_decision` is fixed for the whole cell run, so
    # once an edit sends us round again the re-ask must reach the fallback questioner.
    key: str | None = DECISION_KEY
    for edits in range(MAX_EDITS + 1):
        advice = cost.budget_check(estimate, spec.minutes_per_run, config, flat_schema,
                                   modality)
        ctx.display(cost.render_estimate(estimate, advice, spec.gpu, modality,
                                         basis_run_id=basis_run_id))
        _write_cost(project, last_estimate=estimate.to_dict())
        if runtime.device == "cpu" and not advice.over:
            return "run"
        answer = _ask(ctx, allow_edit=edits < MAX_EDITS, key=key)
        if answer != "edit":
            return answer
        key = None
        config = edit_config(ctx.questioner, config, flat_schema, ctx.display)
        project.write_json(cfg.CONFIG_FILE, config)
        epochs = int(config.get("epochs", epochs))
        # The per-batch cost did not change -- only schema keys did -- so the same
        # measured seconds_per_epoch is re-multiplied; no second dry run is ever launched.
        estimate = (cost.estimate_from_dry_run(dry, epochs, runtime, rates, price,
                                               currency, rounds)
                    if dry is not None
                    else cost.estimate_from_history(estimate.seconds_per_epoch, epochs,
                                                    runtime, rates, price, currency,
                                                    rounds))
    return "run"   # unreachable: the last pass always returns from _ask
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_cost_gate.py -q
ruff check .
```
Expected: `14 passed`, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/cost_gate.py tests/test_cost_gate.py
git commit -m "feat: the cost gate

gate() reads the price and currency once into cost.json, estimates from run
history when the project has a finished run of the same family and from a
subprocess dry run otherwise, checks the estimate against minutes_per_run,
shows the table and asks Run / Edit / Stop. An edit re-estimates from the same
measured timing -- the dry run never runs twice -- and after three edits the
question drops the edit option.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 7: Wire the gate into the train and tune stages

**Files:**
- Modify: `mlagent/stages/train.py:7-14` (imports), `:25-27` (add `__init__`), `:38-80` (`prepare`), `:121-131` (`_narrative`'s headline); `mlagent/stages/tune.py:29-45` (imports), `:218-220` (add `__init__`), `:310-313` (the gate before the `Handoff`), `:493-494` (the debrief headline)
- Test: `tests/test_train_stage.py`, `tests/test_tune_stage.py`, `tests/test_pipeline_e2e.py`, `tests/test_pipeline_e2e_images.py`, `tests/test_report_stage.py`

**Interfaces:**
- Consumes (Tasks 5-6): `cost_gate.gate(ctx, *, rounds_remaining, probes=None, dry_runner=cost.run_dry_run) -> str`, `cost_gate.STOP_TEXT`, `runs.estimate_vs_actual(entry) -> str | None`. Existing repo symbols: `Spec.max_rounds` / `.minutes_per_run` / `.gpu` (`mlagent/spec.py:24-35`), `Handoff` (`mlagent/stages/base.py:29`), `load_tune_state(project, max_rounds)` returning a dict with `round`, `max_rounds`, `decision`, `pending`, `history` (`mlagent/stages/tune.py`).
- Produces (used by Task 8): `TrainStage(gate=cost_gate.gate)`, `TuneStage(gate=cost_gate.gate)`, `TrainStage.prepare(ctx) -> Handoff | None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train_stage.py`, after the imports:

```python
from functools import partial

from mlagent import cost
from mlagent.stages import cost_gate


def cpu_gate(seconds_per_epoch: float = 0.2):
    """The real gate with no hardware probe and no subprocess dry run.

    0.2 s/epoch over the fixture's 3 epochs is 0.012 minutes against a 5 minute budget,
    so the gate is silent and asks nothing -- exactly the beginner-on-CPU path.
    """
    dry = cost.DryRunResult(device="cpu", gpu_name=None, batches_per_epoch=1,
                            seconds_per_batch=seconds_per_epoch,
                            seconds_per_epoch=seconds_per_epoch, n_train=168,
                            script="train.py")
    return partial(cost_gate.gate, probes=(), dry_runner=lambda project_dir, **kw: dry)


def stopping_gate(ctx, *, rounds_remaining):
    ctx.display(cost_gate.STOP_TEXT)
    return "stop"
```

Change `prepared` and `prepared_image` (`tests/test_train_stage.py:19-37`) to leave a price
in place so the gate never asks for one, by adding this line to each, just before `return project`:

```python
    project.write_json("cost.json", {"price_per_unit": 0.0999, "currency": "$"})
```

Every bare `TrainStage()` in `tests/test_train_stage.py` and in `tests/test_report_stage.py`
(its `trained()` helper around line 34 calls `stage.prepare(ctx)`) must construct the stage with
the fake gate; grep `TrainStage(` in `tests/` to find them all and replace each with
`TrainStage(gate=cpu_gate())` (or an equivalent no-hardware fake gate imported into
`tests/test_report_stage.py`). This matters because a bare `TrainStage()` would run
`cost.detect_runtime()` on the machine's real hardware, shell out a real `train.py --dry-run`
subprocess, and call `questioner.number` on a `ScriptedQuestioner` that has no answer queued for
it — failing or hanging every one of these tests. Also have `test_report_stage.py`'s `trained()`
helper write `cost.json` with a price and currency (the same fixture line
`tests/test_train_stage.py`'s `prepared`/`prepared_image` fixtures use, see below) so the gate
never asks a question there either. Then replace the 6a placeholder assertion at line 74:

```python
    assert "cost" in text.lower() and "cpu" in text.lower()
```

with:

```python
    assert "no [[compute unit]] cost gate" not in text
    assert "This runtime has no [[GPU]]." in text
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text
    assert "Training runs on the [[CPU]] for tabular data." in text
```

Append these tests to `tests/test_train_stage.py`:

```python
def test_the_gate_is_asked_for_every_remaining_tune_round(clean_project):
    project = prepared(clean_project)
    seen: list[int] = []

    def recording_gate(ctx, *, rounds_remaining):
        seen.append(rounds_remaining)
        return "run"

    ctx, _shown, _figures = make_ctx(project)
    TrainStage(gate=recording_gate).prepare(ctx)
    assert seen == [project.read_json("spec.json")["max_rounds"]]


def test_a_cost_gate_stop_hands_off_nothing_and_touches_no_outputs(clean_project):
    project = prepared(clean_project)
    project.write_json("metrics.json", {"started_at": "earlier"})
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage(gate=stopping_gate)

    assert stage.prepare(ctx) is None
    assert cost_gate.STOP_TEXT in shown
    # The stale metrics.json is left exactly as it was: nothing committed to a run.
    assert project.read_json("metrics.json") == {"started_at": "earlier"}
    assert not stage.is_complete(ctx)


def test_the_debrief_compares_the_estimate_with_the_actual_time(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage(gate=cpu_gate())
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    text = "\n".join(shown)
    assert "Estimated 0.0 min, actual " in text
    assert "min (" in text and ("% under)." in text or "% over)." in text)
    entry = runlog.read_runs(project.runs_path)[0]
    assert entry["estimated_minutes"] == pytest.approx(0.012)
    assert entry["estimated_units"] == 0.0     # a CPU run spends none


def test_the_debrief_omits_the_comparison_when_the_gate_made_no_estimate(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage(gate=lambda ctx, *, rounds_remaining: "run")   # never estimates
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert "Estimated" not in "\n".join(shown)
```

`pytest` must be imported at the top of `tests/test_train_stage.py` (it is not today):
add `import pytest` after `from pathlib import Path`.

Add to `tests/test_tune_stage.py`, after the imports:

```python
from functools import partial

from mlagent import cost
from mlagent.stages import cost_gate


def cpu_gate(seconds_per_epoch: float = 0.2):
    dry = cost.DryRunResult(device="cpu", gpu_name=None, batches_per_epoch=1,
                            seconds_per_batch=seconds_per_epoch,
                            seconds_per_epoch=seconds_per_epoch, n_train=168,
                            script="train.py")
    return partial(cost_gate.gate, probes=(), dry_runner=lambda project_dir, **kw: dry)
```

In `with_run_one` (`tests/test_tune_stage.py:59-75`), add the price file and pass the gate:

```python
    project.write_json("cost.json", {"price_per_unit": 0.0999, "currency": "$"})
    ctx, _s, _f = make_ctx(project)
    stage = TrainStage(gate=cpu_gate())
```

and replace every bare `TuneStage()` in the file with `TuneStage(gate=cpu_gate())`.

Append to `tests/test_tune_stage.py`:

```python
def test_a_cost_gate_stop_ends_the_round_without_a_decision(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _figures = make_ctx(project, answers=[apply_label(1)])

    def stopping_gate(ctx, *, rounds_remaining):
        ctx.display(cost_gate.STOP_TEXT)
        return "stop"

    assert TuneStage(gate=stopping_gate).prepare(ctx) is None
    assert cost_gate.STOP_TEXT in shown
    state = project.read_json("tune_state.json")
    assert state["decision"] == "continue"        # a cost stop is not a tuning decision
    assert state["pending"] is not None
    assert not TuneStage(gate=cpu_gate()).is_complete(ctx)


def test_the_gate_is_asked_for_the_rounds_that_are_left(clean_project):
    project = with_run_one(clean_project)
    seen: list[int] = []

    def recording_gate(ctx, *, rounds_remaining):
        seen.append(rounds_remaining)
        return "run"

    ctx, _shown, _figures = make_ctx(project, answers=[apply_label(1)])
    TuneStage(gate=recording_gate).prepare(ctx)
    # max_rounds is 3 in the fixture spec and this is round 1, so two rounds remain.
    assert seen == [2]


def test_the_tune_debrief_shows_the_estimate_vs_actual_line(clean_project):
    project = with_run_one(clean_project)
    ctx, shown, _figures = make_ctx(project, answers=[apply_label(1)])
    stage = TuneStage(gate=cpu_gate())
    handoff = stage.prepare(ctx)
    assert handoff is not None
    run_cells(project, handoff)
    stage.debrief(ctx)
    assert "Estimated 0.0 min, actual " in "\n".join(shown)
```

In `tests/test_pipeline_e2e.py`, add the two price answers plus the decision to
`FORM_ANSWERS` (after `"codegen.model_type"`):

```python
    "train.price_per_unit": 0.0999,
    "train.currency": "$",
    "train.cost_decision": "Run it",
```

and give `make_orchestrator` a real gate with no hardware probe (the dry run itself is
real: this is the only place `train.py --dry-run` runs inside the full pipeline):

```python
from functools import partial

from mlagent.stages import cost_gate

E2E_GATE = partial(cost_gate.gate, probes=())


def make_orchestrator(project, stages=None, questioner=None, display=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=questioner or AutoApproveQuestioner([]),
        explainer=None,
        display=display or (lambda s: None),
        display_figure=lambda path, caption="": None,
    )
    stages = stages or [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                        TrainStage(gate=E2E_GATE), TuneStage(gate=E2E_GATE), ReportStage()]
    return Orchestrator(ctx, stages)


def test_the_gate_estimates_before_training_and_the_debrief_compares_afterwards(
        project, advance):
    shown: list[str] = []
    orch = make_orchestrator(project, display=shown.append)
    assert advance(orch, project, answers=FORM_ANSWERS) == ALL_STAGES
    text = "\n".join(shown)
    assert "Timing a short dry run..." in text
    assert "| minutes | [[compute units]] | cost |" in text
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text
    assert "Estimated " in text and " min, actual " in text
    record = project.read_json("cost.json")
    assert record["price_per_unit"] == pytest.approx(0.0999)
    assert record["currency"] == "$"
    assert record["last_estimate"]["basis"] == "dry_run"
    assert project.read_json("runs/run1_metrics.json")["seconds_per_epoch"] > 0
    assert read_runs(project.runs_path)[0]["estimated_minutes"] is not None
```

In `tests/test_pipeline_e2e_images.py`, make exactly the same three `FORM_ANSWERS`
additions, the same `E2E_GATE = partial(cost_gate.gate, probes=())` and the same
`TrainStage(gate=E2E_GATE)` / `TuneStage(gate=E2E_GATE)` construction, plus a `display`
parameter on its `make_orchestrator`.

Do not add a second full-pipeline test here: the file already has exactly one real image
training run, `test_the_image_pipeline_runs_through_every_handoff`, and it keeps that run
fast by calling `advance_to_codegen` + `small_config(project)` (`epochs=2`) before resuming
with the plain `advance` fixture (lines 105-110). A second test built on the plain `advance`
fixture alone would run codegen's *default* epochs (10) through a real CPU CNN training
subprocess a second time in the same file — slow and redundant. Instead, extend the existing
test: pass `display=shown.append` into its `make_orchestrator(project)` call, keep the same
`advance_to_codegen` / `small_config` / `advance` sequence, and add these assertions after the
existing ones:

```python
    text = "\n".join(shown)
    assert "Timing a short dry run..." in text
    assert "This runtime has no [[GPU]]." in text
    assert "Estimated " in text and " min, actual " in text
    stored = project.read_json("cost.json")["last_estimate"]
    assert stored["basis"] == "dry_run" and stored["device"] == "cpu"
```

No extra `CUDA_VISIBLE_DEVICES` wiring is needed for the dry run the gate now launches:
the file's autouse `cpu_only` fixture (`tests/test_pipeline_e2e_images.py:59-65`)
monkeypatches `os.environ`, and `cost.run_dry_run`'s default `env=None` makes the child
inherit the parent's environment — the same way the existing `run_handoff` subprocesses
already pick it up.

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_train_stage.py tests/test_tune_stage.py -q
```
Expected: every test in both files errors at construction with
`TypeError: TrainStage() takes no arguments` / `TypeError: TuneStage() takes no arguments`
(neither class defines `__init__` today), and the rewritten assertion in
`test_real_training_run_is_logged_archived_and_debriefed` fails on
`assert "This runtime has no [[GPU]]." in text`.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/stages/train.py`, add to the imports (after `from mlagent.runlog import read_runs`):

```python
from mlagent.runs import EVAL_VAL_FILE, estimate_vs_actual, log_finished_run, run_problem
from mlagent.stages import cost_gate
```

(replacing the existing `from mlagent.runs import ...` line at `:10`).

Add a constructor to `TrainStage`, immediately under `name = "train"`:

```python
    def __init__(self, gate=cost_gate.gate):
        # Injectable the way `DataStage` takes `hf_load`: tests must never read real
        # hardware or shell out to a real training script.
        self.gate = gate
```

Change `prepare`'s signature to `def prepare(self, ctx: StageContext) -> Handoff | None:`
and drop the cost clause from both `device_sentence` branches (lines 46-50 and 57-60):

```python
        if modality.name == "image":
            device_sentence = (
                "Training picks its [[device]] at run time -- the [[GPU]] if this runtime "
                "has one, otherwise the [[CPU]]."
            )
```

```python
        else:
            device_sentence = "Training runs on the [[CPU]] for tabular data."
```

Insert the gate call between `ctx.teaching().preamble(...)` (line 70) and the comment
above the two `unlink` calls (line 71):

```python
        if self.gate(ctx, rounds_remaining=spec.max_rounds) == "stop":
            return None
```

In `_narrative`, replace the `return` (line 131) with:

```python
        comparison = estimate_vs_actual(entry)
        if comparison:
            headline += " " + comparison
        narrative = ctx.teaching().debrief("train", summary, figures, fallback=fallback)
        return headline + "\n\n" + narrative
```

(the existing `narrative = ...` line at 130 moves below the comparison so the headline is
finished before it is used; nothing else in `_narrative` changes).

In `mlagent/stages/tune.py`, add to the `from mlagent.runs import (...)` block (lines 29-35),
in alphabetical order between `archive_metrics` and `log_finished_run`:

```python
    archive_metrics,
    estimate_vs_actual,
    log_finished_run,
```

and add `from mlagent.stages import cost_gate` as its own line *before* the existing
`from mlagent.stages.base import ...` line, not after — isort (ruff I001) sorts
`mlagent.stages` ahead of `mlagent.stages.base` (the shorter dotted path sorts first):

```python
from mlagent.stages import cost_gate
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
```

(`mlagent/stages/train.py`'s edit above already inserts `from mlagent.stages import cost_gate`
ahead of its own `from mlagent.stages.base import ...` line, so no change is needed there —
call this out explicitly so Task 7 does not repeat tune.py's ordering mistake in train.py.)

Add a constructor to `TuneStage`, immediately under `name = "tune"`:

```python
    def __init__(self, gate=cost_gate.gate):
        self.gate = gate
```

The gate must run *before* `config.json` and `pending` are mutated, not after: the code at
lines 296-297 (`new_config, diff, reason = chosen` / `project.write_json(cfg.CONFIG_FILE,
new_config)`) already applies the proposal to disk before the old plan text's gate call, so a
cost-gate stop there would leave the refused proposal sitting in `config.json` and the next
`prepare` would diagnose and propose against the *mutated* config. Move the gate immediately
after `chosen` is unpacked and before anything is written, keeping `config` (read at line 245,
still the pre-proposal config — `chosen`/`new_config` does not mutate it) so a stop can restore
it unchanged. Replace lines 296-313 with:

```python
        new_config, diff, reason = chosen
        if self.gate(ctx, rounds_remaining=state["max_rounds"] - round_no) == "stop":
            # Not a tuning decision: `decision` stays "continue" so the next orch.run()
            # diagnoses and proposes again. Nothing was written yet, so config.json and
            # tune_state.json are already untouched -- nothing to restore.
            return None
        project.write_json(cfg.CONFIG_FILE, new_config)
        state["pending"] = {
            "round": round_no,
            "applied_diff": diff,
            "reason": reason,
            "diagnosis": diagnosis.label,
            "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        project.write_json(cfg.TUNE_STATE_FILE, state)
        # The cells regenerate these; a stale copy must not satisfy outputs_ready.
        (project.root / cfg.METRICS_FILE).unlink(missing_ok=True)
        (project.root / EVAL_VAL_FILE).unlink(missing_ok=True)
        changed = ", ".join(f"`{k}` {_fmt(v['from'])} -> {_fmt(v['to'])}" for k, v in diff.items())
        ctx.display(f"Applied to `config.json`: {changed}. Run the `train.py` and "
                    "`evaluate.py` cells again, then run this cell to see whether it helped.")
        return Handoff(stage=self.name, commands=[list(c) for c in TUNE_COMMANDS],
                       outputs=list(TUNE_OUTPUTS))
```

With the gate moved ahead of the writes, a stop is simply a return before anything changes — there
is no `config.json` or `pending` to restore. (An earlier draft of this plan put the gate after the
write and `pending` assignment, which would have required restoring `config.json` from the
pre-proposal `config` and clearing `pending` on a stop; moving the gate earlier avoids that
entirely.) Add a test in Task 7's tune tests asserting that after a cost-gate stop, `config.json`
on disk is byte-for-byte the pre-proposal config (i.e. unchanged from before `prepare` ran) and
`tune_state.json`'s `pending` is still whatever it was before this `prepare` call (`None` on a
fresh round).

Replace `debrief`'s display (lines 493-494) with:

```python
        headline = self._headline(spec, entry, previous_best, improved,
                                  more_rounds=more_rounds)
        comparison = estimate_vs_actual(entry)
        if comparison:
            headline += " " + comparison
        ctx.display(headline + "\n\n" + narrative)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_train_stage.py tests/test_tune_stage.py -q
python -m pytest tests/test_pipeline_e2e.py -q
python -m pytest tests/test_pipeline_e2e_images.py -q
ruff check .
```
Expected: all pass, ruff clean. `tests/test_train_stage.py` gains four tests,
`tests/test_tune_stage.py` three, `tests/test_pipeline_e2e.py` one new test;
`tests/test_pipeline_e2e_images.py` gains no new test (its existing full-run test grows
new assertions instead, per the note above).

```
python -m pytest -q
```
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/train.py mlagent/stages/tune.py tests/test_train_stage.py tests/test_tune_stage.py tests/test_pipeline_e2e.py tests/test_pipeline_e2e_images.py
git commit -m "feat: the train and tune stages run the cost gate before handing off

TrainStage.prepare and TuneStage.prepare call the gate immediately before
building their Handoff and return None on a stop, so the stage stays current
with nothing committed. The 6a placeholder sentences saying there is no cost
gate are gone, and both debriefs print the estimate-vs-actual line whenever the
gate managed an estimate.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 8: The notebook's price fields, the smoke checklist and CLAUDE.md

**Files:**
- Modify: `scripts/build_notebook.py:127-129` (the `MINUTES_PER_RUN` hint), `:252-266` (the "4. Model" cell), `notebooks/ML_Training_Agent.ipynb` (regenerated), `docs/colab-smoke.md` (new section at the end, plus the Milestone 3 item ~line 30), `CLAUDE.md:7`, `:11-24`, `:31`, `:34`, `:36`, `pyproject.toml:33` (package-data)
- Test: `tests/test_colab.py`

**Interfaces:**
- Consumes (Tasks 1, 6): `cost.load_rates()["default_price_per_unit"] == 0.0999`, `["default_currency"] == "$"`; the questioner keys `train.price_per_unit` and `train.currency` that `cost_gate.price_and_currency` reads.
- Produces: a rebuilt 21-cell `notebooks/ML_Training_Agent.ipynb` whose "4. Model" cell passes `{'codegen.model_type': MODEL, 'train.price_per_unit': PRICE_PER_UNIT, 'train.currency': CURRENCY}` to `orch.run(until='train', ...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_colab.py` (it already defines `notebook_sources()` at line 191 and
`model_cell()` at line 247):

```python
def test_the_model_cell_has_the_price_and_currency_fields_with_hints():
    source = model_cell()
    for field in ("PRICE_PER_UNIT", "CURRENCY"):
        assert f"{field} = " in source, field
        assert f"**{field}**" in source, field
    assert "Resources panel" in source
    assert "next to the cost estimate" in source


def test_the_model_cell_defaults_match_rates_json():
    from mlagent.cost import load_rates

    rates = load_rates()
    source = model_cell()
    assert f"PRICE_PER_UNIT = {rates['default_price_per_unit']}  #@param" in source
    assert f"CURRENCY = '{rates['default_currency']}'  #@param" in source


def test_the_model_cell_passes_every_cost_answer_key():
    source = model_cell()
    for key in ("codegen.model_type", "train.price_per_unit", "train.currency"):
        assert f"'{key}'" in source, key


def test_the_train_answer_keys_the_model_cell_sends_are_ones_the_gate_reads():
    import re

    source = Path("mlagent/stages/cost_gate.py").read_text(encoding="utf-8")
    used = set(re.findall(r'key="(train\.[a-z_]+)"', source))
    sent = set(re.findall(r"'(train\.[a-z_]+)'", model_cell()))
    assert sent <= used, sent - used


def test_the_tune_cell_gains_no_form_fields():
    tune = next(s for s in notebook_sources() if "#@title 6. Tune" in s)
    assert "#@param" not in tune
    assert "PRICE_PER_UNIT" not in tune


def test_the_minutes_per_run_hint_mentions_the_cost_gate():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    line = next(ln for ln in intake.splitlines() if "**MINUTES_PER_RUN**" in ln)
    assert "cost gate" in line


def test_the_notebook_is_still_twenty_one_cells():
    assert len(notebook_sources()) == 21
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_colab.py -q -k "price or currency or cost or twenty"
```
Expected: `5 failed, 2 passed` — the four new `PRICE_PER_UNIT` / `CURRENCY` / answer-key
tests fail with `AssertionError: PRICE_PER_UNIT` (and on `'train.price_per_unit'`),
`test_the_minutes_per_run_hint_mentions_the_cost_gate` fails with
`AssertionError: assert 'cost gate' in "... the assistant keeps its settings within this budget."`;
`test_the_tune_cell_gains_no_form_fields` and `test_the_notebook_is_still_twenty_one_cells`
already pass.

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_notebook.py`, replace the `MINUTES_PER_RUN` hint (lines 127-128):

```python
        "#@markdown **MINUTES_PER_RUN** — How long one training run may take. `10` is plenty "
        "for small tables; the assistant keeps its settings within this budget, and the cost "
        "gate compares each run's estimate against this before it starts.",
```

Replace the "4. Model" cell (lines 252-266) with:

```python
    code(
        "#@title 4. Model  { display-mode: 'form' }",
        "#@markdown Choose a model. The assistant explains the options for your kind of "
        "data and recommends one. If unsure, keep *Ask me after the explanation* — that "
        "works for tables and images alike.",
        "#@markdown **MODEL** — For tables: *Linear / logistic regression* is simple and "
        "easy to read, *Random forest* is robust, *Gradient boosting* is usually the most "
        "accurate. For images: *Tiny CNN* is a fast sanity check, *Small CNN* is the "
        "default, *Pretrained ResNet-18* is the most accurate but wants the GPU runtime.",
        "MODEL = 'Ask me after the explanation'  #@param "
        "['Ask me after the explanation', 'Linear / logistic regression', 'Random forest', "
        "'Gradient boosting', 'Tiny CNN', 'Small CNN', 'Pretrained ResNet-18']",
        "#@markdown **PRICE_PER_UNIT** — What one Colab compute unit costs you, so the "
        "estimate can show money. Colab Pro is $9.99 for 100 compute units, so `0.0999` "
        "per unit; check your plan in the Resources panel.",
        "PRICE_PER_UNIT = 0.0999  #@param {type:'number'}",
        "#@markdown **CURRENCY** — Shown next to the cost estimate, e.g. `$` or `GBP`.",
        "CURRENCY = '$'  #@param {type:'string'}",
        "",
        "orch.run(until='train', answers={",
        "    'codegen.model_type': MODEL,",
        "    'train.price_per_unit': PRICE_PER_UNIT,",
        "    'train.currency': CURRENCY,",
        "})",
    ),
```

No cell is added or removed: the notebook stays at 21 cells. The "6. Tune" cell keeps its
plain `orch.run(until='tune')` — by the time a tune round runs, `cost.json` holds the
price and currency, so `cost_gate.price_and_currency` never asks again.

Rebuild:

```
python scripts/build_notebook.py
```

Append a new section to `docs/colab-smoke.md`, in the style of the Milestone 6a section:

```markdown
## Milestone 6b: the cost gate

1. **Tabular on a CPU runtime: silent and free.** Fresh project, TASK *Tabular
   classification*, DATA_SOURCE *Synthetic data*, MINUTES_PER_RUN `10`. At *4. Model*
   keep the defaults and run the cell. The output shows `Timing a short dry run...`, then
   "This runtime has no GPU", the minutes/compute-units/cost table with a `$0.00` cost,
   and "A CPU runtime uses no compute units, so this run is free." — with **no question
   asked**. Run `train.py` and `evaluate.py`, then the *4. Model* cell again: the debrief
   headline ends with "Estimated N min, actual M min (P% under/over)."
2. **Images on a T4 runtime: the table and the confirm.** Runtime > Change runtime type >
   T4 GPU. Fresh project, TASK *Image classification*, GPU *T4 GPU*, N_IMAGES `300`,
   IMAGE_SIZE `64`, *Small CNN*. The gate prints `Timing a short dry run...`, names the
   Tesla T4, shows units and a non-zero cost for this run and for the remaining rounds,
   and asks *Run it?*. Choose *Run it*, run both train cells, then *4. Model* again: the
   estimate-vs-actual line appears.
3. **Over budget names the epochs that fit.** Fresh project with MINUTES_PER_RUN `1` at
   intake. At *4. Model* the gate shows an **Over budget** block naming the largest
   `epochs` value that fits the 1-minute limit, and still asks rather than refusing.
4. **Editing re-estimates without a second dry run.** At step 3's question choose *Edit
   the config first*, lower `epochs` to the suggested value, choose *Done*. A second
   table appears with the smaller number — and **no second** `Timing a short dry run...`
   line anywhere in the cell's output. Then choose *Run it*.
5. **Stop leaves nothing half-done.** On a fresh over-budget project choose *Stop*. The
   cell ends with "Change the config and run this cell again, or switch runtime, then run
   this cell again.", `orch.waiting()` returns `None`, and no `metrics.json` was written.
   Edit `config.json` by hand and run *4. Model* again: the gate asks its question again.
   Also try this in a tune round: on step 6's project apply a proposal that would be
   over budget and choose *Stop* at the gate. `config.json` still reads exactly what it did
   before the proposal — the refused proposal is not applied — and `tune_state.json`'s
   `pending` is unchanged (still `None` for a fresh round).
6. **A tune round estimates from history.** On step 2's project run *6. Tune*, apply
   proposal 1. The basis line reads "From run 1's measured time." and **no** dry run
   runs. `runs.jsonl`'s second entry carries `estimated_minutes` and `estimated_units`.
7. **The price and currency come from the form.** Fresh project: at *4. Model* set
   PRICE_PER_UNIT `0.08` and CURRENCY `GBP`. The cost column reads `0.xx GBP`, and
   afterwards `cost.json` holds `{"price_per_unit": 0.08, "currency": "GBP", ...}` with a
   `last_estimate` block. Running *6. Tune* later never re-asks for either.
```

Update `CLAUDE.md`:

- Line 7, the spec list: add `and \`docs/superpowers/specs/2026-09-11-milestone-6b-cost-gate-design.md\`` to the sentence naming the design specs.
- The Commands block (lines 11-24): add after the image-pipeline line
  ```
  python -m pytest tests/test_cost.py tests/test_cost_gate.py tests/test_template_dry_run.py -v   # the cost gate and the dry runs
  ```
- Line 31 (the Pipeline paragraph): after "`train` hands off `[["train.py"], ["evaluate.py"]]` -> `metrics.json` and `eval_val.json`", insert: "Before either `train` or a `tune` round hands off, `mlagent/stages/cost_gate.py::gate` estimates the run — from `train.py --dry-run` the first time, from the latest finished run of the same family after that — checks it against `spec.minutes_per_run`, shows minutes / [[compute units]] / money from `mlagent/rates.json`, and asks Run / Edit / Stop; a stop returns `None` from `prepare` so the stage stays current with nothing written, and the estimate is kept in `cost.json` (`price_per_unit`, `currency`, `last_estimate`) for the debrief's estimate-vs-actual line."
- Line 34 (the generated-scripts paragraph): after the `cli_argv()` sentence, add: "Both `train.py` templates also take `--dry-run`, which times a warm-up batch plus three real training batches and writes `dry_run.json` (`device`, `gpu_name`, `batches_per_epoch`, `seconds_per_batch`, `seconds_per_epoch`, `n_train`, `script`) — no `metrics.json`, no checkpoint, no figure — and is the one place a generated script may exit non-zero."
- Line 36 (the pure-modules paragraph): add `cost.py` (rates, `detect_runtime`, the estimate formula, `budget_check`, `render_estimate`) to the list of pure, unit-tested modules.

Package `mlagent/rates.json` so a plain `pip install .` ships it: add `"rates.json"` to
`[tool.setuptools.package-data]` in `pyproject.toml` (currently `mlagent = ["prompts/**/*.md",
"templates/*/*"]`):

```toml
[tool.setuptools.package-data]
mlagent = ["prompts/**/*.md", "templates/*/*", "rates.json"]
```

Check it with:

```
python -c "from mlagent.cost import load_rates; print(load_rates()['units_per_hour']['T4'])"
```
Expected: prints the T4 rate with no `FileNotFoundError`. (This check is meaningful once
the package is actually installed, e.g. inside `python -m pip install -e ".[dev]"` — add a
one-line note in this step that the editable install already picks up `rates.json` from the
source tree either way, so this check mainly guards a future non-editable / wheel install.)

Update `docs/colab-smoke.md`'s existing Milestone 3 item (~line 30) that currently says the
train stage prints a note about "no cost gate on CPU": rewrite it to describe the new estimate
line instead — the debrief headline now ends with "Estimated N min, actual M min (P%
under/over)." rather than a CPU-only disclaimer, since the cost gate (Milestone 6b) replaces
that note.

- [ ] **Step 4: Run test to verify it passes**

```
python scripts/build_notebook.py
python -m pytest tests/test_colab.py -q
python -m pytest -q
ruff check .
python -c "from mlagent.cost import load_rates; print(load_rates()['units_per_hour']['T4'])"
```
Expected: the builder prints `wrote ... (21 cells)`; `tests/test_colab.py` passes with its
seven new tests; the whole suite is green; ruff clean; the rates check prints a number.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebook.py notebooks/ML_Training_Agent.ipynb tests/test_colab.py docs/colab-smoke.md CLAUDE.md pyproject.toml
git commit -m "feat: price and currency fields in the 4. Model cell, smoke checklist and docs

The 4. Model cell gains PRICE_PER_UNIT and CURRENCY with beginner hints and
passes both to orch.run as train.price_per_unit / train.currency, so the gate
never needs a console prompt; the MINUTES_PER_RUN hint says the gate compares
against it. Seven Milestone 6b smoke items and the CLAUDE.md notes for
cost.py, cost_gate.py, rates.json and --dry-run.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

## Self-review

### 1. Spec coverage — every section, and the task that implements it

| Spec section | Implemented by |
|---|---|
| §1 `RuntimeInfo` | Task 1 |
| §1 `detect_runtime(probes=None)`, torch / nvidia-smi / none probes | Task 1 (ruling 1 on the `probes` shape) |
| §1 `mlagent/rates.json`, `load_rates`, `rate_for` + unknown-GPU note | Task 1 |
| §1 removal of `config.DEFAULT_RATES` / `PRICE_PER_100_UNITS_USD` and the `tests/test_config.py` lines | Task 1 |
| §1 `DryRunResult`, `read_dry_run` | Task 1 |
| §1 `run_dry_run(project_dir, python, timeout, env)` incl. the error paths | Task 2 |
| §1 `Estimate`, `estimate_run` formula, all-rounds figures | Task 2 |
| §1 `estimate_from_dry_run`, `estimate_from_history` | Task 2 |
| §1 `budget_check` / `BudgetAdvice`, epochs + image `batch_size`, never `image_size`, schema-presence rule | Task 2 |
| §1 `render_estimate` lines 1-7 (runtime, mismatch, table, basis, CPU-free, tabular-on-GPU, over-budget block, unknown-GPU note), `[[term]]` markup | Task 2 (ruling 2 on the signature, ruling 4 on the verb) |
| §2 `--dry-run` contract (warm-up + 3 timed batches, `dry_run.json` keys, one printed line, no metrics/figure/checkpoint, non-zero exit on failure) | Tasks 3 and 4 |
| §2 `tabular_sklearn/train.py` rework of `dry_run_timing`, docstring line 5 | Task 3 (ruling 10) |
| §2 `image_torch/train.py` argparse flag, `pick_device` reuse, `cli_argv` unchanged | Task 4 |
| §2 `# --- Dry run ---` section markers in both templates | Tasks 3 and 4 (asserted through `codewalk.split_sections`) |
| §2 "not a handoff output" | Global Constraints; `HANDOFF_COMMANDS` untouched in every task |
| §3 `gate` steps 1-7 (context, price/currency, runtime, basis, budget, display, decision incl. the edit loop and the failed-dry-run path) | Task 6 (rulings 3, 6, 8, 9) |
| §3 estimate persistence and `cost.json`'s shape | Task 6 |
| §3 `TrainStage.prepare` return type, gate placement, 6a placeholder removal | Task 7 |
| §3 `TuneStage.prepare` gate placement, no `decision` on a cost stop | Task 7 (ruling 7) |
| §3 the orchestrator's one-line change | Task 5 |
| §3 `runs.build_run_entry` estimate keys and both call sites | Task 5 |
| §3 the estimate-vs-actual line in both debriefs | Task 5 (the formatter) + Task 7 (the two call sites) |
| §3 `tune_state.json` unchanged | Task 7 — no key is added; asserted by `test_a_cost_gate_stop_ends_the_round_without_a_decision` |
| §3 level behaviour (fixed text, no extra LLM call, `preamble` untouched) | Task 2 (`render_estimate` is pure f-strings) + Task 7 (the `preamble` call is not moved) |
| §4 `scripts/build_notebook.py` "4. Model" fields, hints, answers; `MINUTES_PER_RUN` hint; 21 cells; "6. Tune" unchanged | Task 8 |
| §4 dependencies: none added | No task adds one; `torch` stays a lazy import in `cost.torch_gpu_name`, `nvidia-smi` is a subprocess |
| §4 `tests/test_cost.py` | Tasks 1 and 2 |
| §4 `tests/test_template_dry_run.py` | Tasks 3 and 4 |
| §4 `tests/test_cost_gate.py` (all eight listed paths) | Task 6 |
| §4 the two e2e tests gain the answers and the estimate-vs-actual assertion | Task 7 |
| §4 `docs/colab-smoke.md` seven items | Task 8 |
| §4 `CLAUDE.md` | Task 8 |
| Risks: 1.2 safety factor, absent `nvidia-smi`, rates as data, family mismatch falling back to a dry run | Tasks 1, 2, 6 (`cost.SAFETY_FACTOR`, `detect_runtime`'s `"none"` path, `rates.json`, `history_basis`'s `model_type` filter with its own test) |

### 2. Placeholder scan

No task contains `TBD`, `TODO`, `...` standing in for code, "add appropriate error
handling", "similar to Task N", or "write tests for the above". Every code block is
complete as written; the two places that say "unchanged" (`train`'s body after line 189 in
Task 3, `_narrative`'s `summary` dict in Task 7) name the exact existing line range being
left alone rather than eliding required new lines.

### 3. Signature consistency across tasks

- `cost.DryRunResult(device, gpu_name, batches_per_epoch, seconds_per_batch, seconds_per_epoch, n_train, script, error)` — defined Task 1, constructed in Tasks 6 and 7's fakes with exactly these keyword names, produced by Tasks 3 and 4 as the same seven JSON keys (`error` is the reader's own field, never written by a template).
- `cost.RuntimeInfo(device, gpu_name, gpu_type, source)` — Task 1; compared field-for-field in Task 1's tests and constructed in Task 2's tests.
- `cost.estimate_run(seconds_per_epoch, epochs, runtime, rates, price_per_unit, currency, rounds_remaining, basis)` — the two wrappers and every call site in Task 6 pass this order positionally.
- `cost.budget_check(estimate, minutes_per_run, config, schema, modality)` and `cost.render_estimate(estimate, advice, spec_gpu, modality, basis_run_id=None)` — one order, used identically in Tasks 2 and 6.
- `cost_gate.gate(ctx, *, rounds_remaining, probes=None, dry_runner=cost.run_dry_run)` — Task 6; called with only `rounds_remaining` from the stages (Task 7) and bound with `partial(..., probes=..., dry_runner=...)` in Tasks 6 and 7's tests, so the stage-side signature `(ctx, *, rounds_remaining) -> str` is what every fake gate in Task 7 implements.
- Questioner keys: `train.price_per_unit`, `train.currency` (Task 6, sent by Task 8's cell, asserted by Task 8's key test) and `train.cost_decision` (Task 6, dropped to `None` after the first edit per ruling 9; supplied by the e2e answers in Task 7).
- `cost.json` keys: `price_per_unit`, `currency`, `last_estimate` — written only by `cost_gate._write_cost` / `price_and_currency` (Task 6), read by `runs.last_estimate` (Task 5) and asserted in Tasks 6, 7 and 8.
- `dry_run.json` keys: the same seven in Task 3's writer, Task 4's writer, Task 1's reader and `DRY_RUN_KEYS` in Tasks 3-4's tests.
- `runs.jsonl` keys: `estimated_minutes`, `estimated_units` — added in Task 5, asserted in Tasks 5 and 7.
- Labels: `cost_gate.RUN_LABEL` / `EDIT_LABEL` / `STOP_LABEL` are referenced by name (never as literals) in every test that answers the question.

