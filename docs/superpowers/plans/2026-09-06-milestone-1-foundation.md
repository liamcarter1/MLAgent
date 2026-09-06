# ML Training Agent — Milestone 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `mlagent` package foundation so that, in Colab, the notebook can run the intake interview, write `spec.json`, explain any clicked term, and resume after a runtime reset.

**Architecture:** A pure-Python package `mlagent/` with a manual Claude tool-use loop (`llm.py`) behind an `LLM` protocol so tests use `FakeLLM`. Stages implement a small `Stage` protocol and are sequenced by `Orchestrator`, which checkpoints to `state.json`. User questions go through a `Questioner` protocol (console `input()` in Colab, scripted in tests). Assistant text is rendered to HTML with `[[term]]` markup turned into clickable spans that call back into Python to fetch a cached, project-contextual explanation.

**Tech Stack:** Python >= 3.10, `anthropic` SDK (1.x), `markdown`, `ipython`, `pytest`, `ruff`, `nbformat` (dev only, to build the notebook).

**Spec:** `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md`

## Global Constraints

- Python `>=3.10`. Package name `mlagent`. All tests run with `python -m pytest`.
- Default model id `claude-opus-5`; thinking `{"type": "adaptive"}`; `output_config={"effort": "medium"}`; `max_tokens=16000`; server-side refusal fallbacks via `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`.
- Never use assistant prefill or `tool_choice` forcing. Always parse tool inputs as dicts from the SDK, never string-match serialized JSON.
- No network in tests. Every LLM-using test uses `FakeLLM`.
- Every assistant-facing prompt lives in `mlagent/prompts/*.md` and is loaded at runtime, never inlined in Python.
- User-visible assistant text may contain `[[term]]` markup; terms are single words or short phrases, no nesting.
- Windows dev machine: paths via `pathlib`, text files written with `encoding="utf-8"`.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2
  ```

---

## File Structure (Milestone 1)

```
pyproject.toml
.gitignore
CLAUDE.md                         (update)
mlagent/
  __init__.py
  config.py        constants: model, tokens, effort, drive root, rates
  project.py       Project: folder layout + JSON helpers
  spec.py          Spec dataclass, enums, validation
  llm.py           ToolSpec, LLMResult, LLM protocol, AnthropicLLM, FakeLLM, errors
  prompts_io.py    load_prompt(name) -> str
  orchestrator.py  Orchestrator with state.json
  colab.py         setup(), start() helpers used by the notebook
  ui/
    __init__.py
    questions.py   Questioner protocol, ConsoleQuestioner, ScriptedQuestioner
    render.py      extract_terms, to_html, display_message
    explain.py     Glossary, Explainer
  stages/
    __init__.py
    base.py        StageContext, Stage protocol
    intake.py      IntakeStage
  prompts/
    intake.md
    explain.md
scripts/build_notebook.py
notebooks/ML_Training_Agent.ipynb   (generated)
docs/colab-smoke.md
tests/
  conftest.py
  test_config.py test_project.py test_spec.py test_llm.py
  test_questions.py test_render.py test_explain.py
  test_orchestrator.py test_intake.py test_colab.py
```

---

### Task 1: Repository scaffold and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `mlagent/__init__.py`, `mlagent/config.py`, `tests/conftest.py`, `tests/test_config.py`
- Delete: `main.py` (PyCharm placeholder)

**Interfaces:**
- Produces: `mlagent.config.MODEL_ID: str`, `MAX_TOKENS: int`, `EFFORT: str`, `DRIVE_ROOT: str`, `DEFAULT_RATES: dict[str, float]`, `PRICE_PER_100_UNITS_USD: float`, `STATE_FILE = "state.json"`, `SPEC_FILE = "spec.json"`, `GLOSSARY_FILE = "glossary.json"`.

- [ ] **Step 1: Initialise git and write scaffold files**

```bash
cd "C:/Users/liamc/projects/ML training agent"
git init
rm -f main.py
```

`pyproject.toml`:
```toml
[project]
name = "mlagent"
version = "0.1.0"
description = "AI/data-engineer assistant for ML training in Google Colab"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=1.0",
    "markdown>=3.5",
    "ipython>=8.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "nbformat>=5.9"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["mlagent*"]

[tool.setuptools.package-data]
mlagent = ["prompts/*.md"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/
.idea/
.venv/
venv/
.env
```

`mlagent/__init__.py`:
```python
"""mlagent: an AI/data-engineer assistant for ML training in Google Colab."""

__version__ = "0.1.0"
```

`mlagent/config.py`:
```python
"""Project-wide constants. Override the model with the MLAGENT_MODEL env var."""

import os

MODEL_ID: str = os.environ.get("MLAGENT_MODEL", "claude-opus-5")
MAX_TOKENS: int = 16000
EFFORT: str = "medium"  # low | medium | high | xhigh | max
MAX_TOOL_ROUNDS: int = 20

DRIVE_ROOT: str = "/content/drive/MyDrive/ml_agent"
PROJECTS_DIRNAME: str = "projects"

SPEC_FILE = "spec.json"
STATE_FILE = "state.json"
GLOSSARY_FILE = "glossary.json"

# Colab compute-unit consumption per hour by GPU. Conservative defaults;
# the cost gate asks the user to confirm the live figure from Colab's Resources panel.
DEFAULT_RATES: dict[str, float] = {"T4": 2.0, "L4": 4.8, "A100": 13.0}
PRICE_PER_100_UNITS_USD: float = 9.99
```

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

from mlagent.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project(tmp_path / "projects" / "demo")
    p.ensure_dirs()
    return p
```

`tests/test_config.py`:
```python
from mlagent import config


def test_defaults_are_sane():
    assert config.MODEL_ID.startswith("claude-")
    assert config.MAX_TOKENS >= 4096
    assert config.EFFORT in {"low", "medium", "high", "xhigh", "max"}
    assert set(config.DEFAULT_RATES) == {"T4", "L4", "A100"}
    assert config.PRICE_PER_100_UNITS_USD > 0
```

- [ ] **Step 2: Install and run the test**

Run:
```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_config.py -v
```
Expected: `test_config.py` errors at collection because `mlagent.project` does not exist yet (conftest imports it). That is expected; Task 2 creates it. To confirm config alone, run `python -c "from mlagent import config; print(config.MODEL_ID)"` → prints `claude-opus-5`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml .gitignore mlagent/__init__.py mlagent/config.py tests/conftest.py tests/test_config.py docs/ CLAUDE.md
git commit -m "chore: scaffold mlagent package, config, and design docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 2: Project folder and JSON helpers

**Files:**
- Create: `mlagent/project.py`, `tests/test_project.py`

**Interfaces:**
- Produces: `class Project(root: Path)` with `name: str`, `root: Path`, `spec_path`, `state_path`, `glossary_path`, `data_raw`, `data_clean`, `plots_dir`, `checkpoints_dir`, `ensure_dirs() -> None`, `read_json(filename: str, default=None) -> Any`, `write_json(filename: str, obj) -> Path`, `exists(filename: str) -> bool`.

- [ ] **Step 1: Write the failing test**

`tests/test_project.py`:
```python
import json
from pathlib import Path

from mlagent.project import Project


def test_paths_and_dirs(tmp_path: Path):
    p = Project(tmp_path / "projects" / "cats")
    assert p.name == "cats"
    p.ensure_dirs()
    assert p.data_raw.is_dir()
    assert p.data_clean.is_dir()
    assert p.plots_dir.is_dir()
    assert p.checkpoints_dir.is_dir()
    assert p.spec_path == p.root / "spec.json"
    assert p.state_path == p.root / "state.json"
    assert p.glossary_path == p.root / "glossary.json"


def test_json_roundtrip(project: Project):
    assert project.read_json("spec.json") is None
    assert project.read_json("spec.json", default={}) == {}
    path = project.write_json("spec.json", {"goal": "classify cats", "n": 3})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["goal"] == "classify cats"
    assert project.read_json("spec.json")["n"] == 3
    assert project.exists("spec.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_project.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.project'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/project.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_project.py tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/project.py tests/test_project.py
git commit -m "feat: add Project folder layout and JSON helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 3: Spec dataclass

**Files:**
- Create: `mlagent/spec.py`, `tests/test_spec.py`

**Interfaces:**
- Produces: `TASK_TYPES`, `DATA_SOURCES`, `METRICS_FOR_TASK: dict[str, list[str]]`, `GPU_CHOICES`, `@dataclass Spec(goal, task_type, metric, target_value: float, data_source, minutes_per_run: int, max_rounds: int, gpu: str, notes: str = "")`, `Spec.validate() -> list[str]` (empty list = valid), `Spec.to_dict()`, `Spec.from_dict(d)`, `class SpecError(ValueError)`.

- [ ] **Step 1: Write the failing test**

`tests/test_spec.py`:
```python
import pytest

from mlagent.spec import METRICS_FOR_TASK, Spec, SpecError


def make_spec(**overrides) -> Spec:
    base = dict(
        goal="Predict whether a customer churns",
        task_type="tabular_classification",
        metric="accuracy",
        target_value=0.9,
        data_source="synthetic",
        minutes_per_run=10,
        max_rounds=5,
        gpu="none",
    )
    base.update(overrides)
    return Spec(**base)


def test_valid_spec_roundtrips():
    s = make_spec(notes="keep it simple")
    assert s.validate() == []
    d = s.to_dict()
    assert d["task_type"] == "tabular_classification"
    assert Spec.from_dict(d) == s


def test_invalid_values_are_reported():
    s = make_spec(task_type="video", metric="bleu", data_source="ftp", gpu="H100",
                  minutes_per_run=0, max_rounds=0, goal="")
    problems = s.validate()
    assert len(problems) == 7
    assert any("task_type" in p for p in problems)


def test_metric_must_match_task():
    s = make_spec(task_type="tabular_regression", metric="accuracy")
    assert any("metric" in p for p in s.validate())
    assert "rmse" in METRICS_FOR_TASK["tabular_regression"]


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(SpecError):
        Spec.from_dict({**make_spec().to_dict(), "bogus": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.spec'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/spec.py`:
```python
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
    def from_dict(cls, d: dict[str, Any]) -> "Spec":
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/spec.py tests/test_spec.py
git commit -m "feat: add Spec dataclass with validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 4: LLM wrapper with manual tool loop and FakeLLM

**Files:**
- Create: `mlagent/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `@dataclass ToolSpec(name: str, description: str, input_schema: dict, handler: Callable[[dict], str])`
  - `@dataclass LLMResult(text: str, tool_calls: list[tuple[str, dict]], stop_reason: str)`
  - `class LLMError(RuntimeError)`, `class LLMRefused(LLMError)`
  - `class LLM(Protocol)`: `run(system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult`
  - `class AnthropicLLM(client=None, model=config.MODEL_ID, max_tokens=config.MAX_TOKENS, effort=config.EFFORT, max_rounds=config.MAX_TOOL_ROUNDS)` implementing `LLM`
  - `class FakeLLM(script: list[list[tuple]])` implementing `LLM`; each turn is a list of items `("text", str)` or `("tool", name, input_dict)`; records `calls: list[dict]` with keys `system`, `messages`, `tools`.
  - `ask_text(llm, system, prompt) -> str` convenience wrapper (no tools).

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
import pytest

from mlagent.llm import FakeLLM, LLMError, ToolSpec, ask_text


def echo_tool(store: dict) -> ToolSpec:
    def handler(inp: dict) -> str:
        store.update(inp)
        return "stored"

    return ToolSpec(
        name="store",
        description="store values",
        input_schema={"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
        handler=handler,
    )


def test_fake_llm_runs_tool_then_finishes():
    store: dict = {}
    llm = FakeLLM(script=[
        [("text", "calling"), ("tool", "store", {"k": "v"})],
        [("text", "done [[epoch]]")],
    ])
    result = llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[echo_tool(store)])
    assert store == {"k": "v"}
    assert result.text == "done [[epoch]]"
    assert result.tool_calls == [("store", {"k": "v"})]
    assert result.stop_reason == "end_turn"
    assert llm.calls[0]["system"] == "sys"
    assert [t.name for t in llm.calls[0]["tools"]] == ["store"]


def test_fake_llm_unknown_tool_raises():
    llm = FakeLLM(script=[[("tool", "nope", {})]])
    with pytest.raises(LLMError):
        llm.run(system="s", messages=[{"role": "user", "content": "x"}], tools=[])


def test_fake_llm_script_exhausted_raises():
    llm = FakeLLM(script=[])
    with pytest.raises(LLMError):
        llm.run(system="s", messages=[{"role": "user", "content": "x"}], tools=[])


def test_ask_text_returns_plain_text():
    llm = FakeLLM(script=[[("text", "an answer")]])
    assert ask_text(llm, "sys", "what?") == "an answer"
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "what?"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.llm'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/llm.py`:
```python
"""Claude API wrapper: a manual tool-use loop behind a small protocol, plus a fake for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from mlagent import config


class LLMError(RuntimeError):
    pass


class LLMRefused(LLMError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class LLMResult:
    text: str
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLM(Protocol):
    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult: ...


def _dispatch(name: str, inp: dict, tools: list[ToolSpec]) -> tuple[str, bool]:
    """Run a tool handler. Returns (content, is_error)."""
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        raise LLMError(f"model called unknown tool {name!r}")
    try:
        return by_name[name].handler(dict(inp)), False
    except Exception as exc:  # tool errors go back to the model, not up the stack
        return f"Error: {exc}", True


class AnthropicLLM:
    def __init__(
        self,
        client: Any = None,
        model: str = config.MODEL_ID,
        max_tokens: int = config.MAX_TOKENS,
        effort: str = config.EFFORT,
        max_rounds: int = config.MAX_TOOL_ROUNDS,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_rounds = max_rounds

    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult:
        history = list(messages)
        tool_calls: list[tuple[str, dict]] = []
        last_text = ""
        for _ in range(self.max_rounds):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=history,
                tools=[t.to_api() for t in tools],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            texts = [b.text for b in response.content if b.type == "text"]
            if texts:
                last_text = "\n".join(texts)
            if response.stop_reason == "refusal":
                raise LLMRefused("the model declined this request")
            if response.stop_reason == "max_tokens":
                raise LLMError("response truncated at max_tokens")
            if response.stop_reason == "pause_turn":
                history.append({"role": "assistant", "content": response.content})
                continue
            uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not uses:
                return LLMResult(text=last_text, tool_calls=tool_calls, stop_reason="end_turn")
            history.append({"role": "assistant", "content": response.content})
            results = []
            for use in uses:
                content, is_error = _dispatch(use.name, use.input, tools)
                tool_calls.append((use.name, dict(use.input)))
                block = {"type": "tool_result", "tool_use_id": use.id, "content": content}
                if is_error:
                    block["is_error"] = True
                results.append(block)
            history.append({"role": "user", "content": results})
        raise LLMError(f"tool loop exceeded {self.max_rounds} rounds")


class FakeLLM:
    """Scripted stand-in. Each turn: list of ("text", str) or ("tool", name, input)."""

    def __init__(self, script: list[list[tuple]]):
        self.script = list(script)
        self.calls: list[dict] = []

    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        tool_calls: list[tuple[str, dict]] = []
        last_text = ""
        while True:
            if not self.script:
                raise LLMError("FakeLLM script exhausted")
            turn = self.script.pop(0)
            uses = []
            for item in turn:
                if item[0] == "text":
                    last_text = item[1]
                elif item[0] == "tool":
                    uses.append((item[1], item[2]))
                else:
                    raise LLMError(f"bad FakeLLM item {item!r}")
            if not uses:
                return LLMResult(text=last_text, tool_calls=tool_calls, stop_reason="end_turn")
            for name, inp in uses:
                _dispatch(name, inp, tools)
                tool_calls.append((name, dict(inp)))


def ask_text(llm: LLM, system: str, prompt: str) -> str:
    return llm.run(system=system, messages=[{"role": "user", "content": prompt}], tools=[]).text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/llm.py tests/test_llm.py
git commit -m "feat: add LLM protocol, Anthropic tool loop, and FakeLLM

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 5: Questioner (console and scripted)

**Files:**
- Create: `mlagent/ui/__init__.py` (empty), `mlagent/ui/questions.py`, `tests/test_questions.py`

**Interfaces:**
- Produces: `class Questioner(Protocol)`: `choice(question: str, options: list[str], allow_other: bool = True) -> str`, `text(prompt: str, default: str | None = None) -> str`, `confirm(question: str, default: bool = True) -> bool`.
  `class ConsoleQuestioner(input_fn=input, print_fn=print)`; `class ScriptedQuestioner(answers: list[str])` with `.asked: list[str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_questions.py`:
```python
import pytest

from mlagent.ui.questions import ConsoleQuestioner, ScriptedQuestioner


def make_console(inputs: list[str]):
    printed: list[str] = []
    it = iter(inputs)
    q = ConsoleQuestioner(input_fn=lambda _prompt="": next(it), print_fn=printed.append)
    return q, printed


def test_console_choice_by_number_and_text():
    q, printed = make_console(["2"])
    assert q.choice("Pick", ["alpha", "beta"]) == "beta"
    assert any("1) alpha" in line for line in printed)
    q, _ = make_console(["alpha"])
    assert q.choice("Pick", ["alpha", "beta"]) == "alpha"


def test_console_choice_other_and_retry():
    q, printed = make_console(["9", "my own"])
    assert q.choice("Pick", ["alpha", "beta"], allow_other=True) == "my own"
    q, printed = make_console(["9", "1"])
    assert q.choice("Pick", ["alpha", "beta"], allow_other=False) == "alpha"
    assert any("Please" in line for line in printed)


def test_console_text_default_and_confirm():
    q, _ = make_console([""])
    assert q.text("Name?", default="demo") == "demo"
    q, _ = make_console(["n"])
    assert q.confirm("Go?") is False
    q, _ = make_console([""])
    assert q.confirm("Go?", default=True) is True


def test_scripted_records_and_exhausts():
    q = ScriptedQuestioner(["beta", "hello", "y"])
    assert q.choice("Pick", ["alpha", "beta"]) == "beta"
    assert q.text("Say") == "hello"
    assert q.confirm("Ok?") is True
    assert q.asked == ["Pick", "Say", "Ok?"]
    with pytest.raises(RuntimeError):
        q.text("more")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_questions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.ui'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/ui/__init__.py`: empty file.

`mlagent/ui/questions.py`:
```python
"""Ask the user questions. Console version works in Colab via the inline input() box."""

from __future__ import annotations

from typing import Callable, Protocol


class Questioner(Protocol):
    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str: ...
    def text(self, prompt: str, default: str | None = None) -> str: ...
    def confirm(self, question: str, default: bool = True) -> bool: ...


class ConsoleQuestioner:
    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print):
        self._input = input_fn
        self._print = print_fn

    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str:
        self._print(question)
        for i, opt in enumerate(options, 1):
            self._print(f"  {i}) {opt}")
        hint = "number or text" + (", or type your own answer" if allow_other else "")
        while True:
            raw = self._input(f"[{hint}] > ").strip()
            if raw.isdigit():
                if 1 <= int(raw) <= len(options):
                    return options[int(raw) - 1]
                self._print(f"Please enter a number from 1 to {len(options)}.")
                continue
            for opt in options:
                if raw.lower() == opt.lower():
                    return opt
            if allow_other and raw:
                return raw
            self._print(f"Please enter a number from 1 to {len(options)}.")

    def text(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            raw = self._input(f"{prompt}{suffix} > ").strip()
            if raw:
                return raw
            if default is not None:
                return default
            self._print("Please enter a value.")

    def confirm(self, question: str, default: bool = True) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = self._input(f"{question}{suffix} > ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}


class ScriptedQuestioner:
    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.asked: list[str] = []

    def _next(self, question: str) -> str:
        self.asked.append(question)
        if not self._answers:
            raise RuntimeError(f"ScriptedQuestioner has no answer for: {question}")
        return self._answers.pop(0)

    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str:
        return self._next(question)

    def text(self, prompt: str, default: str | None = None) -> str:
        answer = self._next(prompt)
        return answer if answer or default is None else default

    def confirm(self, question: str, default: bool = True) -> bool:
        answer = self._next(question).strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes", "true"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_questions.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/ui/__init__.py mlagent/ui/questions.py tests/test_questions.py
git commit -m "feat: add Questioner protocol with console and scripted implementations

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 6: Renderer with clickable `[[term]]` markup

**Files:**
- Create: `mlagent/ui/render.py`, `tests/test_render.py`

**Interfaces:**
- Produces: `TERM_RE`, `extract_terms(text: str) -> list[str]` (unique, in order), `strip_terms(text: str) -> str`, `to_html(text: str) -> str`, `display_message(text: str) -> None` (uses `IPython.display.display(HTML(...))`), `COLAB_CLICK_JS: str`, `CSS: str`.
- HTML contract: each term becomes `<span class="mlagent-term" data-term="learning rate">learning rate</span>`; the JS calls `google.colab.kernel.invokeFunction('mlagent.explain', [term], {})` when `window.google?.colab` exists, else does nothing.

- [ ] **Step 1: Write the failing test**

`tests/test_render.py`:
```python
from mlagent.ui import render


def test_extract_and_strip_terms():
    text = "Lower the [[learning rate]] to stop [[overfitting]]; [[learning rate]] again."
    assert render.extract_terms(text) == ["learning rate", "overfitting"]
    assert render.strip_terms(text) == "Lower the learning rate to stop overfitting; learning rate again."


def test_to_html_marks_terms_and_renders_markdown():
    html = render.to_html("**Bold** and a [[batch size]] term\n\n- item")
    assert '<span class="mlagent-term" data-term="batch size">batch size</span>' in html
    assert "<strong>Bold</strong>" in html
    assert "<li>item</li>" in html
    assert "mlagent.explain" in html  # click hook present
    assert "<style>" in html


def test_to_html_escapes_html_in_terms():
    html = render.to_html("[[<b>x</b>]]")
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_display_message_calls_ipython(monkeypatch):
    shown: list = []
    monkeypatch.setattr(render, "_display", lambda obj: shown.append(obj))
    render.display_message("hello [[epoch]]")
    assert len(shown) == 1
    assert "epoch" in shown[0].data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render.py -v`
Expected: FAIL with `ImportError: cannot import name 'render'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/ui/render.py`:
```python
"""Render assistant messages: markdown plus [[term]] markup -> HTML with clickable terms."""

from __future__ import annotations

import html
import re

import markdown as _markdown

TERM_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

CSS = """<style>
.mlagent-msg { font-family: system-ui, sans-serif; line-height: 1.5; max-width: 60rem; }
.mlagent-term { border-bottom: 1px dotted #2a7; color: #2a7; cursor: pointer; }
.mlagent-term:hover { background: rgba(34,170,119,0.12); }
.mlagent-explain { border-left: 3px solid #2a7; padding: 0.5rem 0.75rem; margin: 0.5rem 0; background: rgba(34,170,119,0.06); }
</style>"""

COLAB_CLICK_JS = """<script>
(function(){
  if (window.__mlagentClickBound) return;
  window.__mlagentClickBound = true;
  document.addEventListener('click', function(ev){
    var el = ev.target.closest && ev.target.closest('.mlagent-term');
    if (!el) return;
    var term = el.getAttribute('data-term');
    if (window.google && window.google.colab && window.google.colab.kernel) {
      window.google.colab.kernel.invokeFunction('mlagent.explain', [term], {});
    } else {
      console.log('mlagent explain (no Colab kernel):', term);
    }
  });
})();
</script>"""


def extract_terms(text: str) -> list[str]:
    seen: list[str] = []
    for m in TERM_RE.finditer(text):
        term = m.group(1).strip()
        if term and term not in seen:
            seen.append(term)
    return seen


def strip_terms(text: str) -> str:
    return TERM_RE.sub(lambda m: m.group(1).strip(), text)


def _term_span(m: re.Match) -> str:
    term = html.escape(m.group(1).strip(), quote=True)
    return f'<span class="mlagent-term" data-term="{term}">{term}</span>'


def to_html(text: str) -> str:
    # Protect spans from the markdown processor by inserting them after conversion.
    placeholders: dict[str, str] = {}

    def stash(m: re.Match) -> str:
        key = f"MLAGENTTERM{len(placeholders)}X"
        placeholders[key] = _term_span(m)
        return key

    body = _markdown.markdown(TERM_RE.sub(stash, text))
    for key, span in placeholders.items():
        body = body.replace(key, span)
    return f'{CSS}<div class="mlagent-msg">{body}</div>{COLAB_CLICK_JS}'


def _display(obj) -> None:  # separated so tests can monkeypatch
    from IPython.display import display

    display(obj)


def display_message(text: str) -> None:
    from IPython.display import HTML

    _display(HTML(to_html(text)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/ui/render.py tests/test_render.py
git commit -m "feat: render markdown with clickable [[term]] spans

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 7: Prompt loader, glossary, and explainer

**Files:**
- Create: `mlagent/prompts_io.py`, `mlagent/prompts/explain.md`, `mlagent/ui/explain.py`, `tests/test_explain.py`

**Interfaces:**
- Produces: `load_prompt(name: str) -> str` (reads `mlagent/prompts/<name>.md`).
  `class Glossary(path: Path)`: `get(term) -> str | None`, `set(term, text) -> None` (saves immediately), `terms() -> list[str]`.
  `class Explainer(llm: LLM, glossary: Glossary, context_provider: Callable[[], dict], display: Callable[[str], None] = display_message)`: `explain(term: str, refresh: bool = False) -> str`, `show(term: str, refresh: bool = False) -> None` (explain then display in a panel), `register_colab_callback() -> bool` (True if registered).
- Explanation text may itself contain `[[term]]` markup so users can keep drilling down.

- [ ] **Step 1: Write the failing test**

`tests/test_explain.py`:
```python
import json

from mlagent.llm import FakeLLM
from mlagent.prompts_io import load_prompt
from mlagent.ui.explain import Explainer, Glossary


def test_load_prompt_reads_markdown():
    text = load_prompt("explain")
    assert "{term}" in text and "{context}" in text


def test_glossary_persists(tmp_path):
    g = Glossary(tmp_path / "glossary.json")
    assert g.get("epoch") is None
    g.set("epoch", "one pass over the data")
    assert Glossary(tmp_path / "glossary.json").get("epoch") == "one pass over the data"
    assert g.terms() == ["epoch"]
    assert json.loads((tmp_path / "glossary.json").read_text(encoding="utf-8"))["epoch"]


def test_explainer_caches_and_uses_context(tmp_path):
    llm = FakeLLM(script=[[("text", "An [[epoch]] is one pass.")]])
    g = Glossary(tmp_path / "glossary.json")
    shown: list[str] = []
    ex = Explainer(llm, g, context_provider=lambda: {"stage": "intake", "task_type": "tabular_classification"},
                   display=shown.append)
    assert ex.explain("epoch") == "An [[epoch]] is one pass."
    assert ex.explain("epoch") == "An [[epoch]] is one pass."  # cached, no second call
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "epoch" in prompt and "tabular_classification" in prompt
    ex.show("epoch")
    assert shown and "epoch" in shown[0]


def test_explainer_refresh_reasks(tmp_path):
    llm = FakeLLM(script=[[("text", "v1")], [("text", "v2")]])
    ex = Explainer(llm, Glossary(tmp_path / "g.json"), context_provider=dict, display=lambda s: None)
    assert ex.explain("loss") == "v1"
    assert ex.explain("loss", refresh=True) == "v2"


def test_register_colab_callback_without_colab_returns_false(tmp_path):
    ex = Explainer(FakeLLM([]), Glossary(tmp_path / "g.json"), context_provider=dict, display=lambda s: None)
    assert ex.register_colab_callback() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.prompts_io'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/prompts_io.py`:
```python
"""Load system prompts from mlagent/prompts/*.md."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
```

`mlagent/prompts/explain.md`:
```markdown
You are a patient machine-learning mentor embedded in a Google Colab training assistant.
The user clicked a technical term and wants to understand it in the context of their current project.

Explain the term "{term}" using this structure, in plain language, under 200 words:

1. **What it is** — a one or two sentence definition.
2. **Why it matters here** — its relevance to this project right now, using the context below.
3. **Why this value or choice** — if the context shows a current value or setting for it, say why that is a sensible starting point; otherwise say what a typical starting point is.
4. **What changes if you alter it** — the effect of increasing or decreasing it, or choosing an alternative.
5. **Read more** — one short pointer (a well-known doc page or book chapter), no URL required.

Wrap other technical terms you use in double square brackets like [[learning rate]] so the user can click them too. Do not wrap the term being explained.

Project context (JSON):
{context}
```

`mlagent/ui/explain.py`:
```python
"""Click-to-explain: cached, project-contextual explanations of technical terms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mlagent.llm import LLM, ask_text
from mlagent.prompts_io import load_prompt
from mlagent.ui.render import display_message


class Glossary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, term: str) -> str | None:
        return self._data.get(term.strip().lower())

    def set(self, term: str, text: str) -> None:
        self._data[term.strip().lower()] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def terms(self) -> list[str]:
        return sorted(self._data)


class Explainer:
    CALLBACK_NAME = "mlagent.explain"

    def __init__(
        self,
        llm: LLM,
        glossary: Glossary,
        context_provider: Callable[[], dict],
        display: Callable[[str], None] = display_message,
    ):
        self.llm = llm
        self.glossary = glossary
        self.context_provider = context_provider
        self.display = display

    def explain(self, term: str, refresh: bool = False) -> str:
        if not refresh:
            cached = self.glossary.get(term)
            if cached:
                return cached
        context = json.dumps(self.context_provider(), indent=2, sort_keys=True, default=str)
        system = load_prompt("explain").format(term=term, context=context)
        text = ask_text(self.llm, system, f"Explain: {term}\n\nContext:\n{context}")
        self.glossary.set(term, text)
        return text

    def show(self, term: str, refresh: bool = False) -> None:
        text = self.explain(term, refresh=refresh)
        self.display(f"### {term}\n\n{text}")

    def _on_click(self, term: str) -> None:
        self.show(term)

    def register_colab_callback(self) -> bool:
        try:
            from google.colab import output  # type: ignore
        except Exception:
            return False
        output.register_callback(self.CALLBACK_NAME, self._on_click)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_explain.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/prompts_io.py mlagent/prompts/explain.md mlagent/ui/explain.py tests/test_explain.py
git commit -m "feat: add glossary and click-to-explain Explainer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 8: Stage protocol and Orchestrator with checkpointing

**Files:**
- Create: `mlagent/stages/__init__.py` (empty), `mlagent/stages/base.py`, `mlagent/orchestrator.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Produces:
  - `@dataclass StageContext(project: Project, llm: LLM, questioner: Questioner, explainer: Explainer | None, display: Callable[[str], None])`
  - `class Stage(Protocol)`: `name: str`; `run(ctx: StageContext) -> None`; `is_complete(ctx: StageContext) -> bool`
  - `class Orchestrator(ctx: StageContext, stages: list[Stage])`: `completed() -> list[str]`, `run(until: str | None = None) -> list[str]` (returns names run this call), `reset(stage_name: str) -> None` (drops that stage and all later ones from completed), `mark_complete(name)`.
  - `state.json` shape: `{"completed": ["intake", ...], "current": "data" | null, "forced": [...]}`. `forced` lists stages that must rerun even though their artifact exists (set by `reset`, cleared when the stage completes).

- [ ] **Step 1: Write the failing test**

`tests/test_orchestrator.py`:
```python
import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.stages.base import StageContext
from mlagent.ui.questions import ScriptedQuestioner


class RecordingStage:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.runs = 0

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        if self.fail:
            raise RuntimeError("boom")
        ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


def make_ctx(project):
    return StageContext(project=project, llm=FakeLLM([]), questioner=ScriptedQuestioner([]),
                        explainer=None, display=lambda s: None)


def test_runs_stages_in_order_and_checkpoints(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["a", "b"]
    assert orch.completed() == ["a", "b"]
    assert project.read_json("state.json")["completed"] == ["a", "b"]
    assert project.read_json("state.json")["current"] is None


def test_resumes_after_restart(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    Orchestrator(make_ctx(project), [a, b]).run(until="a")
    a2, b2 = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a2, b2])
    assert orch.run() == ["b"]
    assert a2.runs == 0 and b2.runs == 1


def test_failure_leaves_current_and_does_not_mark_complete(project):
    a, b = RecordingStage("a"), RecordingStage("b", fail=True)
    orch = Orchestrator(make_ctx(project), [a, b])
    with pytest.raises(RuntimeError):
        orch.run()
    state = project.read_json("state.json")
    assert state["completed"] == ["a"]
    assert state["current"] == "b"


def test_reset_drops_later_stages(project):
    a, b, c = RecordingStage("a"), RecordingStage("b"), RecordingStage("c")
    orch = Orchestrator(make_ctx(project), [a, b, c])
    orch.run()
    orch.reset("b")
    assert orch.completed() == ["a"]
    assert orch.run() == ["b", "c"]


def test_stage_with_artifact_but_no_state_is_skipped(project):
    project.write_json("a.json", {"ok": True})
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["b"]
    assert a.runs == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.orchestrator'`

- [ ] **Step 3: Write minimal implementation**

`mlagent/stages/__init__.py`: empty file.

`mlagent/stages/base.py`:
```python
"""Shared types for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from mlagent.llm import LLM
from mlagent.project import Project
from mlagent.ui.explain import Explainer
from mlagent.ui.questions import Questioner


@dataclass
class StageContext:
    project: Project
    llm: LLM
    questioner: Questioner
    explainer: Optional[Explainer]
    display: Callable[[str], None]


class Stage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None: ...

    def is_complete(self, ctx: StageContext) -> bool: ...
```

`mlagent/orchestrator.py`:
```python
"""Run stages in order, checkpointing to state.json so a Colab reset can resume."""

from __future__ import annotations

from mlagent import config
from mlagent.stages.base import Stage, StageContext


class Orchestrator:
    def __init__(self, ctx: StageContext, stages: list[Stage]):
        self.ctx = ctx
        self.stages = list(stages)

    def _state(self) -> dict:
        state = self.ctx.project.read_json(config.STATE_FILE, default=None)
        if state is None:
            return {"completed": [], "current": None, "forced": []}
        state.setdefault("forced", [])
        return state

    def _save(self, state: dict) -> None:
        self.ctx.project.write_json(config.STATE_FILE, state)

    def completed(self) -> list[str]:
        return list(self._state()["completed"])

    def mark_complete(self, name: str) -> None:
        state = self._state()
        if name not in state["completed"]:
            state["completed"].append(name)
        state["forced"] = [n for n in state["forced"] if n != name]
        state["current"] = None
        self._save(state)

    def reset(self, stage_name: str) -> None:
        """Forget this stage and every later one; they rerun even if their artifacts exist."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            raise ValueError(f"unknown stage {stage_name!r}; known: {names}")
        idx = names.index(stage_name)
        state = self._state()
        state["completed"] = [n for n in state["completed"] if n in names[:idx]]
        state["forced"] = names[idx:]
        state["current"] = None
        self._save(state)

    def run(self, until: str | None = None) -> list[str]:
        ran: list[str] = []
        for stage in self.stages:
            state = self._state()
            done = stage.name in state["completed"]
            forced = stage.name in state["forced"]
            if not done and not forced and stage.is_complete(self.ctx):
                self.mark_complete(stage.name)
                done = True
            if not done:
                state = self._state()
                state["current"] = stage.name
                self._save(state)
                self.ctx.display(f"**Stage: {stage.name}**")
                stage.run(self.ctx)
                self.mark_complete(stage.name)
                ran.append(stage.name)
            if until is not None and stage.name == until:
                break
        return ran
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/__init__.py mlagent/stages/base.py mlagent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add Stage protocol and checkpointing Orchestrator

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 9: Intake stage

**Files:**
- Create: `mlagent/stages/intake.py`, `mlagent/prompts/intake.md`, `tests/test_intake.py`

**Interfaces:**
- Consumes: `Questioner`, `LLM.run`, `ToolSpec`, `Spec`, `StageContext`, `load_prompt`.
- Produces: `class IntakeStage` with `name = "intake"`, `run(ctx)`, `is_complete(ctx)` (true when `spec.json` exists and parses as a valid `Spec`), and module function `collect_draft(q: Questioner) -> dict` (the fixed interview).
- LLM tools exposed during intake: `ask_user(question: str, options: list[str] | null) -> str` and `write_spec(<all Spec fields>) -> "ok" | "invalid: ..."`.

Behaviour: fixed interview → draft dict → one LLM run with the intake prompt, the draft as JSON, and the two tools. Claude may ask up to 2 follow-ups via `ask_user`, then must call `write_spec`. If it never calls `write_spec`, the stage validates and writes the draft itself. Finally display a summary.

- [ ] **Step 1: Write the failing test**

`tests/test_intake.py`:
```python
import pytest

from mlagent.llm import FakeLLM
from mlagent.spec import Spec
from mlagent.stages.base import StageContext
from mlagent.stages.intake import IntakeStage, collect_draft
from mlagent.ui.questions import ScriptedQuestioner

ANSWERS = [
    "Predict customer churn from account data",  # goal
    "Tabular classification",                     # task type label
    "accuracy",                                   # metric
    "0.9",                                        # target
    "Synthetic data",                             # data source label
    "10",                                         # minutes per run
    "5",                                          # max rounds
    "No GPU (CPU only)",                          # gpu label
]


def make_ctx(project, llm, answers):
    shown: list[str] = []
    ctx = StageContext(project=project, llm=llm, questioner=ScriptedQuestioner(answers),
                       explainer=None, display=shown.append)
    return ctx, shown


def test_collect_draft_maps_labels_to_codes():
    draft = collect_draft(ScriptedQuestioner(ANSWERS))
    assert draft["task_type"] == "tabular_classification"
    assert draft["data_source"] == "synthetic"
    assert draft["gpu"] == "none"
    assert draft["target_value"] == 0.9
    assert draft["minutes_per_run"] == 10 and draft["max_rounds"] == 5


def test_intake_writes_spec_via_tool(project):
    spec_fields = dict(goal="Predict customer churn from account data", task_type="tabular_classification",
                       metric="accuracy", target_value=0.9, data_source="synthetic",
                       minutes_per_run=10, max_rounds=5, gpu="none", notes="binary target")
    llm = FakeLLM(script=[
        [("tool", "ask_user", {"question": "Is churn binary?", "options": ["yes", "no"]})],
        [("tool", "write_spec", spec_fields)],
        [("text", "Spec saved. Next we get [[training data]].")],
    ])
    ctx, shown = make_ctx(project, llm, ANSWERS + ["yes"])
    stage = IntakeStage()
    assert not stage.is_complete(ctx)
    stage.run(ctx)
    assert stage.is_complete(ctx)
    saved = Spec.from_dict(project.read_json("spec.json"))
    assert saved.notes == "binary target"
    assert "Is churn binary?" in ctx.questioner.asked
    assert any("[[training data]]" in s for s in shown)
    assert "intake" in llm.calls[0]["system"].lower()
    assert "churn" in llm.calls[0]["messages"][0]["content"]


def test_intake_falls_back_to_draft_when_model_writes_nothing(project):
    llm = FakeLLM(script=[[("text", "Looks good.")]])
    ctx, _ = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    saved = Spec.from_dict(project.read_json("spec.json"))
    assert saved.goal.startswith("Predict customer churn")


def test_write_spec_tool_rejects_invalid_and_model_can_retry(project):
    bad = dict(goal="x", task_type="tabular_regression", metric="accuracy", target_value=1,
               data_source="synthetic", minutes_per_run=5, max_rounds=2, gpu="none")
    good = {**bad, "metric": "rmse"}
    llm = FakeLLM(script=[[("tool", "write_spec", bad)], [("tool", "write_spec", good)], [("text", "ok")]])
    ctx, _ = make_ctx(project, llm, ANSWERS)
    IntakeStage().run(ctx)
    assert Spec.from_dict(project.read_json("spec.json")).metric == "rmse"


def test_is_complete_false_for_corrupt_spec(project):
    project.write_json("spec.json", {"goal": "only"})
    ctx, _ = make_ctx(project, FakeLLM([]), [])
    assert IntakeStage().is_complete(ctx) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_intake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.stages.intake'`

- [ ] **Step 3: Write the prompt and implementation**

`mlagent/prompts/intake.md`:
```markdown
You are the intake stage of an ML training assistant that runs inside Google Colab. Your job is to turn the user's interview answers into a precise project specification.

You receive a JSON draft of the user's answers. Review it for gaps or contradictions (for example a regression task with an accuracy metric, an unrealistic target, or a goal that does not match the task type).

Rules:
- You may call `ask_user` at most twice, only when an answer is missing, contradictory, or too vague to act on. Offer options when the question has a small set of sensible answers.
- Then call `write_spec` exactly once with the final values. Keep the user's wording for `goal`. Put anything useful you learned into `notes` (for example the positive class, units of the target, or constraints).
- If `write_spec` returns an error, fix the values and call it again.
- Finish with a short message (under 120 words) that summarises the spec in plain language and says what happens next: obtaining the data. Wrap technical terms in double square brackets like [[validation set]] so the user can click them for an explanation.
```

`mlagent/stages/intake.py`:
```python
"""Intake: fixed interview -> draft -> Claude checks it and writes spec.json."""

from __future__ import annotations

import json

from mlagent import config
from mlagent.llm import ToolSpec
from mlagent.prompts_io import load_prompt
from mlagent.spec import DATA_SOURCES, GPU_CHOICES, METRICS_FOR_TASK, TASK_TYPES, Spec, SpecError
from mlagent.stages.base import StageContext
from mlagent.ui.questions import Questioner

TASK_LABELS = {
    "Tabular classification": "tabular_classification",
    "Tabular regression": "tabular_regression",
    "Image classification": "image_classification",
}
SOURCE_LABELS = {
    "Synthetic data": "synthetic",
    "Upload or Google Drive path": "drive",
    "HuggingFace Hub dataset": "huggingface",
}
GPU_LABELS = {
    "No GPU (CPU only)": "none",
    "T4 GPU": "T4",
    "Any available GPU": "any",
}


def _label_to_code(answer: str, mapping: dict[str, str], allowed: tuple[str, ...]) -> str:
    if answer in mapping:
        return mapping[answer]
    if answer in allowed:
        return answer
    return mapping[list(mapping)[0]]


def _to_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except ValueError:
        return default


def _to_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except ValueError:
        return default


def collect_draft(q: Questioner) -> dict:
    goal = q.text("In one or two sentences, what do you want the model to do?")
    task_type = _label_to_code(q.choice("What kind of task is it?", list(TASK_LABELS), allow_other=False),
                               TASK_LABELS, TASK_TYPES)
    metric = q.choice("Which metric defines success?", METRICS_FOR_TASK[task_type], allow_other=False)
    target_value = _to_float(q.text(f"What {metric} value would count as good enough?", default="0.9"), 0.9)
    data_source = _label_to_code(q.choice("Where will the data come from?", list(SOURCE_LABELS), allow_other=False),
                                 SOURCE_LABELS, DATA_SOURCES)
    minutes = _to_int(q.text("Roughly how many minutes per training run are acceptable?", default="10"), 10)
    rounds = _to_int(q.text("How many tuning rounds at most?", default="5"), 5)
    gpu = _label_to_code(q.choice("GPU preference?", list(GPU_LABELS), allow_other=False), GPU_LABELS, GPU_CHOICES)
    return {
        "goal": goal,
        "task_type": task_type,
        "metric": metric,
        "target_value": target_value,
        "data_source": data_source,
        "minutes_per_run": minutes,
        "max_rounds": rounds,
        "gpu": gpu,
        "notes": "",
    }


SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "metric": {"type": "string"},
        "target_value": {"type": "number"},
        "data_source": {"type": "string", "enum": list(DATA_SOURCES)},
        "minutes_per_run": {"type": "integer"},
        "max_rounds": {"type": "integer"},
        "gpu": {"type": "string", "enum": list(GPU_CHOICES)},
        "notes": {"type": "string"},
    },
    "required": ["goal", "task_type", "metric", "target_value", "data_source",
                 "minutes_per_run", "max_rounds", "gpu"],
}


class IntakeStage:
    name = "intake"
    MAX_FOLLOWUPS = 2

    def is_complete(self, ctx: StageContext) -> bool:
        data = ctx.project.read_json(config.SPEC_FILE)
        if not data:
            return False
        try:
            return Spec.from_dict(data).validate() == []
        except (SpecError, TypeError, ValueError):
            return False

    def run(self, ctx: StageContext) -> None:
        draft = collect_draft(ctx.questioner)
        written: dict = {}
        followups = {"n": 0}

        def ask_user(inp: dict) -> str:
            if followups["n"] >= self.MAX_FOLLOWUPS:
                return "No more follow-up questions allowed; call write_spec now."
            followups["n"] += 1
            options = inp.get("options")
            if options:
                return ctx.questioner.choice(inp["question"], [str(o) for o in options])
            return ctx.questioner.text(inp["question"])

        def write_spec(inp: dict) -> str:
            try:
                spec = Spec.from_dict(inp)
            except SpecError as exc:
                return f"invalid: {exc}"
            problems = spec.validate()
            if problems:
                return "invalid: " + "; ".join(problems)
            ctx.project.write_json(config.SPEC_FILE, spec.to_dict())
            written.update(spec.to_dict())
            return "ok"

        tools = [
            ToolSpec(
                name="ask_user",
                description="Ask the user one clarifying question. Provide options when sensible.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
                handler=ask_user,
            ),
            ToolSpec(
                name="write_spec",
                description="Save the final project specification. Returns 'ok' or 'invalid: ...'.",
                input_schema=SPEC_SCHEMA,
                handler=write_spec,
            ),
        ]
        prompt = "Interview answers (draft spec):\n" + json.dumps(draft, indent=2, sort_keys=True)
        result = ctx.llm.run(system=load_prompt("intake"), messages=[{"role": "user", "content": prompt}],
                             tools=tools)
        if not written:
            spec = Spec.from_dict(draft)
            problems = spec.validate()
            if problems:
                raise SpecError("draft spec invalid: " + "; ".join(problems))
            ctx.project.write_json(config.SPEC_FILE, spec.to_dict())
        if result.text:
            ctx.display(result.text)
        else:
            ctx.display("Spec saved. Next: obtaining the [[training data]].")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_intake.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mlagent/stages/intake.py mlagent/prompts/intake.md tests/test_intake.py
git commit -m "feat: add intake stage with fixed interview and Claude follow-ups

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

### Task 10: Colab entry points, notebook template, docs

**Files:**
- Create: `mlagent/colab.py`, `scripts/build_notebook.py`, `notebooks/ML_Training_Agent.ipynb` (generated), `docs/colab-smoke.md`, `tests/test_colab.py`
- Modify: `CLAUDE.md` (replace placeholder sections with real commands and architecture)

**Interfaces:**
- Produces: `colab.setup(drive_root: str = config.DRIVE_ROOT, mount: bool = True) -> Path` (mounts Drive if possible, adds `drive_root` to `sys.path`, copies `ANTHROPIC_API_KEY` from Colab Secrets to the environment if not already set, returns the projects dir), `colab.make_context(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> StageContext`, `colab.start(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> Orchestrator` (builds context, registers the explain callback, returns the orchestrator with `[IntakeStage()]`), `colab.explain(term: str, refresh: bool = False) -> None` (uses the most recent context).

- [ ] **Step 1: Write the failing test**

`tests/test_colab.py`:
```python
from pathlib import Path

from mlagent import colab
from mlagent.llm import FakeLLM
from mlagent.ui.questions import ScriptedQuestioner


def test_setup_without_colab_adds_path_and_makes_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    projects = colab.setup(drive_root=str(tmp_path), mount=False)
    assert projects == tmp_path / "projects"
    assert projects.is_dir()
    import sys

    assert str(tmp_path) in sys.path


def test_make_context_and_start(tmp_path):
    ctx = colab.make_context("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert ctx.project.root == tmp_path / "projects" / "demo"
    assert ctx.project.root.is_dir()
    assert ctx.explainer is not None
    orch = colab.start("demo", drive_root=str(tmp_path), llm=FakeLLM([]))
    assert [s.name for s in orch.stages] == ["intake"]


def test_explain_uses_last_context(tmp_path, monkeypatch):
    llm = FakeLLM(script=[[("text", "One pass over the data.")]])
    shown: list[str] = []
    orch = colab.start("demo", drive_root=str(tmp_path), llm=llm)
    orch.ctx.questioner = ScriptedQuestioner([])
    orch.ctx.explainer.display = shown.append
    colab.explain("epoch")
    assert shown and "One pass" in shown[0]
    assert Path(tmp_path, "projects", "demo", "glossary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_colab.py -v`
Expected: FAIL with `ImportError: cannot import name 'colab'`

- [ ] **Step 3: Write the implementation, notebook builder, and docs**

`mlagent/colab.py`:
```python
"""Entry points used by the notebook. Safe to import outside Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mlagent import config
from mlagent.llm import LLM, AnthropicLLM
from mlagent.orchestrator import Orchestrator
from mlagent.project import Project
from mlagent.stages.base import StageContext
from mlagent.stages.intake import IntakeStage
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ConsoleQuestioner
from mlagent.ui.render import display_message

_LAST_CTX: StageContext | None = None


def _try_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def _load_api_key_from_secrets() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("ANTHROPIC_API_KEY")
    except Exception:
        return
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key


def setup(drive_root: str = config.DRIVE_ROOT, mount: bool = True) -> Path:
    if mount:
        _try_mount_drive()
    root = Path(drive_root)
    projects = root / config.PROJECTS_DIRNAME
    projects.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _load_api_key_from_secrets()
    return projects


def _context_snapshot(project: Project, stage_name: str = "") -> dict:
    return {
        "project": project.name,
        "stage": stage_name,
        "spec": project.read_json(config.SPEC_FILE),
        "state": project.read_json(config.STATE_FILE),
    }


def make_context(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> StageContext:
    global _LAST_CTX
    projects = setup(drive_root=drive_root, mount=False)
    project = Project(projects / project_name)
    project.ensure_dirs()
    llm = llm or AnthropicLLM()
    explainer = Explainer(
        llm=llm,
        glossary=Glossary(project.glossary_path),
        context_provider=lambda: _context_snapshot(project),
        display=display_message,
    )
    ctx = StageContext(project=project, llm=llm, questioner=ConsoleQuestioner(),
                       explainer=explainer, display=display_message)
    _LAST_CTX = ctx
    return ctx


def start(project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None) -> Orchestrator:
    ctx = make_context(project_name, drive_root=drive_root, llm=llm)
    ctx.explainer.register_colab_callback()
    return Orchestrator(ctx, [IntakeStage()])


def explain(term: str, refresh: bool = False) -> None:
    if _LAST_CTX is None or _LAST_CTX.explainer is None:
        raise RuntimeError("call start(project_name) first")
    _LAST_CTX.explainer.show(term, refresh=refresh)
```

`scripts/build_notebook.py`:
```python
"""Generate notebooks/ML_Training_Agent.ipynb. Run: python scripts/build_notebook.py"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "ML_Training_Agent.ipynb"

cells = [
    nbf.v4.new_markdown_cell(
        "# ML Training Agent\n\n"
        "Copy this notebook once per project. Before running:\n\n"
        "1. Copy the `mlagent/` folder from this repo to `MyDrive/ml_agent/mlagent/`.\n"
        "2. Add your Anthropic key in Colab **Secrets** (key icon) as `ANTHROPIC_API_KEY` with notebook access on.\n"
        "3. Run the cells in order. Click any highlighted term in the assistant's messages to get an explanation."
    ),
    nbf.v4.new_code_cell("%pip -q install anthropic markdown"),
    nbf.v4.new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import sys\n"
        "sys.path.insert(0, '/content/drive/MyDrive/ml_agent')\n"
        "from mlagent import colab\n"
        "colab.setup()"
    ),
    nbf.v4.new_code_cell(
        "PROJECT_NAME = 'my_first_project'  # change per project\n"
        "orch = colab.start(PROJECT_NAME)\n"
        "orch.run()"
    ),
    nbf.v4.new_markdown_cell("### Ask about any term"),
    nbf.v4.new_code_cell("colab.explain('validation set')"),
    nbf.v4.new_markdown_cell("### Redo a stage\nResets that stage and everything after it, then reruns."),
    nbf.v4.new_code_cell("# orch.reset('intake'); orch.run()"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
```

Run: `python scripts/build_notebook.py` → prints the path; commit the generated notebook.

`docs/colab-smoke.md`:
```markdown
# Colab smoke checklist

Run after each milestone. Copy `mlagent/` to `MyDrive/ml_agent/mlagent/` first.

## Milestone 1 (Foundation)
- [ ] Open `notebooks/ML_Training_Agent.ipynb` in Colab, run install and setup cells without error.
- [ ] `colab.start('smoke1'); orch.run()` asks the intake questions inline; answer them.
- [ ] `MyDrive/ml_agent/projects/smoke1/spec.json` exists and matches the answers.
- [ ] The closing message shows highlighted terms; clicking one shows an explanation panel.
- [ ] `colab.explain('learning rate')` shows an explanation; `glossary.json` now contains it.
- [ ] Runtime > Disconnect and delete runtime. Rerun setup and start cells: `orch.run()` reports nothing to do (intake already complete).
- [ ] `orch.reset('intake'); orch.run()` reruns the interview.
```

Update `CLAUDE.md` to:
```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlagent` is an AI/data-engineer assistant that runs inside a Google Colab notebook. It interviews the user, obtains or synthesises data, audits and cleans it, generates real training files onto Google Drive, runs them in a subprocess, plots results, estimates Colab cost before GPU runs, coaches tuning with approved config diffs, and explains any clicked technical term. Design spec: `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md`. Plans: `docs/superpowers/plans/`.

## Commands

```
python -m pip install -e ".[dev]"      # install with dev tools
python -m pytest                       # all tests
python -m pytest tests/test_intake.py -v            # one file
python -m pytest tests/test_intake.py::test_name -v # one test
ruff check .                           # lint
python scripts/build_notebook.py       # regenerate notebooks/ML_Training_Agent.ipynb
```

Tests never hit the network: anything that talks to Claude takes an `LLM` and tests pass `FakeLLM` (`mlagent/llm.py`).

## Architecture

- `mlagent/orchestrator.py` runs `stages/*` in order and checkpoints to `state.json` in the project folder so a Colab runtime reset resumes. A stage is complete when its artifact exists and validates (`Stage.is_complete`).
- `mlagent/llm.py` is the only place that calls the Anthropic SDK. It runs a manual tool-use loop: stages hand it `ToolSpec`s whose handlers do the real work (ask the user, write files). Structured results come back through tools, never by parsing prose.
- `mlagent/stages/base.py` defines `StageContext` (project, llm, questioner, explainer, display). Stages take everything from the context so they are testable with `ScriptedQuestioner` and `FakeLLM`.
- User questions go through the `Questioner` protocol (`ui/questions.py`). In Colab this is `input()`-based because widget clicks cannot block a running cell.
- Assistant text uses `[[term]]` markup. `ui/render.py` turns it into clickable spans; in Colab a click invokes the registered `mlagent.explain` callback, and `ui/explain.py` returns a cached, project-contextual explanation stored in `glossary.json`.
- System prompts live in `mlagent/prompts/*.md` and are loaded with `prompts_io.load_prompt`; never inline prompts in Python.
- `mlagent/colab.py` is the notebook's entry point (`setup`, `start`, `explain`). It must stay importable outside Colab.
- Per-project files live under `<drive_root>/projects/<name>/` (see `project.py` for the layout).

## Conventions

- Default model `claude-opus-5` with adaptive thinking; override with `MLAGENT_MODEL`.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere (Windows dev machine, Linux in Colab).
- Commit messages end with the Claude co-author trailer used in `docs/superpowers/plans/`.
```

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass (approximately 30), ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add mlagent/colab.py scripts/build_notebook.py notebooks/ML_Training_Agent.ipynb docs/colab-smoke.md tests/test_colab.py CLAUDE.md
git commit -m "feat: add Colab entry points, notebook template, and docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PnMncX4T3GzVVaShEBNTb2"
```

---

## Milestone verification

1. `python -m pytest` green and `ruff check .` clean.
2. Run the Milestone 1 section of `docs/colab-smoke.md` in a real Colab session (the user does this; the implementer cannot).
3. Milestone 2 plan starts from the `Stage` and `StageContext` interfaces defined here, adding `stages/data.py` and `stages/clean.py` to the list passed in `colab.start`.
