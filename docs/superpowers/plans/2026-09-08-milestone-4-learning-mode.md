# ML Training Agent — Milestone 4 (Learning Mode and Notebook-First Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the assistant from a black box into a teacher: every analysis step becomes a real script the user loads and runs in their own notebook cell, the agent explains it before and debriefs it after at the user's chosen learning level, every figure carries a caption, and the user picks between three model families on the agent's recommendation.

**Architecture:** Stages become two-phase. `prepare(ctx)` interviews the user, writes scripts and explains them, then returns a `Handoff` naming the cells the user must run; `debrief(ctx)` reads the outputs those cells produced, shows figures with captions, narrates, and completes the stage's artifact. The orchestrator persists `prepared` and `handoff` in `state.json` so a Colab runtime reset resumes at the right phase. All profiling, cleaning, training and evaluation code moves out of `mlagent` and into template scripts (`profile.py`, `clean.py`, `data.py`, `model.py`, `train.py`, `evaluate.py`) that import only numpy, pandas, scikit-learn, matplotlib and joblib. A new `Teaching` object issues the level-appropriate LLM calls (preamble, code walkthrough, debrief with figure notes) and a new `FormQuestioner` lets Colab `#@param` form fields answer the fixed questions.

**Tech Stack:** scikit-learn (`HistGradientBoosting*` warm start, `RandomForest*` warm start, `SGDClassifier`/`SGDRegressor` with `partial_fit`, `SimpleImputer`, `StandardScaler`), joblib, pandas, numpy, matplotlib (Agg in tests, inline in Colab), IPython as a soft guarded dependency inside templates, nbformat for notebook generation.

**Spec:** `docs/superpowers/specs/2026-09-08-milestone-4-learning-mode-design.md` (all sections: Decisions 1-8, Architecture — two-phase stages, notebook layout, scripts, model choice, learning level/captions/code walkthrough, questioner — and Testing).

## Global Constraints

- Python `>=3.10`; all tests run with `python -m pytest`; `ruff check .` must pass (`E, F, W, I, B, UP`, line length 100). Template files under `mlagent/templates/` are linted too.
- No network in tests. LLM calls use `FakeLLM`. Training in tests runs for real on tiny data (a few hundred rows) on CPU and must finish in seconds.
- **Generated scripts import only** the standard library, numpy, pandas, scikit-learn, matplotlib and joblib. They never import `mlagent`. IPython is a soft dependency: it may only be imported inside a `try/except ImportError` and the script must work without it.
- Every script gets: a `# --- settings ---` constants block whose values argparse can override, section markers of the exact form `# --- Title ---` at column 0 (the code walkthrough splits on them), a `cli_argv()` that returns `[]` under ipykernel, and a `__main__` guard of the form `code = main(cli_argv())` / `if code: sys.exit(code)` — never a bare `sys.exit(0)`, which would kill the notebook kernel's cell.
- Every prompt lives in `mlagent/prompts/**.md`, loaded with `prompts_io.load_prompt`; never inline prompts in Python. Teaching material lives in `mlagent/prompts/teaching/*.md`, level guidance in `mlagent/prompts/levels/*.md`.
- User-visible text may contain `[[term]]` markup for click-to-explain. Figure captions may too.
- Raw data is never modified. The test split is evaluated only by the report stage, once per best run.
- Charts: static matplotlib; house palette (series `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`, sequential blue ramp `#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b`, diverging `#1c5cab #86b6ef #f0efec #f3a17f #d95926`, ink `#0b0b0b`, ink-2 `#52514e`, muted `#898781`, grid `#e1e0d9`, axis `#c3c2b7`, surface `#fcfcfb`); single-series charts have no legend, multi-series charts always do; text uses ink colours, never series colours. Templates carry their own copy of the palette — they must not import `mlagent.plots`.
- Windows dev machine: `pathlib` everywhere, `encoding="utf-8"` on every text read and write, `sys.executable` for subprocesses. Project conventions: `from __future__ import annotations`, `from collections.abc import Callable`.
- The suite must be green at the end of every task. Tasks 4-12 migrate one stage at a time; the legacy `run()` bridge added in Task 3 keeps unmigrated stages working in between.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp
  ```

## Rulings made while planning

The spec leaves these open; each was decided here so no task has to improvise.

- **`outputs_ready` takes the handoff as an argument.** The spec writes `outputs_ready(self, ctx)`, but the handoff lives in `state.json` under the orchestrator's control, not on the stage instance (a stage object is rebuilt on every runtime reset). The signature is `outputs_ready(self, ctx, handoff) -> bool` and `ScriptStageBase` implements it by delegating to the module function `outputs_ready(root, handoff)`. Cost if wrong: a one-line signature change.
- **Legacy single-phase bridge.** `stages/base.py` gains adapter functions `stage_prepare`, `stage_debrief`, `stage_outputs_ready` that the orchestrator calls. `stage_prepare` calls `stage.prepare(ctx)` when the stage defines it and otherwise falls back to `stage.run(ctx)` and returns `None`. This keeps Tasks 4-12 able to migrate one stage per commit with the whole suite green, and it stays afterwards so simple test doubles can keep a single `run` method.
- **Freshness tolerance.** `outputs_ready` is true when every listed output exists and `min(output mtimes) + 1.0 >= max(script mtimes)`. `MTIME_TOLERANCE = 1.0` second absorbs Google Drive's mtime lag; `orch.debrief(name)` is the escape hatch when mtimes lie.
- **Handoff outputs are the smallest sufficient set** so `outputs_ready` never depends on a task-type-specific figure name: `data` -> `["profile_raw.json"]`; `clean` -> `["data/clean/data.csv", "profile_clean.json"]`; `train` -> `["metrics.json", "eval_val.json"]`; `report` -> `["eval_test.json"]`. Intake and codegen have no handoff.
- **`eval_val.json` is written by `evaluate.py`, not `train.py`.** The spec gives the train stage two handoff cells (`train.py`, then `evaluate.py`) and lists `eval_{split}.json` as `evaluate.py`'s output, so `train.py` stops writing it. `--eval-test` is deleted; `--dry-run` stays.
- **Run identity.** `metrics.json` gains `started_at` (UTC ISO seconds). `TrainStage.debrief` logs a run only when that `started_at` is not already in `runs.jsonl`; `is_complete` is "at least one `done` run **and** the current `metrics.json`'s `started_at` is logged". `build_run_entry` is re-signed as `build_run_entry(metrics: dict, checkpoint: str | None) -> dict` — the config, seconds and `started_at` all come from `metrics.json` now that no `RunResult` exists.
- **`run_id` inside `eval_test.json`.** `evaluate.py --split test` with no `--checkpoint` picks the best `done` run from `runs.jsonl` and records that run's `run_id` and checkpoint path. For `--split val` (straight after training, before the run is logged) `run_id` is `null`. `ReportStage.debrief` uses `eval_test["run_id"]` for its staleness check.
- **`split_seed` is now written.** `CleanStage.prepare` writes `split_seed: 42` into `data_meta.json`. Generated `data.py` keeps its `42` default for older projects.
- **`profile_clean.json` changes shape** to `{"before": {...}, "after": {...}, "steps": [...], "figures": [...]}` where each half is a compact profile (`n_rows`, `n_cols`, `duplicate_rows`, `columns[{name,dtype,missing,missing_pct,n_unique}]`). Nothing but the clean stage's own summary reads it.
- **`profile_{tag}.json` keeps the Milestone 2 shape** (`profile_dataframe`'s exact keys) plus a `figures` list, so `profile.profile_markdown` still renders it unchanged. `tests/test_template_profile.py` pins this by comparing the template's output with `mlagent.profile.profile_dataframe`.
- **`clean.py` is a rendered template, not a copy.** `mlagent/templates/common/clean.py` is a runnable, lintable, importable file containing the op functions and the line `STEPS_JSON = r"""[]"""`. `cleaning.render_clean_py(steps)` replaces exactly that line. `cleaning.apply_steps` stays as the reference implementation the template is tested against.
- **`plots.py` keeps only** the palette constants, the two colormaps, `style_axes`, `save_figure` and `present`. Every data and evaluation figure function is deleted; the templates carry their own. `tests/test_plots_training.py` is deleted and `tests/test_plots.py` is rewritten.
- **Teaching material trimming.** `prompts/teaching/*.md` may wrap paragraphs in `<!--level:beginner,intermediate-->` ... `<!--/level-->`; `teaching.trim_levels(text, level)` keeps matching blocks and drops the rest. Unfenced text is always kept.
- **LLM calls per level.** `preamble` is skipped at `expert`. `walkthrough` is one LLM call per script at `beginner` (explain every section), one call for all scripts at `intermediate` (a paragraph per file), and no call at `expert` (file names and section titles only). `debrief` is always one call. That gives the spec's 1 / 3 / 3-4 calls per stage.
- **`StageContext.display_figure` uses `default_factory`.** A plain function used as a dataclass default would be bound as a method; `field(default_factory=lambda: render.display_figure)` stores it as an instance attribute so `ctx.display_figure(path, caption)` works.
- **`max_features` is a fraction.** `validate_config` only understands numbers and choices, so the random forest's `max_features` is a number in `0.05..1.0` (the fraction of features tried per split), not scikit-learn's `"sqrt"`/`"log2"` strings.
- **`runner.py` stays in the tree** with its tests, unused by any stage after Task 12. Milestone 5's traceback-to-fix loop wants a subprocess runner and deleting it now would be churn.

## File Structure

| File | Responsibility |
|---|---|
| `mlagent/ui/questions.py` (modify) | `key=` on every `Questioner` method; `FormQuestioner`. |
| `mlagent/spec.py` (modify) | `learning_level` field, `LEARNING_LEVELS`, validation. |
| `mlagent/prompts_io.py` (modify) | `load_prompt(name, **params)`, `audience(level)`. |
| `mlagent/prompts/levels/{beginner,intermediate,expert}.md` (new) | Level guidance injected as `{audience}`. |
| `mlagent/prompts/{preamble,walkthrough}.md` (new) | Teaching prompts. |
| `mlagent/prompts/teaching/model_choices.md` (new) | The three model families, level-fenced. |
| `mlagent/stages/base.py` (modify) | `Handoff`, two-phase `Stage`, `ScriptStageBase`, adapters, `waiting_message`, `StageContext.display_figure`/`teaching`. |
| `mlagent/orchestrator.py` (modify) | Phase loop, `prepared`/`handoff` state, `waiting()`, `debrief(name)`, `run(answers=...)`. |
| `mlagent/captions.py` (new) | `CAPTIONS` table and `caption_for(path)`. |
| `mlagent/codewalk.py` (new) | `split_sections`, `render_walkthrough`. |
| `mlagent/teaching.py` (new) | `Teaching`, `trim_levels`, `material`. |
| `mlagent/ui/render.py` (modify) | `display_figure(path, caption)`. |
| `mlagent/cleaning.py` (modify) | `render_clean_py` renders the common template. |
| `mlagent/templates_io.py` (modify) | `COMMON_DIR`, `common_file`, `copy_common`, `schema_for`, choice-aware validate/coerce/default; `CODE_FILES` gains `evaluate.py`. |
| `mlagent/templates/common/profile.py` (new) | Standalone profiling script and its four figures. |
| `mlagent/templates/common/clean.py` (new) | Standalone cleaning script: ops, `STEPS_JSON`, before/after profile, figure. |
| `mlagent/templates/tabular_sklearn/model.py` (rewrite) | `EpochModel` and `build_model` for three families. |
| `mlagent/templates/tabular_sklearn/config_schema.json` (rewrite) | Nested `common` / `models`. |
| `mlagent/templates/tabular_sklearn/train.py` (modify) | `started_at`, `model_type`, per-epoch curve PNG plus live redraw, no `--eval-test`. |
| `mlagent/templates/tabular_sklearn/evaluate.py` (new) | `--split val\|test`, metrics, five evaluation figures. |
| `mlagent/plots.py` (modify) | Palette, `style_axes`, `save_figure`, `present` only. |
| `mlagent/stages/{intake,data,clean,codegen,train,report}.py` (modify) | Two-phase, teaching, form keys. |
| `mlagent/colab.py` (modify) | `chdir`, module purge hook, `SCRIPT_CELL_FORMATS`, `HANDOFF_COMMANDS`, `cell_source`, `script_cells`. |
| `scripts/build_notebook.py` (rewrite) | The 17-cell notebook from the spec. |
| `tests/conftest.py` (modify) | `run_handoff(project, handoff)`. |
| `tests/test_captions.py`, `test_codewalk.py`, `test_teaching.py`, `test_template_profile.py`, `test_template_clean.py`, `test_template_model.py`, `test_template_evaluate.py` (new) | Unit tests for the new modules and templates. |
| `tests/test_plots_training.py` (delete), `tests/test_plots.py` (rewrite) | Figure tests move into the template tests. |
| `CLAUDE.md`, `docs/colab-smoke.md` (modify) | Architecture paragraph and Milestone 4 smoke items. |

---

### Task 1: `key=` on the questioner and `FormQuestioner`

**Files:**
- Modify: `mlagent/ui/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Questioner` protocol methods all end with `key: str | None = None`:
  `choice(question: str, options: list[str], allow_other: bool = True, key: str | None = None) -> str`,
  `text(prompt: str, default: str | None = None, key: str | None = None) -> str`,
  `confirm(question: str, default: bool = True, key: str | None = None) -> bool`,
  `number(prompt: str, default: float | None = None, minimum: float | None = None, maximum: float | None = None, key: str | None = None) -> float`.
  `ConsoleQuestioner` and `ScriptedQuestioner` accept `key` and ignore it.
  `FormQuestioner(answers: dict[str, object], fallback: Questioner, note: Callable[[str], None] = _silent)` with attribute `used: list[str]` (keys it answered from the form).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_questions.py`:

```python
from mlagent.ui.questions import ConsoleQuestioner, FormQuestioner, ScriptedQuestioner


def make_form(answers, fallback_answers):
    notes: list[str] = []
    fallback = ScriptedQuestioner(list(fallback_answers))
    q = FormQuestioner(answers, fallback=fallback, note=notes.append)
    return q, fallback, notes


def test_form_answers_every_kind_of_question():
    q, fallback, notes = make_form(
        {
            "intake.goal": "Predict churn",
            "intake.task_type": "tabular classification",
            "intake.target_value": 0.85,
            "data.inject_quirks": True,
        },
        [],
    )
    assert q.text("Goal?", key="intake.goal") == "Predict churn"
    assert q.choice(
        "Task?", ["Tabular classification", "Image classification"], key="intake.task_type"
    ) == "Tabular classification"
    assert q.number("Target?", default=0.9, key="intake.target_value") == 0.85
    assert q.confirm("Quirks?", key="data.inject_quirks") is True
    assert q.used == ["intake.goal", "intake.task_type", "intake.target_value",
                      "data.inject_quirks"]
    assert fallback.asked == [] and notes == []


def test_form_falls_back_silently_when_the_key_is_absent():
    q, fallback, notes = make_form({}, ["typed answer"])
    assert q.text("Goal?", key="intake.goal") == "typed answer"
    assert fallback.asked == ["Goal?"] and notes == []
    assert q.used == []


def test_form_notes_and_falls_back_on_empty_or_invalid_values():
    q, fallback, notes = make_form(
        {"intake.goal": "", "intake.minutes_per_run": "soon", "clean.train_fraction": 5.0,
         "codegen.model_type": "quantum forest"},
        ["typed goal", "10", "0.7", "Random forest"],
    )
    assert q.text("Goal?", key="intake.goal") == "typed goal"
    assert q.number("Minutes?", default=10, key="intake.minutes_per_run") == 10.0
    assert q.number("Train?", minimum=0.5, maximum=0.9, key="clean.train_fraction") == 0.7
    assert q.choice("Model?", ["Random forest", "Gradient boosting"], allow_other=False,
                    key="codegen.model_type") == "Random forest"
    assert len(notes) == 4
    assert all("asking instead" in n for n in notes)
    assert q.used == []


def test_form_choice_allows_other_when_permitted():
    q, _fallback, notes = make_form({"data.hf_query": "credit card fraud"}, [])
    assert q.choice("Dataset?", ["iris", "titanic"], allow_other=True,
                    key="data.hf_query") == "credit card fraud"
    assert notes == []


def test_form_confirm_accepts_strings_and_bools():
    q, _fallback, _notes = make_form({"a": "yes", "b": False, "c": "N"}, [])
    assert q.confirm("A?", key="a") is True
    assert q.confirm("B?", key="b") is False
    assert q.confirm("C?", key="c") is False


def test_form_without_a_key_always_falls_back():
    q, fallback, notes = make_form({"intake.goal": "unused"}, ["typed"])
    assert q.text("Anything?") == "typed"
    assert fallback.asked == ["Anything?"] and notes == []


def test_console_and_scripted_accept_and_ignore_key():
    q, _printed = make_console(["2"])
    assert q.choice("Pick", ["alpha", "beta"], key="x.y") == "beta"
    s = ScriptedQuestioner(["hi", "0.5", "y"])
    assert s.text("Say", key="x.y") == "hi"
    assert s.number("Num", key="x.y") == 0.5
    assert s.confirm("Ok?", key="x.y") is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_questions.py -q`
Expected: FAIL with `ImportError: cannot import name 'FormQuestioner' from 'mlagent.ui.questions'`.

- [ ] **Step 3: Add `key=` to the protocol and both existing questioners**

In `mlagent/ui/questions.py`, replace the `Questioner` protocol and add `key` to every `ConsoleQuestioner` and `ScriptedQuestioner` method signature (the bodies do not change — `key` is accepted and ignored):

```python
class Questioner(Protocol):
    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str: ...
    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str: ...
    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool: ...
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float: ...
```

`ConsoleQuestioner`:

```python
    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str:
```
```python
    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
```
```python
    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
```
```python
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float:
```

`ScriptedQuestioner`: the same four signatures, bodies unchanged.

- [ ] **Step 4: Add `FormQuestioner`**

Append to `mlagent/ui/questions.py`:

```python
_MISSING = object()
TRUTHY = {"y", "yes", "true", "1", "on"}
FALSEY = {"n", "no", "false", "0", "off"}


def _silent(_message: str) -> None:
    return None


class FormQuestioner:
    """Answers fixed questions from a Colab `#@param` form, falling back to `fallback`.

    A key that is absent from `answers` falls back silently: the form simply does not
    cover that question. A key that is present but empty or unusable falls back with a
    one-line note, because the user did fill the form in and deserves to know why they
    are being asked again.
    """

    def __init__(
        self,
        answers: dict[str, object],
        fallback: Questioner,
        note: Callable[[str], None] = _silent,
    ):
        self.answers = dict(answers or {})
        self.fallback = fallback
        self.note = note
        self.used: list[str] = []

    def _raw(self, key: str | None):
        if key is None or key not in self.answers:
            return _MISSING
        value = self.answers[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            self.note(f"The form field for '{key}' is empty; asking instead.")
            return _MISSING
        return value

    def _reject(self, key: str, value) -> None:
        self.note(f"'{value}' is not a valid answer for '{key}'; asking instead.")

    def _accept(self, key: str, value):
        self.used.append(key)
        return value

    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str:
        raw = self._raw(key)
        if raw is not _MISSING:
            text = str(raw).strip()
            for option in options:
                if text.lower() == option.lower():
                    return self._accept(key, option)
            if allow_other:
                return self._accept(key, text)
            self._reject(key, text)
        return self.fallback.choice(question, options, allow_other=allow_other, key=key)

    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
        raw = self._raw(key)
        if raw is not _MISSING:
            return self._accept(key, str(raw).strip())
        return self.fallback.text(prompt, default=default, key=key)

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        raw = self._raw(key)
        if raw is not _MISSING:
            if isinstance(raw, bool):
                return self._accept(key, raw)
            text = str(raw).strip().lower()
            if text in TRUTHY:
                return self._accept(key, True)
            if text in FALSEY:
                return self._accept(key, False)
            self._reject(key, raw)
        return self.fallback.confirm(question, default=default, key=key)

    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float:
        raw = self._raw(key)
        if raw is not _MISSING:
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                self._reject(key, raw)
            else:
                too_low = minimum is not None and value < minimum
                too_high = maximum is not None and value > maximum
                if too_low or too_high:
                    self._reject(key, raw)
                else:
                    return self._accept(key, value)
        return self.fallback.number(
            prompt, default=default, minimum=minimum, maximum=maximum, key=key
        )
```

Note: `bool` is a subclass of `int`, so `float(str(True))` raises — a boolean given to `number` is correctly rejected with a note.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_questions.py -q`
Expected: PASS (13 tests).

- [ ] **Step 6: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, ruff clean. Nothing else calls the questioner with `key`, so existing call sites are unaffected.

- [ ] **Step 7: Commit**

```bash
git add mlagent/ui/questions.py tests/test_questions.py
git commit -m "feat: optional key= on questioner methods and a Colab-form questioner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 2: Learning level in the spec, prompts and intake

**Files:**
- Modify: `mlagent/spec.py`
- Modify: `mlagent/prompts_io.py`
- Modify: `mlagent/stages/intake.py`
- Create: `mlagent/prompts/levels/beginner.md`, `mlagent/prompts/levels/intermediate.md`, `mlagent/prompts/levels/expert.md`
- Modify: `mlagent/prompts/{intake,data,clean,codegen,train,report}.md`
- Modify: `mlagent/stages/{data,clean,codegen,train,report}.py` (pass `audience=`)
- Test: `tests/test_spec.py`, `tests/test_intake.py`

**Interfaces:**
- Consumes: `Questioner.key` from Task 1.
- Produces:
  `spec.LEARNING_LEVELS = ("beginner", "intermediate", "expert")`;
  `Spec.learning_level: str = "intermediate"` (validated, optional in `from_dict` like `notes`);
  `prompts_io.load_prompt(name: str, **params: object) -> str` (formats only when `params` is non-empty);
  `prompts_io.audience(level: str) -> str` returning the text of `prompts/levels/<level>.md`, falling back to `intermediate` for an unknown level;
  `intake.LEVEL_LABELS: dict[str, str]` mapping the three question labels to codes;
  every fixed intake question now passes a `key="intake.<field>"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spec.py`:

```python
from mlagent.spec import LEARNING_LEVELS


def test_learning_level_defaults_and_validates():
    s = make_spec()
    assert s.learning_level == "intermediate"
    assert s.validate() == []
    assert LEARNING_LEVELS == ("beginner", "intermediate", "expert")
    bad = make_spec(learning_level="guru")
    assert any("learning_level" in p for p in bad.validate())


def test_learning_level_round_trips_and_is_optional_in_from_dict():
    s = make_spec(learning_level="beginner")
    assert Spec.from_dict(s.to_dict()).learning_level == "beginner"
    legacy = {k: v for k, v in make_spec().to_dict().items() if k != "learning_level"}
    assert Spec.from_dict(legacy).learning_level == "intermediate"
```

Create `tests/test_prompts_io.py`:

```python
from __future__ import annotations

import pytest

from mlagent.prompts_io import PROMPTS_DIR, audience, load_prompt

STAGE_PROMPTS = ["intake", "data", "clean", "codegen", "train", "report"]


def test_load_prompt_without_params_does_not_format():
    text = (PROMPTS_DIR / "data.md").read_text(encoding="utf-8")
    assert load_prompt("data") == text
    assert "{audience}" in text


def test_load_prompt_with_params_formats():
    text = load_prompt("data", audience="Write for an expert.")
    assert "{audience}" not in text
    assert "Write for an expert." in text


def test_every_stage_prompt_has_an_audience_line():
    for name in STAGE_PROMPTS:
        assert "Audience: {audience}" in (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def test_audience_returns_level_text_and_falls_back():
    for level in ("beginner", "intermediate", "expert"):
        assert audience(level).strip()
    assert audience("guru") == audience("intermediate")


def test_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")
```

Append to `tests/test_intake.py`:

```python
def test_intake_asks_the_learning_level_and_stores_it(project):
    from mlagent.stages.intake import LEVEL_LABELS

    assert LEVEL_LABELS["Beginner - explain everything as we go"] == "beginner"
    ctx, _shown = make_ctx(
        project,
        FakeLLM([]),
        [
            "Predict churn",
            "Beginner - explain everything as we go",
            "Tabular classification",
            "accuracy",
            "0.9",
            "Synthetic data",
            "10",
            "5",
            "No GPU (CPU only)",
        ],
    )
    IntakeStage().run(ctx)
    assert project.read_json("spec.json")["learning_level"] == "beginner"
    assert project.read_json("draft_spec.json")["learning_level"] == "beginner"
```

Adjust the existing tests in `tests/test_intake.py`: the module-level `ANSWERS` list gains `"Intermediate - explain the key ideas",  # learning level` as its second entry (immediately after the goal), and `test_collect_draft_maps_labels_to_codes` gains `assert draft["learning_level"] == "intermediate"`. `IntakeStage.run` keeps its name in this task; Task 3 renames it to `prepare` and updates every call site including this one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_spec.py tests/test_prompts_io.py tests/test_intake.py -q`
Expected: FAIL with `ImportError: cannot import name 'LEARNING_LEVELS' from 'mlagent.spec'`.

- [ ] **Step 3: Add `learning_level` to `Spec`**

In `mlagent/spec.py`, after `GPU_CHOICES`:

```python
LEARNING_LEVELS = ("beginner", "intermediate", "expert")
DEFAULT_LEARNING_LEVEL = "intermediate"
```

Add the field to the dataclass (after `gpu`, before `notes` so both defaulted fields sit together):

```python
    learning_level: str = DEFAULT_LEARNING_LEVEL
    notes: str = ""
```

In `validate`, after the `gpu` check:

```python
        if self.learning_level not in LEARNING_LEVELS:
            problems.append(f"learning_level must be one of {LEARNING_LEVELS}")
```

In `from_dict`, make it optional and read it:

```python
        missing = known - set(d) - {"notes", "learning_level"}
```
```python
            gpu=str(d["gpu"]),
            learning_level=str(d.get("learning_level", DEFAULT_LEARNING_LEVEL)),
            notes=str(d.get("notes", "")),
```

- [ ] **Step 4: Teach `load_prompt` about parameters and levels**

Replace `mlagent/prompts_io.py` entirely:

```python
"""Load system prompts from mlagent/prompts/**.md."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
LEVELS_DIRNAME = "levels"
DEFAULT_LEVEL = "intermediate"


def load_prompt(name: str, **params: object) -> str:
    """Read `prompts/<name>.md`. With no params the text is returned verbatim, so
    prompts that contain literal braces are safe until someone actually formats them."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if not params:
        return text
    return text.format(**params)


def audience(level: str) -> str:
    """The level guidance injected into every stage prompt as `{audience}`."""
    path = PROMPTS_DIR / LEVELS_DIRNAME / f"{level}.md"
    if not path.exists():
        path = PROMPTS_DIR / LEVELS_DIRNAME / f"{DEFAULT_LEVEL}.md"
    return path.read_text(encoding="utf-8").strip()
```

- [ ] **Step 5: Write the three level prompts**

`mlagent/prompts/levels/beginner.md`:

```markdown
The user is new to machine learning. Assume no jargon is known. Define every technical
term the first time you use it, in one short clause, and wrap it in double square
brackets so they can click it. Prefer concrete comparisons to formulas. Say what a
number means before you say whether it is good. Use short sentences and no more than
two ideas per paragraph. It is fine to be 30% longer than you would be otherwise.
```

`mlagent/prompts/levels/intermediate.md`:

```markdown
The user has used machine learning before but is not an expert. Use standard terms
without defining them, but wrap the less common ones in double square brackets so they
can click for a refresher. Explain *why* a number or a shape of curve matters rather
than restating it. Keep to the word limit; skip background they will already have.
```

`mlagent/prompts/levels/expert.md`:

```markdown
The user is an experienced practitioner. Be terse. State the numbers, the diagnosis and
the recommended next action, nothing else. Do not define terms and do not explain what a
metric is. Wrap a term in double square brackets only when it is genuinely project
specific. Aim for half the word limit.
```

- [ ] **Step 6: Add the audience line to every stage prompt**

Append these two lines (a blank line, then the audience line) to the end of each of `mlagent/prompts/intake.md`, `data.md`, `clean.md`, `codegen.md`, `train.md`, `report.md`:

```markdown

Audience: {audience}
```

`explain.md` is not a stage prompt and already uses `.format(term=..., context=...)`; leave it alone.

- [ ] **Step 7: Pass the audience at every existing call site**

`mlagent/stages/data.py`, in `_narrate`, replace the `ask_text` line:

```python
            text = ask_text(
                ctx.llm,
                load_prompt("data", audience=audience(ctx.spec().learning_level)),
                prompt,
            )
```
and add `from mlagent.prompts_io import audience, load_prompt` to the imports.

`mlagent/stages/clean.py`, in `_explain`:

```python
            text = ask_text(
                ctx.llm,
                load_prompt("clean", audience=audience(ctx.spec().learning_level)),
                payload,
            )
```
with the same import change.

`mlagent/stages/codegen.py`, in `_propose`:

```python
            result = ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}],
                [tool],
            )
```

`mlagent/stages/train.py`, in `_debrief`:

```python
            narrative = ask_text(
                ctx.llm,
                load_prompt("train", audience=audience(spec.learning_level)),
                json.dumps(summary, default=str),
            )
```

`mlagent/stages/report.py`, in `_lessons`:

```python
            return ask_text(
                ctx.llm,
                load_prompt("report", audience=audience(spec.learning_level)),
                json.dumps(summary, default=str),
            )
```

`mlagent/stages/intake.py`, in `run`:

```python
                system=load_prompt("intake", audience=audience(draft["learning_level"])),
```

Each file's import line becomes `from mlagent.prompts_io import audience, load_prompt`.

- [ ] **Step 8: Ask the level at intake and key every fixed question**

In `mlagent/stages/intake.py`, after `GPU_LABELS`:

```python
LEVEL_LABELS = {
    "Beginner - explain everything as we go": "beginner",
    "Intermediate - explain the key ideas": "intermediate",
    "Expert - just the numbers": "expert",
}
```

Replace `collect_draft` with:

```python
def collect_draft(q: Questioner) -> dict:
    goal = q.text(
        "In one or two sentences, what do you want the model to do?", key="intake.goal"
    )
    learning_level = _label_to_code(
        q.choice(
            "How much explanation do you want as we go?",
            list(LEVEL_LABELS),
            allow_other=False,
            key="intake.learning_level",
        ),
        LEVEL_LABELS,
        LEARNING_LEVELS,
    )
    task_type = _label_to_code(
        q.choice("What kind of task is it?", list(TASK_LABELS), allow_other=False,
                 key="intake.task_type"),
        TASK_LABELS, TASK_TYPES,
    )
    metric = q.choice("Which metric defines success?", METRICS_FOR_TASK[task_type],
                      allow_other=False, key="intake.metric")
    target_value = q.number(f"What {metric} value would count as good enough?", default=0.9,
                            key="intake.target_value")
    data_source = _label_to_code(
        q.choice("Where will the data come from?", list(SOURCE_LABELS), allow_other=False,
                 key="intake.data_source"),
        SOURCE_LABELS, DATA_SOURCES,
    )
    minutes = int(q.number("Roughly how many minutes per training run are acceptable?",
                           default=10, minimum=1, key="intake.minutes_per_run"))
    rounds = int(q.number("How many tuning rounds at most?", default=5, minimum=1,
                          key="intake.max_rounds"))
    gpu = _label_to_code(q.choice("GPU preference?", list(GPU_LABELS), allow_other=False,
                                  key="intake.gpu"),
                         GPU_LABELS, GPU_CHOICES)
    return {
        "goal": goal,
        "learning_level": learning_level,
        "task_type": task_type,
        "metric": metric,
        "target_value": target_value,
        "data_source": data_source,
        "minutes_per_run": minutes,
        "max_rounds": rounds,
        "gpu": gpu,
        "notes": "",
    }
```

Import `LEARNING_LEVELS` from `mlagent.spec` alongside the other constants, and add the key to `SPEC_SCHEMA`:

```python
        "learning_level": {"type": "string", "enum": list(LEARNING_LEVELS)},
```
(leave `required` unchanged: the level is optional for the model, and `Spec.from_dict` defaults it.)

- [ ] **Step 9: Run the tests**

Run: `python -m pytest tests/test_spec.py tests/test_prompts_io.py tests/test_intake.py -q`
Expected: PASS.

- [ ] **Step 10: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass. `tests/test_pipeline_e2e.py`'s `ANSWERS` list needs the learning level inserted after the goal — add `"Intermediate - explain the key ideas",  # intake: learning level` as the second entry.

- [ ] **Step 11: Commit**

```bash
git add mlagent/spec.py mlagent/prompts_io.py mlagent/prompts mlagent/stages tests
git commit -m "feat: learning level in the spec, level-aware prompts and keyed intake questions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---
### Task 3: Two-phase stages, `Handoff`, and the orchestrator phase loop

**Files:**
- Modify: `mlagent/stages/base.py`
- Modify: `mlagent/orchestrator.py`
- Modify: `mlagent/stages/intake.py`, `mlagent/stages/codegen.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `FormQuestioner` (Task 1); `Project.root`, `Project.read_json/write_json`; `config.STATE_FILE`.
- Produces, in `mlagent/stages/base.py`:
  - `MTIME_TOLERANCE = 1.0`
  - `@dataclass class Handoff: stage: str; commands: list[list[str]]; outputs: list[str]` with `to_dict() -> dict` and `classmethod from_dict(d: dict | None) -> Handoff | None`
  - `outputs_ready(root: Path, handoff: Handoff) -> bool`
  - `waiting_message(handoff: Handoff) -> str`
  - `class ScriptStageBase` with `outputs_ready(self, ctx, handoff) -> bool`
  - adapters `stage_prepare(stage, ctx) -> Handoff | None`, `stage_debrief(stage, ctx) -> None`, `stage_outputs_ready(stage, ctx, handoff) -> bool`
  - `Stage` protocol: `name`, `prepare(ctx) -> Handoff | None`, `outputs_ready(ctx, handoff) -> bool`, `debrief(ctx) -> None`, `is_complete(ctx) -> bool`
- Produces, in `mlagent/orchestrator.py`: `Orchestrator.run(until: str | None = None, answers: dict[str, object] | None = None) -> list[str]`, `Orchestrator.waiting() -> Handoff | None`, `Orchestrator.debrief(name: str) -> None`; `state.json` gains `"prepared": list[str]` and `"handoff": dict | None`.
- Produces: `IntakeStage.prepare(ctx) -> None` (renamed from `run`) plus a no-op `debrief`; `CodegenStage.prepare(ctx) -> None` plus a no-op `debrief`.

- [ ] **Step 1: Write the failing tests**

Replace the top of `tests/test_orchestrator.py` (imports and stage doubles) and append the new tests:

```python
import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.stages.base import Handoff, StageContext, outputs_ready, waiting_message
from mlagent.ui.explain import Explainer, Glossary
from mlagent.ui.questions import ScriptedQuestioner


class RecordingStage:
    """A legacy single-phase stage: only `run` and `is_complete`."""

    def __init__(self, name: str, fail: bool = False, write: bool = True):
        self.name = name
        self.fail = fail
        self.write = write
        self.runs = 0

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        if self.fail:
            raise RuntimeError("boom")
        if self.write:
            ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


class ScriptStage:
    """A two-phase stage: prepare writes a script, the 'user' runs it, debrief reads it."""

    def __init__(self, name: str):
        self.name = name
        self.prepared = 0
        self.debriefed = 0

    def prepare(self, ctx: StageContext) -> Handoff:
        self.prepared += 1
        (ctx.project.root / f"{self.name}.py").write_text("print('hi')\n", encoding="utf-8")
        return Handoff(stage=self.name, commands=[[f"{self.name}.py"]],
                       outputs=[f"{self.name}_out.json"])

    def debrief(self, ctx: StageContext) -> None:
        self.debriefed += 1
        ctx.project.write_json(f"{self.name}.json", {"ok": True})

    def is_complete(self, ctx: StageContext) -> bool:
        return ctx.project.exists(f"{self.name}.json")


def user_runs(project, stage_name: str) -> None:
    """Stand in for the user running the handoff cell."""
    project.write_json(f"{stage_name}_out.json", {"done": True})
```

Then append:

```python
def test_handoff_round_trips_through_state_json():
    h = Handoff(stage="train", commands=[["train.py"], ["evaluate.py"]],
                outputs=["metrics.json"])
    assert Handoff.from_dict(h.to_dict()) == h
    assert Handoff.from_dict(None) is None
    assert Handoff.from_dict({"stage": "x"}) == Handoff(stage="x", commands=[], outputs=[])


def test_outputs_ready_checks_existence_and_freshness(tmp_path):
    import os
    import time

    script = tmp_path / "s.py"
    script.write_text("x = 1\n", encoding="utf-8")
    h = Handoff(stage="s", commands=[["s.py"]], outputs=["out.json"])
    assert outputs_ready(tmp_path, h) is False
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")
    assert outputs_ready(tmp_path, h) is True
    # Editing the script after the output was produced makes the output stale.
    future = time.time() + 60
    os.utime(script, (future, future))
    assert outputs_ready(tmp_path, h) is False


def test_waiting_message_names_every_cell():
    h = Handoff(stage="train", commands=[["train.py"], ["evaluate.py", "--split", "val"]],
                outputs=[])
    text = waiting_message(h)
    assert "`train.py`" in text and "`evaluate.py --split val`" in text
    assert "run this cell again" in text


def test_script_stage_pauses_until_outputs_exist(project):
    stage = ScriptStage("s")
    shown: list[str] = []
    orch = Orchestrator(make_ctx(project, display=shown.append), [stage])
    assert orch.run() == []
    assert stage.prepared == 1 and stage.debriefed == 0
    assert orch.waiting() == Handoff(stage="s", commands=[["s.py"]], outputs=["s_out.json"])
    assert any("run this cell again" in s for s in shown)
    state = project.read_json("state.json")
    assert state["prepared"] == ["s"] and state["handoff"]["stage"] == "s"

    # Running again without the outputs does not re-prepare.
    assert orch.run() == []
    assert stage.prepared == 1

    user_runs(project, "s")
    assert orch.run() == ["s"]
    assert stage.prepared == 1 and stage.debriefed == 1
    state = project.read_json("state.json")
    assert state["completed"] == ["s"] and state["handoff"] is None and state["prepared"] == []


def test_prepared_survives_a_fresh_orchestrator(project):
    first = ScriptStage("s")
    Orchestrator(make_ctx(project), [first]).run()
    user_runs(project, "s")
    second = ScriptStage("s")
    assert Orchestrator(make_ctx(project), [second]).run() == ["s"]
    assert second.prepared == 0 and second.debriefed == 1


def test_debrief_by_name_forces_a_stage_whose_mtimes_lie(project):
    stage = ScriptStage("s")
    orch = Orchestrator(make_ctx(project), [stage])
    orch.run()
    orch.debrief("s")  # outputs missing, but the user says they ran it
    assert stage.debriefed == 1
    assert orch.completed() == ["s"]
    with pytest.raises(ValueError):
        orch.debrief("nope")


def test_reset_forgets_prepared_and_handoff(project):
    a, b = ScriptStage("a"), ScriptStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    orch.run()
    user_runs(project, "a")
    orch.run()
    assert orch.completed() == ["a"]
    orch.reset("a")
    state = project.read_json("state.json")
    assert state["prepared"] == [] and state["handoff"] is None
    assert state["forced"] == ["a", "b"]


def test_answers_wrap_the_questioner_for_one_call_only(project):
    class AskingStage:
        name = "ask"

        def __init__(self):
            self.seen: list[str] = []

        def prepare(self, ctx):
            self.seen.append(ctx.questioner.text("Goal?", key="intake.goal"))
            ctx.project.write_json("ask.json", {"ok": True})
            return None

        def debrief(self, ctx):
            return None

        def is_complete(self, ctx):
            return ctx.project.exists("ask.json")

    stage = AskingStage()
    ctx = make_ctx(project)
    ctx.questioner = ScriptedQuestioner(["typed"])
    orch = Orchestrator(ctx, [stage])
    assert orch.run(answers={"intake.goal": "from the form"}) == ["ask"]
    assert stage.seen == ["from the form"]
    assert isinstance(orch.ctx.questioner, ScriptedQuestioner)


def test_legacy_single_phase_stages_still_run(project):
    a, b = RecordingStage("a"), RecordingStage("b")
    orch = Orchestrator(make_ctx(project), [a, b])
    assert orch.run() == ["a", "b"]
    assert orch.waiting() is None
```

Every existing test in the file keeps working unchanged: they all use `RecordingStage`, which the legacy bridge still supports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: FAIL with `ImportError: cannot import name 'Handoff' from 'mlagent.stages.base'`.

- [ ] **Step 3: Add `Handoff` and the stage adapters**

Replace `mlagent/stages/base.py`:

```python
"""Shared types for pipeline stages.

A stage runs in two phases. `prepare` interviews the user, writes the scripts they will
run, and returns a `Handoff` naming the notebook cells to run (or `None` when the stage
needs no cells). `debrief` reads whatever those cells produced, shows it, and writes the
stage's completion artifact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mlagent import config
from mlagent.llm import LLM
from mlagent.project import Project
from mlagent.spec import DEFAULT_LEARNING_LEVEL, Spec, SpecError
from mlagent.ui.explain import Explainer
from mlagent.ui.questions import Questioner
from mlagent.ui.render import display_figure as render_display_figure

# Drive's mtimes can lag behind the write that produced them; allow a second of slack
# before declaring an output stale relative to the script that made it.
MTIME_TOLERANCE = 1.0


@dataclass
class Handoff:
    """Cells the user must run before the stage can be debriefed."""

    stage: str
    commands: list[list[str]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "commands": [list(c) for c in self.commands],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Handoff | None:
        if not isinstance(d, dict) or not d.get("stage"):
            return None
        return cls(
            stage=str(d["stage"]),
            commands=[[str(part) for part in c] for c in (d.get("commands") or [])],
            outputs=[str(o) for o in (d.get("outputs") or [])],
        )

    def scripts(self) -> list[str]:
        return [c[0] for c in self.commands if c]


def outputs_ready(root: Path, handoff: Handoff) -> bool:
    """True when every output exists and none is older than the scripts that make it."""
    root = Path(root)
    out_paths = [root / name for name in handoff.outputs]
    if not out_paths or not all(p.exists() for p in out_paths):
        return False
    script_paths = [root / name for name in handoff.scripts()]
    existing_scripts = [p for p in script_paths if p.exists()]
    if not existing_scripts:
        return True
    newest_script = max(p.stat().st_mtime for p in existing_scripts)
    oldest_output = min(p.stat().st_mtime for p in out_paths)
    return oldest_output + MTIME_TOLERANCE >= newest_script


def waiting_message(handoff: Handoff) -> str:
    cells = ", ".join("`" + " ".join(c) + "`" for c in handoff.commands)
    return (
        f"Run the next cell(s) ({cells}) to see and run the code, then run this cell again "
        "to continue."
    )


@dataclass
class StageContext:
    project: Project
    llm: LLM
    questioner: Questioner
    explainer: Explainer | None
    display: Callable[[str], None]
    display_figure: Callable[[Path, str], None] = field(
        default_factory=lambda: render_display_figure
    )
    stage: str = ""

    def spec(self) -> Spec:
        data = self.project.read_json(config.SPEC_FILE)
        if not data:
            raise SpecError("spec.json not found; run the intake stage first")
        return Spec.from_dict(data)

    def learning_level(self) -> str:
        """The user's chosen level, or the default before intake has run."""
        try:
            return self.spec().learning_level
        except (SpecError, TypeError, ValueError):
            return DEFAULT_LEARNING_LEVEL


class Stage(Protocol):
    name: str

    def prepare(self, ctx: StageContext) -> Handoff | None: ...

    def outputs_ready(self, ctx: StageContext, handoff: Handoff) -> bool: ...

    def debrief(self, ctx: StageContext) -> None: ...

    def is_complete(self, ctx: StageContext) -> bool: ...


class ScriptStageBase:
    """Mixin for stages whose handoff is 'run these scripts'."""

    name = ""

    def outputs_ready(self, ctx: StageContext, handoff: Handoff) -> bool:
        return outputs_ready(ctx.project.root, handoff)


def stage_prepare(stage, ctx: StageContext) -> Handoff | None:
    """Run a stage's first phase. Single-phase stages that only define `run` still work."""
    prepare = getattr(stage, "prepare", None)
    if prepare is not None:
        return prepare(ctx)
    stage.run(ctx)
    return None


def stage_debrief(stage, ctx: StageContext) -> None:
    debrief = getattr(stage, "debrief", None)
    if debrief is not None:
        debrief(ctx)


def stage_outputs_ready(stage, ctx: StageContext, handoff: Handoff) -> bool:
    ready = getattr(stage, "outputs_ready", None)
    if ready is not None:
        return ready(ctx, handoff)
    return outputs_ready(ctx.project.root, handoff)
```

- [ ] **Step 4: Rewrite the orchestrator loop**

Replace `mlagent/orchestrator.py`:

```python
"""Run stages in order, checkpointing to state.json so a Colab reset can resume."""

from __future__ import annotations

from mlagent import config
from mlagent.stages.base import (
    Handoff,
    Stage,
    StageContext,
    stage_debrief,
    stage_outputs_ready,
    stage_prepare,
    waiting_message,
)
from mlagent.ui.questions import FormQuestioner

EMPTY_STATE = {"completed": [], "current": None, "forced": [], "prepared": [], "handoff": None}


class Orchestrator:
    def __init__(self, ctx: StageContext, stages: list[Stage]):
        self.ctx = ctx
        self.stages = list(stages)

    def _state(self) -> dict:
        state = self.ctx.project.read_json(config.STATE_FILE, default=None)
        if not isinstance(state, dict) or not isinstance(state.get("completed"), list):
            return dict(EMPTY_STATE, completed=[], forced=[], prepared=[])
        for key, default in EMPTY_STATE.items():
            state.setdefault(key, [] if isinstance(default, list) else default)
        if not isinstance(state.get("prepared"), list):
            state["prepared"] = []
        return state

    def _save(self, state: dict) -> None:
        self.ctx.project.write_json(config.STATE_FILE, state)

    def completed(self) -> list[str]:
        return list(self._state()["completed"])

    def waiting(self) -> Handoff | None:
        """The handoff the user still has to run, or None."""
        return Handoff.from_dict(self._state().get("handoff"))

    def mark_complete(self, name: str) -> None:
        state = self._state()
        if name not in state["completed"]:
            state["completed"].append(name)
        state["forced"] = [n for n in state["forced"] if n != name]
        state["prepared"] = [n for n in state["prepared"] if n != name]
        state["current"] = None
        state["handoff"] = None
        self._save(state)

    def reset(self, stage_name: str) -> None:
        """Forget this stage and every later one; they rerun even if their artifacts exist."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            raise ValueError(f"unknown stage {stage_name!r}; known: {names}")
        idx = names.index(stage_name)
        state = self._state()
        state["completed"] = [n for n in state["completed"] if n in names[:idx]]
        state["prepared"] = [n for n in state["prepared"] if n in names[:idx]]
        state["forced"] = names[idx:]
        state["current"] = None
        state["handoff"] = None
        self._save(state)

    def _stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"unknown stage {name!r}; known: {[s.name for s in self.stages]}")

    def debrief(self, name: str) -> None:
        """Force a stage's second phase, skipping the freshness check.

        The escape hatch for when Drive's mtimes lie about outputs the user really did make.
        """
        stage = self._stage(name)
        self.ctx.stage = name
        stage_debrief(stage, self.ctx)
        if stage.is_complete(self.ctx):
            self.mark_complete(name)

    def run(
        self, until: str | None = None, answers: dict[str, object] | None = None
    ) -> list[str]:
        """Advance the pipeline. `answers` supplies Colab form values for this call only."""
        if answers is None:
            return self._run(until)
        original = self.ctx.questioner
        self.ctx.questioner = FormQuestioner(
            answers, fallback=original, note=self.ctx.display
        )
        try:
            return self._run(until)
        finally:
            self.ctx.questioner = original

    def _run(self, until: str | None) -> list[str]:
        ran: list[str] = []
        for stage in self.stages:
            state = self._state()
            done = stage.name in state["completed"]
            forced = stage.name in state["forced"]
            if not done and not forced and stage.is_complete(self.ctx):
                self.mark_complete(stage.name)
                done = True
            if not done:
                self.ctx.stage = stage.name
                state = self._state()
                state["current"] = stage.name
                self._save(state)
                if stage.name in state["prepared"]:
                    handoff = Handoff.from_dict(state.get("handoff"))
                else:
                    self.ctx.display(f"**Stage: {stage.name}**")
                    handoff = stage_prepare(stage, self.ctx)
                    state = self._state()
                    state["prepared"] = [*state["prepared"], stage.name]
                    state["handoff"] = handoff.to_dict() if handoff is not None else None
                    self._save(state)
                if handoff is not None and not stage_outputs_ready(stage, self.ctx, handoff):
                    self.ctx.display(waiting_message(handoff))
                    return ran
                stage_debrief(stage, self.ctx)
                if stage.is_complete(self.ctx):
                    self.mark_complete(stage.name)
                    ran.append(stage.name)
                else:
                    self.ctx.display(
                        f"Stage {stage.name} did not finish; rerun orch.run() to continue."
                    )
                    return ran
            if until is not None and stage.name == until:
                break
        if not ran:
            state = self._state()
            if all(s.name in state["completed"] for s in self.stages):
                self.ctx.display("Nothing to do: all stages complete.")
        return ran
```

- [ ] **Step 5: Move intake and codegen to the two-phase shape**

In `mlagent/stages/intake.py`, rename `def run(self, ctx)` to `def prepare(self, ctx) -> None` and add below it:

```python
    def debrief(self, ctx: StageContext) -> None:
        """Intake needs no cells from the user; everything happened in prepare."""
        return None
```

Do exactly the same in `mlagent/stages/codegen.py`: rename `run` to `prepare` (its two early `return` statements stay as they are — returning `None` means "no handoff") and add the same no-op `debrief`.

Update the call sites in tests: `tests/test_intake.py` and `tests/test_codegen_stage.py` call `IntakeStage().run(ctx)` / `CodegenStage().run(ctx)` — change every one to `.prepare(ctx)`. `tests/test_train_stage.py`'s `prepared()` helper and `tests/test_report_stage.py`'s `trained()` helper both call `CodegenStage().run(ctx)`; change those to `CodegenStage().prepare(ctx)`.

- [ ] **Step 6: Run the orchestrator tests**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: PASS (17 tests).

- [ ] **Step 7: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass. `data`, `clean`, `train` and `report` still define only `run` and go through the legacy bridge; the end-to-end test is unaffected because none of them returns a handoff yet.

- [ ] **Step 8: Commit**

```bash
git add mlagent/stages/base.py mlagent/orchestrator.py mlagent/stages/intake.py \
        mlagent/stages/codegen.py tests
git commit -m "feat: two-phase stages with a Handoff and orchestrator phase state

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 4: The `profile.py` template, `copy_common`, and the caption table

**Files:**
- Create: `mlagent/templates/common/profile.py` (the directory holds copyable scripts, not a package — do not add an `__init__.py`)
- Create: `mlagent/captions.py`
- Modify: `mlagent/templates_io.py`
- Test: `tests/test_template_profile.py`, `tests/test_captions.py`, `tests/test_templates_io.py`

**Interfaces:**
- Consumes: `mlagent.profile.profile_dataframe` (as the contract the template must reproduce); `data_meta.json` keys `target`, `raw_path`, `clean_path`, `task_type`.
- Produces:
  - `templates_io.COMMON_DIR: Path` (`mlagent/templates/common`), `COMMON_FILES = ("profile.py",)`, `common_file(name: str) -> Path`, `copy_common(names: Sequence[str], project_root: Path) -> list[Path]`.
  - `mlagent/templates/common/profile.py` CLI: `python profile.py [--project DIR] [--input CSV] [--tag TAG]`; defaults to the raw CSV named by `data_meta.json` and `--tag raw`; writes `profile_{tag}.json` and `plots/{tag}_histograms.png`, `plots/{tag}_missing.png`, `plots/{tag}_class_balance.png` (classification) or `plots/{tag}_target_distribution.png` (regression), `plots/{tag}_correlation.png` (only with two or more numeric columns).
  - `profile_{tag}.json` = exactly `mlagent.profile.profile_dataframe(df, target)`'s keys (`n_rows`, `n_cols`, `duplicate_rows`, `memory_mb`, `columns`, `target`) plus `"figures": list[str]` of plot filenames.
  - `captions.CAPTIONS: dict[str, str]`, `captions.caption_for(path: Path | str) -> str` (longest matching filename suffix; `""` when nothing matches).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_captions.py`:

```python
from __future__ import annotations

from pathlib import Path

from mlagent.captions import CAPTIONS, caption_for

EXPECTED_KINDS = {
    "histograms", "missing", "class_balance", "target_distribution", "correlation",
    "clean_before_after_missing", "training_curves", "confusion", "roc_pr", "per_class",
    "pred_vs_actual", "residuals",
}


def test_every_figure_kind_has_a_caption():
    assert set(CAPTIONS) == EXPECTED_KINDS
    for kind, text in CAPTIONS.items():
        assert text.strip(), kind
        assert len(text) < 400, kind


def test_caption_for_matches_by_filename_suffix():
    assert caption_for(Path("plots/raw_histograms.png")) == CAPTIONS["histograms"]
    assert caption_for("plots/clean_histograms.png") == CAPTIONS["histograms"]
    assert caption_for(Path("plots/run1_val_confusion.png")) == CAPTIONS["confusion"]
    assert caption_for(Path("plots/test_roc_pr.png")) == CAPTIONS["roc_pr"]
    assert caption_for(Path("plots/training_curves.png")) == CAPTIONS["training_curves"]
    assert caption_for(Path("plots/run3_training.png")) == CAPTIONS["training_curves"]


def test_longest_suffix_wins_and_unknown_returns_empty():
    # "clean_before_after_missing" also ends in "missing"; the longer key must win.
    assert caption_for("plots/clean_before_after_missing.png") == (
        CAPTIONS["clean_before_after_missing"]
    )
    assert caption_for("plots/something_else.png") == ""
```

Create `tests/test_template_profile.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from mlagent.profile import profile_dataframe
from mlagent.templates_io import COMMON_FILES, copy_common


def run_profile(project, args=()):
    result = subprocess.run(
        [sys.executable, "profile.py", *args],
        cwd=str(project.root), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_copy_common_writes_the_scripts(project):
    written = copy_common(COMMON_FILES, project.root)
    assert [p.name for p in written] == list(COMMON_FILES)
    assert (project.root / "profile.py").exists()
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source


def test_profile_script_matches_the_library_profile(clean_project):
    project = clean_project
    (project.data_raw).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(project.data_clean / "data.csv")
    df.to_csv(project.data_raw / "data.csv", index=False)
    copy_common(COMMON_FILES, project.root)
    run_profile(project)

    written = json.loads((project.root / "profile_raw.json").read_text(encoding="utf-8"))
    figures = written.pop("figures")
    assert written == profile_dataframe(df, "target")
    assert set(figures) == {
        "raw_histograms.png", "raw_missing.png", "raw_class_balance.png",
        "raw_correlation.png",
    }
    for name in figures:
        assert (project.plots_dir / name).exists()


def test_profile_script_tag_and_input_flags_and_regression_target(regression_project):
    project = regression_project
    copy_common(COMMON_FILES, project.root)
    run_profile(project, ["--input", "data/clean/data.csv", "--tag", "clean"])
    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["target"]["kind"] == "numeric"
    assert "clean_target_distribution.png" in written["figures"]
    assert (project.plots_dir / "clean_target_distribution.png").exists()
    assert not (project.plots_dir / "clean_class_balance.png").exists()


def test_profile_script_has_walkthrough_sections_and_an_argv_guard(project):
    copy_common(COMMON_FILES, project.root)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5
```

Append to `tests/test_templates_io.py`:

```python
def test_common_files_are_locatable_and_copied(tmp_path):
    assert tio.COMMON_DIR.is_dir()
    assert tio.common_file("profile.py").exists()
    written = tio.copy_common(tio.COMMON_FILES, tmp_path)
    assert [p.name for p in written] == list(tio.COMMON_FILES)
    with pytest.raises(FileNotFoundError):
        tio.common_file("no_such_script.py")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_captions.py tests/test_template_profile.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.captions'`.

- [ ] **Step 3: Write the caption table**

Create `mlagent/captions.py`:

```python
"""Fixed 'how to read this chart' text, one entry per figure kind.

The agent adds a sentence about *this* dataset below expert level (see `teaching.py`);
these captions are the part that never changes and never needs an LLM call.
"""

from __future__ import annotations

from pathlib import Path

CAPTIONS: dict[str, str] = {
    "histograms": (
        "One [[histogram]] per numeric column: the x axis is the value, the height is how "
        "many rows fall in that bucket. Look for columns that are all one value, long tails "
        "on one side ([[skew]]), or two separate humps."
    ),
    "missing": (
        "Each row of the chart is a column of your data and each pixel across is a row; dark "
        "means the value is missing. Solid dark bands mean a column is mostly empty; vertical "
        "stripes mean whole rows are missing values together."
    ),
    "class_balance": (
        "How many rows carry each label. A large gap between the bars is [[class imbalance]]: "
        "a model can score well just by always predicting the biggest class, so accuracy alone "
        "will flatter it."
    ),
    "target_distribution": (
        "The spread of the value you are predicting. Check the range is what you expect and "
        "watch for a long tail: a few extreme targets pull [[regression]] errors around."
    ),
    "correlation": (
        "How strongly each pair of numeric columns moves together, from -1 (opposite) through "
        "0 (unrelated) to +1 (identical). A column almost perfectly correlated with the target "
        "is often [[data leakage]]."
    ),
    "clean_before_after_missing": (
        "The percentage of missing values per column before and after cleaning. Bars that "
        "shrink to zero were filled or dropped; bars that did not move were left alone on "
        "purpose."
    ),
    "training_curves": (
        "Left: [[loss]] per [[epoch]] for the training and validation splits. Right: the same "
        "for your chosen metric. Training loss falling while validation loss rises is "
        "[[overfitting]]; both flat is a [[plateau]]."
    ),
    "confusion": (
        "Rows are the true label, columns are what the model predicted, so the diagonal is "
        "correct. A bright off-diagonal cell names the two classes the model keeps confusing."
    ),
    "roc_pr": (
        "Left: the [[ROC curve]] — true positives against false positives as the decision "
        "threshold moves; further above the dashed line is better. Right: [[precision]] "
        "against [[recall]], which is the more honest view when classes are imbalanced."
    ),
    "per_class": (
        "Precision and recall for each class. Precision is how often a prediction of that "
        "class is right; recall is how much of that class the model finds. Small classes with "
        "low bars are the ones to fix."
    ),
    "pred_vs_actual": (
        "Each point is one row: actual value across, predicted value up. Perfect predictions "
        "sit on the dashed diagonal. Points bending away from it at one end mean the model is "
        "biased in that part of the range."
    ),
    "residuals": (
        "Left: the spread of actual minus predicted; a bell centred on zero is what you want. "
        "Right: the same errors against the prediction; a funnel or a curve means the model is "
        "missing structure rather than just being noisy."
    ),
}


# The train stage archives `training_curves.png` as `run{N}_training.png`; the shortened
# suffix maps back to the same caption.
ALIASES = {"training": "training_curves"}


def caption_for(path: Path | str) -> str:
    """The caption for a figure, matched on the longest key that ends its filename stem."""
    stem = Path(path).stem
    best = ""
    for kind in CAPTIONS:
        if (stem == kind or stem.endswith(f"_{kind}")) and len(kind) > len(best):
            best = kind
    if not best:
        best = ALIASES.get(stem.rsplit("_", 1)[-1], "")
    return CAPTIONS.get(best, "")
```

- [ ] **Step 4: Add the common-template helpers**

In `mlagent/templates_io.py`, after `TEMPLATES_DIR`:

```python
COMMON_DIRNAME = "common"
COMMON_DIR = TEMPLATES_DIR / COMMON_DIRNAME
COMMON_FILES = ("profile.py",)
```

and after `copy_template`:

```python
def common_file(name: str) -> Path:
    """Path to a shared template script (`profile.py`, `clean.py`)."""
    path = COMMON_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no common template named {name!r} under {COMMON_DIR}")
    return path


def copy_common(names: Sequence[str], project_root: Path) -> list[Path]:
    """Copy shared scripts into the project folder, overwriting; return the paths."""
    written: list[Path] = []
    for filename in names:
        target = Path(project_root) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common_file(filename), target)
        written.append(target)
    return written
```

Add `from collections.abc import Sequence` to the imports.

- [ ] **Step 5: Write the profiling script**

Create `mlagent/templates/common/profile.py`:

```python
"""Profile a dataset and draw its four overview figures. Generated by mlagent; safe to edit.

Run it from the project folder:

    python profile.py                                     profile the raw data
    python profile.py --input data/clean/data.csv --tag clean

Writes `profile_{tag}.json` and PNGs into `plots/`. Imports only pandas, numpy and
matplotlib, so you can run it anywhere the project folder exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pandas.api import types as ptypes

# --- settings ---
PROJECT_DIR = Path(".")
META_FILE = "data_meta.json"
TAG = "raw"
INPUT = None  # None means "the raw CSV named in data_meta.json"
MAX_HIST_COLS = 12
MAX_MISSING_ROWS = 500
MAX_MISSING_COLS = 60
MAX_CORR_COLS = 20
CATEGORICAL_MAX_UNIQUE = 50
MAX_LISTED_CATEGORIES = 20

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#f3a17f", "#d95926"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
DIV_CMAP = LinearSegmentedColormap.from_list("mlagent_div", DIVERGING)


def style(ax, title=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


# --- reading the data ---
def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame(project_dir: Path, input_path: str | None, meta: dict):
    relative = input_path or meta.get("raw_path") or "data/raw/data.csv"
    path = project_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"no data file at {path}")
    return pd.read_csv(path), path


# --- describing the columns ---
def to_python(value):
    """numpy scalars become plain Python; NaN becomes None so the JSON is valid."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def is_categorical(s: pd.Series) -> bool:
    if ptypes.is_bool_dtype(s) or not ptypes.is_numeric_dtype(s):
        return True
    vals = s.dropna()
    if vals.empty or vals.nunique() > CATEGORICAL_MAX_UNIQUE:
        return False
    return bool((vals % 1 == 0).all())


def column_info(name: str, s: pd.Series) -> dict:
    info = {
        "name": name,
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "missing_pct": round(float(s.isna().mean() * 100), 2),
        "n_unique": int(s.nunique(dropna=True)),
        "sample": [str(v) for v in s.dropna().unique()[:5]],
    }
    if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
        vals = s.dropna()
        if not vals.empty:
            info.update({
                "min": to_python(vals.min()),
                "max": to_python(vals.max()),
                "mean": to_python(round(float(vals.mean()), 6)),
                "std": to_python(round(float(vals.std()), 6)) if len(vals) > 1 else 0.0,
            })
    return info


def target_info(name: str, s: pd.Series) -> dict:
    if is_categorical(s):
        all_counts = s.value_counts(dropna=True)
        counts = all_counts.head(MAX_LISTED_CATEGORIES)
        return {
            "name": name,
            "kind": "categorical",
            "n_classes": int(all_counts.shape[0]),
            "total": int(all_counts.sum()),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
    vals = s.dropna()
    if vals.empty:
        return {"name": name, "kind": "numeric", "min": None, "max": None, "mean": None,
                "std": None}
    return {
        "name": name,
        "kind": "numeric",
        "min": to_python(vals.min()),
        "max": to_python(vals.max()),
        "mean": to_python(round(float(vals.mean()), 6)),
        "std": to_python(round(float(vals.std()), 6)) if len(vals) > 1 else 0.0,
    }


def profile_frame(df: pd.DataFrame, target: str | None) -> dict:
    profile = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3),
        "columns": [column_info(str(c), df[c]) for c in df.columns],
        "target": None,
    }
    if target is not None and target in df.columns:
        profile["target"] = target_info(target, df[target])
    return profile


# --- figures ---
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])
    ]


def show(fig) -> None:
    """Display in a notebook if one is running; do nothing from a plain shell."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is not None:
        display(fig)


def save(fig, plots_dir: Path, name: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig)
    plt.close(fig)
    return f"{name}.png"


def histograms(df: pd.DataFrame):
    cols = numeric_columns(df)[:MAX_HIST_COLS]
    n = max(1, len(cols))
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax, col in zip(axes.flat, cols, strict=False):
        ax.hist(df[col].dropna(), bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        style(ax, col)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    if not cols:
        axes[0][0].set_visible(True)
        axes[0][0].text(0.5, 0.5, "no numeric columns", ha="center", color=MUTED)
    fig.suptitle("Feature distributions", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def missing_matrix(df: pd.DataFrame):
    sample = df if len(df) <= MAX_MISSING_ROWS else df.sample(
        MAX_MISSING_ROWS, random_state=0
    ).sort_index()
    columns = list(df.columns)[:MAX_MISSING_COLS]
    mat = sample[columns].isna().to_numpy().T.astype(float)
    fig = plt.figure(figsize=(7, 0.28 * len(columns) + 1.5), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels([str(c) for c in columns], fontsize=8, color=INK_2)
    ax.set_xlabel(f"rows (showing {len(sample)} of {len(df)})", color=MUTED, fontsize=8)
    style(ax, "Missing values (dark = missing)", grid_axis="none")
    fig.tight_layout()
    return fig


def class_balance(counts: dict, total: int | None):
    fig = plt.figure(figsize=(5, 0.5 * max(3, len(counts)) + 1.2), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    labels = [str(k) for k in counts]
    values = [int(v) for v in counts.values()]
    denom = total if total else (sum(values) or 1)
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v} ({v / denom:.0%})", va="center", color=INK_2, fontsize=8)
    style(ax, "Target class balance", grid_axis="x")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3 if values else 1)
    fig.tight_layout()
    return fig


def target_distribution(series: pd.Series):
    fig = plt.figure(figsize=(5, 2.8), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.hist(series.dropna(), bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    style(ax, "Target distribution")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric = df[numeric_columns(df)[:MAX_CORR_COLS]]
    fig = plt.figure(figsize=(6, 5), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    corr = numeric.corr().to_numpy()
    im = ax.imshow(corr, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ticks = range(numeric.shape[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(numeric.columns, rotation=60, ha="right", fontsize=8, color=INK_2)
    ax.set_yticklabels(numeric.columns, fontsize=8, color=INK_2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=MUTED, labelsize=8)
    style(ax, "Correlation between numeric columns", grid_axis="none")
    fig.tight_layout()
    return fig


def draw_all(df: pd.DataFrame, profile: dict, plots_dir: Path, tag: str) -> list[str]:
    figures = [
        save(histograms(df), plots_dir, f"{tag}_histograms"),
        save(missing_matrix(df), plots_dir, f"{tag}_missing"),
    ]
    target = profile.get("target")
    if target and target["kind"] == "categorical":
        figures.append(
            save(class_balance(target["counts"], target.get("total")), plots_dir,
                 f"{tag}_class_balance")
        )
    elif target:
        figures.append(
            save(target_distribution(df[target["name"]]), plots_dir,
                 f"{tag}_target_distribution")
        )
    if len(numeric_columns(df)) >= 2:
        figures.append(save(correlation_heatmap(df), plots_dir, f"{tag}_correlation"))
    return figures


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script; nothing when the notebook kernel runs this file."""
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if not name.endswith(".py"):
        return []
    return sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a dataset and draw its figures.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--input", default=INPUT, help="CSV path relative to --project")
    parser.add_argument("--tag", default=TAG, help="prefix for the output files")
    args = parser.parse_args(argv)

    project_dir = Path(args.project).resolve()
    meta = read_json(project_dir / META_FILE, default={}) or {}
    df, path = load_frame(project_dir, args.input, meta)
    profile = profile_frame(df, meta.get("target"))
    profile["figures"] = draw_all(df, profile, project_dir / "plots", args.tag)
    out = project_dir / f"profile_{args.tag}.json"
    out.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"profiled {path.name}: {profile['n_rows']} rows x {profile['n_cols']} columns "
        f"-> {out.name} and {len(profile['figures'])} figures",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

`show()` is the only place IPython is touched, and it is guarded: with the notebook's inline backend the figure appears in the cell as soon as it is saved; from a plain shell nothing happens and the PNG on disk is the output.

- [ ] **Step 6: Run the new tests**

Run: `python -m pytest tests/test_captions.py tests/test_template_profile.py tests/test_templates_io.py -q`
Expected: PASS. If `test_profile_script_matches_the_library_profile` fails on a key, the difference is a real divergence between `profile_frame` here and `mlagent.profile.profile_dataframe`; fix the template, not the test.

- [ ] **Step 7: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add mlagent/captions.py mlagent/templates/common mlagent/templates_io.py tests
git commit -m "feat: standalone profile.py template, copy_common, and the figure caption table

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---
### Task 5: A self-contained `clean.py` template

**Files:**
- Create: `mlagent/templates/common/clean.py`
- Modify: `mlagent/cleaning.py`
- Modify: `mlagent/templates_io.py` (`COMMON_FILES` documentation only — `clean.py` is rendered, not copied)
- Test: `tests/test_template_clean.py`, `tests/test_cleaning.py`

**Interfaces:**
- Consumes: `templates_io.common_file` (Task 4); `cleaning.apply_steps` (unchanged, the reference implementation).
- Produces:
  - `cleaning.STEPS_MARKER = 'STEPS_JSON = r"""[]"""'` and `cleaning.render_clean_py(steps: list[dict]) -> str`, which reads `templates/common/clean.py` and substitutes the JSON into that one line. It raises `ValueError` if the marker is missing.
  - `mlagent/templates/common/clean.py` CLI: `python clean.py [--project DIR]`. Reads `data/raw/data.csv` (or `data_meta.json`'s `raw_path`) and writes `data/clean/data.csv`, `profile_clean.json` and `plots/clean_before_after_missing.png`.
  - `profile_clean.json` = `{"before": SUMMARY, "after": SUMMARY, "steps": [...], "figures": ["clean_before_after_missing.png"]}` where `SUMMARY = {"n_rows": int, "n_cols": int, "duplicate_rows": int, "columns": [{"name", "dtype", "missing", "missing_pct", "n_unique"}]}`.
  - The rendered script exposes `STEPS: list[dict]` and `clean(df) -> pd.DataFrame` at module level, so the existing "clean.py reproduces the cleaned CSV" tests keep working by `exec`-ing it.
- `CLEAN_PY_TEMPLATE` is deleted from `cleaning.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_template_clean.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from mlagent.cleaning import apply_steps, render_clean_py

STEPS = [
    {"op": "drop_columns", "params": {"columns": ["row_id"]}},
    {"op": "drop_duplicates", "params": {}},
    {"op": "fill_missing", "params": {"column": "f1", "strategy": "median"}},
    {"op": "normalise_categories", "params": {"column": "cat"}},
    {"op": "clip_outliers", "params": {"column": "f2", "lower": -3.0, "upper": 3.0}},
    {"op": "coerce_numeric", "params": {"column": "f2"}},
    {"op": "drop_rows_missing_target", "params": {"target": "target"}},
]


def messy(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n) * 4,
        "cat": rng.choice(["A ", " b", "C"], size=n),
        "target": rng.integers(0, 2, size=n).astype(float),
    })
    df.loc[:4, "f1"] = np.nan
    df.loc[5, "target"] = np.nan
    return pd.concat([df, df.iloc[:3]], ignore_index=True)


def prepare(project) -> pd.DataFrame:
    df = messy()
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / "data.csv", index=False)
    project.write_json("data_meta.json", {
        "target": "target", "task_type": "tabular_classification",
        "raw_path": "data/raw/data.csv",
    })
    (project.root / "clean.py").write_text(render_clean_py(STEPS), encoding="utf-8")
    return df


def run_clean(project):
    result = subprocess.run(
        [sys.executable, "clean.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_rendered_clean_py_matches_apply_steps_and_writes_artifacts(project):
    raw = prepare(project)
    run_clean(project)

    produced = pd.read_csv(project.data_clean / "data.csv")
    expected = apply_steps(raw, STEPS).reset_index(drop=True)
    expected["target"] = expected["target"].astype("int64")
    pd.testing.assert_frame_equal(produced, expected, check_dtype=False)

    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["steps"] == STEPS
    assert written["figures"] == ["clean_before_after_missing.png"]
    assert written["before"]["n_rows"] == len(raw)
    assert written["after"]["n_rows"] == len(produced)
    assert {c["name"] for c in written["after"]["columns"]} == set(produced.columns)
    assert (project.plots_dir / "clean_before_after_missing.png").exists()


def test_rendered_clean_py_is_importable_and_standalone(project):
    prepare(project)
    source = (project.root / "clean.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    namespace: dict = {}
    exec(compile(source, "clean.py", "exec"), namespace)
    assert namespace["STEPS"] == STEPS
    out = namespace["clean"](messy())
    pd.testing.assert_frame_equal(out, apply_steps(messy(), STEPS), check_dtype=False)


def test_empty_steps_render_and_run(project):
    df = messy()
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / "data.csv", index=False)
    project.write_json("data_meta.json", {"target": "target",
                                          "task_type": "tabular_regression",
                                          "raw_path": "data/raw/data.csv"})
    (project.root / "clean.py").write_text(render_clean_py([]), encoding="utf-8")
    run_clean(project)
    produced = pd.read_csv(project.data_clean / "data.csv")
    assert len(produced) == len(df)
```

In `tests/test_cleaning.py`, replace the body of the `render_clean_py` test (currently around line 78) with:

```python
def test_render_clean_py_embeds_the_steps_and_reproduces_apply_steps():
    steps = [
        {"op": "drop_columns", "params": {"columns": ["id"]}},
        {"op": "fill_missing", "params": {"column": "x", "strategy": "median"}},
    ]
    source = render_clean_py(steps)
    assert "from mlagent" not in source
    namespace: dict = {}
    exec(compile(source, "clean.py", "exec"), namespace)
    pd.testing.assert_frame_equal(namespace["clean"](df()), apply_steps(df(), steps))
    assert namespace["STEPS"] == steps
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_template_clean.py -q`
Expected: FAIL — the rendered script still calls `from mlagent.cleaning import apply_steps`, so the subprocess run inside a bare project folder fails with `ModuleNotFoundError`, and `test_rendered_clean_py_is_importable_and_standalone` fails on `"from mlagent" not in source`.

- [ ] **Step 3: Write the clean template**

Create `mlagent/templates/common/clean.py`:

```python
"""Apply the cleaning steps you approved. Generated by mlagent; safe to edit.

Run it from the project folder:

    python clean.py

Reads the raw CSV named in `data_meta.json`, applies `STEPS` in order, and writes
`data/clean/data.csv`, `profile_clean.json` (a before/after summary) and
`plots/clean_before_after_missing.png`. Every operation is a plain pandas function you
can read and change; editing `STEPS` and rerunning is the supported way to redo the
cleaning differently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# --- settings ---
PROJECT_DIR = Path(".")
META_FILE = "data_meta.json"
CLEAN_PATH = Path("data") / "clean" / "data.csv"
PROFILE_FILE = "profile_clean.json"
FIGURE_NAME = "clean_before_after_missing"
MAX_PLOTTED_COLS = 60

# --- the steps mlagent recorded ---
# Each step is {"op": name, "params": {...}} and runs in order. Edit and rerun freely.
STEPS_JSON = r"""[]"""
STEPS: list[dict] = json.loads(STEPS_JSON)

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)


# --- cleaning operations ---
def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing(df: pd.DataFrame, column: str, strategy: str = "median",
                 value=None) -> pd.DataFrame:
    if column not in df.columns:
        return df
    s = df[column]
    if strategy == "median":
        fill = s.median()
    elif strategy == "mode":
        mode = s.mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else value
    elif strategy == "constant":
        fill = value
    else:
        raise ValueError(f"unknown fill strategy {strategy!r}")
    if fill is None:
        return df
    out = df.copy()
    out[column] = s.fillna(fill)
    return out


def drop_rows_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.dropna(subset=[target]).reset_index(drop=True)


def normalise_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    s = df[column]
    if not (s.dtype == "object" or s.dtype.name == "string"):
        return df
    out = df.copy()
    s = out[column]
    out[column] = s.where(s.isna(), s.astype(str).str.strip().str.lower())
    return out


def clip_outliers(df: pd.DataFrame, column: str, lower: float, upper: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = out[column].clip(lower=lower, upper=upper)
    return out


def coerce_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


OPS = {
    "drop_columns": drop_columns,
    "drop_duplicates": drop_duplicates,
    "fill_missing": fill_missing,
    "drop_rows_missing_target": drop_rows_missing_target,
    "normalise_categories": normalise_categories,
    "clip_outliers": clip_outliers,
    "coerce_numeric": coerce_numeric,
}


def clean(df: pd.DataFrame, steps: list[dict] | None = None) -> pd.DataFrame:
    """Run every step in order and return a new frame; the input is never modified."""
    steps = STEPS if steps is None else steps
    if not steps:
        return df.copy()
    out = df
    for step in steps:
        op = step.get("op")
        if op not in OPS:
            raise ValueError(f"unknown cleaning op {op!r}")
        out = OPS[op](out, **step.get("params", {}))
    return out


def cast_classification_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Whole-number float labels (a side effect of NaNs) become ints, so a label reads
    as `1`, not `1.0`."""
    if target not in df.columns:
        return df
    s = df[target]
    if not ptypes.is_float_dtype(s) or s.isna().any():
        return df
    non_null = s.dropna()
    if non_null.empty or not (non_null % 1 == 0).all():
        return df
    out = df.copy()
    out[target] = s.astype("int64")
    return out


# --- before and after summary ---
def summarise(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "missing": int(df[c].isna().sum()),
                "missing_pct": round(float(df[c].isna().mean() * 100), 2),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
    }


# --- figure ---
def before_after_missing(before: dict, after: dict):
    all_names = [c["name"] for c in before["columns"]]
    names = all_names[:MAX_PLOTTED_COLS]
    before_pct = {c["name"]: c["missing_pct"] for c in before["columns"]}
    after_pct = {c["name"]: c["missing_pct"] for c in after["columns"]}
    b = [before_pct[n] for n in names]
    a = [after_pct.get(n, 0.0) for n in names]
    y = np.arange(len(names))
    fig = plt.figure(figsize=(6, 0.35 * max(3, len(names)) + 1.4), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.barh(y - 0.18, b, height=0.34, color=SERIES[0], label="before cleaning")
    ax.barh(y + 0.18, a, height=0.34, color=SERIES[1], label="after cleaning")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK_2)
    ax.set_xlabel("% missing", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    title = "Missing values before and after cleaning"
    if len(names) < len(all_names):
        title += f" - showing {len(names)} of {len(all_names)} columns"
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def show(fig) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is not None:
        display(fig)


def save(fig, plots_dir: Path, name: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig)
    plt.close(fig)
    return f"{name}.png"


# --- command line ---
def cli_argv() -> list[str]:
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if not name.endswith(".py"):
        return []
    return sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the approved cleaning steps.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()

    meta_path = project_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    raw_path = project_dir / str(meta.get("raw_path") or "data/raw/data.csv")
    raw = pd.read_csv(raw_path)

    cleaned = clean(raw).reset_index(drop=True)
    target = meta.get("target")
    if target and meta.get("task_type") == "tabular_classification":
        cleaned = cast_classification_target(cleaned, str(target))

    out_path = project_dir / CLEAN_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    before, after = summarise(raw), summarise(cleaned)
    figures = [save(before_after_missing(before, after), project_dir / "plots", FIGURE_NAME)]
    (project_dir / PROFILE_FILE).write_text(
        json.dumps(
            {"before": before, "after": after, "steps": STEPS, "figures": figures},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"cleaned {raw_path.name}: {before['n_rows']} x {before['n_cols']} -> "
        f"{after['n_rows']} x {after['n_cols']} rows x columns "
        f"after {len(STEPS)} step(s) -> {CLEAN_PATH.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

- [ ] **Step 4: Render the template instead of the old string**

In `mlagent/cleaning.py`, delete `CLEAN_PY_TEMPLATE` and replace `render_clean_py` with:

```python
STEPS_MARKER = 'STEPS_JSON = r"""[]"""'


def render_clean_py(steps: list[dict]) -> str:
    """The standalone cleaning script for this project: the shared template with the
    approved steps substituted into its one placeholder line."""
    source = common_file("clean.py").read_text(encoding="utf-8")
    if STEPS_MARKER not in source:
        raise ValueError(
            f"templates/common/clean.py no longer contains the line {STEPS_MARKER!r}"
        )
    payload = json.dumps(steps, indent=2)
    return source.replace(STEPS_MARKER, f'STEPS_JSON = r"""{payload}"""', 1)
```

Add `from mlagent.templates_io import common_file` to the imports of `cleaning.py`. `templates_io` imports nothing from `cleaning`, so there is no cycle.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_template_clean.py tests/test_cleaning.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: `tests/test_clean_stage.py` and `tests/test_pipeline_e2e.py` still pass — both `exec` the rendered `clean.py` and call `namespace["clean"](df)`, which the template still provides with the same meaning. `CleanStage` itself is untouched in this task, so `profile_clean.json` is still written by the stage in the old shape; Task 7 moves it.

- [ ] **Step 7: Commit**

```bash
git add mlagent/templates/common/clean.py mlagent/cleaning.py tests
git commit -m "feat: clean.py is a self-contained template with embedded ops, steps and figure

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 6: `display_figure` and the two-phase data stage

**Files:**
- Modify: `mlagent/ui/render.py`
- Modify: `mlagent/stages/base.py` (already carries the `display_figure` field from Task 3 — nothing further)
- Modify: `mlagent/stages/data.py`
- Test: `tests/test_render.py`, `tests/test_data_stage.py`

**Interfaces:**
- Consumes: `Handoff`, `ScriptStageBase`, `StageContext.display_figure` (Task 3); `templates_io.copy_common`, `COMMON_FILES` (Task 4); `captions.caption_for` (Task 4); `profile.profile_markdown`.
- Produces:
  - `render.figure_html(path: Path, caption: str = "") -> str` and `render.display_figure(path: Path | str, caption: str = "") -> None` (embeds the PNG as a base64 `data:` URI inside a `<figure>` and renders the caption through `to_html` so `[[term]]`s stay clickable).
  - `DataStage.prepare(ctx) -> Handoff` returning `Handoff("data", [["profile.py"]], ["profile_raw.json"])`; `DataStage.debrief(ctx) -> None`; `DataStage.is_complete` now also requires `profile_raw.json`.
  - Every fixed data question carries a key: `data.n_rows`, `data.n_features`, `data.n_classes`, `data.class_balance`, `data.noise`, `data.inject_quirks`, `data.drive_path`, `data.hf_query`, `data.target_column`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
def test_figure_html_embeds_the_png_and_renders_the_caption(tmp_path):
    from mlagent.ui.render import figure_html

    png = tmp_path / "raw_histograms.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    html = figure_html(png, "Each [[histogram]] shows one column.")
    assert "data:image/png;base64," in html
    assert "<figure" in html and "</figure>" in html
    assert 'data-term="histogram"' in html
    assert "raw_histograms.png" in html  # the filename is shown as the figure's label


def test_figure_html_without_a_caption_and_for_a_missing_file(tmp_path):
    from mlagent.ui.render import figure_html

    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    assert "<figcaption" not in figure_html(png)
    missing = figure_html(tmp_path / "nope.png", "caption")
    assert "nope.png" in missing and "data:image/png" not in missing


def test_display_figure_passes_html_to_ipython(tmp_path, monkeypatch):
    from mlagent.ui import render

    png = tmp_path / "test_confusion.png"
    png.write_bytes(b"\x89PNG")
    captured = []
    monkeypatch.setattr(render, "_display", lambda obj: captured.append(obj))
    render.display_figure(png, "Rows are the true label.")
    assert captured and "figure" in captured[0].data
```

Rewrite `tests/test_data_stage.py` around the two phases. Replace `make_ctx` and every test body that called `DataStage().run(ctx)`:

```python
from pathlib import Path

from mlagent.stages.base import Handoff
from mlagent.templates_io import COMMON_FILES


def make_ctx(project, answers, source="synthetic", task="tabular_classification", llm=None):
    project.write_json("spec.json", {**SPEC, "data_source": source, "task_type": task})
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Narrative [[class balance]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return ctx, shown, figures


def run_profile_script(project):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "profile.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_synthetic_classification_prepares_then_debriefs(project):
    ctx, shown, figures = make_ctx(project, ["300", "5", "2", "0.6", "0.1", "y"])
    stage = DataStage()
    assert not stage.is_complete(ctx)
    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="data", commands=[["profile.py"]],
                              outputs=["profile_raw.json"])
    assert (project.root / "profile.py").exists()
    for name in COMMON_FILES:
        assert (project.root / name).exists()
    df = pd.read_csv(project.data_raw / RAW_FILE)
    assert len(df) > 300 and "row_id" in df.columns and "target" in df.columns
    meta = project.read_json(META_FILE)
    assert meta["source"] == "synthetic" and meta["target"] == "target"
    assert meta["synth_config"]["n_samples"] == 300
    assert meta["raw_n_rows"] == len(df) and meta["raw_n_cols"] == df.shape[1]
    assert meta["raw_path"] == "data/raw/data.csv"
    assert not stage.is_complete(ctx)  # the profile has not been produced yet
    assert not stage.outputs_ready(ctx, handoff)

    run_profile_script(project)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert project.read_json(PROFILE_RAW_FILE)["target"]["kind"] == "categorical"
    shown_figures = {Path(p).name for p, _c in figures}
    assert shown_figures == {"raw_histograms.png", "raw_missing.png",
                             "raw_class_balance.png", "raw_correlation.png"}
    captions = {Path(p).name: c for p, c in figures}
    assert "histogram" in captions["raw_histograms.png"]
    assert any("[[class balance]]" in s for s in shown)
    assert any("Data profile" in s for s in shown)
```

Update the remaining tests in the file the same way: `test_synthetic_regression_without_quirks`, `test_drive_source_lists_files_and_asks_target`, `test_huggingface_source_searches_picks_and_loads` and `test_image_task_is_not_supported_yet` call `DataStage(...).prepare(ctx)` and assert on `data_meta.json` / raw CSV only (they never needed the profile). `test_llm_failure_still_completes` becomes:

```python
def test_llm_failure_still_completes(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"],
                                    llm=FakeLLM([]))
    stage = DataStage()
    stage.prepare(ctx)
    run_profile_script(project)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("Couldn't reach Claude" in s for s in shown)
```

And add:

```python
def test_missing_profile_debrief_says_so_and_stays_incomplete(project):
    ctx, shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"])
    stage = DataStage()
    stage.prepare(ctx)
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("profile.py" in s for s in shown)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render.py tests/test_data_stage.py -q`
Expected: FAIL with `ImportError: cannot import name 'figure_html' from 'mlagent.ui.render'`.

- [ ] **Step 3: Add `figure_html` and `display_figure`**

Append to `mlagent/ui/render.py` (and add `import base64` and `from pathlib import Path` at the top):

```python
FIGURE_CSS = """<style>
.mlagent-fig { margin: 0.75rem 0; max-width: 60rem; }
.mlagent-fig img { max-width: 100%; height: auto; display: block; }
.mlagent-fig figcaption {
  font-family: system-ui, sans-serif; font-size: 0.85rem; color: #52514e;
  line-height: 1.45; margin-top: 0.35rem;
}
.mlagent-fig .mlagent-figname { color: #898781; font-size: 0.75rem; }
</style>"""


def figure_html(path: Path | str, caption: str = "") -> str:
    """A PNG with its caption beneath, as one self-contained HTML block.

    The image is embedded as a data URI because a Colab cell cannot load a file from
    Google Drive by path, and the caption goes through `to_html` so `[[term]]` markup
    stays clickable.
    """
    path = Path(path)
    name = html.escape(path.name)
    if not path.exists():
        return f'{FIGURE_CSS}<div class="mlagent-fig">missing figure: {name}</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    parts = [
        FIGURE_CSS,
        '<figure class="mlagent-fig">',
        f'<img src="data:image/png;base64,{encoded}" alt="{name}">',
    ]
    if caption.strip():
        parts.append(
            f'<figcaption>{to_html(caption)}'
            f'<div class="mlagent-figname">{name}</div></figcaption>'
        )
    else:
        parts.append(f'<div class="mlagent-figname">{name}</div>')
    parts.append("</figure>")
    return "".join(parts)


def display_figure(path: Path | str, caption: str = "") -> None:
    from IPython.display import HTML

    _display(HTML(figure_html(path, caption)))
```

- [ ] **Step 4: Split the data stage into prepare and debrief**

Replace the body of `mlagent/stages/data.py` from the imports down to the end of `run`, keeping `_synthetic`, `_drive`, `_huggingface` and `_ask_target` (with keys added) and deleting `_plots` entirely:

```python
"""Data stage: obtain raw tabular data, then hand the user profile.py to run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlagent.captions import caption_for
from mlagent.datasources.drive import list_candidates, load_table
from mlagent.datasources.hf import load_tabular, search_datasets
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_markdown
from mlagent.prompts_io import audience, load_prompt
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate
from mlagent.templates_io import COMMON_FILES, copy_common

RAW_FILE = "data.csv"
META_FILE = "data_meta.json"
PROFILE_RAW_FILE = "profile_raw.json"
DEFAULT_QUIRKS = ("missing", "duplicates", "id_column", "categorical", "whitespace", "outliers")
DEFAULT_SEARCH_ROOTS = (Path("/content/drive/MyDrive"), Path("/content"))
TABULAR_TASKS = {"tabular_classification": "classification", "tabular_regression": "regression"}


class DataStage(ScriptStageBase):
    name = "data"

    def __init__(self, search_roots=None, hf_search=search_datasets, hf_load=load_tabular):
        self.search_roots = (
            list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        )
        self.hf_search = hf_search
        self.hf_load = hf_load

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        return (
            bool(meta and meta.get("target"))
            and (ctx.project.data_raw / RAW_FILE).exists()
            and ctx.project.exists(PROFILE_RAW_FILE)
        )

    def prepare(self, ctx: StageContext) -> Handoff:
        spec = ctx.spec()
        if spec.task_type not in TABULAR_TASKS:
            raise NotImplementedError(
                f"{spec.task_type} data is not supported yet (image tasks arrive in Milestone 6)"
            )
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx)
        else:
            df, target, meta = self._huggingface(ctx)

        ctx.project.data_raw.mkdir(parents=True, exist_ok=True)
        path = ctx.project.data_raw / RAW_FILE
        df.to_csv(path, index=False)
        meta.update(
            {
                "target": target,
                "raw_path": path.relative_to(ctx.project.root).as_posix(),
                "raw_n_rows": int(len(df)),
                "raw_n_cols": int(df.shape[1]),
                "task_type": spec.task_type,
            }
        )
        ctx.project.write_json(META_FILE, meta)
        copy_common(COMMON_FILES, ctx.project.root)
        ctx.display(
            f"I saved {len(df)} rows and {df.shape[1]} columns to `data/raw/data.csv` and wrote "
            "`profile.py`, which measures the data and draws four figures. Run it in the next "
            "cell; nothing about the raw data is changed."
        )
        return Handoff(
            stage=self.name, commands=[["profile.py"]], outputs=[PROFILE_RAW_FILE]
        )

    def debrief(self, ctx: StageContext) -> None:
        profile = ctx.project.read_json(PROFILE_RAW_FILE)
        if not isinstance(profile, dict):
            ctx.display(
                "I can't see `profile_raw.json` yet. Run the `profile.py` cell, then run this "
                "cell again."
            )
            return
        ctx.display(profile_markdown(profile))
        for name in profile.get("figures") or []:
            path = ctx.project.plots_dir / str(name)
            ctx.display_figure(path, caption_for(path))
        self._narrate(ctx, ctx.spec().to_dict(), profile)

    def _narrate(self, ctx: StageContext, spec: dict, profile: dict) -> None:
        payload = {k: v for k, v in profile.items() if k != "figures"}
        prompt = (
            "Project spec:\n" + json.dumps(spec, indent=2)
            + "\n\nData profile:\n" + json.dumps(payload, indent=2)
        )
        try:
            text = ask_text(
                ctx.llm,
                load_prompt("data", audience=audience(ctx.learning_level())),
                prompt,
            )
        except LLMError as exc:
            text = (
                f"(Couldn't reach Claude for a narrative: {exc}) "
                "Data saved. Next: the [[data cleaning]] audit."
            )
        ctx.display(text)
```

Add the form keys to the question calls in `_synthetic`, `_drive`, `_huggingface` and `_ask_target`:

```python
        n_samples = int(q.number("How many rows?", default=1000, minimum=100, maximum=200000,
                                 key="data.n_rows"))
        n_features = int(q.number("How many numeric features?", default=8, minimum=2,
                                  maximum=100, key="data.n_features"))
```
```python
            n_classes = int(q.number("How many classes?", default=2, minimum=2, maximum=10,
                                     key="data.n_classes"))
```
```python
                class_balance = q.number(
                    "Fraction of rows in the majority class (0.5 = balanced)?",
                    default=0.5, minimum=0.5, maximum=0.95, key="data.class_balance",
                )
```
```python
        noise = q.number(
            "Label/measurement noise (0 = clean, 0.3 = very noisy)?",
            default=0.1, minimum=0.0, maximum=1.0, key="data.noise",
        )
        inject = q.confirm(
            "Inject realistic data problems (missing values, duplicates, an ID column, messy "
            "categories, outliers) so the cleaning stage has work to do?",
            default=True, key="data.inject_quirks",
        )
```
In `_ask_target`: `..., cols, allow_other=False, key="data.target_column")`.
In `_drive`: both the `q.choice(...)` and the `q.text(...)` fallback take `key="data.drive_path"`.
In `_huggingface`: the `q.text(...)` search takes `key="data.hf_query"` and the `q.choice("Which dataset?", ...)` takes no key (the options depend on the search results, so a form field cannot answer it).

Note `mlagent.plots` and `mlagent.profile.profile_dataframe` are no longer imported by this module.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_render.py tests/test_data_stage.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: `tests/test_pipeline_e2e.py` now fails at the data stage, because the orchestrator returns as soon as it hits the handoff. Update that test to the handoff loop introduced properly in Task 14; for now, add the loop helper inline at the top of the test:

```python
def advance(orch, project, limit=12):
    """Run the orchestrator, running each handoff's cells as the user would."""
    import subprocess
    import sys

    ran: list[str] = []
    for _ in range(limit):
        ran += orch.run()
        handoff = orch.waiting()
        if handoff is None:
            return ran
        for command in handoff.commands:
            result = subprocess.run(
                [sys.executable, *command], cwd=str(project.root),
                capture_output=True, text=True, encoding="utf-8",
            )
            assert result.returncode == 0, result.stdout + result.stderr
    raise AssertionError("pipeline did not settle")
```
and call `ran = advance(orch, project)` in place of `ran = orch.run()`. The `ran == ALL_STAGES` assertion stays true because each stage is appended exactly once. Task 14 moves `advance` into `tests/conftest.py` as `run_handoff` plus a loop.

- [ ] **Step 7: Commit**

```bash
git add mlagent/ui/render.py mlagent/stages/data.py tests
git commit -m "feat: figures with captions in the notebook; the data stage hands off profile.py

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 7: The two-phase clean stage

**Files:**
- Modify: `mlagent/stages/clean.py`
- Test: `tests/test_clean_stage.py`

**Interfaces:**
- Consumes: `Handoff`, `ScriptStageBase` (Task 3); `render_clean_py` (Task 5); `caption_for`, `ctx.display_figure` (Tasks 4, 6); `audit_tabular`, `describe_step`, `profile.profile_dataframe` (for the pre-cleaning audit only).
- Produces:
  - `CleanStage.prepare(ctx) -> Handoff` returning `Handoff("clean", [["clean.py"]], ["data/clean/data.csv", "profile_clean.json"])`. It writes `audit.json`, `clean.py`, and the *decisions* half of `data_meta.json`: `splits`, `split_seed`, `dropped_columns`.
  - `CleanStage.debrief(ctx) -> None` reads `data/clean/data.csv` and completes `data_meta.json` with `clean_path`, `clean_n_rows`, `clean_n_cols`, `feature_columns`, `categorical_columns` and, for classification, `n_classes` and `class_labels`; shows the before/after figure with its caption; displays the summary.
  - `CleanStage.is_complete` requires the clean CSV, `audit.json`, `data_meta.json["splits"]` **and** `data_meta.json["feature_columns"]`.
  - `SPLIT_SEED = 42` module constant.
  - Question keys: `clean.drop_columns`, `clean.train_fraction`, `clean.val_fraction`. The per-fix "Apply this fix?" confirms stay unkeyed (they are invented per dataset, so a form cannot answer them).
  - `_apply_steps_safely` and `_find_failing_step` are deleted: the stage no longer applies steps in process.

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_clean_stage.py`'s helpers and first test; the `messy_df`, `user_id_collision_df` and `prepare` helpers stay, with `prepare` gaining the task type:

```python
import subprocess
import sys
from pathlib import Path

from mlagent.stages.base import Handoff
from mlagent.stages.clean import (
    AUDIT_FILE,
    CLEAN_FILE,
    CLEAN_PY,
    PROFILE_CLEAN_FILE,
    SPLIT_SEED,
    CleanStage,
)


def prepare(project, df, task_type="tabular_classification"):
    project.write_json("spec.json", {**SPEC, "task_type": task_type})
    project.data_raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.data_raw / RAW_FILE, index=False)
    project.write_json(META_FILE, {"source": "synthetic", "target": "target",
                                   "task_type": task_type,
                                   "raw_path": "data/raw/data.csv"})


def make_ctx(project, answers, llm=None):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(
        project=project, llm=llm or FakeLLM([[("text", "Report card [[missing values]]")]]),
        questioner=ScriptedQuestioner(answers), explainer=None, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return ctx, shown, figures


def run_clean_script(project):
    result = subprocess.run(
        [sys.executable, "clean.py"], cwd=str(project.root),
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_approved_fixes_are_written_to_clean_py_and_debriefed(project):
    df = messy_df()
    prepare(project, df)
    n_fixable = sum(1 for i in audit_tabular(df, "target") if i.fix)
    assert n_fixable >= 4
    ctx, shown, figures = make_ctx(project, ["y"] * n_fixable + ["", "0.7", "0.15"])
    stage = CleanStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(
        stage="clean", commands=[["clean.py"]],
        outputs=["data/clean/data.csv", "profile_clean.json"],
    )
    audit = project.read_json(AUDIT_FILE)
    assert len(audit["decisions"]) == len(audit["issues"]) and len(audit["steps"]) == n_fixable
    assert all(d["approved"] for d in audit["decisions"] if d["fix"])
    meta = project.read_json(META_FILE)
    assert meta["splits"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert meta["dropped_columns"] == [] and meta["split_seed"] == SPLIT_SEED
    assert "feature_columns" not in meta  # only known after clean.py has run
    assert (project.root / CLEAN_PY).exists()
    assert not stage.is_complete(ctx)

    run_clean_script(project)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    cleaned = pd.read_csv(project.data_clean / CLEAN_FILE)
    assert "row_id" not in cleaned.columns and "const" not in cleaned.columns
    assert cleaned.duplicated().sum() == 0 and cleaned["f1"].isna().sum() == 0
    assert set(cleaned["cat"].unique()) == {"a", "b", "c"}
    meta = project.read_json(META_FILE)
    assert meta["clean_path"] == "data/clean/data.csv"
    assert meta["clean_n_rows"] == len(cleaned) and meta["clean_n_cols"] == cleaned.shape[1]
    assert meta["feature_columns"] == [c for c in cleaned.columns if c != "target"]
    assert meta["categorical_columns"] == ["cat"]
    assert meta["n_classes"] == 2 and meta["class_labels"] == ["0", "1"]
    assert project.read_json(PROFILE_CLEAN_FILE)["after"]["n_rows"] == len(cleaned)
    assert [Path(p).name for p, _c in figures] == ["clean_before_after_missing.png"]
    assert "missing" in figures[0][1]
    assert any("[[missing values]]" in s for s in shown)
    assert any("Cleaning summary" in s for s in shown)

    namespace: dict = {}
    code = compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec")
    exec(code, namespace)
    pd.testing.assert_frame_equal(
        namespace["clean"](df).reset_index(drop=True), cleaned, check_dtype=False
    )
```

The other tests in the file become prepare/run/debrief sequences in the same way:
`test_skipped_fixes_and_manual_drops` asserts on `audit.json` and `data_meta.json["dropped_columns"]` straight after `prepare` (no script run needed);
`test_clean_data_has_no_issues_and_splits_are_adjusted` likewise;
`test_approved_drop_and_percolumn_fix_on_same_column_does_not_crash` runs the script and then asserts `"user_id" not in cleaned.columns`;
`test_target_becomes_float_via_nan_drop_is_saved_as_int` runs the script and asserts the CSV's target column parses as `int64`, because the cast now lives in the template.

Add:

```python
def test_debrief_without_the_clean_csv_says_so(project):
    prepare(project, messy_df())
    n_fixable = sum(1 for i in audit_tabular(messy_df(), "target") if i.fix)
    ctx, shown, _figures = make_ctx(project, ["n"] * n_fixable + ["", "0.7", "0.15"])
    stage = CleanStage()
    stage.prepare(ctx)
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("clean.py" in s for s in shown)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_clean_stage.py -q`
Expected: FAIL with `ImportError: cannot import name 'SPLIT_SEED' from 'mlagent.stages.clean'`.

- [ ] **Step 3: Rewrite the clean stage**

Replace `mlagent/stages/clean.py`:

```python
"""Clean stage: audit the raw data, agree the fixes, write clean.py, then read its output."""

from __future__ import annotations

import json

import pandas as pd
from pandas.api import types as ptypes

from mlagent.audit import Issue, audit_tabular
from mlagent.captions import caption_for
from mlagent.cleaning import describe_step, render_clean_py
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe
from mlagent.prompts_io import audience, load_prompt
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.stages.data import META_FILE, RAW_FILE

CLEAN_FILE = "data.csv"
CLEAN_REL_PATH = "data/clean/data.csv"
AUDIT_FILE = "audit.json"
PROFILE_CLEAN_FILE = "profile_clean.json"
CLEAN_PY = "clean.py"
MIN_TEST_FRACTION = 0.05
# Frozen independently of config.json's tunable `seed` so the held-out test split is the
# same for every run of this project.
SPLIT_SEED = 42


class CleanStage(ScriptStageBase):
    name = "clean"

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE) or {}
        return (
            (ctx.project.data_clean / CLEAN_FILE).exists()
            and ctx.project.exists(AUDIT_FILE)
            and bool(meta.get("splits"))
            and bool(meta.get("feature_columns"))
        )

    def prepare(self, ctx: StageContext) -> Handoff:
        meta = ctx.project.read_json(META_FILE) or {}
        target = meta.get("target")
        if not target:
            raise RuntimeError("data_meta.json has no target; run the data stage first")
        df = pd.read_csv(ctx.project.data_raw / RAW_FILE)
        before = profile_dataframe(df, target)

        issues = audit_tabular(df, target)
        decisions = self._review_issues(ctx, issues, before)
        steps = self._collect_steps(decisions)

        drops = self._ask_drops(ctx, df, target)
        if drops:
            steps.append({"op": "drop_columns", "params": {"columns": drops}})
        splits = self._ask_splits(ctx)

        ctx.project.write_json(
            AUDIT_FILE,
            {"issues": [i.to_dict() for i in issues], "decisions": decisions, "steps": steps},
        )
        (ctx.project.root / CLEAN_PY).write_text(render_clean_py(steps), encoding="utf-8")
        meta.update(
            {"splits": splits, "dropped_columns": drops, "split_seed": SPLIT_SEED}
        )
        ctx.project.write_json(META_FILE, meta)

        listed = "\n".join(f"- {describe_step(s)}" for s in steps) or "- (no changes)"
        ctx.display(
            f"I wrote `clean.py` with {len(steps)} step(s):\n\n{listed}\n\n"
            "Run it in the next cell. It reads `data/raw/data.csv`, writes "
            "`data/clean/data.csv`, and never touches the raw file — so if you change your "
            "mind you can edit `STEPS` in `clean.py` and run it again."
        )
        return Handoff(
            stage=self.name,
            commands=[[CLEAN_PY]],
            outputs=[CLEAN_REL_PATH, PROFILE_CLEAN_FILE],
        )

    def debrief(self, ctx: StageContext) -> None:
        clean_path = ctx.project.data_clean / CLEAN_FILE
        if not clean_path.exists():
            ctx.display(
                "I can't see `data/clean/data.csv` yet. Run the `clean.py` cell, then run this "
                "cell again."
            )
            return
        meta = ctx.project.read_json(META_FILE) or {}
        target = str(meta.get("target"))
        cleaned = pd.read_csv(clean_path)
        feature_columns = [str(c) for c in cleaned.columns if c != target]
        categorical_columns = [
            c for c in feature_columns
            if ptypes.is_object_dtype(cleaned[c])
            or ptypes.is_string_dtype(cleaned[c])
            or ptypes.is_bool_dtype(cleaned[c])
        ]
        meta.update(
            {
                "clean_path": CLEAN_REL_PATH,
                "clean_n_rows": int(len(cleaned)),
                "clean_n_cols": int(cleaned.shape[1]),
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
            }
        )
        if meta.get("task_type") == "tabular_classification" and target in cleaned.columns:
            labels = sorted(str(v) for v in cleaned[target].dropna().unique())
            meta["n_classes"] = len(labels)
            meta["class_labels"] = labels
        ctx.project.write_json(META_FILE, meta)

        profile = ctx.project.read_json(PROFILE_CLEAN_FILE) or {}
        for name in profile.get("figures") or []:
            path = ctx.project.plots_dir / str(name)
            ctx.display_figure(path, caption_for(path))
        ctx.display(self._summary(profile, meta))

    def _collect_steps(self, decisions: list[dict]) -> list[dict]:
        """Approved fixes become steps, unless the fix's column was itself dropped by
        another approved fix; those are recorded as skipped so decisions stay meaningful."""
        dropped_cols: set[str] = set()
        for d in decisions:
            if d["approved"] and d["fix"] and d["fix"]["op"] == "drop_columns":
                dropped_cols.update(d["fix"]["params"].get("columns", []))
        steps: list[dict] = []
        for d in decisions:
            if not d["approved"] or not d["fix"]:
                d["applied"] = False
                continue
            fix = d["fix"]
            col = fix.get("params", {}).get("column")
            if fix["op"] != "drop_columns" and col is not None and col in dropped_cols:
                d["applied"] = False
                d["reason"] = "column dropped"
                continue
            d["applied"] = True
            steps.append(fix)
        return steps

    def _review_issues(self, ctx: StageContext, issues: list[Issue], profile: dict) -> list[dict]:
        if not issues:
            ctx.display("The audit found no problems. Nice and clean.")
            return []
        self._explain(ctx, issues, profile)
        decisions: list[dict] = []
        for issue in issues:
            where = f" in `{issue.column}`" if issue.column else ""
            line = f"**[{issue.severity}] {issue.kind}**{where}: {issue.message}"
            if issue.fix:
                ctx.display(f"{line}\n\nProposed fix: {describe_step(issue.fix)}")
                approved = ctx.questioner.confirm(
                    "Apply this fix?", default=issue.severity != "low"
                )
            else:
                ctx.display(f"{line}\n\nNo automatic fix; noted for the modelling stage.")
                approved = False
            decisions.append(
                {"kind": issue.kind, "column": issue.column, "fix": issue.fix, "approved": approved}
            )
        return decisions

    def _explain(self, ctx: StageContext, issues: list[Issue], profile: dict) -> None:
        payload = json.dumps(
            {
                "issues": [i.to_dict() for i in issues],
                "spec": ctx.spec().to_dict(),
                "profile_summary": {
                    "n_rows": profile["n_rows"],
                    "n_cols": profile["n_cols"],
                    "target": profile.get("target"),
                },
            },
            indent=2,
            default=str,
        )
        try:
            text = ask_text(
                ctx.llm, load_prompt("clean", audience=audience(ctx.learning_level())), payload
            )
        except LLMError as exc:
            text = f"(Couldn't reach Claude for the report card: {exc}) Here are the issues found:"
        ctx.display(text)

    def _ask_drops(self, ctx: StageContext, df: pd.DataFrame, target: str) -> list[str]:
        cols = [str(c) for c in df.columns if c != target]
        raw = ctx.questioner.text(
            "Any other columns to drop before training? (comma-separated names, or leave blank)",
            default="", key="clean.drop_columns",
        )
        chosen = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in chosen if c not in cols]
        if unknown:
            ctx.display(f"Ignoring unknown columns: {', '.join(unknown)}")
        return [c for c in chosen if c in cols]

    def _ask_splits(self, ctx: StageContext) -> dict:
        q = ctx.questioner
        train = q.number("Fraction of rows for training?", default=0.7, minimum=0.5, maximum=0.9,
                         key="clean.train_fraction")
        val = q.number(
            "Fraction for validation (used during tuning)?", default=0.15, minimum=0.05,
            maximum=0.3, key="clean.val_fraction",
        )
        test = round(1 - train - val, 4)
        if test < MIN_TEST_FRACTION:
            val = round(max(MIN_TEST_FRACTION, 1 - MIN_TEST_FRACTION - train), 4)
            test = round(1 - train - val, 4)
            ctx.display(
                f"Adjusted so the [[test set]] keeps at least {MIN_TEST_FRACTION:.0%}: "
                f"train {train:g}, validation {val:g}, test {test:g}."
            )
        return {"train": round(train, 4), "val": round(val, 4), "test": test}

    def _summary(self, profile: dict, meta: dict) -> str:
        before = profile.get("before") or {}
        after = profile.get("after") or {}
        steps = profile.get("steps") or []
        lines = [
            "### Cleaning summary",
            "",
            f"- Rows: {before.get('n_rows', '?')} -> {after.get('n_rows', '?')}",
            f"- Columns: {before.get('n_cols', '?')} -> {after.get('n_cols', '?')}",
            f"- Steps applied: {len(steps)}",
        ]
        lines.extend(f"  - {describe_step(s)}" for s in steps)
        splits = meta.get("splits") or {}
        lines += [
            "",
            f"Splits frozen at train {splits.get('train')}, validation {splits.get('val')}, "
            f"test {splits.get('test')} with seed {meta.get('split_seed', SPLIT_SEED)}, so every "
            "run sees the same rows. Next: choosing a model and generating the "
            "[[training pipeline]].",
        ]
        return "\n".join(lines)
```

`_cast_classification_target` is deleted from this module — the template owns it now.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_clean_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass. `tests/test_pipeline_e2e.py` uses the `advance` helper from Task 6, so the clean handoff is run for it automatically.

- [ ] **Step 6: Commit**

```bash
git add mlagent/stages/clean.py tests/test_clean_stage.py
git commit -m "feat: the clean stage hands off clean.py and completes the meta from its output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---
### Task 8: Three model families and a nested config schema

**Files:**
- Rewrite: `mlagent/templates/tabular_sklearn/model.py`
- Rewrite: `mlagent/templates/tabular_sklearn/config_schema.json`
- Modify: `mlagent/templates_io.py`
- Modify: `mlagent/templates/tabular_sklearn/train.py` (epoch loop only)
- Modify: `mlagent/stages/codegen.py` (`is_complete` and `_propose` use the flat schema)
- Test: `tests/test_template_model.py` (new), `tests/test_templates_io.py`, `tests/test_template_data.py` (drop the model test), `tests/test_codegen_stage.py`, `tests/test_template_train.py`

**Interfaces:**
- Consumes: `data.load_data`'s `categorical_mask`, `task_type`, `classes`.
- Produces, in `model.py`:
  - `MODEL_TYPES = ("gradient_boosting", "random_forest", "linear")`
  - `class EpochModel` with attributes `estimator`, `model_type`, `task_type`, `step`, `epochs_fitted`, and methods `fit_epoch(X, y) -> EpochModel`, `predict(X)`, `predict_proba(X)`, property `classes_`.
  - `build_model(config: dict, task_type: str, categorical_mask: list[bool]) -> EpochModel`.
  - `grow` is deleted.
- Produces, in `templates_io.py`:
  - `schema_for(schema: dict, model_type: str) -> dict` — the flat rules for one family (`common` merged with `models[model_type]`); raises `ValueError` for an unknown family.
  - `model_types(schema: dict) -> list[str]`.
  - `validate_config`, `coerce_config`, `default_config` accept a rule of `{"type": "choice", "choices": [...]}` alongside `number` and `integer`, and take the **flat** schema returned by `schema_for`.
  - `CODE_FILES` is unchanged in this task (`evaluate.py` joins it in Task 9).
- `config.json` stays flat: the four common keys plus the chosen family's keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_template_model.py`:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_module():
    return load_module("model")


def classification_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X[0, 1] = np.nan  # every family must tolerate a missing value
    return X, y


def regression_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.2, size=n)
    return X, y


CONFIGS = {
    "gradient_boosting": {"model_type": "gradient_boosting", "seed": 1, "iters_per_epoch": 3,
                          "learning_rate": 0.2, "max_leaf_nodes": 8, "max_depth": None,
                          "min_samples_leaf": 5, "l2_regularization": 0.0},
    "random_forest": {"model_type": "random_forest", "seed": 1, "trees_per_epoch": 4,
                      "max_depth": 4, "min_samples_leaf": 2, "max_features": 0.8},
    "linear": {"model_type": "linear", "seed": 1, "learning_rate": 0.01, "alpha": 0.0001},
}


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "linear"])
def test_every_family_trains_epoch_by_epoch_on_classification(model_module, model_type):
    X, y = classification_data()
    model = model_module.build_model(CONFIGS[model_type], "tabular_classification",
                                     [False] * 4)
    for epoch in range(1, 4):
        model.fit_epoch(X, y)
        assert model.epochs_fitted == epoch
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert list(model.classes_) == [0, 1]


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "linear"])
def test_every_family_trains_on_regression(model_module, model_type):
    X, y = regression_data()
    model = model_module.build_model(CONFIGS[model_type], "tabular_regression", [False] * 4)
    model.fit_epoch(X, y)
    model.fit_epoch(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    assert model.classes_ is None


def test_capacity_grows_with_each_epoch(model_module):
    X, y = classification_data()
    gb = model_module.build_model(CONFIGS["gradient_boosting"], "tabular_classification",
                                  [False] * 4)
    gb.fit_epoch(X, y)
    first = gb.estimator.max_iter
    gb.fit_epoch(X, y)
    assert gb.estimator.max_iter == first + CONFIGS["gradient_boosting"]["iters_per_epoch"]

    rf = model_module.build_model(CONFIGS["random_forest"], "tabular_classification",
                                  [False] * 4)
    rf.fit_epoch(X, y)
    trees = rf.estimator.n_estimators
    rf.fit_epoch(X, y)
    assert rf.estimator.n_estimators == trees + CONFIGS["random_forest"]["trees_per_epoch"]

    lin = model_module.build_model(CONFIGS["linear"], "tabular_classification", [False] * 4)
    lin.fit_epoch(X, y)
    lin.fit_epoch(X, y)
    assert lin.epochs_fitted == 2  # one partial_fit pass per epoch, no capacity change


def test_categorical_mask_is_passed_to_gradient_boosting(model_module):
    model = model_module.build_model(CONFIGS["gradient_boosting"], "tabular_classification",
                                     [True, False, False, False])
    assert list(model.estimator.categorical_features) == [True, False, False, False]
    none_mask = model_module.build_model(CONFIGS["gradient_boosting"],
                                         "tabular_classification", [False] * 4)
    assert none_mask.estimator.categorical_features is None


def test_unknown_model_type_and_task_type_raise(model_module):
    with pytest.raises(ValueError):
        model_module.build_model({"model_type": "quantum"}, "tabular_classification", [])
    with pytest.raises(ValueError):
        model_module.build_model(CONFIGS["linear"], "image_classification", [])


def test_model_types_constant(model_module):
    assert model_module.MODEL_TYPES == ("gradient_boosting", "random_forest", "linear")
```

Append to `tests/test_templates_io.py`:

```python
def test_schema_is_nested_and_schema_for_flattens_it():
    schema = tio.load_schema("tabular_sklearn")
    assert set(schema) == {"common", "models"}
    assert set(schema["common"]) == {"model_type", "epochs", "early_stopping_patience", "seed"}
    assert tio.model_types(schema) == ["gradient_boosting", "random_forest", "linear"]

    flat = tio.schema_for(schema, "random_forest")
    assert set(flat) == {"model_type", "epochs", "early_stopping_patience", "seed",
                         "trees_per_epoch", "max_depth", "min_samples_leaf", "max_features"}
    assert "learning_rate" not in flat
    assert tio.schema_for(schema, "linear")["alpha"]["default"] == 0.0001
    with pytest.raises(ValueError):
        tio.schema_for(schema, "quantum")


def test_choice_rules_validate_and_coerce():
    schema = tio.load_schema("tabular_sklearn")
    flat = tio.schema_for(schema, "gradient_boosting")
    cfg = tio.default_config(flat)
    assert cfg["model_type"] == "gradient_boosting"
    assert tio.validate_config(cfg, flat) == []

    bad = {**cfg, "model_type": "quantum"}
    assert any("model_type" in p for p in tio.validate_config(bad, flat))
    bad_type = {**cfg, "model_type": 3}
    assert any("model_type" in p for p in tio.validate_config(bad_type, flat))

    coerced, notes = tio.coerce_config({"model_type": "quantum", "learning_rate": 0.05}, flat)
    assert coerced["model_type"] == "gradient_boosting"
    assert coerced["learning_rate"] == 0.05
    assert any("model_type" in n for n in notes)
```

Delete `test_build_model_and_grow` from `tests/test_template_data.py` (its replacement lives in `tests/test_template_model.py`).

In `tests/test_template_train.py`, `SMALL` gains the model type:

```python
SMALL = {"model_type": "gradient_boosting", "epochs": 4, "iters_per_epoch": 3,
         "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
         "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
         "l2_regularization": 0.0}
```

In `tests/test_codegen_stage.py`, add:

```python
def test_config_is_flat_for_the_chosen_model(clean_project):
    from mlagent.templates_io import load_schema, schema_for, validate_config

    ctx, _shown = make_ctx(clean_project, FakeLLM([]), ["y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    schema = schema_for(load_schema("tabular_sklearn"), cfg["model_type"])
    assert validate_config(cfg, schema) == []
    assert "models" not in cfg and "common" not in cfg
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_template_model.py tests/test_templates_io.py -q`
Expected: FAIL with `AttributeError: module 'model' has no attribute 'MODEL_TYPES'` and `AttributeError: module 'mlagent.templates_io' has no attribute 'schema_for'`.

- [ ] **Step 3: Write the three-family model template**

Replace `mlagent/templates/tabular_sklearn/model.py`:

```python
"""Build the model. Generated by mlagent; safe to edit.

One wrapper, `EpochModel`, gives all three families the same shape so `train.py` can add
capacity one epoch at a time and record a real learning curve:

- gradient_boosting: HistGradientBoosting with warm_start; an epoch adds `iters_per_epoch`
  boosting rounds to the same ensemble.
- random_forest: RandomForest with warm_start; an epoch adds `trees_per_epoch` trees.
- linear: SGDClassifier/SGDRegressor; an epoch is one `partial_fit` pass over the training
  rows. Linear models cannot cope with missing values or wildly different scales, so this
  family fills gaps with each column's median and standardises before fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

# --- settings ---
MODEL_TYPES = ("gradient_boosting", "random_forest", "linear")
CLASSIFICATION = "tabular_classification"
REGRESSION = "tabular_regression"


# --- one epoch at a time ---
class EpochModel:
    """A scikit-learn estimator plus the rule for what 'one more epoch' means."""

    def __init__(self, estimator, model_type: str, task_type: str, step: int = 0):
        self.estimator = estimator
        self.model_type = model_type
        self.task_type = task_type
        self.step = int(step)
        self.epochs_fitted = 0
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    # Linear models need numbers with no gaps and comparable scales.
    def _prepare(self, X, fit: bool = False):
        if fit:
            return self.scaler.fit_transform(self.imputer.fit_transform(X))
        return self.scaler.transform(self.imputer.transform(X))

    def fit_epoch(self, X, y) -> EpochModel:
        first = self.epochs_fitted == 0
        if self.model_type == "gradient_boosting":
            if not first:
                self.estimator.set_params(max_iter=int(self.estimator.max_iter) + self.step)
            self.estimator.fit(X, y)
        elif self.model_type == "random_forest":
            if not first:
                self.estimator.set_params(
                    n_estimators=int(self.estimator.n_estimators) + self.step
                )
            self.estimator.fit(X, y)
        else:
            Xs = self._prepare(X, fit=first)
            if self.task_type == CLASSIFICATION and first:
                self.estimator.partial_fit(Xs, y, classes=np.unique(np.asarray(y)))
            else:
                self.estimator.partial_fit(Xs, y)
        self.epochs_fitted += 1
        return self

    def _features(self, X):
        return self._prepare(X) if self.model_type == "linear" else X

    def predict(self, X):
        return self.estimator.predict(self._features(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._features(X))

    @property
    def classes_(self):
        return getattr(self.estimator, "classes_", None)


# --- building each family ---
def _gradient_boosting(config: dict, task_type: str, categorical_mask: list[bool]):
    params = {
        "learning_rate": float(config.get("learning_rate", 0.1)),
        "max_iter": int(config.get("iters_per_epoch", 10)),
        "max_leaf_nodes": int(config.get("max_leaf_nodes", 31)),
        "max_depth": config.get("max_depth"),
        "min_samples_leaf": int(config.get("min_samples_leaf", 20)),
        "l2_regularization": float(config.get("l2_regularization", 0.0)),
        "categorical_features": list(categorical_mask) if any(categorical_mask) else None,
        "warm_start": True,
        "early_stopping": False,
        "random_state": int(config.get("seed", 42)),
    }
    cls = HistGradientBoostingClassifier if task_type == CLASSIFICATION \
        else HistGradientBoostingRegressor
    return cls(**params), int(config.get("iters_per_epoch", 10))


def _random_forest(config: dict, task_type: str, _categorical_mask: list[bool]):
    trees = int(config.get("trees_per_epoch", 20))
    params = {
        "n_estimators": trees,
        "max_depth": config.get("max_depth"),
        "min_samples_leaf": int(config.get("min_samples_leaf", 1)),
        "max_features": float(config.get("max_features", 1.0)),
        "warm_start": True,
        "random_state": int(config.get("seed", 42)),
    }
    cls = RandomForestClassifier if task_type == CLASSIFICATION else RandomForestRegressor
    return cls(**params), trees


def _linear(config: dict, task_type: str, _categorical_mask: list[bool]):
    rate = float(config.get("learning_rate", 0.01))
    alpha = float(config.get("alpha", 0.0001))
    seed = int(config.get("seed", 42))
    if task_type == CLASSIFICATION:
        estimator = SGDClassifier(
            loss="log_loss", alpha=alpha, learning_rate="constant", eta0=rate,
            random_state=seed,
        )
    else:
        estimator = SGDRegressor(
            alpha=alpha, learning_rate="constant", eta0=rate, random_state=seed
        )
    return estimator, 1


BUILDERS = {
    "gradient_boosting": _gradient_boosting,
    "random_forest": _random_forest,
    "linear": _linear,
}


def build_model(config: dict, task_type: str, categorical_mask: list[bool]) -> EpochModel:
    model_type = str(config.get("model_type", "gradient_boosting"))
    if model_type not in BUILDERS:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")
    if task_type not in (CLASSIFICATION, REGRESSION):
        raise ValueError(f"unsupported task_type {task_type!r} for the tabular_sklearn template")
    estimator, step = BUILDERS[model_type](config, task_type, list(categorical_mask))
    return EpochModel(estimator, model_type, task_type, step)
```

The random forest's `max_features` is a fraction of the columns rather than scikit-learn's
`"sqrt"` string so that the tuner in Milestone 5 can move it continuously inside the schema's bounds.

- [ ] **Step 4: Write the nested config schema**

Replace `mlagent/templates/tabular_sklearn/config_schema.json`:

```json
{
  "common": {
    "model_type": {"type": "choice", "default": "gradient_boosting",
      "choices": ["gradient_boosting", "random_forest", "linear"],
      "description": "Which family of model to train."},
    "epochs": {"type": "integer", "default": 10, "min": 1, "max": 100,
      "description": "Number of epochs; each one adds capacity or does another pass."},
    "early_stopping_patience": {"type": "integer", "default": 5, "min": 0, "max": 50,
      "description": "Stop when validation has not improved for this many epochs; 0 disables."},
    "seed": {"type": "integer", "default": 42, "min": 0, "max": 1000000,
      "description": "Random seed for the model (the data split has its own frozen seed)."}
  },
  "models": {
    "gradient_boosting": {
      "learning_rate": {"type": "number", "default": 0.1, "min": 0.001, "max": 1.0,
        "description": "Shrinkage applied to each boosting round; lower is slower but steadier."},
      "iters_per_epoch": {"type": "integer", "default": 10, "min": 1, "max": 100,
        "description": "Boosting rounds (trees) added per epoch."},
      "max_leaf_nodes": {"type": "integer", "default": 31, "min": 2, "max": 255,
        "description": "Maximum leaves per tree; more leaves fit more detail."},
      "max_depth": {"type": "integer", "default": null, "min": 1, "max": 32, "nullable": true,
        "description": "Maximum tree depth; null means unlimited."},
      "min_samples_leaf": {"type": "integer", "default": 20, "min": 1, "max": 200,
        "description": "Minimum rows per leaf; higher values regularise."},
      "l2_regularization": {"type": "number", "default": 0.0, "min": 0.0, "max": 10.0,
        "description": "L2 penalty on leaf values; higher values regularise."}
    },
    "random_forest": {
      "trees_per_epoch": {"type": "integer", "default": 20, "min": 1, "max": 200,
        "description": "Trees added to the forest each epoch."},
      "max_depth": {"type": "integer", "default": null, "min": 1, "max": 32, "nullable": true,
        "description": "Maximum tree depth; null means grow until the leaves are pure."},
      "min_samples_leaf": {"type": "integer", "default": 1, "min": 1, "max": 200,
        "description": "Minimum rows per leaf; higher values make each tree simpler."},
      "max_features": {"type": "number", "default": 0.5, "min": 0.05, "max": 1.0,
        "description": "Fraction of columns each split may consider; lower decorrelates trees."}
    },
    "linear": {
      "learning_rate": {"type": "number", "default": 0.01, "min": 0.0001, "max": 1.0,
        "description": "Step size for each gradient update."},
      "alpha": {"type": "number", "default": 0.0001, "min": 0.0, "max": 1.0,
        "description": "Strength of the L2 penalty on the coefficients."}
    }
  }
}
```

- [ ] **Step 5: Teach `templates_io` about nesting and choices**

In `mlagent/templates_io.py`, add after `load_schema`:

```python
def model_types(schema: dict) -> list[str]:
    return list((schema.get("models") or {}).keys())


def schema_for(schema: dict, model_type: str) -> dict:
    """The flat rules for one model family: the common keys plus that family's keys.

    Everything downstream — `default_config`, `validate_config`, `coerce_config`, the
    codegen proposal and Milestone 5's tuner — works on this flat form.
    """
    models = schema.get("models") or {}
    if model_type not in models:
        raise ValueError(
            f"unknown model_type {model_type!r}; expected one of {sorted(models)}"
        )
    return {**(schema.get("common") or {}), **models[model_type]}
```

Extend `_check_value` with the choice branch, as its first check after the null test:

```python
def _check_value(key: str, value, rule: dict) -> str | None:
    if value is None:
        return None if rule.get("nullable") else f"{key}: must not be null"
    if rule.get("type") == "choice":
        choices = list(rule.get("choices") or [])
        if not isinstance(value, str):
            return f"{key}: expected one of {choices}, got {value!r}"
        if value not in choices:
            return f"{key}: {value!r} is not one of {choices}"
        return None
    if isinstance(value, bool):
        return f"{key}: expected a number, got a boolean"
```
Everything from the existing `if rule.get("type") == "integer":` line to the end of the
function stays exactly as it is.

Extend `coerce_config`'s per-key loop with the choice branch, immediately after `rule = schema[key]`:

```python
        rule = schema[key]
        if rule.get("type") == "choice":
            choices = list(rule.get("choices") or [])
            if isinstance(value, str) and value in choices:
                config[key] = value
            else:
                notes.append(
                    f"Ignored {key}={value!r}: not one of {choices}; kept {config[key]!r}."
                )
            continue
```
The existing `try: cast = _cast(value, rule)` block and everything after it in the loop
stays exactly as it is.

`default_config` needs no change: it already reads each rule's `default`, and a choice rule has one.

- [ ] **Step 6: Point `train.py` and codegen at the new shapes**

In `mlagent/templates/tabular_sklearn/train.py`:

- change the import to `from model import build_model`
- replace the two lines in the epoch loop

```python
        if epoch > 1:
            grow(model, iters)
        model.fit(data["X_train"], data["y_train"])
```
with

```python
        model.fit_epoch(data["X_train"], data["y_train"])
```

- `copy.deepcopy(model)` still works: `EpochModel` holds only picklable scikit-learn objects.

In `mlagent/stages/codegen.py`:

- import `schema_for` and `model_types` alongside the existing names;
- in `is_complete`, replace the schema line with

```python
            spec = ctx.spec()
            schema = load_schema(TEMPLATE_FOR_TASK[spec.task_type])
            flat = schema_for(schema, str(config.get("model_type", "")))
        except Exception:  # noqa: BLE001 - missing spec, unknown model or task: not complete
            return False
        return validate_config(config, flat) == []
```

- in `prepare`, replace `schema = load_schema(template)` with

```python
        nested = load_schema(template)
        schema = schema_for(nested, model_types(nested)[0])
```
(Task 10 replaces that line with the user's chosen family; this keeps the stage working in the meantime.)

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_template_model.py tests/test_templates_io.py tests/test_template_data.py tests/test_template_train.py tests/test_codegen_stage.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add mlagent/templates/tabular_sklearn/model.py \
        mlagent/templates/tabular_sklearn/config_schema.json \
        mlagent/templates/tabular_sklearn/train.py mlagent/templates_io.py \
        mlagent/stages/codegen.py tests
git commit -m "feat: three model families behind EpochModel and a nested, choice-aware config schema

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 9: `evaluate.py` and the notebook-friendly `train.py`

**Files:**
- Create: `mlagent/templates/tabular_sklearn/evaluate.py`
- Modify: `mlagent/templates/tabular_sklearn/train.py`
- Modify: `mlagent/templates_io.py` (`CODE_FILES`)
- Test: `tests/test_template_evaluate.py` (new), `tests/test_template_train.py`

**Interfaces:**
- Consumes: `data.load_data`, `model.build_model` (Task 8); `runs.jsonl` entries' `run_id`, `status`, `best_val_metric`, `checkpoint`.
- Produces, in `evaluate.py`:
  - `compute_metric(name: str, y_true, y_pred) -> float`
  - `full_proba(model, X, n_classes: int) -> np.ndarray`
  - `evaluate_split(model, X, y, task_type: str, metric: str, n_classes: int | None) -> dict` with keys `loss`, `value`, `y_true`, `y_pred`, `y_proba`
  - `best_checkpoint(project_dir: Path) -> tuple[str | None, int | None]` — the best `done` run's checkpoint path and `run_id` from `runs.jsonl`
  - `save_figures(record: dict, plots_dir: Path, split: str) -> list[str]`
  - CLI `python evaluate.py [--project DIR] [--split val|test] [--checkpoint PATH]`
  - `eval_{split}.json` = `split`, `task_type`, `metric`, `value`, `loss`, `classes`, `y_true`, `y_pred`, `y_proba`, plus `checkpoint` (project-relative string) and `run_id` (`int | None`) and `figures` (list of PNG filenames).
  - Figure names: `{split}_confusion.png`, `{split}_roc_pr.png`, `{split}_per_class.png` for classification; `{split}_pred_vs_actual.png`, `{split}_residuals.png` for regression.
- Produces, in `train.py`: `metrics.json` gains `started_at` (UTC ISO seconds) and `model_type`; `plots/training_curves.png` is written after every epoch and redrawn live when IPython is present; `--eval-test` and the `eval_val.json` write are removed; `--dry-run` is unchanged.
- `templates_io.CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_template_evaluate.py`:

```python
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
SMALL = {"model_type": "gradient_boosting", "epochs": 3, "iters_per_epoch": 3,
         "learning_rate": 0.2, "seed": 1, "early_stopping_patience": 0,
         "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
         "l2_regularization": 0.0}


def install(project, config=None) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config or SMALL)
    return project.root


def run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=180,
    )


def train(project) -> Path:
    root = install(project)
    proc = run(root, "train.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root


def test_evaluate_val_writes_record_and_classification_figures(clean_project):
    root = train(clean_project)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert record["split"] == "val" and record["task_type"] == "tabular_classification"
    assert record["metric"] == "accuracy" and 0.0 <= record["value"] <= 1.0
    assert record["checkpoint"] == "checkpoints/best.joblib"
    assert record["run_id"] is None
    assert len(record["y_true"]) == len(record["y_pred"]) == len(record["y_proba"])
    assert len(record["y_proba"][0]) == 2
    assert set(record["figures"]) == {"val_confusion.png", "val_roc_pr.png",
                                      "val_per_class.png"}
    for name in record["figures"]:
        assert (root / "plots" / name).exists()


def test_evaluate_test_uses_the_best_run_checkpoint(clean_project):
    root = train(clean_project)
    shutil.copy(root / "checkpoints" / "best.joblib", root / "checkpoints" / "run1.joblib")
    (root / "runs.jsonl").write_text(
        json.dumps({"run_id": 1, "status": "done", "best_val_metric": 0.7,
                    "checkpoint": "checkpoints/run1.joblib"}) + "\n"
        + json.dumps({"run_id": 2, "status": "failed", "best_val_metric": 0.99,
                      "checkpoint": None}) + "\n",
        encoding="utf-8",
    )
    proc = run(root, "evaluate.py", "--split", "test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["split"] == "test"
    assert record["checkpoint"] == "checkpoints/run1.joblib"
    assert record["run_id"] == 1
    assert "test_confusion.png" in record["figures"]


def test_evaluate_regression_figures_and_explicit_checkpoint(regression_project):
    root = train(regression_project)
    proc = run(root, "evaluate.py", "--split", "test",
               "--checkpoint", "checkpoints/best.joblib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["y_proba"] is None and record["metric"] == "rmse"
    assert set(record["figures"]) == {"test_pred_vs_actual.png", "test_residuals.png"}
    assert record["run_id"] is None


def test_evaluate_without_a_checkpoint_fails_clearly(clean_project):
    root = install(clean_project)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()


def test_full_proba_handles_a_class_absent_from_training():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate", TEMPLATE / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)

    class Stub:
        classes_ = np.array([0, 2])

        def predict_proba(self, X):
            return np.tile([0.25, 0.75], (len(X), 1))

    proba = module.full_proba(Stub(), np.zeros((4, 2)), 3)
    assert proba.shape == (4, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] < 1e-6


def test_evaluate_script_shape(clean_project):
    root = install(clean_project)
    source = (root / "evaluate.py").read_text(encoding="utf-8")
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert "import mlagent" not in source


@pytest.mark.parametrize("model_type", ["random_forest", "linear"])
def test_other_families_train_and_evaluate(clean_project, model_type):
    configs = {
        "random_forest": {"model_type": "random_forest", "epochs": 2, "trees_per_epoch": 5,
                          "max_depth": 4, "min_samples_leaf": 1, "max_features": 0.8,
                          "seed": 1, "early_stopping_patience": 0},
        "linear": {"model_type": "linear", "epochs": 3, "learning_rate": 0.05,
                   "alpha": 0.0001, "seed": 1, "early_stopping_patience": 0},
    }
    root = install(clean_project, configs[model_type])
    assert run(root, "train.py").returncode == 0
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == model_type
    assert run(root, "evaluate.py").returncode == 0
```

Change `tests/test_template_train.py`:

- `CODE_FILES` becomes the four-file tuple.
- `test_classification_run_writes_metrics_checkpoint_and_eval` drops every `eval_val.json` assertion and gains:

```python
    assert metrics["started_at"] and metrics["started_at"].endswith("+00:00")
    assert metrics["model_type"] == "gradient_boosting"
    assert (root / "plots" / "training_curves.png").exists()
    assert not (root / "eval_val.json").exists()
    assert not (root / "eval_test.json").exists()
```
- `test_regression_run_and_eval_test` becomes `test_regression_run` and loses its `--eval-test` half.
- `test_eval_test_without_checkpoint_fails_clearly`, `test_eval_test_with_explicit_checkpoint` and `test_eval_test_with_missing_explicit_checkpoint_names_it` are deleted — `tests/test_template_evaluate.py` covers those paths now.
- `test_full_proba_handles_class_absent_from_training` moves to `tests/test_template_evaluate.py` (already written above) and is deleted here.
- Add:

```python
def test_each_run_gets_a_new_started_at(clean_project):
    root = install(clean_project, SMALL)
    assert run(root).returncode == 0
    first = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["started_at"]
    assert run(root).returncode == 0
    second = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["started_at"]
    assert first and second and first != second


def test_train_script_shape(clean_project):
    root = install(clean_project, SMALL)
    source = (root / "train.py").read_text(encoding="utf-8")
    assert "--eval-test" not in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert "# --- settings ---" in source
```

(If two runs in the same second could collide, `started_at` uses microsecond precision — see Step 3.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_template_evaluate.py -q`
Expected: FAIL with `FileNotFoundError` on `mlagent/templates/tabular_sklearn/evaluate.py`.

- [ ] **Step 3: Write `evaluate.py`**

Create `mlagent/templates/tabular_sklearn/evaluate.py`:

```python
"""Score a saved model on one split and draw its evaluation figures.

Run it from the project folder:

    python evaluate.py                                  the validation split, latest checkpoint
    python evaluate.py --split test                     the test split, best run's checkpoint
    python evaluate.py --split test --checkpoint P      a specific checkpoint

Writes `eval_{split}.json` (the numbers plus the raw predictions, so the figures can be
redrawn without retraining) and PNGs into `plots/`. `train.py` imports `compute_metric`,
`full_proba` and `evaluate_split` from here so the two scripts can never disagree about
what a metric means.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from data import load_data
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
)

# --- settings ---
PROJECT_DIR = Path(".")
CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
RUNS_FILE = "runs.jsonl"
DEFAULT_CHECKPOINT = "checkpoints/best.joblib"
SPLITS = ("val", "test")
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"tabular_classification": "accuracy", "tabular_regression": "rmse"}
EPS = 1e-12
MAX_ROC_CLASSES = 8

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)


# --- small helpers ---
def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    runs = []
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


def metric_for(project_dir: Path, task_type: str) -> str:
    spec = read_json(project_dir / SPEC_FILE, default={}) or {}
    return str(spec.get("metric") or DEFAULT_METRIC[task_type])


# --- scoring ---
def compute_metric(name: str, y_true, y_pred) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if name == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"unknown metric {name!r}")


def full_proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with one column per class even if a class was absent from training."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    out = np.clip(out, EPS, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def evaluate_split(model, X, y, task_type: str, metric: str, n_classes: int | None) -> dict:
    y_pred = model.predict(X)
    if task_type == "tabular_classification":
        proba = full_proba(model, X, n_classes)
        loss = float(log_loss(y, proba, labels=list(range(n_classes))))
        y_proba = proba.tolist()
    else:
        loss = float(mean_squared_error(y, y_pred))
        y_proba = None
    return {
        "loss": loss,
        "value": compute_metric(metric, y, y_pred),
        "y_true": np.asarray(y).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": y_proba,
    }


def eval_record(split: str, task_type: str, metric: str, classes, ev: dict) -> dict:
    return {
        "split": split,
        "task_type": task_type,
        "metric": metric,
        "value": ev["value"],
        "loss": ev["loss"],
        "classes": classes,
        "y_true": ev["y_true"],
        "y_pred": ev["y_pred"],
        "y_proba": ev["y_proba"],
    }


# --- choosing a checkpoint ---
def best_checkpoint(project_dir: Path) -> tuple[str | None, int | None]:
    """The checkpoint of the best finished run, and its run id."""
    task_type = (read_json(project_dir / "data_meta.json", default={}) or {}).get(
        "task_type", "tabular_classification"
    )
    metric = metric_for(project_dir, task_type)
    higher = HIGHER_IS_BETTER.get(metric, True)
    best = None
    for run in read_runs(project_dir / RUNS_FILE):
        if run.get("status") != "done" or not run.get("checkpoint"):
            continue
        value = run.get("best_val_metric")
        if best is None:
            best = run
            continue
        incumbent = best.get("best_val_metric")
        if value is None:
            continue
        if incumbent is None or (value > incumbent if higher else value < incumbent):
            best = run
    if best is None:
        return None, None
    return str(best["checkpoint"]), best.get("run_id")


# --- figures ---
def frame(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def show(fig) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is not None:
        display(fig)


def save(fig, plots_dir: Path, name: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig)
    plt.close(fig)
    return f"{name}.png"


def confusion_figure(y_true, y_pred, labels: list[str]):
    n = len(labels)
    m = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    size = min(2.5 + 0.35 * n, 9)
    fig, ax = plt.subplots(figsize=(size, size), facecolor=SURFACE)
    ax.imshow(m, cmap=SEQ_CMAP)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK_2, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=INK_2, fontsize=8)
    ax.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax.set_ylabel("actual", color=INK_2, fontsize=8)
    if n <= 20 and m.size:
        threshold = m.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color=SURFACE if m[i, j] > threshold else INK)
    ax.set_title("Confusion matrix", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def roc_pr_figure(y_true, y_proba, labels: list[str]):
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    n = p.shape[1]
    if n == 2:
        curves = [(1, p[:, 1], labels[1])]
    else:
        curves = [(k, p[:, k], labels[k]) for k in range(min(n, MAX_ROC_CLASSES))]
    drawn = 0
    for (k, score, label), colour in zip(curves, SERIES, strict=False):
        positive = (y == k).astype(int)
        if positive.sum() in (0, len(positive)):
            continue
        drawn += 1
        fpr, tpr, _ = roc_curve(positive, score)
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, label=label)
        precision, recall, _ = precision_recall_curve(positive, score)
        ax_pr.plot(recall, precision, color=colour, linewidth=2, label=label)
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    suffix = f" (showing {len(curves)} of {n} classes)" if n > 2 and len(curves) < n else ""
    frame(ax_roc, "ROC curve" + suffix)
    ax_roc.set_xlabel("false positive rate", color=INK_2, fontsize=8)
    ax_roc.set_ylabel("true positive rate", color=INK_2, fontsize=8)
    frame(ax_pr, "Precision-recall curve")
    ax_pr.set_xlabel("recall", color=INK_2, fontsize=8)
    ax_pr.set_ylabel("precision", color=INK_2, fontsize=8)
    if drawn > 1:
        ax_roc.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        ax_pr.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def per_class_figure(y_true, y_pred, labels: list[str]):
    n = len(labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * n + 2), 3.2), facecolor=SURFACE)
    x = np.arange(n)
    width = 0.38
    ax.bar(x - width / 2, precision, width=width, color=SERIES[0], label="precision")
    ax.bar(x + width / 2, recall, width=width, color=SERIES[1], label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 6 else 0,
                       ha="right" if n > 6 else "center", color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.05)
    frame(ax, "Precision and recall per class")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def pred_vs_actual_figure(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=SURFACE)
    ax.scatter(y, p, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
    frame(ax, "Predicted vs actual")
    ax.set_xlabel("actual", color=INK_2, fontsize=8)
    ax.set_ylabel("predicted", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def residual_figure(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    residuals = y - p
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    ax_hist.hist(residuals, bins=min(30, max(5, len(residuals) // 5)), color=SERIES[0])
    frame(ax_hist, "Residual distribution")
    ax_hist.set_xlabel("actual - predicted", color=INK_2, fontsize=8)
    ax_scatter.scatter(p, residuals, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    ax_scatter.axhline(0, color=AXIS, linestyle="--", linewidth=1)
    frame(ax_scatter, "Residuals vs predicted")
    ax_scatter.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax_scatter.set_ylabel("residual", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def save_figures(record: dict, plots_dir: Path, split: str) -> list[str]:
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    if record["task_type"] == "tabular_classification":
        labels = [str(c) for c in (record.get("classes") or sorted({*y_true, *y_pred}))]
        names = [save(confusion_figure(y_true, y_pred, labels), plots_dir,
                      f"{split}_confusion")]
        if record.get("y_proba"):
            names.append(save(roc_pr_figure(y_true, record["y_proba"], labels), plots_dir,
                              f"{split}_roc_pr"))
        names.append(save(per_class_figure(y_true, y_pred, labels), plots_dir,
                          f"{split}_per_class"))
        return names
    return [
        save(pred_vs_actual_figure(y_true, y_pred), plots_dir, f"{split}_pred_vs_actual"),
        save(residual_figure(y_true, y_pred), plots_dir, f"{split}_residuals"),
    ]


# --- command line ---
def cli_argv() -> list[str]:
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if not name.endswith(".py"):
        return []
    return sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on one split.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--split", default="val", choices=list(SPLITS))
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path relative to --project")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()

    run_id = None
    checkpoint = args.checkpoint
    if checkpoint is None and args.split == "test":
        checkpoint, run_id = best_checkpoint(project_dir)
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    checkpoint_path = project_dir / checkpoint
    if not checkpoint_path.exists():
        print(f"error: no checkpoint at {checkpoint_path}; run train.py first", flush=True)
        return 1

    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    task_type = data["task_type"]
    metric = metric_for(project_dir, task_type)
    n_classes = len(data["classes"]) if data["classes"] is not None else None
    model = joblib.load(checkpoint_path)
    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]

    ev = evaluate_split(model, X, y, task_type, metric, n_classes)
    record = eval_record(args.split, task_type, metric, data["classes"], ev)
    record["checkpoint"] = Path(checkpoint).as_posix()
    record["run_id"] = run_id
    record["figures"] = save_figures(record, project_dir / "plots", args.split)
    out = project_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{args.split} {metric}={record['value']:.4f} loss={record['loss']:.4f} "
        f"({len(record['y_true'])} rows, checkpoint {record['checkpoint']}) -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

- [ ] **Step 4: Rework `train.py`**

Apply these edits to `mlagent/templates/tabular_sklearn/train.py`:

1. Docstring and usage lines lose `--eval-test`:

```python
"""Train, evaluate per epoch, write metrics.json. Generated by mlagent; safe to edit.

Usage (run from the project folder):
  python train.py                one full training run
  python train.py --dry-run      one round; prints timing and n_train, writes nothing

After training, run `evaluate.py` to score the saved checkpoint and draw its figures.
"""
```

2. Imports: drop the `sklearn.metrics` block and `from model import build_model, grow`; add matplotlib and the shared scoring functions:

```python
import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from data import load_data
from evaluate import evaluate_split
from model import build_model
```

`evaluate_split` is the single definition of "score this model on these rows"; it calls
`compute_metric` and `full_proba` internally, so `train.py` and `evaluate.py` can never
disagree about what a metric means.

3. Constants: delete `EVAL_VAL_FILE`, `EVAL_TEST_FILE` and `EPS`; add the curve settings and palette:

```python
# --- settings ---
CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
METRICS_FILE = "metrics.json"
CHECKPOINT = Path("checkpoints") / "best.joblib"
CURVES_FIGURE = Path("plots") / "training_curves.png"
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"tabular_classification": "accuracy", "tabular_regression": "rmse"}

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
```

4. Delete `compute_metric`, `full_proba`, `evaluate`, `eval_record` and `eval_test` from this file — `evaluate.py` owns them. Replace every `evaluate(model, ...)` call in the epoch loop with `evaluate_split(model, ...)`.

5. Add the curve figure and its live redraw, in a new section:

```python
# --- the live training curve ---
_HANDLE = None


def training_curves(epochs: list[dict], metric: str):
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
    xs = [e["epoch"] for e in epochs]
    panels = (
        (ax_loss, ("train_loss", "val_loss"), "Loss per epoch"),
        (ax_metric, ("train_metric", "val_metric"), f"{metric} per epoch"),
    )
    for ax, keys, title in panels:
        for key, colour, label in zip(keys, SERIES[:2], ("train", "validation"), strict=True):
            ax.plot(xs, [e.get(key) for e in epochs], color=colour, linewidth=2,
                    marker="o", markersize=4, label=label)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("epoch", color=INK_2, fontsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    if not epochs:
        ax_loss.text(0.5, 0.5, "no epochs yet", ha="center", va="center", color=MUTED,
                     transform=ax_loss.transAxes)
    fig.tight_layout()
    return fig


def redraw(fig) -> None:
    """Update one output area in place, so the curve animates instead of stacking up."""
    global _HANDLE
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is None:
        return
    if _HANDLE is None:
        _HANDLE = display(fig, display_id=True)
    else:
        _HANDLE.update(fig)


def draw_curves(project_dir: Path, epochs: list[dict], metric: str) -> None:
    fig = training_curves(epochs, metric)
    path = project_dir / CURVES_FIGURE
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    redraw(fig)
    plt.close(fig)
```

6. `empty_metrics` gains the two new keys (`started_at` and `model_type`), so the whole
   returned dict becomes:

```python
    return {
        "status": "running",
        "started_at": None,
        "model_type": None,
        "task_type": None,
        "metric": None,
        "higher_is_better": None,
        "config": None,
        "classes": None,
        "n_train": None,
        "n_val": None,
        "n_test": None,
        "epochs": [],
        "best_epoch": None,
        "best_val_metric": None,
        "stopped_early": False,
        "error": None,
        "seconds": None,
        "seconds_per_epoch": None,
    }
```

7. In `train`, set them and draw the curve each epoch. After the existing `metrics.update({...})` block:

```python
    metrics["started_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    metrics["model_type"] = str(config.get("model_type", "gradient_boosting"))
```
and inside the epoch loop, right after `save()` and before the `print(...)`:

```python
        if not dry_run:
            draw_curves(project_dir, metrics["epochs"], metric)
```

8. In `train`'s tail, delete the `eval_val.json` write and the final `evaluate(...)` call; keep the checkpoint dump:

```python
    (project_dir / CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, project_dir / CHECKPOINT)
    metrics["status"] = "done"
    save()
    return metrics
```

9. `main` loses `--eval-test` and `--checkpoint` and gains the argv guard:

```python
def cli_argv() -> list[str]:
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if not name.endswith(".py"):
        return []
    return sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the model and record every epoch.")
    parser.add_argument("--project", default=".", help="project folder (default: cwd)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
    try:
        if not args.dry_run:
            # Fresh placeholders before training starts, so a failure before the first
            # save() cannot leave a previous run's epochs/best metric on disk.
            write_json(project_dir / METRICS_FILE, empty_metrics())
        metrics = train(project_dir, dry_run=args.dry_run)
        return 0 if metrics["status"] != "failed" else 1
    except Exception as exc:  # noqa: BLE001 - record any failure for the debrief
        traceback.print_exc()
        if not args.dry_run:
            read_metrics = read_json(project_dir / METRICS_FILE, default=None) or {}
            existing = {**empty_metrics(), **read_metrics}
            existing["status"] = "failed"
            existing["error"] = f"{type(exc).__name__}: {exc}"
            write_json(project_dir / METRICS_FILE, existing)
        else:
            print(f"error: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

`started_at` uses `timespec="microseconds"` so two runs a second apart are still distinct
run identities.

- [ ] **Step 5: Add `evaluate.py` to the copied file list**

In `mlagent/templates_io.py`:

```python
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
```

- [ ] **Step 6: Run the template tests**

Run: `python -m pytest tests/test_template_evaluate.py tests/test_template_train.py tests/test_template_data.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: `tests/test_train_stage.py`, `tests/test_report_stage.py` and `tests/test_pipeline_e2e.py` now fail: the train stage still expects `train.py` to write `eval_val.json`, and the report stage still shells out to `train.py --eval-test`. Tasks 11 and 12 fix those two stages. To keep this commit green, mark the affected tests with `@pytest.mark.xfail(reason="train/report stages move to handoffs in Tasks 11-12", strict=False)` — every test in `tests/test_train_stage.py` and `tests/test_report_stage.py`, and `test_full_pipeline_runs_and_is_reproducible_and_resumable` — and delete those markers again in Tasks 11, 12 and 14 respectively.

- [ ] **Step 8: Commit**

```bash
git add mlagent/templates/tabular_sklearn/evaluate.py \
        mlagent/templates/tabular_sklearn/train.py mlagent/templates_io.py tests
git commit -m "feat: evaluate.py owns scoring and evaluation figures; train.py draws live curves

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---
### Task 10: The code walkthrough splitter and a codegen stage that recommends a model

**Files:**
- Create: `mlagent/codewalk.py`
- Create: `mlagent/prompts/teaching/model_choices.md`
- Modify: `mlagent/prompts/codegen.md`
- Modify: `mlagent/stages/codegen.py`
- Test: `tests/test_codewalk.py` (new), `tests/test_codegen_stage.py`

**Interfaces:**
- Consumes: `schema_for`, `model_types`, `load_schema`, `copy_template`, `coerce_config`, `validate_config`, `CODE_FILES` (Tasks 8, 9); `prompts_io.load_prompt/audience` (Task 2).
- Produces, in `codewalk.py`:
  - `SECTION_RE`, `HEADER_TITLE = "Header"`
  - `split_sections(source: str) -> list[tuple[str, str]]` — `(title, code)` pairs split on lines exactly matching `# --- Title ---`; any code before the first marker becomes a `Header` section; a file with no markers is one `Header` section.
  - `render_walkthrough(sections: list[tuple[str, str]], explanations: dict[str, str]) -> str` — markdown: `#### Title`, a fenced python block, then the explanation for that title if `explanations` has one.
- Produces, in `codegen.py`:
  - `MODEL_LABELS: dict[str, str]` = `{"Linear / logistic regression": "linear", "Random forest": "random_forest", "Gradient boosting": "gradient_boosting"}`
  - `ASK_LABEL = "Ask me after the explanation"` (maps to the recommendation)
  - `SMALL_DATA_ROWS = 300`; `fallback_model(meta: dict) -> str` — `"linear"` when `clean_n_rows < SMALL_DATA_ROWS`, else `"gradient_boosting"`
  - `RECOMMEND_TOOL: ToolSpec` named `recommend_model` with input `{model_type: choice, reason: string}`
  - `CodegenStage.prepare` sequence: check data -> show `prompts/teaching/model_choices.md` -> `recommend_model` -> ask the user -> `propose_config` bounded by `schema_for` -> copy the template, write `config.json` -> show the config table -> optional edit -> code walkthrough of the four files.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codewalk.py`:

```python
from __future__ import annotations

from mlagent.codewalk import HEADER_TITLE, render_walkthrough, split_sections

SOURCE = '''"""A script."""

import json

# --- settings ---
NAME = "demo"

# --- doing the work ---
def work():
    return 1
'''


def test_split_sections_keeps_the_header_and_each_marked_block():
    sections = split_sections(SOURCE)
    assert [title for title, _code in sections] == [HEADER_TITLE, "settings", "doing the work"]
    assert sections[0][1].startswith('"""A script."""')
    assert "import json" in sections[0][1]
    assert sections[1][1] == 'NAME = "demo"'
    assert sections[2][1].splitlines()[0] == "def work():"


def test_a_file_without_markers_is_one_header_section():
    assert split_sections("x = 1\n") == [(HEADER_TITLE, "x = 1")]
    assert split_sections("   \n") == []


def test_a_marker_like_line_that_is_not_at_column_zero_is_not_a_section():
    source = "def f():\n    # --- not a section ---\n    return 1\n"
    assert [t for t, _c in split_sections(source)] == [HEADER_TITLE]


def test_render_walkthrough_fences_the_code_and_adds_explanations():
    sections = split_sections(SOURCE)
    md = render_walkthrough(sections, {"settings": "Constants you can change."})
    assert "#### settings" in md
    assert "```python" in md and "```" in md
    assert "Constants you can change." in md
    assert "#### doing the work" in md
    # Sections with no explanation still show their code.
    assert "def work():" in md


def test_render_walkthrough_with_no_sections():
    assert render_walkthrough([], {}) == ""
```

Append to `tests/test_codegen_stage.py`:

```python
def test_model_choices_material_is_shown_and_the_recommendation_is_named(clean_project):
    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "random_forest",
                                       "reason": "Small, noisy, mixed [[features]]."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {"trees_per_epoch": 30},
                                      "rationale": "More trees for a small set."})],
        [("text", "done")],
    ])
    ctx, shown = make_ctx(clean_project, llm, ["Random forest", "y"])
    CodegenStage().prepare(ctx)
    text = "\n".join(shown)
    assert "Gradient boosting" in text and "Random forest" in text
    assert "Linear" in text
    assert "Small, noisy" in text
    cfg = clean_project.read_json("config.json")
    assert cfg["model_type"] == "random_forest"
    assert cfg["trees_per_epoch"] == 30
    assert "learning_rate" not in cfg  # not a random-forest key
    asked = [q for q in ctx.questioner.asked if "Which model" in q]
    assert asked and "Random forest" in asked[0]  # the question names the recommendation


def test_ask_me_label_takes_the_recommendation(clean_project):
    llm = FakeLLM([
        [("tool", "recommend_model", {"model_type": "linear", "reason": "Few rows."})],
        [("text", "chosen")],
        [("tool", "propose_config", {"config": {}, "rationale": "Defaults."})],
        [("text", "done")],
    ])
    ctx, _shown = make_ctx(clean_project, llm, ["Ask me after the explanation", "y"])
    CodegenStage().prepare(ctx)
    assert clean_project.read_json("config.json")["model_type"] == "linear"


def test_fallback_heuristic_when_the_llm_is_unavailable(clean_project):
    from mlagent.stages.codegen import fallback_model

    assert fallback_model({"clean_n_rows": 120}) == "linear"
    assert fallback_model({"clean_n_rows": 5000}) == "gradient_boosting"
    assert fallback_model({}) == "gradient_boosting"

    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    cfg = clean_project.read_json("config.json")
    assert cfg["model_type"] == "gradient_boosting"
    assert any("defaults" in s for s in shown)


def test_walkthrough_lists_every_generated_file(clean_project):
    ctx, shown = make_ctx(clean_project, FakeLLM([]), ["Gradient boosting", "y"])
    CodegenStage().prepare(ctx)
    text = "\n".join(shown)
    for name in ("data.py", "model.py", "train.py", "evaluate.py"):
        assert name in text
    assert "#### settings" in text
```

Update the existing codegen tests: every `make_ctx(..., answers)` list gains the model answer `"Gradient boosting"` before the `"y"` confirm, and every `FakeLLM` script gains a leading `recommend_model` turn (or stays empty to exercise the fallback). `test_llm_proposal_is_coerced_and_written` asserts `cfg["epochs"] == 100` (clamped) exactly as before.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_codewalk.py tests/test_codegen_stage.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.codewalk'`.

- [ ] **Step 3: Write the section splitter**

Create `mlagent/codewalk.py`:

```python
"""Split a generated script into the sections the walkthrough explains.

Every template marks its parts with a line of exactly `# --- Title ---` at column zero.
Splitting on those markers gives short, quotable blocks: short enough to explain one at a
time, long enough to be worth reading.
"""

from __future__ import annotations

import re

SECTION_RE = re.compile(r"^# --- (.+?) ---[ \t]*$", re.MULTILINE)
HEADER_TITLE = "Header"


def split_sections(source: str) -> list[tuple[str, str]]:
    """`(title, code)` per section, with anything before the first marker as `Header`."""
    matches = list(SECTION_RE.finditer(source))
    if not matches:
        stripped = source.strip()
        return [(HEADER_TITLE, stripped)] if stripped else []
    sections: list[tuple[str, str]] = []
    head = source[: matches[0].start()].strip()
    if head:
        sections.append((HEADER_TITLE, head))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[match.end() : end].strip("\n").rstrip()
        sections.append((match.group(1).strip(), body))
    return sections


def render_walkthrough(
    sections: list[tuple[str, str]], explanations: dict[str, str]
) -> str:
    """Markdown: each section's code in a fenced block, with its explanation beneath."""
    parts: list[str] = []
    for title, code in sections:
        parts.append(f"#### {title}")
        parts.append("")
        parts.append("```python")
        parts.append(code)
        parts.append("```")
        explanation = (explanations.get(title) or "").strip()
        if explanation:
            parts.append("")
            parts.append(explanation)
        parts.append("")
    return "\n".join(parts).strip()
```

- [ ] **Step 4: Write the model-choices teaching material**

Create `mlagent/prompts/teaching/model_choices.md`:

```markdown
### Three kinds of model

<!--level:beginner,intermediate-->
A model is a recipe for turning the columns of a row into a prediction. These three
recipes differ in how much shape they can learn and how much data they need to learn it.
<!--/level-->

#### Linear / logistic regression

Fits one weight per column and adds them up. Fast, hard to break, and the weights are
readable: a positive weight means "more of this column, higher prediction".

<!--level:beginner,intermediate-->
It can only draw a straight line (or a flat plane), so it misses interactions like "high
income *and* short tenure". It is the right first choice on small datasets, because with
few rows a flexible model mostly learns the noise — that is [[overfitting]].
<!--/level-->

#### Random forest

Grows many decision trees on different random slices of the data and averages them. Each
tree is a chain of yes/no questions, so it handles interactions and needs no scaling.

<!--level:beginner,intermediate-->
Averaging many noisy trees cancels out their individual mistakes, which is why a forest
is hard to overfit badly. It is slower to predict than a linear model and the individual
trees are no longer readable once you have hundreds of them.
<!--/level-->

#### Gradient boosting

Also builds trees, but one at a time, each one trained to fix the errors the previous
ones made. Usually the strongest option on tabular data.

<!--level:beginner,intermediate-->
Because every tree chases the remaining error, boosting *can* overfit if you let it run
too long — which is exactly what the [[validation]] curve is for. The [[learning rate]]
controls how much of each correction is applied: lower is slower but steadier.
<!--/level-->
```

- [ ] **Step 5: Update the codegen prompt**

Replace `mlagent/prompts/codegen.md`:

```markdown
You are the code-generation stage of an ML training assistant that runs inside Google Colab. The training code is a fixed, tested template. The user picks one of three model families and you choose sensible starting hyperparameters for it and explain them.

You receive JSON with the project spec (goal, task type, metric, target value, minutes per run), a data summary (rows, columns, categorical columns, classes, split fractions), and — depending on the step — the model choices or the config schema.

When you are asked to recommend a model, call `recommend_model` exactly once:
- `model_type`: one of `linear`, `random_forest`, `gradient_boosting`.
- `reason`: under 60 words, about THIS dataset. Weigh the number of rows (a few hundred rows favours `linear`), the number and kind of columns, class imbalance, and the minutes-per-run budget. Say what you would switch to if the first run disappoints.

When you are asked for a configuration, call `propose_config` exactly once:
- `config`: only keys from the schema you were given. Omit keys you would leave at their default. Keep values inside the min/max bounds. Prefer defaults unless the data suggests otherwise (small datasets want a lower learning rate and larger min_samples_leaf; many classes or rows can afford more epochs; a short minutes-per-run budget wants fewer epochs).
- `rationale`: under 120 words. Say what you changed from the defaults and why, or say you kept the defaults and why they fit.

Wrap technical terms in double square brackets like [[learning rate]] or [[overfitting]] so the user can click them. Do not write code and do not restate the raw JSON.

Audience: {audience}
```

- [ ] **Step 6: Rewrite the codegen stage**

In `mlagent/stages/codegen.py`, keep `check_data`, `meta_summary`, `config_table` and `_edit_config` unchanged, and replace the tool block, the class's `prepare`, and `_propose`:

```python
from mlagent.codewalk import render_walkthrough, split_sections
from mlagent.prompts_io import audience, load_prompt
from mlagent.templates_io import (
    CODE_FILES,
    TEMPLATE_FOR_TASK,
    coerce_config,
    copy_template,
    load_schema,
    model_types,
    schema_for,
    validate_config,
)

MAX_LISTED = 20
SMALL_DATA_ROWS = 300
MODEL_LABELS = {
    "Linear / logistic regression": "linear",
    "Random forest": "random_forest",
    "Gradient boosting": "gradient_boosting",
}
LABEL_FOR_MODEL = {code: label for label, code in MODEL_LABELS.items()}
ASK_LABEL = "Ask me after the explanation"

RECOMMEND_TOOL = ToolSpec(
    name="recommend_model",
    description=(
        "Recommend one model family for this dataset. `reason` is shown to the user before "
        "they choose, so make it about their data, not about models in general."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_type": {"type": "string", "enum": list(MODEL_LABELS.values())},
            "reason": {"type": "string"},
        },
        "required": ["model_type", "reason"],
    },
    handler=lambda inp: "recorded",
)


def fallback_model(meta: dict) -> str:
    """What to recommend when Claude is unreachable: flexibility needs rows."""
    rows = meta.get("clean_n_rows") or 0
    return "linear" if 0 < int(rows) < SMALL_DATA_ROWS else "gradient_boosting"
```

The class body:

```python
class CodegenStage:
    name = "codegen"

    def is_complete(self, ctx: StageContext) -> bool:
        root = ctx.project.root
        if not all((root / f).exists() for f in CODE_FILES):
            return False
        config = ctx.project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict):
            return False
        try:
            spec = ctx.spec()
            nested = load_schema(TEMPLATE_FOR_TASK[spec.task_type])
            schema = schema_for(nested, str(config.get("model_type", "")))
        except Exception:  # noqa: BLE001 - missing spec, unknown model or task: not complete
            return False
        return validate_config(config, schema) == []

    def prepare(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        template = TEMPLATE_FOR_TASK.get(spec.task_type)
        if template is None:
            ctx.display(
                f"No training template for task type `{spec.task_type}` yet; "
                "this milestone covers tabular tasks only."
            )
            return None
        meta = ctx.project.read_json(META_FILE) or {}
        problems = check_data(meta, ctx.project.root)
        if problems:
            ctx.display("The data is not ready for training:\n- " + "\n- ".join(problems))
            return None

        nested = load_schema(template)
        model_type = self._choose_model(ctx, spec, meta, nested)
        schema = schema_for(nested, model_type)

        proposal, rationale = self._propose(ctx, spec, meta, schema)
        proposal = {**proposal, "model_type": model_type}
        config, notes = coerce_config(proposal, schema)
        written = copy_template(template, ctx.project.root)
        ctx.project.write_json(cfg.CONFIG_FILE, config)

        files = ", ".join(f"`{p.name}`" for p in written) + ", `config.json`"
        message = [
            f"I wrote the training project into the project folder: {files}.",
            f"`train.py` trains a **{LABEL_FOR_MODEL[model_type]}** model; each [[epoch]] "
            "adds capacity and records train and validation [[loss]] so we can watch for "
            "[[overfitting]]. `evaluate.py` scores a saved model on one split.",
            "",
            rationale,
            "",
            config_table(config, schema),
        ]
        if notes:
            message += ["", "Adjustments to keep values inside the schema:"]
            message += [f"- {n}" for n in notes]
        ctx.display("\n".join(message))

        if not ctx.questioner.confirm("Happy with this configuration? (No lets you change values)"):
            config = self._edit_config(ctx, config, schema)
            ctx.project.write_json(cfg.CONFIG_FILE, config)
            ctx.display("Updated configuration:\n\n" + config_table(config, schema))

        self._walkthrough(ctx, written)
        return None

    def debrief(self, ctx: StageContext) -> None:
        """Codegen needs no cells from the user; everything happened in prepare."""
        return None

    def _walkthrough(self, ctx: StageContext, written: list[Path]) -> None:
        for path in written:
            sections = split_sections(path.read_text(encoding="utf-8"))
            titles = ", ".join(title for title, _code in sections)
            ctx.display(f"### `{path.name}`\n\nSections: {titles}")
            ctx.display(render_walkthrough(sections, {}))

    def _choose_model(self, ctx: StageContext, spec: Spec, meta: dict, nested: dict) -> str:
        recommended, reason = self._recommend(ctx, spec, meta, nested)
        material = load_prompt("teaching/model_choices")
        ctx.display(material)
        ctx.display(f"**My recommendation: {LABEL_FOR_MODEL[recommended]}.** {reason}")
        options = [ASK_LABEL, *MODEL_LABELS]
        answer = ctx.questioner.choice(
            f"Which model shall I set up? (I recommend {LABEL_FOR_MODEL[recommended]})",
            options,
            allow_other=False,
            key="codegen.model_type",
        )
        if answer == ASK_LABEL:
            return recommended
        return MODEL_LABELS.get(answer, recommended)

    def _recommend(
        self, ctx: StageContext, spec: Spec, meta: dict, nested: dict
    ) -> tuple[str, str]:
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["model_type"] = str(inp.get("model_type") or "")
            captured["reason"] = str(inp.get("reason") or "")
            return "recorded"

        tool = ToolSpec(
            name=RECOMMEND_TOOL.name,
            description=RECOMMEND_TOOL.description,
            input_schema=RECOMMEND_TOOL.input_schema,
            handler=handler,
        )
        prompt = json.dumps(
            {
                "spec": spec.to_dict(),
                "data": meta_summary(meta),
                "model_choices": model_types(nested),
                "task": "recommend one model family",
            },
            indent=2,
            default=str,
        )
        try:
            ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}],
                [tool],
            )
        except LLMError as exc:
            chosen = fallback_model(meta)
            rows = meta.get("clean_n_rows")
            return chosen, (
                f"(The assistant was unavailable: {exc}.) Going by the size of the dataset "
                f"({rows} rows), {LABEL_FOR_MODEL[chosen]} is the safe default."
            )
        chosen = captured.get("model_type") or ""
        if chosen not in MODEL_LABELS.values():
            chosen = fallback_model(meta)
        return chosen, captured.get("reason") or "It suits the shape of this dataset."

    def _propose(self, ctx: StageContext, spec: Spec, meta: dict, schema: dict) -> tuple[dict, str]:
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["config"] = inp.get("config") or {}
            captured["rationale"] = str(inp.get("rationale") or "")
            return "recorded"

        tool = ToolSpec(
            name=PROPOSE_TOOL.name,
            description=PROPOSE_TOOL.description,
            input_schema=PROPOSE_TOOL.input_schema,
            handler=handler,
        )
        prompt = json.dumps(
            {
                "spec": spec.to_dict(),
                "data": meta_summary(meta),
                "config_schema": schema,
                "task": "propose starting hyperparameters",
            },
            indent=2,
            default=str,
        )
        try:
            result = ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}],
                [tool],
            )
        except LLMError as exc:
            return {}, f"Using the template defaults (the assistant was unavailable: {exc})."
        rationale = captured.get("rationale") or result.text or "Using the template defaults."
        return dict(captured.get("config") or {}), rationale
```

`config_table` needs one tweak so a choice value renders: its `isinstance(value, float)` branch already falls through to `str(value)` for strings, so no change is needed.

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_codewalk.py tests/test_codegen_stage.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, with `tests/test_train_stage.py`, `tests/test_report_stage.py` and the end-to-end test still xfailed from Task 9. `tests/test_train_stage.py`'s `prepared()` and `tests/test_report_stage.py`'s `trained()` helpers now need a model answer in their scripted answers: `ScriptedQuestioner(["Gradient boosting", "y"])`.

- [ ] **Step 9: Commit**

```bash
git add mlagent/codewalk.py mlagent/prompts/teaching mlagent/prompts/codegen.md \
        mlagent/stages/codegen.py tests
git commit -m "feat: model-choice teaching, a recommend_model tool and a code walkthrough

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 11: The two-phase train stage

**Files:**
- Modify: `mlagent/stages/train.py`
- Test: `tests/test_train_stage.py`

**Interfaces:**
- Consumes: `Handoff`, `ScriptStageBase` (Task 3); `caption_for`, `ctx.display_figure` (Tasks 4, 6); `runlog.append_run/read_runs`; `metrics.json`'s `started_at`, `model_type`, `epochs`, `best_epoch`, `best_val_metric`, `seconds`, `config`, `status`, `error`; `eval_val.json`'s `value`, `loss`, `metric`, `figures`.
- Produces:
  - `EVAL_VAL_FILE = "eval_val.json"`, `CURVES_FIGURE = "training_curves.png"`
  - `build_run_entry(metrics: dict, checkpoint: str | None) -> dict` with exactly the Milestone 3 keys (`started_at`, `status`, `config`, `epochs_run`, `best_epoch`, `best_val_metric`, `final_train_loss`, `final_val_loss`, `seconds`, `error`, `applied_diff`, `checkpoint`)
  - `archive_run(project, run_id: int) -> list[Path]` — copies `plots/training_curves.png` to `plots/run{N}_training.png`, `checkpoints/best.joblib` to `checkpoints/run{N}.joblib`, and every `plots/val_*.png` to `plots/run{N}_val_*.png`; returns the archived plot paths in display order
  - `TrainStage()` takes no constructor arguments; `prepare(ctx) -> Handoff` returns `Handoff("train", [["train.py"], ["evaluate.py"]], ["metrics.json", "eval_val.json"])`; `debrief(ctx) -> None`; `is_complete(ctx)` is "at least one `done` run **and** the current `metrics.json`'s `started_at` is logged".
- `mlagent.runner`, `mlagent.plots` and `matplotlib` are no longer imported by this module. `LivePlotter` and `IPythonDisplay` are deleted (`train.py` draws its own curve).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_train_stage.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlagent import runlog
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.train import EVAL_VAL_FILE, TrainStage, archive_run, build_run_entry
from mlagent.ui.questions import ScriptedQuestioner

SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


def prepared(project):
    """Run codegen with defaults so the training project exists."""
    ctx = StageContext(project=project, llm=FakeLLM([]),
                       questioner=ScriptedQuestioner(["Gradient boosting", "y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(SMALL)
    project.write_json("config.json", cfg)
    return project


def make_ctx(project, llm=None, answers=()):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append,
                       display_figure=lambda path, caption="": figures.append((path, caption)))
    return ctx, shown, figures


def run_cells(project, handoff):
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command], cwd=str(project.root),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_real_training_run_is_logged_archived_and_debriefed(clean_project):
    project = prepared(clean_project)
    llm = FakeLLM([[("text", "Best [[validation accuracy]] beat the target.")]])
    ctx, shown, figures = make_ctx(project, llm)
    stage = TrainStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="train", commands=[["train.py"], ["evaluate.py"]],
                              outputs=["metrics.json", "eval_val.json"])
    text = "\n".join(shown)
    assert "cost" in text.lower() and "cpu" in text.lower()
    assert not stage.outputs_ready(ctx, handoff)

    run_cells(project, handoff)
    assert stage.outputs_ready(ctx, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    runs = runlog.read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["run_id"] == 1
    assert runs[0]["epochs_run"] == 3 and runs[0]["best_val_metric"] is not None
    assert runs[0]["config"]["epochs"] == 3
    assert runs[0]["checkpoint"] == "checkpoints/run1.joblib"
    assert runs[0]["started_at"]
    assert (project.checkpoints_dir / "run1.joblib").exists()
    names = sorted(p.name for p in project.plots_dir.glob("run1_*.png"))
    assert names == ["run1_training.png", "run1_val_confusion.png",
                     "run1_val_per_class.png", "run1_val_roc_pr.png"]
    shown_figures = [Path(p).name for p, _c in figures]
    assert shown_figures[0] == "run1_training.png"
    assert set(shown_figures) == set(names)
    assert all(caption for _p, caption in figures)
    assert "[[validation accuracy]]" in "\n".join(shown)
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "best_epoch" in prompt


def test_a_second_run_is_logged_as_run_two(clean_project):
    project = prepared(clean_project)
    ctx, _shown, _figures = make_ctx(project)
    stage = TrainStage()
    handoff = stage.prepare(ctx)
    run_cells(project, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    # The user edits config.json and runs the cells again without asking the agent first.
    cfg = project.read_json("config.json")
    cfg["epochs"] = 2
    project.write_json("config.json", cfg)
    run_cells(project, handoff)
    assert not stage.is_complete(ctx)  # the new run is not logged yet

    stage.debrief(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["started_at"] != runs[1]["started_at"]
    assert (project.plots_dir / "run2_training.png").exists()
    assert stage.is_complete(ctx)


def test_debriefing_twice_does_not_log_the_same_run_twice(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    stage.debrief(ctx)
    assert len(runlog.read_runs(project.runs_path)) == 1


def test_failed_run_is_logged_and_the_stage_stays_incomplete(clean_project):
    project = prepared(clean_project)
    project.write_json("metrics.json", {
        "status": "failed", "started_at": "2026-09-08T10:00:00.000000+00:00",
        "model_type": "gradient_boosting", "config": {"epochs": 3}, "epochs": [],
        "error": "ValueError: boom", "seconds": 0.2,
    })
    ctx, shown, _figures = make_ctx(project)
    stage = TrainStage()
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    runs = runlog.read_runs(project.runs_path)
    assert runs[0]["status"] == "failed" and "boom" in runs[0]["error"]
    assert any("boom" in s for s in shown)


def test_debrief_without_metrics_says_so(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project)
    TrainStage().debrief(ctx)
    assert any("train.py" in s for s in shown)


def test_prepare_without_a_config_raises(clean_project):
    ctx, _shown, _figures = make_ctx(clean_project)
    try:
        TrainStage().prepare(ctx)
    except RuntimeError as exc:
        assert "codegen" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_build_run_entry_reads_everything_from_metrics():
    metrics = {
        "status": "done", "started_at": "2026-09-08T10:00:00.000000+00:00",
        "config": {"epochs": 2}, "best_epoch": 2, "best_val_metric": 0.9, "seconds": 1.5,
        "error": None,
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
                   {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4}],
    }
    entry = build_run_entry(metrics, "checkpoints/run3.joblib")
    assert entry["status"] == "done" and entry["epochs_run"] == 2
    assert entry["final_val_loss"] == 0.4 and entry["seconds"] == 1.5
    assert entry["applied_diff"] is None and entry["error"] is None
    assert entry["checkpoint"] == "checkpoints/run3.joblib"
    assert entry["started_at"] == metrics["started_at"]

    failed = build_run_entry({"status": "failed", "error": "boom", "epochs": []},
                             "checkpoints/run4.joblib")
    assert failed["status"] == "failed" and failed["checkpoint"] is None


def test_archive_run_copies_curves_checkpoint_and_val_figures(project):
    project.ensure_dirs()
    (project.plots_dir / "training_curves.png").write_bytes(b"a")
    (project.plots_dir / "val_confusion.png").write_bytes(b"b")
    (project.plots_dir / "val_roc_pr.png").write_bytes(b"c")
    (project.checkpoints_dir / "best.joblib").write_bytes(b"d")
    archived = archive_run(project, 7)
    assert [p.name for p in archived] == ["run7_training.png", "run7_val_confusion.png",
                                          "run7_val_roc_pr.png"]
    assert (project.checkpoints_dir / "run7.joblib").exists()


def test_llm_failure_still_completes(clean_project):
    project = prepared(clean_project)
    ctx, shown, _figures = make_ctx(project, FakeLLM([]))
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    assert any("best" in s.lower() for s in shown)
    assert json.loads(project.metrics_path.read_text(encoding="utf-8"))["status"] == "done"
    assert project.exists(EVAL_VAL_FILE)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_train_stage.py -q`
Expected: FAIL with `ImportError: cannot import name 'archive_run' from 'mlagent.stages.train'`.

- [ ] **Step 3: Rewrite the train stage**

Replace `mlagent/stages/train.py`:

```python
"""Train stage: hand the user train.py and evaluate.py, then log and explain the run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mlagent import config as cfg
from mlagent.captions import caption_for
from mlagent.llm import LLMError, ask_text
from mlagent.prompts_io import audience, load_prompt
from mlagent.runlog import append_run, read_runs
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.templates_io import CODE_FILES

EVAL_VAL_FILE = "eval_val.json"
CURVES_FIGURE = "training_curves.png"
BEST_CHECKPOINT = "best.joblib"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_run_entry(metrics: dict, checkpoint: str | None = None) -> dict:
    """One `runs.jsonl` line, entirely derived from the metrics.json that train.py wrote."""
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
        "applied_diff": None,
        "checkpoint": checkpoint if ok else None,
    }


def archive_run(project, run_id: int) -> list[Path]:
    """Freeze this run's outputs under `run{N}` names so later runs cannot overwrite them."""
    archived: list[Path] = []
    curves = project.plots_dir / CURVES_FIGURE
    if curves.exists():
        target = project.plots_dir / f"run{run_id}_training.png"
        shutil.copy2(curves, target)
        archived.append(target)
    for source in sorted(project.plots_dir.glob("val_*.png")):
        target = project.plots_dir / f"run{run_id}_{source.name}"
        shutil.copy2(source, target)
        archived.append(target)
    best = project.checkpoints_dir / BEST_CHECKPOINT
    if best.exists():
        shutil.copy2(best, project.checkpoints_dir / f"run{run_id}.joblib")
    return archived


class TrainStage(ScriptStageBase):
    name = "train"

    def is_complete(self, ctx: StageContext) -> bool:
        runs = read_runs(ctx.project.runs_path)
        if not any(r.get("status") == "done" for r in runs):
            return False
        metrics = ctx.project.read_json(cfg.METRICS_FILE) or {}
        started = metrics.get("started_at")
        if not started:
            return True
        return any(r.get("started_at") == started for r in runs)

    def prepare(self, ctx: StageContext) -> Handoff:
        project = ctx.project
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not all((project.root / f).exists() for f in CODE_FILES):
            raise RuntimeError("training project not found; run the codegen stage first")
        spec = ctx.spec()
        ctx.display(
            "Training runs on the [[CPU]] for tabular data, so there is no [[compute unit]] "
            f"cost gate for this run. `train.py` runs {config.get('epochs')} [[epoch]]s, "
            "redrawing the loss and metric curves as it goes, and saves the best model to "
            "`checkpoints/best.joblib`. `evaluate.py` then scores that model on the "
            f"[[validation set]] and draws the {spec.metric} figures. Run both cells."
        )
        return Handoff(
            stage=self.name,
            commands=[["train.py"], ["evaluate.py"]],
            outputs=[cfg.METRICS_FILE, EVAL_VAL_FILE],
        )

    def debrief(self, ctx: StageContext) -> None:
        project = ctx.project
        metrics = project.read_json(cfg.METRICS_FILE)
        if not isinstance(metrics, dict) or not metrics.get("started_at"):
            ctx.display(
                "I can't see a finished run in `metrics.json` yet. Run the `train.py` cell, "
                "then run this cell again."
            )
            return
        runs = read_runs(project.runs_path)
        entry = next(
            (r for r in runs if r.get("started_at") == metrics["started_at"]), None
        )
        figures: list[Path] = []
        if entry is None:
            run_id = len(runs) + 1
            figures = archive_run(project, run_id)
            checkpoint = (
                f"checkpoints/run{run_id}.joblib"
                if (project.checkpoints_dir / f"run{run_id}.joblib").exists()
                else None
            )
            entry = append_run(project.runs_path, build_run_entry(metrics, checkpoint))
        else:
            figures = sorted(project.plots_dir.glob(f"run{entry['run_id']}_*.png"))
            figures.sort(key=lambda p: (not p.name.endswith("_training.png"), p.name))

        run_id = entry["run_id"]
        if entry["status"] != "done":
            ctx.display(
                f"Run {run_id} failed: {entry['error']}. Fix the cause (the traceback is in "
                "the `train.py` cell's output) and run the cells again."
            )
            return

        spec = ctx.spec()
        for path in figures:
            ctx.display_figure(path, caption_for(path))
        eval_data = project.read_json(EVAL_VAL_FILE) or {}
        ctx.display(self._narrative(ctx, spec, entry, metrics, eval_data, figures))

    def _narrative(self, ctx: StageContext, spec, entry: dict, metrics: dict,
                   eval_data: dict, figures: list[Path]) -> str:
        summary = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "run_id": entry["run_id"],
            "model_type": metrics.get("model_type"),
            "epochs": metrics.get("epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "stopped_early": metrics.get("stopped_early"),
            "validation": {"metric": eval_data.get("metric"), "value": eval_data.get("value"),
                           "loss": eval_data.get("loss")},
            "figures": [p.name for p in figures],
        }
        headline = (
            f"Run {entry['run_id']} finished: best validation {spec.metric} "
            f"{_fmt(entry['best_val_metric'])} at epoch {_fmt(entry['best_epoch'])} "
            f"(target {spec.target_value:g})."
        )
        try:
            narrative = ask_text(
                ctx.llm,
                load_prompt("train", audience=audience(ctx.learning_level())),
                json.dumps(summary, default=str),
            )
        except LLMError:
            narrative = (
                "Look at the [[loss]] curves: if validation loss rises while training loss "
                "keeps falling, the model is [[overfitting]]."
            )
        return headline + "\n\n" + narrative
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_train_stage.py -q`
Expected: PASS. Remove the `xfail` markers added to this file in Task 9.

- [ ] **Step 5: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass; `tests/test_report_stage.py` and the end-to-end test stay xfailed until Tasks 12 and 14. `mlagent/colab.py` constructs `TrainStage()` with no arguments already, so it still imports.

- [ ] **Step 6: Commit**

```bash
git add mlagent/stages/train.py tests/test_train_stage.py
git commit -m "feat: the train stage hands off train.py and evaluate.py and logs runs by started_at

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 12: The two-phase report stage

**Files:**
- Modify: `mlagent/stages/report.py`
- Test: `tests/test_report_stage.py`

**Interfaces:**
- Consumes: `Handoff`, `ScriptStageBase` (Task 3); `caption_for`, `ctx.display_figure`; `runlog.best_run/read_runs/summarise`; `eval_test.json`'s `run_id`, `checkpoint`, `figures`, `value`, `loss`, `metric`; `report_meta.json`'s `best_run`.
- Produces:
  - `EVAL_TEST_FILE = "eval_test.json"`; `render_report(project_name, spec, runs, best, eval_test, lessons, figures)` unchanged; `_run_number` unchanged.
  - `ReportStage()` takes no constructor arguments.
  - `prepare(ctx) -> Handoff | None`: with no successful run, or when the user declines the confirm, it displays why and returns `None`; when `eval_test.json` already matches the best run it returns `None` (the debrief just rewrites the report); otherwise it returns `Handoff("report", [["evaluate.py", "--split", "test"]], ["eval_test.json"])`.
  - `debrief(ctx) -> None`: refuses to write a report from an `eval_test.json` whose `run_id` is neither `None` nor the best run's id; shows the test figures with captions; writes `report.md` and `report_meta.json`.
  - `is_complete` unchanged from the committed version (report + eval_test + `report_meta["best_run"] == best run id`).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_report_stage.py`'s helpers and its first test, and update the rest in the same shape:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlagent import config as cfg
from mlagent.llm import FakeLLM
from mlagent.stages.base import Handoff, StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.report import EVAL_TEST_FILE, ReportStage, render_report
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner


def run_cells(project, handoff):
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command], cwd=str(project.root),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def trained(project):
    ctx = StageContext(project=project, llm=FakeLLM([]),
                       questioner=ScriptedQuestioner(["Gradient boosting", "y"]),
                       explainer=None, display=lambda s: None)
    CodegenStage().prepare(ctx)
    config = project.read_json("config.json")
    config.update({"epochs": 2, "iters_per_epoch": 3, "early_stopping_patience": 0})
    project.write_json("config.json", config)
    stage = TrainStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    assert stage.is_complete(ctx)
    return project


def make_ctx(project, llm=None, answers=("y",)):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    ctx = StageContext(project=project, llm=llm or FakeLLM([]),
                       questioner=ScriptedQuestioner(list(answers)), explainer=None,
                       display=shown.append,
                       display_figure=lambda path, caption="": figures.append((path, caption)))
    return ctx, shown, figures


def test_report_hands_off_the_test_evaluation_then_writes_markdown(clean_project):
    project = trained(clean_project)
    llm = FakeLLM([[("text", "The [[test set]] score was close to validation.")]])
    ctx, shown, figures = make_ctx(project, llm)
    stage = ReportStage()
    assert not stage.is_complete(ctx)

    handoff = stage.prepare(ctx)
    assert handoff == Handoff(stage="report",
                              commands=[["evaluate.py", "--split", "test"]],
                              outputs=["eval_test.json"])
    assert any("test set" in s.lower() for s in shown)
    assert not stage.outputs_ready(ctx, handoff)

    run_cells(project, handoff)
    stage.debrief(ctx)
    assert stage.is_complete(ctx)

    record = json.loads((project.root / EVAL_TEST_FILE).read_text(encoding="utf-8"))
    assert record["run_id"] == 1
    report = project.report_path.read_text(encoding="utf-8")
    assert report.startswith("# ")
    assert "| run | status |" in report
    assert "## Held-out test result" in report and "accuracy" in report
    assert "## Best configuration" in report and "model_type" in report
    assert "## What we learned" in report and "[[test set]]" in report
    assert "plots/test_confusion.png" in report and "plots/run1_training.png" in report
    assert project.read_json(cfg.REPORT_META_FILE)["best_run"] == 1
    assert {Path(p).name for p, _c in figures} >= {"test_confusion.png"}
    assert all(caption for _p, caption in figures)


def test_declining_the_confirm_skips_without_a_handoff(clean_project):
    project = trained(clean_project)
    ctx, shown, _figures = make_ctx(project, answers=("n",))
    stage = ReportStage()
    assert stage.prepare(ctx) is None
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("Skipped" in s for s in shown)


def test_rewrites_the_report_without_touching_test_again(clean_project):
    project = trained(clean_project)
    ctx, _shown, _figures = make_ctx(project)
    stage = ReportStage()
    run_cells(project, stage.prepare(ctx))
    stage.debrief(ctx)
    first = project.report_path.read_text(encoding="utf-8")

    ctx2, shown2, _figures2 = make_ctx(project)
    assert stage.prepare(ctx2) is None  # already evaluated for the best run
    stage.debrief(ctx2)
    assert any("already evaluated" in s.lower() for s in shown2)
    assert project.report_path.read_text(encoding="utf-8") == first


def test_stale_eval_test_is_refused(clean_project):
    project = trained(clean_project)
    ctx, shown, _figures = make_ctx(project)
    stage = ReportStage()
    run_cells(project, stage.prepare(ctx))
    record = json.loads((project.root / EVAL_TEST_FILE).read_text(encoding="utf-8"))
    record["run_id"] = 99
    (project.root / EVAL_TEST_FILE).write_text(json.dumps(record), encoding="utf-8")
    stage.debrief(ctx)
    assert not stage.is_complete(ctx)
    assert any("run 99" in s for s in shown)


def test_no_successful_run_yet(clean_project):
    ctx, shown, _figures = make_ctx(clean_project)
    stage = ReportStage()
    assert stage.prepare(ctx) is None
    assert any("train" in s.lower() for s in shown)


def test_render_report_shape():
    report = render_report(
        "demo", {"goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
                 "target_value": 0.9},
        [{"run_id": 1, "status": "done", "best_val_metric": 0.8}],
        {"run_id": 1, "config": {"model_type": "linear"}, "best_val_metric": 0.8},
        {"metric": "accuracy", "value": 0.78, "loss": 0.5},
        "Lessons here.", [Path("plots/test_confusion.png")],
    )
    assert "# demo: training report" in report
    assert "![test_confusion](plots/test_confusion.png)" in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_report_stage.py -q`
Expected: FAIL — `ReportStage` has no `prepare`, so `stage.prepare(ctx)` raises `AttributeError`.

- [ ] **Step 3: Rewrite the report stage**

Replace everything from `_load_runs_and_best` down in `mlagent/stages/report.py` (keeping `render_report`, `_fmt` and `_run_number` as they are, and dropping the `runner` and `plots` imports):

```python
"""Report stage: one held-out test evaluation, then report.md with figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mlagent import config as cfg
from mlagent.captions import caption_for
from mlagent.llm import LLMError, ask_text
from mlagent.prompts_io import audience, load_prompt
from mlagent.runlog import best_run, read_runs, summarise
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext

EVAL_TEST_FILE = "eval_test.json"
EVAL_TEST_COMMAND = ["evaluate.py", "--split", "test"]
```

and the class:

```python
def _load_runs_and_best(project, spec) -> tuple[list[dict], dict | None]:
    """The one place that decides "the best run": shared by `is_complete`, prepare and debrief."""
    runs = read_runs(project.runs_path)
    return runs, best_run(runs, spec.metric)


class ReportStage(ScriptStageBase):
    name = "report"

    def is_complete(self, ctx: StageContext) -> bool:
        project = ctx.project
        if not (project.report_path.exists() and project.exists(EVAL_TEST_FILE)):
            return False
        meta = project.read_json(cfg.REPORT_META_FILE)
        if not isinstance(meta, dict) or "best_run" not in meta:
            return False
        _runs, best = _load_runs_and_best(project, ctx.spec())
        if best is None:
            return False
        return meta["best_run"] == best.get("run_id")

    def prepare(self, ctx: StageContext) -> Handoff | None:
        project = ctx.project
        spec = ctx.spec()
        _runs, best = _load_runs_and_best(project, spec)
        if best is None:
            ctx.display("No successful training run yet; run the train stage first.")
            return None
        best_run_id = best["run_id"]
        existing = project.read_json(EVAL_TEST_FILE)
        meta = project.read_json(cfg.REPORT_META_FILE)
        meta_best = meta.get("best_run") if isinstance(meta, dict) else None

        if existing is not None and meta_best == best_run_id:
            ctx.display(
                f"The [[test set]] was already evaluated for run {best_run_id}; I will rewrite "
                "the report with the current run history instead of touching it again."
            )
            return None

        if existing is None:
            detail = (
                "The [[test set]] has been untouched until now; evaluating on it once gives "
                "an honest estimate of real-world performance."
            )
        elif meta_best is None:
            detail = (
                "An earlier test evaluation exists but its run is unknown; run "
                f"{best_run_id} is the best model, so it will be evaluated once more."
            )
        else:
            detail = (
                f"The test set was last evaluated for run {meta_best}; run {best_run_id} is now "
                "the best model, so it needs evaluating once more."
            )
        ctx.display(
            f"The best run so far is run {best_run_id} with validation {spec.metric} "
            f"{_fmt(best.get('best_val_metric'))}. {detail}"
        )
        if not ctx.questioner.confirm(
            "Evaluate the best model on the held-out test set now and write the report?",
            default=True,
        ):
            ctx.display("Skipped. Rerun this stage when you have finished tuning.")
            return None
        ctx.display(
            "Run the next cell. `evaluate.py --split test` loads run "
            f"{best_run_id}'s checkpoint and scores it once on the rows no model has seen."
        )
        return Handoff(
            stage=self.name, commands=[list(EVAL_TEST_COMMAND)], outputs=[EVAL_TEST_FILE]
        )

    def debrief(self, ctx: StageContext) -> None:
        project = ctx.project
        spec = ctx.spec()
        runs, best = _load_runs_and_best(project, spec)
        if best is None:
            return
        eval_test = project.read_json(EVAL_TEST_FILE)
        if not isinstance(eval_test, dict):
            ctx.display(
                "I can't see `eval_test.json` yet. Run the `evaluate.py --split test` cell, "
                "then run this cell again."
            )
            return
        run_id = eval_test.get("run_id")
        if run_id is not None and run_id != best["run_id"]:
            ctx.display(
                f"`eval_test.json` was produced for run {run_id}, but run {best['run_id']} is "
                "the best model. Run the `evaluate.py --split test` cell again so the report "
                "describes the model you would actually ship."
            )
            return
        self._write_report(ctx, project, spec, runs, best, eval_test)

    def _write_report(self, ctx: StageContext, project, spec, runs: list[dict], best: dict,
                      eval_test: dict) -> None:
        test_figures = [
            project.plots_dir / str(name) for name in (eval_test.get("figures") or [])
        ]
        run_figures = sorted(project.plots_dir.glob("run*_training.png"), key=_run_number)
        for path in test_figures:
            ctx.display_figure(path, caption_for(path))

        lessons = self._lessons(ctx, spec, runs, best, eval_test)
        report = render_report(
            project.name, spec.to_dict(), runs, best,
            eval_test, lessons, run_figures + test_figures,
        )
        project.report_path.write_text(report, encoding="utf-8")
        project.write_json(
            cfg.REPORT_META_FILE, {"best_run": best["run_id"], "n_runs": len(runs)}
        )
        ctx.display(
            f"Test {spec.metric}: **{_fmt(eval_test.get('value'))}** "
            f"(target {spec.target_value:g}).\n\n{lessons}\n\n"
            f"Report written to `{cfg.REPORT_FILE}` in the project folder."
        )

    def _lessons(self, ctx: StageContext, spec, runs: list[dict], best: dict,
                 eval_test: dict) -> str:
        summary = {
            "spec": spec.to_dict(),
            "runs": [
                {k: r.get(k) for k in ("run_id", "status", "config", "best_epoch",
                                        "best_val_metric", "final_train_loss",
                                        "final_val_loss", "error")}
                for r in runs
            ],
            "best_run_id": best.get("run_id"),
            "test": {"metric": eval_test.get("metric"), "value": eval_test.get("value"),
                     "loss": eval_test.get("loss")},
        }
        try:
            return ask_text(
                ctx.llm,
                load_prompt("report", audience=audience(ctx.learning_level())),
                json.dumps(summary, default=str),
            )
        except LLMError:
            return (
                f"Best validation {spec.metric} was {_fmt(best.get('best_val_metric'))}; "
                f"the [[test set]] gave {_fmt(eval_test.get('value'))}. A large gap between "
                "them means the model does not [[generalise]] well."
            )
```

`re` is still imported for `_run_number`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_report_stage.py -q`
Expected: PASS. Remove the `xfail` markers added to this file in Task 9.

- [ ] **Step 5: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass, with only the end-to-end test still xfailed (Task 14 restores it). Nothing imports `mlagent.runner` any more except `tests/test_runner.py`, which keeps passing.

- [ ] **Step 6: Commit**

```bash
git add mlagent/stages/report.py tests/test_report_stage.py
git commit -m "feat: the report stage hands off evaluate.py --split test and checks the run id

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---
### Task 13: `Teaching` — preambles, level-aware walkthroughs and figure notes

**Files:**
- Create: `mlagent/teaching.py`
- Create: `mlagent/prompts/preamble.md`, `mlagent/prompts/walkthrough.md`
- Modify: `mlagent/stages/base.py` (`StageContext.teaching()`)
- Modify: `mlagent/stages/{data,clean,codegen,train,report}.py`
- Test: `tests/test_teaching.py` (new), the five stage test files

**Interfaces:**
- Consumes: `prompts_io.load_prompt/audience`; `codewalk.split_sections/render_walkthrough`; `captions.caption_for`; `llm.LLM/ToolSpec/LLMError/ask_text`.
- Produces, in `mlagent/teaching.py`:
  - `LEVEL_BLOCK_RE`, `trim_levels(text: str, level: str) -> str` — keeps `<!--level:a,b-->...<!--/level-->` blocks whose list contains `level`, drops the others, leaves unfenced text alone.
  - `material(name: str, level: str) -> str` — `trim_levels(load_prompt(f"teaching/{name}"), level)`.
  - `WRITE_DEBRIEF_TOOL: ToolSpec` named `write_debrief` with input `{narrative: string, figure_notes: object}`.
  - `@dataclass class Teaching(level: str, llm: LLM, display: Callable[[str], None], display_figure: Callable[[Path, str], None])` with:
    - `preamble(self, stage: str, payload: dict) -> None` — nothing at `expert`; otherwise one LLM call, displayed.
    - `walkthrough(self, paths: list[Path]) -> None` — `expert`: file names and section titles only, no LLM call; `intermediate`: one LLM call for all files, a paragraph each; `beginner`: one LLM call per file explaining every section.
    - `debrief(self, prompt_name: str, payload: dict, figures: list[Path], fallback: str = "") -> str` — one LLM call with `write_debrief`; shows every figure with its fixed caption plus the model's note for it (below `expert`); returns the narrative, or `fallback` on `LLMError`.
    - `show_figures(self, paths: list[Path], notes: dict[str, str]) -> None` — caption plus note, keyed by filename.
  - `StageContext.teaching() -> Teaching`.
- Call sites: `DataStage.prepare` and `CleanStage.prepare` call `preamble`; `CodegenStage.prepare` calls `walkthrough(written)`; `DataStage.debrief`, `CleanStage.debrief`, `TrainStage.debrief` and `ReportStage.debrief` call `debrief(...)` instead of `ask_text` and stop calling `ctx.display_figure` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_teaching.py`:

```python
from __future__ import annotations

from pathlib import Path

from mlagent.llm import FakeLLM
from mlagent.teaching import Teaching, material, trim_levels

TEXT = """Always shown.

<!--level:beginner-->
Only beginners.
<!--/level-->

<!--level:beginner,intermediate-->
Beginners and intermediates.
<!--/level-->

The end.
"""


def make(level, llm):
    shown: list[str] = []
    figures: list[tuple[Path, str]] = []
    teaching = Teaching(
        level=level, llm=llm, display=shown.append,
        display_figure=lambda path, caption="": figures.append((path, caption)),
    )
    return teaching, shown, figures


def test_trim_levels_keeps_matching_blocks_only():
    beginner = trim_levels(TEXT, "beginner")
    assert "Only beginners." in beginner and "Beginners and intermediates." in beginner
    intermediate = trim_levels(TEXT, "intermediate")
    assert "Only beginners." not in intermediate
    assert "Beginners and intermediates." in intermediate
    expert = trim_levels(TEXT, "expert")
    assert "Only beginners." not in expert and "Beginners and intermediates." not in expert
    for level in ("beginner", "intermediate", "expert"):
        text = trim_levels(TEXT, level)
        assert "Always shown." in text and "The end." in text
        assert "<!--" not in text


def test_material_loads_and_trims_a_teaching_file():
    expert = material("model_choices", "expert")
    beginner = material("model_choices", "beginner")
    assert "Random forest" in expert and "Random forest" in beginner
    assert len(beginner) > len(expert)
    assert "<!--" not in beginner


def test_preamble_is_silent_for_experts_and_costs_one_call_otherwise():
    llm = FakeLLM([])
    teaching, shown, _figures = make("expert", llm)
    teaching.preamble("data", {"rows": 100})
    assert shown == [] and llm.calls == []

    llm = FakeLLM([[("text", "Next we look at the shape of your data.")]])
    teaching, shown, _figures = make("beginner", llm)
    teaching.preamble("data", {"rows": 100})
    assert shown == ["Next we look at the shape of your data."]
    assert len(llm.calls) == 1
    assert "data" in llm.calls[0]["messages"][0]["content"]


def test_preamble_is_silent_when_the_llm_fails():
    teaching, shown, _figures = make("beginner", FakeLLM([]))
    teaching.preamble("clean", {})
    assert shown == []


def test_walkthrough_costs_no_call_for_experts(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("# --- settings ---\nX = 1\n# --- loop ---\nY = 2\n", encoding="utf-8")
    llm = FakeLLM([])
    teaching, shown, _figures = make("expert", llm)
    teaching.walkthrough([script])
    assert llm.calls == []
    text = "\n".join(shown)
    assert "train.py" in text and "settings" in text and "loop" in text
    assert "```python" not in text  # experts get the file list, not the source


def test_walkthrough_beginner_explains_every_section(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("# --- settings ---\nX = 1\n# --- loop ---\nY = 2\n", encoding="utf-8")
    llm = FakeLLM([
        [("tool", "write_walkthrough",
          {"explanations": {"settings": "Knobs.", "loop": "The epochs."}})],
        [("text", "done")],
    ])
    teaching, shown, _figures = make("beginner", llm)
    teaching.walkthrough([script])
    text = "\n".join(shown)
    assert "```python" in text and "Knobs." in text and "The epochs." in text
    assert len(llm.calls) == 1  # one call per file


def test_walkthrough_intermediate_uses_one_call_for_all_files(tmp_path):
    a = tmp_path / "data.py"
    b = tmp_path / "model.py"
    a.write_text("# --- load ---\nA = 1\n", encoding="utf-8")
    b.write_text("# --- build ---\nB = 2\n", encoding="utf-8")
    llm = FakeLLM([
        [("tool", "write_walkthrough",
          {"explanations": {"data.py": "Loads and splits.", "model.py": "Builds the model."}})],
        [("text", "done")],
    ])
    teaching, shown, _figures = make("intermediate", llm)
    teaching.walkthrough([a, b])
    text = "\n".join(shown)
    assert "Loads and splits." in text and "Builds the model." in text
    assert len(llm.calls) == 1


def test_debrief_returns_the_narrative_and_annotates_figures(tmp_path):
    fig = tmp_path / "val_confusion.png"
    fig.write_bytes(b"x")
    llm = FakeLLM([
        [("tool", "write_debrief",
          {"narrative": "Validation held up.",
           "figure_notes": {"val_confusion.png": "Class 1 is the weak one here."}})],
        [("text", "done")],
    ])
    teaching, shown, figures = make("intermediate", llm)
    text = teaching.debrief("train", {"run_id": 1}, [fig], fallback="fallback")
    assert text == "Validation held up."
    assert shown == []  # the caller displays the narrative
    assert len(figures) == 1
    path, caption = figures[0]
    assert path == fig
    assert "Rows are the true label" in caption  # the fixed caption
    assert "Class 1 is the weak one here." in caption


def test_debrief_falls_back_to_fixed_captions_when_the_llm_fails(tmp_path):
    fig = tmp_path / "val_confusion.png"
    fig.write_bytes(b"x")
    teaching, _shown, figures = make("beginner", FakeLLM([]))
    text = teaching.debrief("train", {}, [fig], fallback="fallback text")
    assert text == "fallback text"
    assert len(figures) == 1
    assert "Rows are the true label" in figures[0][1]


def test_expert_debrief_asks_for_no_figure_notes(tmp_path):
    fig = tmp_path / "test_residuals.png"
    fig.write_bytes(b"x")
    llm = FakeLLM([
        [("tool", "write_debrief", {"narrative": "rmse 0.4 vs target 0.5.",
                                     "figure_notes": {}})],
        [("text", "done")],
    ])
    teaching, _shown, figures = make("expert", llm)
    assert teaching.debrief("report", {}, [fig]) == "rmse 0.4 vs target 0.5."
    payload = llm.calls[0]["messages"][0]["content"]
    assert "figure_notes" not in payload or "figures" not in payload
    assert figures[0][1].startswith("Left: the spread of actual minus predicted")
```

Add one assertion to each stage test file that the level reaches the prompt:

```python
# tests/test_data_stage.py
def test_learning_level_reaches_the_debrief_prompt(project):
    llm = FakeLLM([[("tool", "write_debrief", {"narrative": "n", "figure_notes": {}})],
                   [("text", "done")]])
    ctx, _shown, _figures = make_ctx(project, ["100", "3", "2", "0.5", "0.0", "n"], llm=llm)
    project.write_json("spec.json", {**SPEC, "learning_level": "beginner"})
    stage = DataStage()
    stage.prepare(ctx)
    run_profile_script(project)
    stage.debrief(ctx)
    assert "new to machine learning" in llm.calls[-1]["system"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_teaching.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlagent.teaching'`.

- [ ] **Step 3: Write the two teaching prompts**

`mlagent/prompts/preamble.md`:

```markdown
You are an ML training assistant running inside Google Colab, about to start one stage of the pipeline. You receive the stage name and a small JSON summary of what is known so far.

Write two or three sentences, under 80 words, that tell the user what is about to happen and why it matters for their project. Say what they will be asked to do and what they will see at the end. Do not list numbers they can already see, do not summarise the JSON and do not promise results.

Wrap technical terms in double square brackets like [[validation set]] so the user can click them.

Audience: {audience}
```

`mlagent/prompts/walkthrough.md`:

```markdown
You are an ML training assistant explaining generated code to the person who will run it. The code is already written and tested; your job is to make it readable, not to review or change it.

You receive JSON with the script's filename and its sections (each section is a title and its source). Call `write_walkthrough` exactly once with an `explanations` object.

- When the JSON lists sections of one file, use each section title as a key and explain that section in one or two sentences: what it does and why it is there. Name the one line worth looking at.
- When the JSON lists several files, use each filename as a key and explain that whole file in one short paragraph: what it is responsible for and what the user would change in it.

Never invent code that is not in the source. Do not restate the code line by line. Wrap technical terms in double square brackets like [[epoch]] so the user can click them.

Audience: {audience}
```

Both stage debrief prompts (`data.md`, `clean.md`, `train.md`, `report.md`) gain a paragraph before their audience line:

```markdown
Call `write_debrief` exactly once. `narrative` is the text above; `figure_notes` maps each figure filename you were given to one sentence about what THIS data shows in it (leave it empty when you were given no figures).
```

- [ ] **Step 4: Write `teaching.py`**

Create `mlagent/teaching.py`:

```python
"""How much to explain, and the LLM calls that do the explaining.

One object per stage run. Everything it does degrades to fixed text on `LLMError`, so a
missing API key costs the user detail, never a stage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mlagent.captions import caption_for
from mlagent.codewalk import render_walkthrough, split_sections
from mlagent.llm import LLM, LLMError, ToolSpec
from mlagent.prompts_io import audience, load_prompt

LEVEL_BLOCK_RE = re.compile(r"<!--level:([a-z, ]+)-->(.*?)<!--/level-->\n?", re.DOTALL)
EXPERT = "expert"
BEGINNER = "beginner"

WRITE_WALKTHROUGH_TOOL = ToolSpec(
    name="write_walkthrough",
    description=(
        "Explain the code you were shown. `explanations` maps each key you were given "
        "(a section title, or a filename) to your explanation of it."
    ),
    input_schema={
        "type": "object",
        "properties": {"explanations": {"type": "object"}},
        "required": ["explanations"],
    },
    handler=lambda inp: "recorded",
)

WRITE_DEBRIEF_TOOL = ToolSpec(
    name="write_debrief",
    description=(
        "Record the debrief. `narrative` is the prose shown to the user; `figure_notes` "
        "maps a figure filename to one sentence about what this data shows in it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
            "figure_notes": {"type": "object"},
        },
        "required": ["narrative"],
    },
    handler=lambda inp: "recorded",
)


def trim_levels(text: str, level: str) -> str:
    """Drop the level-fenced blocks that are not for this reader; keep everything else."""

    def keep(match: re.Match) -> str:
        levels = {part.strip() for part in match.group(1).split(",") if part.strip()}
        return match.group(2).strip() + "\n" if level in levels else ""

    return LEVEL_BLOCK_RE.sub(keep, text).strip() + "\n"


def material(name: str, level: str) -> str:
    """A teaching document from `prompts/teaching/`, trimmed to this level."""
    return trim_levels(load_prompt(f"teaching/{name}"), level)


def _capture(store: dict, key: str) -> Callable[[dict], str]:
    def handler(inp: dict) -> str:
        value = inp.get(key)
        store[key] = value if isinstance(value, dict) else {}
        if "narrative" in inp:
            store["narrative"] = str(inp.get("narrative") or "")
        return "recorded"

    return handler


@dataclass
class Teaching:
    level: str
    llm: LLM
    display: Callable[[str], None]
    display_figure: Callable[[Path, str], None]

    def _system(self, name: str) -> str:
        return load_prompt(name, audience=audience(self.level))

    # --- before a stage ---
    def preamble(self, stage: str, payload: dict) -> None:
        if self.level == EXPERT:
            return
        prompt = json.dumps({"stage": stage, "context": payload}, indent=2, default=str)
        try:
            result = self.llm.run(
                self._system("preamble"), [{"role": "user", "content": prompt}], []
            )
        except LLMError:
            return
        if result.text.strip():
            self.display(result.text.strip())

    # --- explaining generated code ---
    def walkthrough(self, paths: list[Path]) -> None:
        paths = [Path(p) for p in paths]
        if not paths:
            return
        if self.level == EXPERT:
            for path in paths:
                titles = ", ".join(
                    title for title, _code in split_sections(path.read_text(encoding="utf-8"))
                )
                self.display(f"`{path.name}` — sections: {titles}")
            return
        if self.level == BEGINNER:
            for path in paths:
                sections = split_sections(path.read_text(encoding="utf-8"))
                explanations = self._ask_walkthrough(
                    {
                        "file": path.name,
                        "sections": [
                            {"title": title, "source": code} for title, code in sections
                        ],
                    }
                )
                self.display(f"### `{path.name}`")
                self.display(render_walkthrough(sections, explanations))
            return
        files = [
            {"file": path.name, "source": path.read_text(encoding="utf-8")} for path in paths
        ]
        explanations = self._ask_walkthrough({"files": files})
        lines = ["### The generated code", ""]
        for path in paths:
            note = (explanations.get(path.name) or "").strip()
            titles = ", ".join(
                title for title, _code in split_sections(path.read_text(encoding="utf-8"))
            )
            lines.append(f"**`{path.name}`** — sections: {titles}")
            if note:
                lines.append("")
                lines.append(note)
            lines.append("")
        self.display("\n".join(lines).strip())

    def _ask_walkthrough(self, payload: dict) -> dict[str, str]:
        store: dict = {}
        tool = ToolSpec(
            name=WRITE_WALKTHROUGH_TOOL.name,
            description=WRITE_WALKTHROUGH_TOOL.description,
            input_schema=WRITE_WALKTHROUGH_TOOL.input_schema,
            handler=_capture(store, "explanations"),
        )
        try:
            self.llm.run(
                self._system("walkthrough"),
                [{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            return {}
        return {str(k): str(v) for k, v in (store.get("explanations") or {}).items()}

    # --- after a stage ---
    def show_figures(self, paths: list[Path], notes: dict[str, str]) -> None:
        for path in paths:
            path = Path(path)
            caption = caption_for(path)
            note = (notes.get(path.name) or "").strip()
            if note:
                caption = f"{caption} {note}".strip()
            self.display_figure(path, caption)

    def debrief(
        self,
        prompt_name: str,
        payload: dict,
        figures: list[Path],
        fallback: str = "",
    ) -> str:
        figures = [Path(p) for p in figures]
        body = dict(payload)
        if self.level != EXPERT and figures:
            body["figures"] = [p.name for p in figures]
        store: dict = {}
        tool = ToolSpec(
            name=WRITE_DEBRIEF_TOOL.name,
            description=WRITE_DEBRIEF_TOOL.description,
            input_schema=WRITE_DEBRIEF_TOOL.input_schema,
            handler=_capture(store, "figure_notes"),
        )
        narrative = fallback
        notes: dict[str, str] = {}
        try:
            result = self.llm.run(
                self._system(prompt_name),
                [{"role": "user", "content": json.dumps(body, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            pass
        else:
            narrative = store.get("narrative") or result.text.strip() or fallback
            notes = {str(k): str(v) for k, v in (store.get("figure_notes") or {}).items()}
        self.show_figures(figures, notes)
        return narrative
```

Add to `StageContext` in `mlagent/stages/base.py`:

```python
    def teaching(self) -> Teaching:
        """The explainer for this stage, at the user's chosen level."""
        return Teaching(
            level=self.learning_level(),
            llm=self.llm,
            display=self.display,
            display_figure=self.display_figure,
        )
```
with `from mlagent.teaching import Teaching` at the top.

- [ ] **Step 5: Wire `Teaching` into the five stages**

`mlagent/stages/data.py`:
- in `prepare`, before the source questions:
  ```python
        ctx.teaching().preamble("data", {"spec": spec.to_dict()})
  ```
- replace the figure loop and `_narrate` in `debrief` with:
  ```python
        ctx.display(profile_markdown(profile))
        figures = [
            ctx.project.plots_dir / str(name) for name in (profile.get("figures") or [])
        ]
        payload = {
            "spec": ctx.spec().to_dict(),
            "profile": {k: v for k, v in profile.items() if k != "figures"},
        }
        fallback = "Data saved. Next: the [[data cleaning]] audit."
        ctx.display(ctx.teaching().debrief("data", payload, figures, fallback=fallback))
  ```
  and delete `_narrate`, `ask_text`, `caption_for`, `audience` and `load_prompt` from the imports.

`mlagent/stages/clean.py`:
- in `prepare`, before the audit: `ctx.teaching().preamble("clean", {"target": target, "n_rows": int(len(df))})`
- `_explain` keeps `ask_text` (it runs *before* the fixes, with no figures).
- in `debrief`, replace the figure loop and the plain `ctx.display(self._summary(...))` with:
  ```python
        figures = [
            ctx.project.plots_dir / str(name) for name in (profile.get("figures") or [])
        ]
        ctx.display(self._summary(profile, meta))
        payload = {"before": profile.get("before"), "after": profile.get("after"),
                   "steps": profile.get("steps"), "splits": meta.get("splits")}
        note = ctx.teaching().debrief("clean", payload, figures, fallback="")
        if note:
            ctx.display(note)
  ```

`mlagent/stages/codegen.py`: replace `_walkthrough`'s body with `ctx.teaching().walkthrough(written)` and delete the `codewalk` imports from that module.

`mlagent/stages/train.py`:
- in `prepare`, after the CPU line: `ctx.teaching().preamble("train", {"config": config, "metric": spec.metric})`
- in `debrief`, delete the `for path in figures: ctx.display_figure(...)` loop (Teaching
  shows the figures now, with the model's note appended to each caption)
- `_narrative` becomes:
  ```python
    def _narrative(self, ctx: StageContext, spec, entry: dict, metrics: dict,
                   eval_data: dict, figures: list[Path]) -> str:
        summary = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "run_id": entry["run_id"],
            "model_type": metrics.get("model_type"),
            "epochs": metrics.get("epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "stopped_early": metrics.get("stopped_early"),
            "validation": {"metric": eval_data.get("metric"), "value": eval_data.get("value"),
                           "loss": eval_data.get("loss")},
        }
        headline = (
            f"Run {entry['run_id']} finished: best validation {spec.metric} "
            f"{_fmt(entry['best_val_metric'])} at epoch {_fmt(entry['best_epoch'])} "
            f"(target {spec.target_value:g})."
        )
        fallback = (
            "Look at the [[loss]] curves: if validation loss rises while training loss "
            "keeps falling, the model is [[overfitting]]."
        )
        narrative = ctx.teaching().debrief("train", summary, figures, fallback=fallback)
        return headline + "\n\n" + narrative
  ```
  (`Teaching.debrief` adds the figure filenames to the payload itself, so the `"figures"`
  key drops out of `summary`), and `ask_text`, `load_prompt`, `audience` and `caption_for`
  leave this module's imports.

`mlagent/stages/report.py`: in `_write_report`, drop the `ctx.display_figure` loop; `_lessons` becomes

```python
    def _lessons(self, ctx, spec, runs, best, eval_test, figures) -> str:
        summary = {
            "spec": spec.to_dict(),
            "runs": [
                {k: r.get(k) for k in ("run_id", "status", "config", "best_epoch",
                                        "best_val_metric", "final_train_loss",
                                        "final_val_loss", "error")}
                for r in runs
            ],
            "best_run_id": best.get("run_id"),
            "test": {"metric": eval_test.get("metric"), "value": eval_test.get("value"),
                     "loss": eval_test.get("loss")},
        }
        fallback = (
            f"Best validation {spec.metric} was {_fmt(best.get('best_val_metric'))}; "
            f"the [[test set]] gave {_fmt(eval_test.get('value'))}. A large gap between "
            "them means the model does not [[generalise]] well."
        )
        return ctx.teaching().debrief("report", summary, figures, fallback=fallback)
```
called as `lessons = self._lessons(ctx, spec, runs, best, eval_test, test_figures)`.

Every stage test's `FakeLLM` script now needs a `write_debrief` tool turn instead of a plain text turn wherever a debrief narrative is asserted, e.g.

```python
    llm = FakeLLM([
        [("tool", "write_debrief", {"narrative": "Best [[validation accuracy]] beat the target.",
                                     "figure_notes": {"run1_val_confusion.png": "Class 1 lags."}})],
        [("text", "done")],
    ])
```
Empty `FakeLLM([])` scripts keep exercising the fallback path unchanged.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_teaching.py tests/test_data_stage.py tests/test_clean_stage.py tests/test_codegen_stage.py tests/test_train_stage.py tests/test_report_stage.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite and lint**

Run: `python -m pytest -q && python -m ruff check .`
Expected: all pass except the still-xfailed end-to-end test.

- [ ] **Step 8: Commit**

```bash
git add mlagent/teaching.py mlagent/prompts mlagent/stages tests
git commit -m "feat: level-aware preambles, code walkthroughs and figure notes via Teaching

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

### Task 14: Colab entry point, the notebook, the plots clear-out, and the docs

**Files:**
- Modify: `mlagent/colab.py`
- Rewrite: `scripts/build_notebook.py`
- Modify: `mlagent/plots.py`
- Delete: `tests/test_plots_training.py`
- Rewrite: `tests/test_plots.py`
- Modify: `tests/conftest.py`, `tests/test_pipeline_e2e.py`, `tests/test_colab.py`
- Modify: `CLAUDE.md`, `docs/colab-smoke.md`

**Interfaces:**
- Consumes: everything above.
- Produces, in `mlagent/colab.py`:
  - `SCRIPT_CELL_FORMATS = {"load": "%load {script}", "run": "%run {script} {args}"}`
  - `HANDOFF_COMMANDS: dict[str, list[list[str]]]` = `{"intake": [], "data": [["profile.py"]], "clean": [["clean.py"]], "codegen": [], "train": [["train.py"], ["evaluate.py"]], "report": [["evaluate.py", "--split", "test"]]}`
  - `PURGED_MODULES = ("clean", "data", "evaluate", "model", "profile", "train")`
  - `cell_source(command: list[str]) -> str` — `%load script` for a bare script, `%run script args` when there are arguments
  - `script_cells(stage_name: str) -> list[str]`
  - `start()` now also `os.chdir(project.root)` and registers the pre-run-cell purge hook
- Produces, in `mlagent/plots.py`: only `SERIES`, `SEQUENTIAL`, `DIVERGING`, `INK`, `INK_2`, `MUTED`, `GRID`, `AXIS`, `SURFACE`, `SEQ_CMAP`, `DIV_CMAP`, `style_axes(ax, title=None, grid_axis="y")`, `save_figure(fig, path)`, `present(fig, plots_dir, name)`.
- Produces, in `tests/conftest.py`: `run_handoff(project, handoff) -> None` and `advance(orch, project, limit=12) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_colab.py`:

```python
def test_cell_source_and_script_cells():
    from mlagent import colab

    assert colab.cell_source(["profile.py"]) == "%load profile.py"
    assert colab.cell_source(["evaluate.py", "--split", "test"]) == "%run evaluate.py --split test"
    assert colab.script_cells("train") == ["%load train.py", "%load evaluate.py"]
    assert colab.script_cells("intake") == []
    assert colab.script_cells("nope") == []
    assert set(colab.HANDOFF_COMMANDS) == {"intake", "data", "clean", "codegen", "train",
                                           "report"}


def test_handoff_commands_match_what_the_stages_return(tmp_path):
    from mlagent import colab
    from mlagent.stages.clean import CLEAN_PY
    from mlagent.stages.report import EVAL_TEST_COMMAND

    assert colab.HANDOFF_COMMANDS["clean"] == [[CLEAN_PY]]
    assert colab.HANDOFF_COMMANDS["report"] == [list(EVAL_TEST_COMMAND)]


def test_start_changes_into_the_project_folder(tmp_path, monkeypatch):
    import os

    from mlagent import colab
    from mlagent.llm import FakeLLM

    monkeypatch.chdir(tmp_path)
    colab.start("demo", drive_root=str(tmp_path / "drive"), llm=FakeLLM([]))
    assert Path(os.getcwd()).resolve() == (
        tmp_path / "drive" / "projects" / "demo"
    ).resolve()


def test_purge_modules_forgets_generated_scripts(monkeypatch):
    import sys

    from mlagent import colab

    sentinel = object()
    monkeypatch.setitem(sys.modules, "model", sentinel)
    monkeypatch.setitem(sys.modules, "evaluate", sentinel)
    colab._purge_modules()
    assert "model" not in sys.modules and "evaluate" not in sys.modules
```

Rewrite `tests/test_plots.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt

from mlagent import plots


def test_palette_constants_are_present():
    assert plots.SERIES[0] == "#2a78d6"
    assert len(plots.SERIES) == 8 and len(plots.SEQUENTIAL) == 7
    assert len(plots.DIVERGING) == 5
    assert plots.SEQ_CMAP.name == "mlagent_seq" and plots.DIV_CMAP.name == "mlagent_div"


def test_style_axes_hides_the_top_and_right_spines_and_sets_the_title():
    fig, ax = plt.subplots()
    plots.style_axes(ax, "A title")
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.get_title() == "A title"
    plt.close(fig)


def test_present_saves_and_closes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    path = plots.present(fig, tmp_path, "demo")
    assert path == tmp_path / "demo.png" and path.exists()
    assert not plt.fignum_exists(fig.number)


def test_the_figure_functions_have_moved_to_the_templates():
    for gone in ("feature_histograms", "missing_matrix", "correlation_heatmap",
                 "before_after_missing", "training_curves", "confusion_matrix_plot",
                 "roc_pr_curves", "per_class_bars", "predicted_vs_actual",
                 "residual_plots", "present_evaluation"):
        assert not hasattr(plots, gone), gone
```

Rewrite `tests/test_pipeline_e2e.py`'s body (helpers move to `conftest.py`):

```python
"""End-to-end run of intake -> data -> clean -> codegen -> train -> report, no network."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.runlog import read_runs
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CLEAN_FILE, CLEAN_PY, CleanStage
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.data import RAW_FILE, DataStage
from mlagent.stages.intake import IntakeStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
from mlagent.ui.questions import ScriptedQuestioner

FORM_ANSWERS = {
    "intake.goal": "Predict churn from account data",
    "intake.learning_level": "Intermediate - explain the key ideas",
    "intake.task_type": "Tabular classification",
    "intake.metric": "accuracy",
    "intake.target_value": 0.9,
    "intake.data_source": "Synthetic data",
    "intake.minutes_per_run": 10,
    "intake.max_rounds": 5,
    "intake.gpu": "No GPU (CPU only)",
    "data.n_rows": 300,
    "data.n_features": 6,
    "data.n_classes": 2,
    "data.class_balance": 0.6,
    "data.noise": 0.1,
    "data.inject_quirks": True,
    "clean.drop_columns": "",
    "clean.train_fraction": 0.7,
    "clean.val_fraction": 0.15,
    "codegen.model_type": "Gradient boosting",
}
ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Every confirm() is approved without consuming a scripted answer (the number of
    audit fixes varies with the data)."""

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        self.asked.append(question)
        return True


def make_orchestrator(project, stages=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=AutoApproveQuestioner([]),
        explainer=None,
        display=lambda s: None,
        display_figure=lambda path, caption="": None,
    )
    stages = stages or [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                        TrainStage(), ReportStage()]
    return Orchestrator(ctx, stages)


def test_full_pipeline_runs_through_handoffs(project):
    orch = make_orchestrator(project)
    ran = advance(orch, project, answers=FORM_ANSWERS)
    assert ran == ALL_STAGES

    assert project.exists("spec.json") and project.exists("draft_spec.json")
    assert project.read_json("spec.json")["learning_level"] == "intermediate"
    assert (project.data_raw / RAW_FILE).exists()
    for name in ("data_meta.json", "profile_raw.json", "profile.py", AUDIT_FILE,
                 "profile_clean.json", CLEAN_PY, "data.py", "model.py", "train.py",
                 "evaluate.py", "config.json", "metrics.json", "eval_val.json",
                 "eval_test.json", "runs.jsonl", "report.md", "report_meta.json"):
        assert project.exists(name), name
    assert (project.data_clean / CLEAN_FILE).exists()
    assert (project.checkpoints_dir / "best.joblib").exists()
    assert (project.checkpoints_dir / "run1.joblib").exists()
    assert (project.plots_dir / "raw_histograms.png").exists()
    assert (project.plots_dir / "clean_before_after_missing.png").exists()
    assert (project.plots_dir / "run1_training.png").exists()
    assert (project.plots_dir / "test_confusion.png").exists()
    runs = read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done"

    # clean.py reproduces data/clean/data.csv exactly when run on data/raw/data.csv.
    raw_df = pd.read_csv(project.data_raw / RAW_FILE)
    clean_df = pd.read_csv(project.data_clean / CLEAN_FILE)
    namespace: dict = {}
    exec(compile((project.root / CLEAN_PY).read_text(encoding="utf-8"), "clean.py", "exec"),
         namespace)
    reproduced = namespace["clean"](raw_df).reset_index(drop=True)
    assert len(reproduced) == len(clean_df)

    # Deleting state.json: every stage is complete via its artifacts, so nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []


def test_a_second_training_run_is_logged_and_the_report_re_triggers(project):
    orch = make_orchestrator(project)
    advance(orch, project, answers=FORM_ANSWERS)

    orch.reset("train")
    ran = advance(orch, project)
    assert ran == ["train", "report"]
    assert len(read_runs(project.runs_path)) == 2
    assert (project.plots_dir / "run2_training.png").exists()
    meta = project.read_json("report_meta.json")
    best = max(read_runs(project.runs_path), key=lambda r: r["best_val_metric"])
    assert meta["best_run"] == best["run_id"]


def test_editing_a_script_makes_its_outputs_stale(project):
    import os
    import time

    orch = make_orchestrator(project)
    advance(orch, project, answers=FORM_ANSWERS)

    stage = DataStage()
    handoff = json.loads('{"stage": "data", "commands": [["profile.py"]], '
                         '"outputs": ["profile_raw.json"]}')
    from mlagent.stages.base import Handoff

    parsed = Handoff.from_dict(handoff)
    assert stage.outputs_ready(orch.ctx, parsed) is True
    script = project.root / "profile.py"
    future = time.time() + 60
    os.utime(script, (future, future))
    assert stage.outputs_ready(orch.ctx, parsed) is False


@pytest.mark.parametrize("level", ["beginner", "expert"])
def test_the_pipeline_runs_at_every_learning_level(project, level):
    labels = {"beginner": "Beginner - explain everything as we go",
              "expert": "Expert - just the numbers"}
    answers = {**FORM_ANSWERS, "intake.learning_level": labels[level]}
    orch = make_orchestrator(project)
    assert advance(orch, project, answers=answers) == ALL_STAGES
    assert project.read_json("spec.json")["learning_level"] == level
```

Add to `tests/conftest.py`:

```python
import subprocess  # noqa: E402
import sys  # noqa: E402


def run_handoff(project, handoff) -> None:
    """Run every cell of a handoff the way the user would, from the project folder."""
    for command in handoff.commands:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=str(project.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
        assert result.returncode == 0, (
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )


@pytest.fixture
def advance():
    """Drive an orchestrator to a standstill, running each handoff's cells in between."""

    def _advance(orch, project, answers=None, limit=12) -> list[str]:
        ran: list[str] = []
        for _ in range(limit):
            ran += orch.run(answers=answers)
            handoff = orch.waiting()
            if handoff is None:
                return ran
            run_handoff(project, handoff)
        raise AssertionError(f"pipeline did not settle after {limit} rounds")

    return _advance
```

(and the e2e test's functions take `advance` as a fixture argument: `def test_full_pipeline_runs_through_handoffs(project, advance):` and so on.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_colab.py tests/test_plots.py -q`
Expected: FAIL with `AttributeError: module 'mlagent.colab' has no attribute 'cell_source'`.

- [ ] **Step 3: Strip `plots.py` back to the house style**

Replace `mlagent/plots.py` with only:

```python
"""The house chart style. Every figure the *user* sees is drawn by a template script that
carries its own copy of this palette; what stays here is the palette itself plus the two
helpers Milestone 5's agent-side run-comparison plots will use."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#f3a17f", "#d95926"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
DIV_CMAP = LinearSegmentedColormap.from_list("mlagent_div", DIVERGING)


def style_axes(ax, title: str | None = None, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    return path


def _show(fig: Figure) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
    except Exception:  # noqa: BLE001 - display is best effort, never fatal
        pass


def present(fig: Figure, plots_dir: Path, name: str) -> Path:
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    _show(fig)
    plt.close(fig)
    return path
```

Delete `tests/test_plots_training.py`.

- [ ] **Step 4: Update `colab.py`**

In `mlagent/colab.py`, add after the imports:

```python
# The single place the "run a script in its own cell" mechanism is defined. Swapping
# %load for %run here changes every generated cell and the notebook builder at once.
SCRIPT_CELL_FORMATS = {"load": "%load {script}", "run": "%run {script} {args}"}

# What each stage's handoff asks the user to run. Kept in step with the stages by
# tests/test_colab.py, and shared with scripts/build_notebook.py.
HANDOFF_COMMANDS: dict[str, list[list[str]]] = {
    "intake": [],
    "data": [["profile.py"]],
    "clean": [["clean.py"]],
    "codegen": [],
    "train": [["train.py"], ["evaluate.py"]],
    "report": [["evaluate.py", "--split", "test"]],
}

# Scripts the user runs in their own cells. Python caches them after the first import,
# so a regenerated file would be ignored without this purge.
PURGED_MODULES = ("clean", "data", "evaluate", "model", "profile", "train")


def cell_source(command: list[str]) -> str:
    """The cell text for one handoff command."""
    script, *args = command
    if args:
        return SCRIPT_CELL_FORMATS["run"].format(script=script, args=" ".join(args))
    return SCRIPT_CELL_FORMATS["load"].format(script=script)


def script_cells(stage_name: str) -> list[str]:
    return [cell_source(command) for command in HANDOFF_COMMANDS.get(stage_name, [])]


def _purge_modules() -> None:
    for name in PURGED_MODULES:
        sys.modules.pop(name, None)


def _register_purge_hook() -> bool:
    """Forget the generated modules before every cell, so an edited script is re-read."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    shell.events.register("pre_run_cell", lambda *_args, **_kwargs: _purge_modules())
    return True
```

and change `start`:

```python
def start(
    project_name: str, drive_root: str = config.DRIVE_ROOT, llm: LLM | None = None
) -> Orchestrator:
    ctx = make_context(project_name, drive_root=drive_root, llm=llm)
    ctx.explainer.register_colab_callback()
    # The generated scripts read and write project-relative paths, and the user runs them
    # from their own cells, so the notebook's working directory must be the project.
    os.chdir(ctx.project.root)
    _register_purge_hook()
    return Orchestrator(
        ctx,
        [IntakeStage(), DataStage(), CleanStage(), CodegenStage(), TrainStage(), ReportStage()],
    )
```

- [ ] **Step 5: Rebuild the notebook**

Replace `scripts/build_notebook.py`:

```python
"""Generate notebooks/ML_Training_Agent.ipynb. Run: python scripts/build_notebook.py"""

from pathlib import Path

import nbformat as nbf

from mlagent.colab import script_cells

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "ML_Training_Agent.ipynb"


def code(*lines: str):
    return nbf.v4.new_code_cell("\n".join(lines))


def script_cell(stage: str, index: int):
    return code(script_cells(stage)[index])


cells = [
    nbf.v4.new_markdown_cell(
        "# ML Training Agent\n\n"
        "Copy this notebook once per project. Before running:\n\n"
        "1. Copy the `mlagent/` folder from this repo to `MyDrive/ml_agent/mlagent/`.\n"
        "2. Add your Anthropic key in Colab **Secrets** (key icon) as `ANTHROPIC_API_KEY` "
        "with notebook access on.\n"
        "3. Run the cells in order.\n\n"
        "**How it works.** Cells alternate. An *assistant cell* (`orch.run(...)`) asks you "
        "questions, writes real Python scripts into your project folder on Drive, and "
        "explains them. It then stops and names the *script cell* to run next. A script "
        "cell starts as `%load train.py`: run it once to pull the code into the cell so you "
        "can read (and edit) it, then run it again to execute it. When it finishes, go back "
        "to the assistant cell and run it again for the debrief.\n\n"
        "Click any highlighted term in the assistant's messages for an explanation. While a "
        "cell is waiting for you to type an answer, clicking does nothing — click after the "
        "cell finishes, or run `colab.explain('term')` in its own cell."
    ),
    code(
        "%pip -q install anthropic markdown pandas numpy scikit-learn matplotlib pyarrow "
        "openpyxl huggingface_hub datasets"
    ),
    code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "import sys",
        "sys.path.insert(0, '/content/drive/MyDrive/ml_agent')",
        "from mlagent import colab",
        "colab.setup()",
    ),
    code(
        "#@title 1. Project and interview  { display-mode: 'form' }",
        "PROJECT_NAME = 'my_first_project'  #@param {type:'string'}",
        "LEARNING_LEVEL = 'Intermediate - explain the key ideas'  #@param "
        "['Beginner - explain everything as we go', 'Intermediate - explain the key ideas', "
        "'Expert - just the numbers']",
        "GOAL = 'Predict which customers churn'  #@param {type:'string'}",
        "TASK = 'Tabular classification'  #@param ['Tabular classification', "
        "'Tabular regression']",
        "METRIC = 'accuracy'  #@param ['accuracy', 'f1', 'rmse', 'mae', 'r2']",
        "TARGET_VALUE = 0.9  #@param {type:'number'}",
        "DATA_SOURCE = 'Synthetic data'  #@param ['Synthetic data', "
        "'Upload or Google Drive path', 'HuggingFace Hub dataset']",
        "MINUTES_PER_RUN = 10  #@param {type:'integer'}",
        "MAX_ROUNDS = 5  #@param {type:'integer'}",
        "GPU = 'No GPU (CPU only)'  #@param ['No GPU (CPU only)', 'T4 GPU', "
        "'Any available GPU']",
        "",
        "orch = colab.start(PROJECT_NAME)",
        "orch.run(until='intake', answers={",
        "    'intake.goal': GOAL,",
        "    'intake.learning_level': LEARNING_LEVEL,",
        "    'intake.task_type': TASK,",
        "    'intake.metric': METRIC,",
        "    'intake.target_value': TARGET_VALUE,",
        "    'intake.data_source': DATA_SOURCE,",
        "    'intake.minutes_per_run': MINUTES_PER_RUN,",
        "    'intake.max_rounds': MAX_ROUNDS,",
        "    'intake.gpu': GPU,",
        "})",
    ),
    code(
        "#@title 2. Data  { display-mode: 'form' }",
        "N_ROWS = 1000  #@param {type:'integer'}",
        "N_FEATURES = 8  #@param {type:'integer'}",
        "N_CLASSES = 2  #@param {type:'integer'}",
        "CLASS_BALANCE = 0.5  #@param {type:'number'}",
        "NOISE = 0.1  #@param {type:'number'}",
        "INJECT_QUIRKS = True  #@param {type:'boolean'}",
        "DRIVE_PATH = ''  #@param {type:'string'}",
        "HF_QUERY = ''  #@param {type:'string'}",
        "TARGET_COLUMN = ''  #@param {type:'string'}",
        "",
        "orch.run(until='data', answers={",
        "    'data.n_rows': N_ROWS,",
        "    'data.n_features': N_FEATURES,",
        "    'data.n_classes': N_CLASSES,",
        "    'data.class_balance': CLASS_BALANCE,",
        "    'data.noise': NOISE,",
        "    'data.inject_quirks': INJECT_QUIRKS,",
        "    'data.drive_path': DRIVE_PATH,",
        "    'data.hf_query': HF_QUERY,",
        "    'data.target_column': TARGET_COLUMN,",
        "})",
    ),
    script_cell("data", 0),
    code(
        "#@title 3. Clean  { display-mode: 'form' }",
        "DROP_COLUMNS = ''  #@param {type:'string'}",
        "TRAIN_FRACTION = 0.7  #@param {type:'number'}",
        "VAL_FRACTION = 0.15  #@param {type:'number'}",
        "",
        "# Each proposed fix is confirmed one at a time in the output below.",
        "orch.run(until='clean', answers={",
        "    'clean.drop_columns': DROP_COLUMNS,",
        "    'clean.train_fraction': TRAIN_FRACTION,",
        "    'clean.val_fraction': VAL_FRACTION,",
        "})",
    ),
    script_cell("clean", 0),
    code(
        "#@title 4. Model  { display-mode: 'form' }",
        "MODEL = 'Ask me after the explanation'  #@param "
        "['Ask me after the explanation', 'Linear / logistic regression', 'Random forest', "
        "'Gradient boosting']",
        "",
        "orch.run(until='train', answers={'codegen.model_type': MODEL})",
    ),
    script_cell("train", 0),
    script_cell("train", 1),
    code(
        "#@title 5. Report",
        "# Asks before touching the held-out test set; answer in the box below.",
        "orch.run(until='report')",
    ),
    script_cell("report", 0),
    code("orch.run()"),
    nbf.v4.new_markdown_cell(
        "## Train again\n\n"
        "Edit `config.json` in the project folder, rerun the `train.py` and `evaluate.py` "
        "cells above, then run `orch.run()` to log the new run and rewrite the report. "
        "`orch.waiting()` says which cells the assistant is still waiting on; "
        "`orch.debrief('train')` forces the debrief if Drive's timestamps lag."
    ),
    nbf.v4.new_markdown_cell("### Ask about any term"),
    code("colab.explain('validation set')"),
    nbf.v4.new_markdown_cell(
        "### Redo a stage\nResets that stage and everything after it, then reruns."
    ),
    code("# orch.reset('intake'); orch.run()"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
```

Run: `python scripts/build_notebook.py`
Expected: `wrote .../notebooks/ML_Training_Agent.ipynb (18 cells)`.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_colab.py tests/test_plots.py tests/test_pipeline_e2e.py -q`
Expected: PASS. Remove the `xfail` marker from the end-to-end test (added in Task 9).

- [ ] **Step 7: Update `CLAUDE.md`**

In "What this is", replace the pipeline sentence's tail with:

```markdown
Pipeline: `intake -> data -> clean -> codegen -> train -> report`, each stage running in two phases with the user running the generated scripts in their own notebook cells. Design specs: `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md` and `docs/superpowers/specs/2026-09-08-milestone-4-learning-mode-design.md`. Plans: `docs/superpowers/plans/`.
```

In Architecture, replace the first three bullets with:

```markdown
- `mlagent/orchestrator.py` runs `stages/*` in order and checkpoints to `state.json` (`completed`, `current`, `forced`, `prepared`, `handoff`) so a Colab runtime reset resumes at the right phase. Each stage runs in two phases: `prepare(ctx)` interviews the user, writes scripts and explains them, and returns a `Handoff` (`stage`, `commands`, `outputs`) naming the cells the user must run; `debrief(ctx)` reads those outputs, shows figures with captions, narrates and completes the artifact. `orch.run(until=None, answers=None)` wraps the questioner in a `FormQuestioner` for that call; `orch.waiting()` returns the pending handoff; `orch.debrief(name)` forces a debrief when Drive's mtimes lag. A stage is complete when `Stage.is_complete` says its artifact exists and validates.
- Pipeline: `intake` -> `data` -> `clean` -> `codegen` -> `train` -> `report` (`mlagent/stages/`). `data` writes `data/raw/data.csv` and `data_meta.json`, copies `profile.py` and hands off `[["profile.py"]]` -> `profile_raw.json` plus four figures. `clean` audits, writes `audit.json` and a rendered `clean.py`, records `splits`/`split_seed`/`dropped_columns`, and hands off `[["clean.py"]]` -> `data/clean/data.csv` and `profile_clean.json`; its debrief completes `data_meta.json`. `codegen` shows `prompts/teaching/model_choices.md`, asks Claude for a `recommend_model`, lets the user pick one of `linear` / `random_forest` / `gradient_boosting`, copies `mlagent/templates/<family>/{data,model,train,evaluate}.py` and writes a flat `config.json` bounded by `templates_io.schema_for(schema, model_type)`. `train` hands off `[["train.py"], ["evaluate.py"]]` -> `metrics.json` and `eval_val.json`; its debrief logs the run by `metrics.json["started_at"]`, archives `run{N}` figures and checkpoints, and narrates. `report` hands off `[["evaluate.py", "--split", "test"]]` -> `eval_test.json` and writes `report.md`. Raw data is never modified; the test split is evaluated only by the report stage, once per best run.
- Generated scripts (`mlagent/templates/common/{profile,clean}.py`, `mlagent/templates/tabular_sklearn/{data,model,train,evaluate}.py`) import only numpy/pandas/scikit-learn/matplotlib/joblib, never `mlagent`; IPython is imported only inside `try/except`. Each has a `# --- settings ---` block, `# --- Title ---` section markers that `codewalk.split_sections` splits for the walkthrough, a `cli_argv()` that returns `[]` under ipykernel, and a `__main__` guard that never calls `sys.exit(0)`. `mlagent/plots.py` keeps only the palette, `style_axes`, `save_figure` and `present`; every user-facing figure is drawn by a template.
```

Add after the `data_meta.json` bullet:

```markdown
- Teaching: `Spec.learning_level` (`beginner` / `intermediate` / `expert`, asked once at intake) drives `mlagent/teaching.py`. `Teaching.preamble` (skipped at expert), `Teaching.walkthrough` (per-section at beginner, per-file at intermediate, titles only at expert) and `Teaching.debrief` (one `write_debrief` call returning a narrative plus per-figure notes) are the only places stages call the LLM for explanation. `prompts_io.load_prompt(name, **params)` formats only when params are given; every stage prompt ends with `Audience: {audience}` filled from `prompts/levels/<level>.md`. `mlagent/captions.py` holds the fixed "how to read this chart" text per figure kind; `ui/render.display_figure` shows a PNG with its caption.
```

Add to Commands:

```
python -m pytest tests/test_template_evaluate.py -v   # the generated scripts, run for real
```

- [ ] **Step 8: Add the smoke checklist**

Append the "Milestone 4" section from the "Smoke checklist addendum" below to `docs/colab-smoke.md`.

- [ ] **Step 9: Run everything**

Run: `python -m pytest -q && python -m ruff check . && python scripts/build_notebook.py`
Expected: all tests pass, ruff clean, the notebook regenerates. Also run the deprecation-warning gate that `CLAUDE.md` documents:
Run: `python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add mlagent/colab.py mlagent/plots.py scripts/build_notebook.py \
        notebooks/ML_Training_Agent.ipynb CLAUDE.md docs/colab-smoke.md tests
git rm tests/test_plots_training.py
git commit -m "feat: notebook-first Colab entry point, form-driven notebook, docs and e2e handoff loop

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016qyhLYsdaj4pmeMNRqGNtp"
```

---

## Smoke checklist addendum

Append to `docs/colab-smoke.md`:

```markdown
## Milestone 4 (Learning mode and notebook-first pipeline)

Run in a fresh Colab runtime, on a new project name, with `LEARNING_LEVEL` set to
"Beginner - explain everything as we go" unless an item says otherwise.

1. **The two-click script cell works.** After cell 4 (`2. Data`) the assistant stops and
   says which cell to run. Cell 5 shows `%load profile.py`; run it once — the cell body is
   replaced by the source of `profile.py` with `%load` commented out. Run it again: the
   profile prints, four figures appear inline, and `profile_raw.json` lands in the project
   folder on Drive. If `%load` misbehaves, change `SCRIPT_CELL_FORMATS["load"]` in
   `mlagent/colab.py` to the `%run` form and re-run `python scripts/build_notebook.py`.
2. **Form fields answer the fixed questions.** Fill cell 3's form (project name, level,
   goal, task, metric, target, source, minutes, rounds, GPU) and run it: no `input()` box
   appears for any of those, and `spec.json` on Drive contains exactly what you typed,
   including `learning_level`.
3. **A blank form field falls back cleanly.** Clear `GOAL` in cell 3, reset the project
   (`orch.reset('intake')`) and run it: a one-line note names the empty field and an
   `input()` box asks for the goal instead.
4. **The data debrief carries captions.** Re-run cell 4 after cell 5: the profile table,
   then each of the four figures with a caption beneath it, then the narrative. At beginner
   level the caption has a second sentence about *your* data; switch the project to expert
   level and confirm the extra sentence disappears.
5. **Cleaning happens in the user's cell.** Cell 6 asks about each proposed fix in the
   output area, then names cell 7. `%load clean.py` shows the operations and the embedded
   `STEPS` list. Run it: `data/clean/data.csv` appears, and the before/after missing-value
   figure is drawn in the cell. Edit one entry in `STEPS`, run the cell again, then re-run
   cell 6 — the debrief reflects the edited cleaning.
6. **Three models, one recommendation.** Cell 8 with `MODEL = "Ask me after the
   explanation"` prints the three model descriptions, names a recommendation with a reason
   about your dataset, and writes a `config.json` whose keys match that family only
   (no `learning_rate` for a random forest). Re-run with `MODEL = "Random forest"` after
   `orch.reset('codegen')` and confirm `config.json` changes accordingly.
7. **The code walkthrough is level-appropriate.** At beginner level cell 8 ends with each
   of `data.py`, `model.py`, `train.py`, `evaluate.py` broken into its sections with an
   explanation under each. At expert level it prints only the filenames and section titles.
8. **The training curve animates.** Cell 9 (`%load train.py`, run twice) redraws one
   loss/metric figure in place as epochs complete — the transcript above it is not wiped —
   and leaves `plots/training_curves.png` and `checkpoints/best.joblib` on Drive. Cell 10
   (`evaluate.py`) writes `eval_val.json` and the validation figures.
9. **Re-running the scripts logs a second run.** Edit `epochs` in `config.json`, run cells
   9 and 10 again, then run cell 8's `orch.run(until='train')` cell: `runs.jsonl` gains
   run 2, `plots/run2_training.png` exists, and the debrief compares the two.
10. **The test set is touched once, on purpose.** Cell 11 names the best run and asks
    before evaluating; answer "y", run cell 12 (`%run evaluate.py --split test`), then cell
    13. `eval_test.json` records `run_id` equal to the best run, `report.md` opens on Drive
    with every figure rendering, and re-running cell 11 says the test set was already
    evaluated instead of asking again.
11. **A runtime reset resumes at the right phase.** Runtime > Disconnect and delete
    runtime. Re-run cells 1-3 only, then run cell 11: the orchestrator picks up where it
    was (or reports nothing to do) without re-asking any earlier question. If a debrief
    insists the outputs are missing because Drive's timestamps lag, `orch.debrief('train')`
    forces it through.
```

## Self-review

Checked after writing, and fixed inline:

- **Spec coverage.** Decision 1 (notebook-first) -> Tasks 3, 6, 7, 11, 12, 14. Decision 2
  (every step is a script) -> Tasks 4, 5, 9. Decision 3 (three models) -> Tasks 8, 10.
  Decision 4 (learning level at intake) -> Task 2. Decision 5 (form fields) -> Tasks 1, 14.
  Decision 6 (captions) -> Tasks 4, 6, 13. Decision 8 (rulings: no in-process cleaning,
  templates duplicate code) -> Tasks 5, 7, and the Global Constraints. Architecture:
  `Handoff`/`ScriptStageBase`/state keys -> Task 3; notebook layout -> Task 14; the script
  table -> Tasks 4, 5, 9; model choice and `schema_for` -> Tasks 8, 10; learning level,
  captions, walkthrough -> Tasks 2, 4, 10, 13; questioner -> Task 1. Testing section ->
  `run_handoff`/`advance` in Task 14 plus the per-stage test rewrites in Tasks 6-13.
  Decision 7 (sequencing) and the Milestone 5/6 outlines are deliberately out of scope.
- **Placeholder scan.** No "TBD", no "similar to Task N", no "add validation". Every code
  step carries the code. The two places that say "unchanged" (`render_report`, `_fmt`,
  `check_data`, `meta_summary`, `config_table`, `_edit_config`, `_collect_steps`) name the
  exact existing functions being kept rather than describing new work.
- **Type consistency.** `Handoff(stage, commands, outputs)` and `outputs_ready(stage, ctx,
  handoff)` are used identically in Tasks 3, 6, 7, 11, 12 and 14. `build_run_entry(metrics,
  checkpoint)` has the Task 11 signature everywhere. `schema_for(schema, model_type)`
  returns the flat dict consumed by `default_config`/`validate_config`/`coerce_config` in
  Tasks 8 and 10. `Teaching.debrief(prompt_name, payload, figures, fallback)` matches every
  call site in Task 13. `caption_for(path)` takes a path in Tasks 4, 6, 7, 11, 12 and 13.
  `cell_source(command)` takes the same `list[str]` shape as `Handoff.commands`.
