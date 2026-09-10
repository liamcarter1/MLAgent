# Milestone 6a: Image Classification Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A beginner can pick "Image classification" at intake and run the whole existing pipeline (`intake -> data -> clean -> codegen -> train -> tune -> report`) on synthetic shapes, a Drive folder of their own images, or a HuggingFace image dataset, with the same two-phase stages, notebook cells, teaching, captions and tuning loop tabular tasks already have.

**Architecture:** A frozen `Modality` record in `mlagent/modality.py` names, per task type, which generate/load/audit/clean/write/read callables and which template files a stage should use; `DataStage`, `CleanStage` and `CodegenStage` look the record up with `modality_for(spec.task_type)` instead of importing the tabular modules directly. Images travel as an `ImageSet` (uint8 `(N, H, W, 3)` array + labels + class names + a manifest) stored as `data.npz` plus `manifest.csv`, mirroring the tabular `data.csv`. A new `mlagent/templates/image_torch/` family (PyTorch, `torchvision` lazily imported only by the ResNet builder) writes byte-for-byte the same `metrics.json` and `eval_{split}.json` contracts the tabular templates write, so `runs.py`, `diagnose.py`, `plots.py` and the tune and report stages work on image runs unchanged apart from one suffix-aware checkpoint copy.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, matplotlib (Agg in tests), Pillow (image synthesis and ingest), PyTorch (`torch`, and `torchvision` only for ResNet-18), pytest, ruff. Tests never hit the network: every LLM call takes an `LLM` and tests pass `FakeLLM`; the HuggingFace loader is always injected; every `resnet18` test run uses `pretrained=none`.

**Spec:** docs/superpowers/specs/2026-09-10-milestone-6a-image-tasks-design.md

## Global Constraints

- Pipeline order is unchanged: `intake -> data -> clean -> codegen -> train -> tune -> report`. No new stages and no new notebook cells; only the "2. Data" cell's form fields grow.
- `mlagent/colab.py`'s `HANDOFF_COMMANDS` is unchanged: image runs use the same cell names (`profile.py`, `clean.py`, `train.py`, `evaluate.py`) because the copied files always land under those destination names whatever the modality.
- Generated scripts (`mlagent/templates/**`) import only numpy / pandas / scikit-learn / matplotlib / joblib / torch (with `torchvision` imported lazily *inside* `model.py`'s ResNet builder function, never at module scope) and never `mlagent`; IPython is imported only inside `try/except`.
- Every generated script has a `# --- settings ---` block containing a `SCRIPT_NAME` constant, `# --- Title ---` section markers that `codewalk.split_sections` splits, a `cli_argv()` that returns `[]` unless `Path(sys.argv[0]).name.lower()` equals `SCRIPT_NAME.lower()`, and a `__main__` guard that never calls `sys.exit(0)`.
- Raw data is never modified. `data/raw/data.npz` + `data/raw/manifest.csv` are written once; `clean.py` writes `data/clean/data.npz` + `data/clean/manifest.csv`.
- Tabular `data_meta.json` is unchanged: it gains no `modality` key, and every reader defaults to tabular behaviour when `modality` is absent.
- The only fix op image audits propose is `drop_indices`, params `{"indices": [...], "reason": str}`.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere.
- Tests never hit the network: `FakeLLM` for Claude, `loader=` injection for HuggingFace, `pretrained=none` for every ResNet test run.
- Every subprocess test that runs an `image_torch` script sets `CUDA_VISIBLE_DEVICES="-1"` in the child environment (not `""`, which unsets the variable on Windows), so results never depend on the dev machine's GPU.
- `python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py tests/test_template_profile_images.py tests/test_template_evaluate_images.py` must stay clean.
- Prompts live in `mlagent/prompts/*.md` and are loaded with `prompts_io.load_prompt`; never inline a prompt in Python. Every stage prompt ends with `Audience: {audience}`.
- `ruff check .` clean (line-length 100); `python -m pytest` green.
- Commit messages end with exactly:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c
  ```

## Rulings made while planning (deviations from the spec's letter)

Ten rulings, below, resolve places where the spec's letter and the existing codebase disagreed.

1. **`TEMPLATE_FOR_TASK` stays a literal dict in `templates_io.py`**, gaining `"image_classification": "image_torch"`. The spec asks for it to be "a thin dict built from the registry", but `modality.py` must import `cleaning.py`, which imports `templates_io.py`, so building it from the registry would be a circular import. Instead `tests/test_modality.py` asserts the two agree: `TEMPLATE_FOR_TASK == {t: m.template_family for m in MODALITIES for t in m.task_types}`.
2. **The image `clean.py` writes `profile_clean.json` itself**, in the same `{before, after, steps, figures}` shape `mlagent/templates/common/clean.py` already writes (`mlagent/templates/common/clean.py:277-283`). The spec says the clean stage "reuses the same script" as the profile, but the tabular clean stage does not re-run `profile.py` either — its `clean.py` writes the before/after summary directly, and the handoff is a single `[["clean.py"]]` command that must not grow a second command (see the `HANDOFF_COMMANDS` constraint).
3. **A fifth caption kind, `clean_before_after_classes`**, is added alongside the spec's four (`thumbnails`, `intensity`, `class_means`, `misclassified`), because ruling 2's image `clean.py` draws a before/after class-count figure and every figure needs a caption. It is the image analogue of the existing `clean_before_after_missing`.
4. **`mlagent/templates/image_common/profile.py` does not import PIL.** The spec lists PIL among its imports, but the script reads pixels straight out of the `.npz`, so numpy + pandas + matplotlib suffice — and PIL is not on the allowed import list in the Global Constraints. PIL is used only by `mlagent/synth/images.py` and the two image data sources, which are agent-side modules, not generated scripts.
5. **`training_curves.png` is drawn by `image_torch/train.py`, not by `evaluate.py`**, exactly as `tabular_sklearn/train.py:154` does today. The spec's `evaluate.py` section lists it among the evaluate figures; keeping it in `train.py` is what makes `runs.CURVES_FIGURE` archiving work identically for both families.
6. **The `eval_{split}.json` record keeps tabular's exact field names** — `y_pred` (the spec calls it `predictions`) and `y_proba` alongside `y_true` — so `mlagent/stages/report.py` and `evaluate.py`'s `best_checkpoint()` need no branching.
7. **`Modality` gains two fields the spec's sketch omits**: `teaching_material: str` (`"model_choices"` vs `"model_choices_images"`, needed by codegen's `material()` call at `mlagent/stages/codegen.py:199`) and `meta_keys: Callable[..., dict]` is *not* added — the modality-specific `data_meta.json` keys are produced inside the stages, where the questioner's answers already are.
8. **`archive_run` gains a third parameter** rather than reading `metrics.json` itself: `archive_run(project, run_id, checkpoint=None)`. `log_finished_run` already has the metrics dict in scope (`mlagent/runs.py:129`) and passes `metrics.get("checkpoint")`. `BEST_CHECKPOINT = "best.joblib"` stays as the fallback for legacy runs whose `metrics.json` predates the key.
9. **`tabular_sklearn/train.py` also starts writing `metrics["checkpoint"]`** (`"checkpoints/best.joblib"`), so both families feed ruling 8's suffix-aware copy from the same place.
10. **`render_report` gains a "## Data" section** reading `data_meta.json`. The spec says to mention `image_size` and `n_images` "where it currently prints tabular row/column counts from `data_meta.json`" — but `render_report` (`mlagent/stages/report.py:31`) never reads `data_meta.json` today, so the section is new rather than edited.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | `pillow` in core deps; new `images` extra (`torch`, `torchvision`); both added to `dev` |
| `mlagent/modality.py` (new) | `Modality` frozen dataclass, `TABULAR`/`IMAGE` records, `MODALITIES`, `modality_for` |
| `mlagent/imageset.py` (new) | `ImageSet`, `write_pair`, `read_pair`, `prepare_image`, `image_hash`, `duplicate_groups`, `blank_indices`, `stratified_indices` |
| `mlagent/synth/images.py` (new) | `SynthImageConfig`, `generate(cfg) -> ImageSet` — shapes with injected quirks |
| `mlagent/datasources/drive_images.py` (new) | `load_folder(path, image_size, max_images=None) -> (ImageSet, skipped)` |
| `mlagent/datasources/hf_images.py` (new) | `load_image_dataset(...) -> ImageSet`, lazy `datasets` import, injectable `loader` |
| `mlagent/audit_images.py` (new) | `audit_images(imageset, meta, skipped=()) -> list[Issue]`, `drop_indices` fixes only |
| `mlagent/cleaning_images.py` (new) | `apply_steps`, `describe_step`, `render_clean_py` for image steps |
| `mlagent/templates/image_common/profile.py` (new) | Generated image profile: 4 figures + `profile_{tag}.json` |
| `mlagent/templates/image_common/clean.py` (new) | Generated image cleaner: drops indices, writes the clean pair + `profile_clean.json` |
| `mlagent/templates/image_torch/config_schema.json` (new) | Common + per-model tunable keys for the three CNN families |
| `mlagent/templates/image_torch/data.py` (new) | Load the clean pair, stratified split, normalise, augment, `DataLoader`s, `pick_device` |
| `mlagent/templates/image_torch/model.py` (new) | `TinyCNN`, `SmallCNN`, lazy-`torchvision` `resnet18`, `build_model` |
| `mlagent/templates/image_torch/train.py` (new) | Epoch loop, live curve, `metrics.json` (+ `device`, `checkpoint`), `checkpoints/best.pt` |
| `mlagent/templates/image_torch/evaluate.py` (new) | `eval_{split}.json`, confusion / per-class / misclassified figures |
| `mlagent/templates_io.py` | `shared_file`, `copy_shared`, `IMAGE_COMMON_DIRNAME`; `TEMPLATE_FOR_TASK` gains the image task |
| `mlagent/captions.py` | 5 new kinds: `thumbnails`, `intensity`, `class_means`, `misclassified`, `clean_before_after_classes` |
| `mlagent/stages/data.py` | Registry-driven; image source questions; image `data_meta.json` keys |
| `mlagent/stages/clean.py` | Registry-driven audit/clean/render; image `data_meta.json` completion; empty-class error |
| `mlagent/stages/codegen.py` | Family-aware model labels, recommend tool, `check_data`, teaching material |
| `mlagent/stages/report.py` | New "## Data" section in `render_report` |
| `mlagent/runs.py` | Suffix-aware `archive_run(project, run_id, checkpoint=None)` |
| `mlagent/diagnose.py` | Presence-checked `heuristic_proposals` + image moves |
| `mlagent/prompts/teaching/model_choices_images.md` (new) | Tiny CNN vs small CNN vs transfer learning, per level |
| `scripts/build_notebook.py`, `notebooks/ML_Training_Agent.ipynb` | New "2. Data" fields, pip line, TASK choice, intro cell text |
| `docs/colab-smoke.md`, `CLAUDE.md` | Milestone 6a checklist; architecture notes |
| tests | `test_modality.py`, `test_imageset.py`, `test_synth_images.py`, `test_datasources_images.py`, `test_audit_images.py`, `test_cleaning_images.py`, `test_template_profile_images.py`, `test_template_model_images.py`, `test_template_evaluate_images.py`, `test_pipeline_e2e_images.py` (new); `test_captions.py`, `test_templates_io.py`, `test_runs.py`, `test_diagnose.py`, `test_codegen_stage.py`, `test_data_stage.py`, `test_clean_stage.py`, `test_report_stage.py`, `test_colab.py`, `conftest.py` (modified) |

---

### Task 1: Dependencies and the modality registry

**Files:**
- Create: `mlagent/modality.py`, `tests/test_modality.py`
- Modify: `pyproject.toml:6-21`, `mlagent/templates_io.py:14-24`, `mlagent/templates_io.py:168-184`, `mlagent/stages/data.py:10-22`, `mlagent/stages/data.py:62-89`, `mlagent/stages/clean.py:10-11`, `mlagent/stages/clean.py:50-63`, `mlagent/stages/codegen.py:17-28`, `mlagent/stages/codegen.py:135-149`
- Test: `tests/test_modality.py`, `tests/test_templates_io.py`

**Interfaces:**
- Consumes: `synth.tabular.generate` (`mlagent/synth/tabular.py`), `datasources.drive.load_table` (`mlagent/datasources/drive.py:39`), `datasources.hf.load_tabular` (`mlagent/datasources/hf.py:48`), `audit.audit_tabular` (`mlagent/audit.py:331`), `cleaning.apply_steps` (`mlagent/cleaning.py:89`), `cleaning.render_clean_py` (`mlagent/cleaning.py:127`).
- Produces (used by Tasks 5, 6, 7, 10):
  - `@dataclass(frozen=True) Modality` with fields `name, task_types, data_file, profile_template, clean_template, template_family, teaching_material, profile_figures, generate, load_drive, load_hf, audit, apply_steps, render_clean_py, write_raw, read`
  - `TABULAR: Modality`, `MODALITIES: tuple[Modality, ...]`, `modality_for(task_type: str) -> Modality`
  - `templates_io.shared_file(relpath: str) -> Path`, `templates_io.copy_shared(relpath: str, project_root: Path, name: str | None = None) -> Path`, `templates_io.IMAGE_COMMON_DIRNAME = "image_common"`
  - `templates_io.TEMPLATE_FOR_TASK["image_classification"] == "image_torch"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_modality.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from mlagent import audit, cleaning
from mlagent.datasources import drive, hf
from mlagent.modality import MODALITIES, TABULAR, Modality, modality_for
from mlagent.synth import tabular
from mlagent.templates_io import TEMPLATE_FOR_TASK


def test_lookup_by_task_type_returns_the_tabular_record():
    for task in ("tabular_classification", "tabular_regression"):
        m = modality_for(task)
        assert m is TABULAR
        assert m.name == "tabular"
        assert m.data_file == "data.csv"
        assert m.template_family == "tabular_sklearn"
        assert m.profile_template == "common/profile.py"
        assert m.clean_template == "common/clean.py"
        assert m.teaching_material == "model_choices"


def test_unknown_task_type_names_itself_and_the_valid_ones():
    with pytest.raises(ValueError) as exc:
        modality_for("audio_classification")
    message = str(exc.value)
    assert "audio_classification" in message
    assert "tabular_classification" in message


def test_the_tabular_record_wraps_the_existing_functions_unchanged():
    assert TABULAR.generate is tabular.generate
    assert TABULAR.load_drive is drive.load_table
    assert TABULAR.load_hf is hf.load_tabular
    assert TABULAR.audit is audit.audit_tabular
    assert TABULAR.apply_steps is cleaning.apply_steps
    assert TABULAR.render_clean_py is cleaning.render_clean_py


def test_the_tabular_record_writes_and_reads_a_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    TABULAR.write_raw(df, tmp_path)
    assert (tmp_path / "data.csv").exists()
    pd.testing.assert_frame_equal(TABULAR.read(tmp_path), df)


def test_template_for_task_agrees_with_the_registry():
    from_registry = {t: m.template_family for m in MODALITIES for t in m.task_types}
    assert TEMPLATE_FOR_TASK == from_registry


def test_the_record_is_frozen():
    with pytest.raises(Exception):
        TABULAR.name = "other"
    assert isinstance(TABULAR, Modality)
```

Append to `tests/test_templates_io.py`:

```python
def test_shared_file_and_copy_shared_round_trip(tmp_path):
    from mlagent.templates_io import copy_shared, shared_file

    assert shared_file("common/profile.py").is_file()
    written = copy_shared("common/profile.py", tmp_path)
    assert written == tmp_path / "profile.py"
    assert "SCRIPT_NAME" in written.read_text(encoding="utf-8")


def test_copy_shared_can_rename_the_destination(tmp_path):
    from mlagent.templates_io import copy_shared

    written = copy_shared("common/profile.py", tmp_path, name="profile.py")
    assert written.name == "profile.py"


def test_shared_file_rejects_a_missing_path():
    import pytest

    from mlagent.templates_io import shared_file

    with pytest.raises(FileNotFoundError):
        shared_file("common/nope.py")
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_modality.py tests/test_templates_io.py -q
```
Expected: `ModuleNotFoundError: No module named 'mlagent.modality'` collecting `tests/test_modality.py`, and `ImportError: cannot import name 'shared_file' from 'mlagent.templates_io'` in the three new `test_templates_io.py` tests.

- [ ] **Step 3: Write minimal implementation**

Edit `pyproject.toml`: add `"pillow>=10.0",` after `"openpyxl>=3.1",` in `dependencies`, and replace the `[project.optional-dependencies]` block (line 20-21) with:

```toml
[project.optional-dependencies]
images = ["torch>=2.2", "torchvision>=0.17"]
dev = ["pytest>=8.0", "ruff>=0.5", "nbformat>=5.9", "torch>=2.2"]
```

`torchvision` stays only in the `images` extra; `torch` stays in both `images` and `dev`
so tabular-only dev environments can still import it, but pulling in torchvision (and its
heavier native wheel matrix) is opt-in via `images`.

In `mlagent/templates_io.py`, replace lines 14-24 with:

```python
TEMPLATE_FOR_TASK = {
    "tabular_classification": "tabular_sklearn",
    "tabular_regression": "tabular_sklearn",
    "image_classification": "image_torch",
}

COMMON_DIRNAME = "common"
IMAGE_COMMON_DIRNAME = "image_common"
COMMON_DIR = TEMPLATES_DIR / COMMON_DIRNAME
# Scripts copied verbatim into a project via copy_common(). `clean.py` also lives under
# COMMON_DIR but is never copied as-is: cleaning.render_clean_py() reads it and
# substitutes the approved steps into its STEPS_JSON line before writing it out.
COMMON_FILES = ("profile.py",)
```

and replace `common_file` / `copy_common` (lines 168-184) with:

```python
def shared_file(relpath: str) -> Path:
    """A shared template script by its path under `mlagent/templates`.

    e.g. `"common/profile.py"`, `"image_common/clean.py"` -- what a `Modality` record's
    `profile_template` and `clean_template` fields name.
    """
    path = TEMPLATES_DIR / relpath
    if not path.is_file():
        raise FileNotFoundError(f"no shared template at {relpath!r} under {TEMPLATES_DIR}")
    return path


def copy_shared(relpath: str, project_root: Path, name: str | None = None) -> Path:
    """Copy one shared script into the project, overwriting; return the written path.

    The destination name defaults to the source's own filename, so both
    `common/profile.py` and `image_common/profile.py` land as `profile.py`.
    """
    source = shared_file(relpath)
    target = Path(project_root) / (name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def common_file(name: str) -> Path:
    """Path to a shared template script under `common/` (`profile.py`, `clean.py`)."""
    return shared_file(f"{COMMON_DIRNAME}/{name}")


def copy_common(names: Sequence[str], project_root: Path) -> list[Path]:
    """Copy shared scripts into the project folder, overwriting; return the paths."""
    return [copy_shared(f"{COMMON_DIRNAME}/{name}", project_root) for name in names]
```

Create `mlagent/modality.py`:

```python
"""What a stage needs to know to treat one kind of data uniformly.

`DataStage`, `CleanStage` and `CodegenStage` never import `synth.tabular`,
`datasources.drive`, `datasources.hf`, `audit` or `cleaning` directly: they look up a
`Modality` record by task type and call its callables. Adding a modality is adding a
record here plus the modules it points at, never an `if task_type == ...` branch
scattered through the stages.

`TEMPLATE_FOR_TASK` stays a literal dict in `templates_io.py` (this module imports
`cleaning`, which imports `templates_io`, so building it here would be circular);
`tests/test_modality.py` asserts the two agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlagent import audit, cleaning
from mlagent.datasources import drive, hf
from mlagent.synth import tabular

TABULAR_RAW_FILE = "data.csv"


@dataclass(frozen=True)
class Modality:
    name: str                          # "tabular" | "image"
    task_types: tuple[str, ...]
    data_file: str                     # "data.csv" | "data.npz"
    profile_template: str              # source path under mlagent/templates
    clean_template: str
    template_family: str               # "tabular_sklearn" | "image_torch"
    teaching_material: str             # prompts/teaching/<name>.md shown before the model pick
    profile_figures: tuple[str, ...]   # ordered figure kinds the profile draws
    generate: Callable[..., object]
    load_drive: Callable[..., object]
    load_hf: Callable[..., object]
    audit: Callable[..., object]
    apply_steps: Callable[..., object]
    render_clean_py: Callable[..., str]
    write_raw: Callable[..., object]
    read: Callable[[Path], object]


def _write_table(df: pd.DataFrame, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABULAR_RAW_FILE
    df.to_csv(path, index=False)
    return path


def _read_table(directory: Path) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / TABULAR_RAW_FILE)


TABULAR = Modality(
    name="tabular",
    task_types=("tabular_classification", "tabular_regression"),
    data_file=TABULAR_RAW_FILE,
    profile_template="common/profile.py",
    clean_template="common/clean.py",
    template_family="tabular_sklearn",
    teaching_material="model_choices",
    profile_figures=("histograms", "missing", "class_balance", "correlation"),
    generate=tabular.generate,
    load_drive=drive.load_table,
    load_hf=hf.load_tabular,
    audit=audit.audit_tabular,
    apply_steps=cleaning.apply_steps,
    render_clean_py=cleaning.render_clean_py,
    write_raw=_write_table,
    read=_read_table,
)

MODALITIES: tuple[Modality, ...] = (TABULAR,)


def modality_for(task_type: str) -> Modality:
    """The record for a task type; the only lookup function."""
    for modality in MODALITIES:
        if task_type in modality.task_types:
            return modality
    valid = sorted(t for m in MODALITIES for t in m.task_types)
    raise ValueError(f"unknown task type {task_type!r}; expected one of {valid}")
```

Now wire the three stages to the registry, leaving tabular behaviour identical.

In `mlagent/stages/data.py`, replace the import block (lines 10-15) with:

```python
from mlagent.datasources.drive import list_candidates
from mlagent.datasources.hf import search_datasets
from mlagent.modality import modality_for
from mlagent.profile import profile_markdown
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate
from mlagent.templates_io import copy_shared
```

`generate` stays in this import list because `_synthetic` (data.py:145) still calls the
bare tabular `generate(cfg)` directly; it is not yet routed through the modality record.
Task 7 changes `_synthetic` to call `modality.generate(cfg)` instead, at which point this
import can drop back out.

and change `DataStage.__init__` (line 47) to take the loaders from the registry by default:

```python
    def __init__(self, search_roots=None, hf_search=search_datasets, hf_load=None):
        self.search_roots = (
            list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        )
        self.hf_search = hf_search
        # None means "whatever the task type's modality record says"; tests inject a fake.
        self.hf_load = hf_load
```

Replace `prepare`'s first lines (62-74) with:

```python
    def prepare(self, ctx: StageContext) -> Handoff:
        spec = ctx.spec()
        modality = modality_for(spec.task_type)
        ctx.teaching().preamble("data", {"spec": spec.to_dict()})
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx, modality)
        else:
            df, target, meta = self._huggingface(ctx, modality)
```

replace `copy_common(COMMON_FILES, ctx.project.root)` (line 89) with:

```python
        copy_shared(modality.profile_template, ctx.project.root)
```

and change the two source helpers to use the record:

```python
    def _drive(self, ctx: StageContext, modality):
        # mlagent/stages/data.py:164-178 unchanged: list_candidates, the choice/text
        # question, `path = Path(answer.strip())` and the is_file() check.
        df = modality.load_drive(path)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "drive", "source_path": str(path)}
```
```python
    def _huggingface(self, ctx: StageContext, modality):
        # mlagent/stages/data.py:185-200 unchanged: the three search attempts, the
        # "no datasets found" guard, the labels list, the choice and `chosen`.
        loader = self.hf_load or modality.load_hf
        df = loader(chosen.id)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "huggingface", "hf_id": chosen.id}
```

In `mlagent/stages/clean.py`, replace lines 10-11 with:

```python
from mlagent.audit import Issue
from mlagent.cleaning import describe_step
from mlagent.modality import modality_for
```

and in `prepare`, after `before = profile_dataframe(df, target)` (line 47), insert the lookup, then use it at lines 50 and 63:

```python
        modality = modality_for(str(meta.get("task_type") or ctx.spec().task_type))
```
```python
        issues = modality.audit(df, target)
```
```python
        (ctx.project.root / CLEAN_PY).write_text(
            modality.render_clean_py(steps), encoding="utf-8"
        )
```

In `mlagent/stages/codegen.py`, drop `TEMPLATE_FOR_TASK` from the `templates_io` import list (line 19), add `from mlagent.modality import modality_for` after the `mlagent.llm` import, replace line 135 with `schema = load_schema(modality_for(spec.task_type).template_family)`, and replace lines 141-149 with:

```python
    def prepare(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        template = modality_for(spec.task_type).template_family
```

`is_complete`'s existing `except Exception` (line 137) already turns the new `ValueError` from an unknown task type into "not complete", so it needs no change.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pip install -e ".[dev,images]"
python -m pytest tests/test_modality.py tests/test_templates_io.py -q
python -m pytest -q
ruff check .
```
Run `python -m pip install -e ".[dev,images]"` first. If pip cannot resolve `torchvision`
on this Python version (e.g. no wheel for Python 3.14 yet), fall back to
`python -m pip install -e ".[dev]"` instead and record in the task report which of the two
installs actually succeeded. Expected: all green, ruff clean. The install pulls `pillow`
and `torch` unconditionally, plus `torchvision` when the `images` extra resolves; on this
dev machine `torch` 2.9.1 is already importable. Every `resnet18` test (plan ~3676, 3688,
4241 and any others — grep the plan for `resnet18`) starts with
`pytest.importorskip("torchvision")` so the suite still passes when only `[dev]` installed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml mlagent/modality.py mlagent/templates_io.py mlagent/stages/data.py mlagent/stages/clean.py mlagent/stages/codegen.py tests/test_modality.py tests/test_templates_io.py
git commit -m "feat: modality registry, shared-template copying and the image dependencies

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 2: `mlagent/imageset.py` -- the in-memory and on-disk image contract

**Files:**
- Create: `mlagent/imageset.py`, `tests/test_imageset.py`
- Test: `tests/test_imageset.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces (used by Tasks 3, 4, 6, 7, 13):
  - `@dataclass ImageSet(images: np.ndarray, labels: np.ndarray, class_names: list[str], manifest: pd.DataFrame)` with properties `n_images`, `image_size`, `n_channels`; methods `class_counts() -> dict[str, int]`, `validate() -> None`, `take(indices) -> ImageSet`
  - `RAW_FILE = "data.npz"`, `MANIFEST_FILE = "manifest.csv"`, `MANIFEST_COLUMNS`, `BLANK_STD_THRESHOLD = 3.0`, `IMAGE_SIZES = (32, 64, 128)`, `DEFAULT_IMAGE_SIZE = 64`
  - `write_pair(imageset: ImageSet, directory: Path) -> tuple[Path, Path]`
  - `read_pair(directory: Path) -> ImageSet`
  - `prepare_image(pil_image, image_size: int) -> np.ndarray`
  - `image_hash(image: np.ndarray) -> str`
  - `duplicate_groups(images: np.ndarray) -> list[list[int]]`
  - `blank_indices(images: np.ndarray, threshold: float = BLANK_STD_THRESHOLD) -> list[int]`
  - `stratified_indices(labels, max_images: int | None, seed: int = 0) -> list[int]`
  - `make_manifest(labels, class_names: list[str], sources, sizes) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Create `tests/test_imageset.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mlagent.imageset import (
    MANIFEST_COLUMNS,
    ImageSet,
    blank_indices,
    duplicate_groups,
    image_hash,
    make_manifest,
    prepare_image,
    read_pair,
    stratified_indices,
    write_pair,
)


def tiny_set(n: int = 6, size: int = 8) -> ImageSet:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(n, size, size, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)
    class_names = ["circle", "square"]
    manifest = make_manifest(
        labels, class_names,
        sources=[f"synthetic#{i}" for i in range(n)],
        sizes=[(size * 2, size * 3)] * n,
    )
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


def test_properties_and_class_counts():
    s = tiny_set(n=6, size=8)
    assert s.n_images == 6
    assert s.image_size == 8
    assert s.n_channels == 3
    assert s.class_counts() == {"circle": 3, "square": 3}


def test_manifest_columns_and_source_sizes():
    s = tiny_set(n=4)
    assert list(s.manifest.columns) == list(MANIFEST_COLUMNS)
    assert s.manifest["index"].tolist() == [0, 1, 2, 3]
    assert s.manifest["class_name"].tolist() == ["circle", "square", "circle", "square"]
    assert s.manifest["source_width"].tolist() == [16, 16, 16, 16]
    assert s.manifest["source_height"].tolist() == [24, 24, 24, 24]


def test_write_pair_then_read_pair_round_trips(tmp_path):
    s = tiny_set(n=5)
    npz_path, manifest_path = write_pair(s, tmp_path)
    assert npz_path.name == "data.npz" and manifest_path.name == "manifest.csv"
    back = read_pair(tmp_path)
    assert np.array_equal(back.images, s.images)
    assert np.array_equal(back.labels, s.labels)
    assert back.class_names == s.class_names
    assert back.manifest["source"].tolist() == s.manifest["source"].tolist()


def test_prepare_image_makes_a_square_rgb_array_of_the_asked_size():
    src = Image.new("L", (40, 20), color=128)
    out = prepare_image(src, 16)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8
    assert int(out.min()) == int(out.max()) == 128


def test_prepare_image_centre_crops_the_long_side():
    src = Image.new("RGB", (30, 10), color=(0, 0, 0))
    for x in range(10, 20):
        for y in range(10):
            src.putpixel((x, y), (255, 255, 255))
    out = prepare_image(src, 10)
    assert out.mean() > 250


def test_image_hash_and_duplicate_groups():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.ones((4, 4, 3), dtype=np.uint8)
    assert image_hash(a) == image_hash(a.copy())
    assert image_hash(a) != image_hash(b)
    images = np.stack([a, b, a, b, a])
    assert duplicate_groups(images) == [[0, 2, 4], [1, 3]]


def test_duplicate_groups_is_empty_when_every_image_differs():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 255, size=(5, 6, 6, 3), dtype=np.uint8)
    assert duplicate_groups(images) == []


def test_blank_indices_finds_near_constant_images():
    rng = np.random.default_rng(2)
    noisy = rng.integers(0, 255, size=(3, 8, 8, 3), dtype=np.uint8)
    flat = np.full((1, 8, 8, 3), 200, dtype=np.uint8)
    images = np.concatenate([noisy, flat])
    assert blank_indices(images) == [3]


def test_stratified_indices_caps_and_keeps_every_class():
    labels = np.array([0] * 10 + [1] * 4)
    picked = stratified_indices(labels, max_images=6, seed=0)
    assert len(picked) == 6
    assert set(labels[picked].tolist()) == {0, 1}
    assert picked == sorted(picked)


def test_stratified_indices_returns_everything_when_uncapped():
    labels = np.array([0, 1, 0, 1])
    assert stratified_indices(labels, max_images=None) == [0, 1, 2, 3]
    assert stratified_indices(labels, max_images=99) == [0, 1, 2, 3]


def test_take_keeps_the_chosen_rows_and_reindexes_the_manifest():
    s = tiny_set(n=6)
    kept = s.take([1, 3, 5])
    assert kept.n_images == 3
    assert kept.manifest["index"].tolist() == [0, 1, 2]
    assert kept.manifest["source"].tolist() == ["synthetic#1", "synthetic#3", "synthetic#5"]
    assert np.array_equal(kept.images, s.images[[1, 3, 5]])


def test_validate_rejects_a_mismatched_label_count():
    rng = np.random.default_rng(3)
    images = rng.integers(0, 255, size=(3, 4, 4, 3), dtype=np.uint8)
    labels = np.array([0, 1])
    manifest = make_manifest(labels, ["a", "b"], ["x", "y"], [(4, 4), (4, 4)])
    with pytest.raises(ValueError, match="labels"):
        ImageSet(images=images, labels=labels, class_names=["a", "b"],
                 manifest=manifest).validate()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_imageset.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'mlagent.imageset'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/imageset.py`:

```python
"""Images in memory and on disk.

The tabular side of the pipeline passes a `pandas.DataFrame` and stores `data.csv`; the
image side passes an `ImageSet` and stores `data.npz` plus `manifest.csv` in the same
folder. Every source (synthetic, Drive, HuggingFace) normalises to the same shape: RGB
uint8 `(N, H, W, 3)` with `H == W == image_size`, integer labels indexing a sorted
`class_names` list, and one manifest row per image recording where it came from and how
big it was *before* resizing.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = "data.npz"
MANIFEST_FILE = "manifest.csv"
MANIFEST_COLUMNS = (
    "index", "label", "class_name", "source", "source_width", "source_height",
)
# Per-image pixel standard deviation (0-255 scale) below which an image counts as blank.
BLANK_STD_THRESHOLD = 3.0
IMAGE_SIZES = (32, 64, 128)
DEFAULT_IMAGE_SIZE = 64


@dataclass
class ImageSet:
    images: np.ndarray       # uint8, shape (N, H, W, 3)
    labels: np.ndarray       # int, shape (N,)
    class_names: list[str]   # sorted; labels index into this list
    manifest: pd.DataFrame

    @property
    def n_images(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.images.shape[1]) if self.images.ndim == 4 else 0

    @property
    def n_channels(self) -> int:
        return int(self.images.shape[3]) if self.images.ndim == 4 else 0

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for label in np.asarray(self.labels).tolist():
            if 0 <= int(label) < len(self.class_names):
                counts[self.class_names[int(label)]] += 1
        return counts

    def validate(self) -> None:
        """Raise ValueError if the arrays and the manifest do not line up."""
        if self.images.ndim != 4 or self.images.shape[3] != 3:
            raise ValueError(f"images must be (N, H, W, 3); got {self.images.shape}")
        if self.images.shape[1] != self.images.shape[2]:
            raise ValueError("images must be square; run prepare_image on every source")
        if len(self.labels) != self.n_images:
            raise ValueError(f"{len(self.labels)} labels for {self.n_images} images")
        if len(self.manifest) != self.n_images:
            raise ValueError(f"{len(self.manifest)} manifest rows for {self.n_images} images")
        if not self.class_names:
            raise ValueError("class_names is empty")

    def take(self, indices) -> ImageSet:
        """A new ImageSet keeping only `indices`, in order, with the manifest reindexed."""
        idx = [int(i) for i in indices]
        manifest = self.manifest.iloc[idx].copy().reset_index(drop=True)
        manifest["index"] = range(len(manifest))
        return ImageSet(
            images=self.images[idx],
            labels=np.asarray(self.labels)[idx],
            class_names=list(self.class_names),
            manifest=manifest,
        )


def make_manifest(labels, class_names: list[str], sources, sizes) -> pd.DataFrame:
    """One row per image: where it came from and its pre-resize width and height."""
    values = [int(v) for v in np.asarray(labels).tolist()]
    pairs = [(int(w), int(h)) for w, h in sizes]
    return pd.DataFrame(
        {
            "index": list(range(len(values))),
            "label": values,
            "class_name": [class_names[v] for v in values],
            "source": list(sources),
            "source_width": [w for w, _h in pairs],
            "source_height": [h for _w, h in pairs],
        },
        columns=list(MANIFEST_COLUMNS),
    )


def write_pair(imageset: ImageSet, directory: Path) -> tuple[Path, Path]:
    """Write `data.npz` (images, labels, class_names) and `manifest.csv` into `directory`."""
    imageset.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / RAW_FILE
    np.savez_compressed(
        npz_path,
        images=np.ascontiguousarray(imageset.images, dtype=np.uint8),
        labels=np.asarray(imageset.labels, dtype=np.int64),
        class_names=np.asarray(imageset.class_names, dtype="U"),
    )
    manifest_path = directory / MANIFEST_FILE
    imageset.manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    return npz_path, manifest_path


def read_pair(directory: Path) -> ImageSet:
    """Read back what `write_pair` wrote."""
    directory = Path(directory)
    with np.load(directory / RAW_FILE) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]
    manifest = pd.read_csv(directory / MANIFEST_FILE)
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


def prepare_image(pil_image, image_size: int) -> np.ndarray:
    """RGB, centre-cropped to a square, resized to `image_size`, as uint8 (H, W, 3)."""
    from PIL import Image

    img = pil_image.convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((int(image_size), int(image_size)), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def image_hash(image: np.ndarray) -> str:
    """A content hash of one image's raw bytes; equal hashes are exact duplicates."""
    return hashlib.sha1(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest()


def duplicate_groups(images: np.ndarray) -> list[list[int]]:
    """Groups of indices sharing an exact content hash, largest group first, then by first
    index. Images with no duplicate are not listed."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, image in enumerate(np.asarray(images)):
        buckets[image_hash(image)].append(i)
    groups = [sorted(idx) for idx in buckets.values() if len(idx) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def blank_indices(images: np.ndarray, threshold: float = BLANK_STD_THRESHOLD) -> list[int]:
    """Indices of near-constant images: per-image pixel standard deviation below
    `threshold` on the 0-255 scale."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.size == 0:
        return []
    stds = arr.reshape(arr.shape[0], -1).std(axis=1)
    return [int(i) for i in np.flatnonzero(stds < float(threshold)).tolist()]


def stratified_indices(labels, max_images: int | None, seed: int = 0) -> list[int]:
    """At most `max_images` indices, spread evenly over the classes, sorted ascending.

    Every class present keeps at least one image while the budget allows. `None` (or a cap
    at or above the number of images) keeps everything.
    """
    values = np.asarray(labels)
    n = int(values.shape[0])
    if max_images is None or int(max_images) >= n:
        return list(range(n))
    budget = max(1, int(max_images))
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(values.tolist()):
        by_class[int(label)].append(i)
    order = sorted(by_class)
    for key in order:
        rng.shuffle(by_class[key])
    picked: list[int] = []
    while len(picked) < budget and any(by_class[k] for k in order):
        for key in order:
            if len(picked) >= budget:
                break
            if by_class[key]:
                picked.append(by_class[key].pop())
    return sorted(picked)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_imageset.py -q && ruff check mlagent/imageset.py
```
Expected: 12 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/imageset.py tests/test_imageset.py
git commit -m "feat: ImageSet, the data.npz/manifest.csv pair and the duplicate/blank helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 3: `mlagent/synth/images.py` -- synthetic shapes with injected quirks

**Files:**
- Create: `mlagent/synth/images.py`, `tests/test_synth_images.py`
- Test: `tests/test_synth_images.py`

**Interfaces:**
- Consumes: `ImageSet`, `make_manifest`, `duplicate_groups`, `blank_indices` from Task 2 (`mlagent/imageset.py`).
- Produces (used by Tasks 7, 13):
  - `SHAPES = ("circle", "square", "triangle", "star", "cross")`
  - `@dataclass SynthImageConfig(n_images=300, image_size=64, n_classes=3, seed=42, noise=0.1, class_imbalance=0.0, duplicate_fraction=0.0, blank_fraction=0.0)` with `validate() -> None`
  - `generate(cfg: SynthImageConfig) -> ImageSet`

- [ ] **Step 1: Write the failing test**

Create `tests/test_synth_images.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from mlagent.imageset import blank_indices, duplicate_groups
from mlagent.synth.images import SHAPES, SynthImageConfig, generate


def test_generate_returns_a_valid_imageset_of_the_asked_shape():
    cfg = SynthImageConfig(n_images=40, image_size=32, n_classes=3, seed=7)
    s = generate(cfg)
    s.validate()
    assert s.n_images == 40
    assert s.images.shape == (40, 32, 32, 3)
    assert s.images.dtype == np.uint8
    assert s.class_names == sorted(SHAPES[:3])
    assert set(np.unique(s.labels).tolist()) <= set(range(3))


def test_the_manifest_records_the_source_size_before_resizing():
    s = generate(SynthImageConfig(n_images=12, image_size=32, n_classes=2, seed=1))
    assert s.manifest["source"].str.startswith("synthetic#").all()
    assert (s.manifest["source_width"] == 32).all()
    assert (s.manifest["source_height"] == 32).all()


def test_generation_is_deterministic_given_a_seed():
    cfg = SynthImageConfig(n_images=20, image_size=32, n_classes=3, seed=99)
    a, b = generate(cfg), generate(cfg)
    assert np.array_equal(a.images, b.images)
    assert np.array_equal(a.labels, b.labels)
    other = generate(SynthImageConfig(n_images=20, image_size=32, n_classes=3, seed=100))
    assert not np.array_equal(a.images, other.images)


def test_the_shapes_are_actually_different_between_classes():
    s = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=3, noise=0.0))
    means = [s.images[s.labels == k].mean() for k in range(3)]
    assert len({round(float(m), 1) for m in means}) >= 2


def test_injected_duplicates_are_findable_by_the_audit_helper():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=5,
                                  duplicate_fraction=0.25))
    groups = duplicate_groups(s.images)
    assert sum(len(g) - 1 for g in groups) >= 8


def test_injected_blanks_are_findable_by_the_audit_helper():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=5,
                                  blank_fraction=0.2, noise=0.0))
    assert len(blank_indices(s.images)) >= 6


def test_class_imbalance_makes_the_first_class_dominate():
    balanced = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=2))
    counts = list(balanced.class_counts().values())
    assert max(counts) - min(counts) <= 1

    skewed = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=2,
                                       class_imbalance=0.7))
    top = max(skewed.class_counts().values())
    assert top / skewed.n_images >= 0.6


def test_no_quirks_means_no_duplicates_and_no_blanks():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=3, seed=11))
    assert duplicate_groups(s.images) == []
    assert blank_indices(s.images) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_images": 3},
        {"n_classes": 1},
        {"n_classes": 9},
        {"image_size": 17},
        {"noise": 2.0},
        {"duplicate_fraction": 1.5},
    ],
)
def test_validate_rejects_impossible_configs(kwargs):
    with pytest.raises(ValueError):
        generate(SynthImageConfig(**kwargs))
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_synth_images.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'mlagent.synth.images'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/synth/images.py`:

```python
"""Synthetic shape images with optional realistic quirks for the audit stage to find.

The image counterpart of `synth/tabular.py`: five drawable shapes on a noisy coloured
background, with random size, position, rotation and colour, deterministic given a seed.
`class_imbalance`, `duplicate_fraction` and `blank_fraction` are the injected quirks that
give `audit_images` something to report, exactly as `SynthTabularConfig.quirks` does for
tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mlagent.imageset import IMAGE_SIZES, ImageSet, make_manifest

SHAPES = ("circle", "square", "triangle", "star", "cross")
MIN_IMAGES = 10
MAX_CLASSES = len(SHAPES)


@dataclass
class SynthImageConfig:
    n_images: int = 300
    image_size: int = 64
    n_classes: int = 3
    seed: int = 42
    noise: float = 0.1               # background speckle, 0 = flat, 1 = very noisy
    class_imbalance: float = 0.0     # 0 = balanced; 0.7 = the first class takes ~70%
    duplicate_fraction: float = 0.0  # share of images replaced by a copy of an earlier one
    blank_fraction: float = 0.0      # share of images replaced by a near-constant image

    def validate(self) -> None:
        if self.n_images < MIN_IMAGES:
            raise ValueError(f"n_images must be at least {MIN_IMAGES}")
        if not 2 <= self.n_classes <= MAX_CLASSES:
            raise ValueError(f"n_classes must be between 2 and {MAX_CLASSES}")
        if self.image_size not in IMAGE_SIZES:
            raise ValueError(f"image_size must be one of {IMAGE_SIZES}")
        for name in ("noise", "class_imbalance", "duplicate_fraction", "blank_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.duplicate_fraction + self.blank_fraction > 0.6:
            raise ValueError("duplicate_fraction + blank_fraction must not exceed 0.6")


def _class_shares(n_classes: int, imbalance: float) -> np.ndarray:
    """Share of images per class: uniform at imbalance 0, first class dominant at 1."""
    uniform = np.full(n_classes, 1.0 / n_classes)
    if imbalance <= 0:
        return uniform
    dominant = np.full(n_classes, (1.0 - imbalance) / max(1, n_classes - 1))
    dominant[0] = imbalance
    return uniform * (1 - imbalance) + dominant * imbalance


def _label_plan(cfg: SynthImageConfig, rng: np.random.Generator) -> np.ndarray:
    shares = _class_shares(cfg.n_classes, float(cfg.class_imbalance))
    counts = np.maximum(1, np.floor(shares * cfg.n_images).astype(int))
    while counts.sum() < cfg.n_images:
        counts[int(np.argmax(shares))] += 1
    while counts.sum() > cfg.n_images:
        counts[int(np.argmax(counts))] -= 1
    labels = np.repeat(np.arange(cfg.n_classes), counts)
    rng.shuffle(labels)
    return labels.astype(np.int64)


def _polygon(shape: str, cx: float, cy: float, radius: float, angle: float):
    """Vertices for one shape, rotated `angle` radians about its centre."""
    if shape == "square":
        base = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    elif shape == "triangle":
        base = [
            (math.cos(math.pi / 2 + k * 2 * math.pi / 3),
             math.sin(math.pi / 2 + k * 2 * math.pi / 3))
            for k in range(3)
        ]
    elif shape == "star":
        base = []
        for k in range(10):
            r = 1.0 if k % 2 == 0 else 0.45
            theta = math.pi / 2 + k * math.pi / 5
            base.append((r * math.cos(theta), r * math.sin(theta)))
    elif shape == "cross":
        t = 0.34
        base = [
            (-t, -1), (t, -1), (t, -t), (1, -t), (1, t), (t, t),
            (t, 1), (-t, 1), (-t, t), (-1, t), (-1, -t), (-t, -t),
        ]
    else:
        raise ValueError(f"{shape!r} has no polygon; draw it as an ellipse")
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (cx + radius * (x * cos_a - y * sin_a), cy + radius * (x * sin_a + y * cos_a))
        for x, y in base
    ]


def _draw_one(shape: str, size: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    from PIL import Image, ImageDraw

    background = tuple(int(v) for v in rng.integers(200, 256, size=3))
    colour = tuple(int(v) for v in rng.integers(0, 160, size=3))
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    radius = float(rng.uniform(0.22, 0.38)) * size
    cx = float(rng.uniform(radius, size - radius))
    cy = float(rng.uniform(radius, size - radius))
    if shape == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
    else:
        angle = float(rng.uniform(0, 2 * math.pi))
        draw.polygon(_polygon(shape, cx, cy, radius, angle), fill=colour)
    arr = np.asarray(img, dtype=np.int16)
    if noise > 0:
        speckle = rng.normal(0.0, 60.0 * float(noise), size=arr.shape)
        arr = arr + speckle
    return np.clip(arr, 0, 255).astype(np.uint8)


def generate(cfg: SynthImageConfig) -> ImageSet:
    """Draw `cfg.n_images` shape images, then inject the configured quirks."""
    cfg.validate()
    rng = np.random.default_rng(int(cfg.seed))
    class_names = sorted(SHAPES[: cfg.n_classes])
    labels = _label_plan(cfg, rng)
    size = int(cfg.image_size)
    images = np.stack(
        [_draw_one(class_names[int(k)], size, float(cfg.noise), rng) for k in labels]
    )

    n = int(cfg.n_images)
    n_blank = int(round(float(cfg.blank_fraction) * n))
    n_dup = int(round(float(cfg.duplicate_fraction) * n))
    quirk_targets = rng.permutation(n)[: n_blank + n_dup]
    for i in quirk_targets[:n_blank]:
        level = int(rng.integers(40, 220))
        images[int(i)] = np.full((size, size, 3), level, dtype=np.uint8)
    for i in quirk_targets[n_blank:]:
        source = int(rng.integers(0, n))
        if source == int(i):
            source = (source + 1) % n
        images[int(i)] = images[source]
        labels[int(i)] = labels[source]

    manifest = make_manifest(
        labels,
        class_names,
        sources=[f"synthetic#{i}" for i in range(n)],
        sizes=[(size, size)] * n,
    )
    imageset = ImageSet(images=images, labels=labels, class_names=class_names,
                        manifest=manifest)
    imageset.validate()
    return imageset
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_synth_images.py -q && ruff check mlagent/synth/images.py
```
Expected: 14 passed (9 named tests plus the 6 parametrised cases minus overlap -- pytest reports 14), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/synth/images.py tests/test_synth_images.py
git commit -m "feat: synthetic shape images with injected duplicate, blank and imbalance quirks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 4: Image data sources -- a Drive folder and a HuggingFace dataset

**Files:**
- Create: `mlagent/datasources/drive_images.py`, `mlagent/datasources/hf_images.py`, `tests/test_datasources_images.py`
- Test: `tests/test_datasources_images.py`

**Interfaces:**
- Consumes: `ImageSet`, `make_manifest`, `prepare_image`, `stratified_indices` from Task 2 (`mlagent/imageset.py`).
- Produces (used by Task 7):
  - `drive_images.IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")`
  - `drive_images.load_folder(path: Path | str, image_size: int, max_images: int | None = None) -> tuple[ImageSet, list[tuple[str, str]]]`
  - `hf_images.load_image_dataset(dataset_id: str, image_size: int, split: str = "train", max_images: int | None = None, image_column: str | None = None, label_column: str | None = None, loader=None) -> ImageSet`

- [ ] **Step 1: Write the failing test**

Create `tests/test_datasources_images.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mlagent.datasources.drive_images import load_folder
from mlagent.datasources.hf_images import load_image_dataset


def write_image(path, size=(24, 18), colour=(10, 200, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def build_folder(root):
    for i in range(3):
        write_image(root / "cats" / f"c{i}.png", colour=(200, 10, 10))
    for i in range(2):
        write_image(root / "dogs" / f"d{i}.JPG", colour=(10, 10, 200))
    return root


def test_load_folder_reads_class_subfolders_case_insensitively(tmp_path):
    imageset, skipped = load_folder(build_folder(tmp_path), image_size=32)
    imageset.validate()
    assert imageset.n_images == 5
    assert imageset.class_names == ["cats", "dogs"]
    assert imageset.class_counts() == {"cats": 3, "dogs": 2}
    assert imageset.images.shape == (5, 32, 32, 3)
    assert skipped == []


def test_the_manifest_keeps_the_pre_resize_size_and_the_file_path(tmp_path):
    imageset, _skipped = load_folder(build_folder(tmp_path), image_size=32)
    assert (imageset.manifest["source_width"] == 24).all()
    assert (imageset.manifest["source_height"] == 18).all()
    assert imageset.manifest["source"].str.endswith((".png", ".JPG")).all()


def test_unreadable_and_root_level_files_are_skipped_not_raised(tmp_path):
    root = build_folder(tmp_path)
    (root / "cats" / "broken.png").write_bytes(b"not an image at all")
    write_image(root / "stray.png")
    (root / "notes.txt").write_text("hello", encoding="utf-8")

    imageset, skipped = load_folder(root, image_size=32)
    assert imageset.n_images == 5
    reasons = {name: reason for name, reason in skipped}
    assert any(k.endswith("broken.png") for k in reasons)
    assert any(k.endswith("stray.png") and v == "no class folder" for k, v in reasons.items())
    assert not any(k.endswith("notes.txt") for k in reasons)


def test_max_images_caps_with_a_stratified_sample(tmp_path):
    imageset, _skipped = load_folder(build_folder(tmp_path), image_size=32, max_images=4)
    assert imageset.n_images == 4
    assert set(imageset.class_counts()) == {"cats", "dogs"}
    assert min(imageset.class_counts().values()) >= 1


def test_an_empty_folder_raises_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no readable images"):
        load_folder(tmp_path, image_size=32)


class FakeClassLabel:
    def __init__(self, names):
        self.names = list(names)


class FakeImageFeature:
    pass


class FakeDataset:
    """Just enough of a `datasets.Dataset` for the loader: features, len, indexing."""

    def __init__(self, rows, features):
        self.rows = list(rows)
        self.features = dict(features)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def fake_hf_dataset(n=6):
    rows = []
    for i in range(n):
        rows.append({
            "image": Image.new("RGB", (40, 30), (i * 20 % 256, 100, 50)),
            "label": i % 2,
        })
    return FakeDataset(rows, {"image": FakeImageFeature(),
                              "label": FakeClassLabel(["cat", "dog"])})


def test_load_image_dataset_uses_the_injected_loader_and_the_class_label_names():
    calls = []

    def loader(dataset_id, split=None, **kwargs):
        calls.append((dataset_id, split, kwargs))
        return fake_hf_dataset()

    imageset = load_image_dataset("acme/pets", image_size=16, loader=loader)
    imageset.validate()
    assert calls == [("acme/pets", "train", {})]
    assert imageset.class_names == ["cat", "dog"]
    assert imageset.n_images == 6
    assert imageset.images.shape == (6, 16, 16, 3)
    assert imageset.manifest["source"].tolist()[0] == "hf:acme/pets/train#0"


def test_load_image_dataset_caps_with_a_stratified_sample():
    imageset = load_image_dataset(
        "acme/pets", image_size=16, max_images=4, loader=lambda *a, **k: fake_hf_dataset(10)
    )
    assert imageset.n_images == 4
    assert set(np.unique(imageset.labels).tolist()) == {0, 1}


def test_explicit_column_names_win_over_detection():
    rows = [{"pic": Image.new("RGB", (8, 8)), "kind": "a"} for _ in range(4)]
    ds = FakeDataset(rows, {"pic": FakeImageFeature(), "kind": FakeClassLabel(["a"])})
    imageset = load_image_dataset(
        "x/y", image_size=8, image_column="pic", label_column="kind",
        loader=lambda *a, **k: ds,
    )
    assert imageset.class_names == ["a"]
    assert imageset.n_images == 4


def test_a_dataset_with_no_image_column_raises():
    ds = FakeDataset([{"text": "hi", "label": 0}], {"text": object(),
                                                    "label": FakeClassLabel(["x"])})
    with pytest.raises(ValueError, match="image column"):
        load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_datasources_images.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'mlagent.datasources.drive_images'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/datasources/drive_images.py`:

```python
"""Load a folder of images laid out as `<class>/<file>` (Drive, or any directory).

Nothing here ever raises on a bad file: unreadable files, non-image files and images
dropped straight into the root with no class folder are collected in `skipped` as
`(path, reason)` pairs so the audit can report them to the user in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mlagent.imageset import ImageSet, make_manifest, prepare_image, stratified_indices

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
ROOT_LEVEL_REASON = "no class folder"


def _candidate_files(root: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """(class_name, path) pairs plus the root-level strays, both sorted by path."""
    found: list[tuple[str, Path]] = []
    skipped: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        here = Path(dirpath)
        for name in sorted(filenames):
            path = here / name
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:  # pragma: no cover - os.walk always stays under root
                continue
            if len(relative.parts) < 2:
                skipped.append((str(path), ROOT_LEVEL_REASON))
                continue
            found.append((relative.parts[0], path))
    return found, skipped


def load_folder(
    path: Path | str, image_size: int, max_images: int | None = None
) -> tuple[ImageSet, list[tuple[str, str]]]:
    """Read `<class>/<file>` images into an ImageSet, plus the files that could not be read."""
    from PIL import Image

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"no such folder: {root}")
    candidates, skipped = _candidate_files(root)

    arrays: list[np.ndarray] = []
    class_of: list[str] = []
    sources: list[str] = []
    sizes: list[tuple[int, int]] = []
    for class_name, file_path in candidates:
        try:
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                arrays.append(prepare_image(img, image_size))
        except Exception as exc:  # noqa: BLE001 - any unreadable file is reported, never fatal
            skipped.append((str(file_path), f"{type(exc).__name__}: {exc}"))
            continue
        class_of.append(class_name)
        sources.append(str(file_path))
        sizes.append((width, height))

    if not arrays:
        raise ValueError(f"no readable images under {root} (looked for {list(IMAGE_EXTS)})")

    class_names = sorted(set(class_of))
    index = {name: i for i, name in enumerate(class_names)}
    labels = np.array([index[name] for name in class_of], dtype=np.int64)
    imageset = ImageSet(
        images=np.stack(arrays),
        labels=labels,
        class_names=class_names,
        manifest=make_manifest(labels, class_names, sources, sizes),
    )
    keep = stratified_indices(imageset.labels, max_images)
    if len(keep) != imageset.n_images:
        imageset = imageset.take(keep)
    imageset.validate()
    return imageset, skipped
```

Create `mlagent/datasources/hf_images.py`:

```python
"""Load an image classification dataset from the HuggingFace Hub.

The `datasets` import is lazy (as `datasources/hf.py` does for tables) so importing
`mlagent` never requires it installed, and `loader` is injectable so tests never hit the
network.
"""

from __future__ import annotations

import numpy as np

from mlagent.imageset import ImageSet, make_manifest, prepare_image, stratified_indices

IMAGE_COLUMN_NAMES = ("image", "img", "picture")
LABEL_COLUMN_NAMES = ("label", "labels", "fine_label", "class")


def _detect_image_column(features: dict, given: str | None) -> str:
    if given:
        return given
    for name, feature in features.items():
        if type(feature).__name__ == "Image":
            return name
    for name in IMAGE_COLUMN_NAMES:
        if name in features:
            return name
    raise ValueError(
        f"could not find an image column in {sorted(features)}; pass image_column="
    )


def _detect_label_column(features: dict, given: str | None) -> str:
    if given:
        return given
    for name, feature in features.items():
        if hasattr(feature, "names"):
            return name
    for name in LABEL_COLUMN_NAMES:
        if name in features:
            return name
    raise ValueError(
        f"could not find a label column in {sorted(features)}; pass label_column="
    )


def _class_names(features: dict, label_column: str, raw_labels: list) -> list[str]:
    feature = features.get(label_column)
    names = getattr(feature, "names", None)
    if names:
        return [str(n) for n in names]
    return sorted({str(v) for v in raw_labels})


def load_image_dataset(
    dataset_id: str,
    image_size: int,
    split: str = "train",
    max_images: int | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
    loader=None,
) -> ImageSet:
    """One split of a HuggingFace image dataset, normalised to an ImageSet."""
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    ds = loader(dataset_id, split=split)
    if hasattr(ds, "keys") and not hasattr(ds, "features"):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]

    features = dict(getattr(ds, "features", {}) or {})
    image_key = _detect_image_column(features, image_column)
    label_key = _detect_label_column(features, label_column)

    rows = [ds[i] for i in range(len(ds))]
    raw_labels = [row[label_key] for row in rows]
    class_names = _class_names(features, label_key, raw_labels)
    index = {name: i for i, name in enumerate(class_names)}

    arrays: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    sizes: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        image = row[image_key]
        width, height = getattr(image, "size", (image_size, image_size))
        arrays.append(prepare_image(image, image_size))
        value = raw_labels[i]
        labels.append(int(value) if isinstance(value, int) else index[str(value)])
        sources.append(f"hf:{dataset_id}/{split}#{i}")
        sizes.append((int(width), int(height)))

    if not arrays:
        raise ValueError(f"{dataset_id} split {split!r} has no rows")

    label_array = np.array(labels, dtype=np.int64)
    imageset = ImageSet(
        images=np.stack(arrays),
        labels=label_array,
        class_names=class_names,
        manifest=make_manifest(label_array, class_names, sources, sizes),
    )
    keep = stratified_indices(imageset.labels, max_images)
    if len(keep) != imageset.n_images:
        imageset = imageset.take(keep)
    imageset.validate()
    return imageset
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_datasources_images.py -q && ruff check mlagent/datasources/
```
Expected: 9 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/datasources/drive_images.py mlagent/datasources/hf_images.py tests/test_datasources_images.py
git commit -m "feat: image sources - a Drive class-folder walker and an injectable HuggingFace loader

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 5: The image profile template and its four captions

**Files:**
- Create: `mlagent/templates/image_common/profile.py`, `tests/test_template_profile_images.py`
- Modify: `mlagent/captions.py:11-81`
- Test: `tests/test_template_profile_images.py`, `tests/test_captions.py`

**Interfaces:**
- Consumes: `imageset.write_pair` (Task 2), `synth.images.generate` / `SynthImageConfig` (Task 3), `templates_io.copy_shared` (Task 1).
- Produces (used by Tasks 6, 7, 9, 13):
  - `captions.CAPTIONS` gains `"thumbnails"`, `"intensity"`, `"class_means"`, `"misclassified"`, `"clean_before_after_classes"`
  - `mlagent/templates/image_common/profile.py` with `SCRIPT_NAME = "profile.py"`, `CAPTIONS`, `cli_argv()`, `main(argv)`, writing `profile_{tag}.json` with keys `n_images, image_size, n_channels, n_classes, class_counts, duplicate_images, blank_images, channel_mean, channel_std, source_width, source_height, figures`
  - figures `{tag}_thumbnails.png`, `{tag}_class_balance.png`, `{tag}_intensity.png`, `{tag}_class_means.png`

- [ ] **Step 1: Write the failing test**

Create `tests/test_template_profile_images.py`:

```python
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from mlagent import captions
from mlagent.imageset import write_pair
from mlagent.synth.images import SynthImageConfig, generate
from mlagent.templates_io import copy_shared

TEMPLATE = Path("mlagent/templates/image_common").resolve()
IMAGE_PROFILE = "image_common/profile.py"
KINDS = ("thumbnails", "class_balance", "intensity", "class_means")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_image_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_image_{name}"] = module
    spec.loader.exec_module(module)
    return module


def image_project(project, n_images=30, image_size=32, n_classes=3, duplicate_fraction=0.2,
                  blank_fraction=0.1):
    imageset = generate(SynthImageConfig(
        n_images=n_images, image_size=image_size, n_classes=n_classes, seed=4,
        duplicate_fraction=duplicate_fraction, blank_fraction=blank_fraction, noise=0.05,
    ))
    write_pair(imageset, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label",
        "task_type": "image_classification",
        "modality": "image",
        "source": "synthetic",
        "raw_path": "data/raw/data.npz",
        "raw_n_rows": imageset.n_images,
        "raw_n_cols": image_size * image_size * 3,
        "image_size": image_size,
        "n_channels": 3,
    })
    copy_shared(IMAGE_PROFILE, project.root)
    return project, imageset


def run_profile(project, args=()):
    result = subprocess.run(
        [sys.executable, "profile.py", *args],
        cwd=str(project.root), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_the_copied_script_never_imports_mlagent(project):
    copy_shared(IMAGE_PROFILE, project.root)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5


def test_profile_writes_every_key_and_all_four_figures(project):
    project, imageset = image_project(project)
    run_profile(project)
    written = json.loads((project.root / "profile_raw.json").read_text(encoding="utf-8"))

    assert written["n_images"] == imageset.n_images
    assert written["image_size"] == 32
    assert written["n_channels"] == 3
    assert written["n_classes"] == 3
    assert sum(written["class_counts"].values()) == imageset.n_images
    assert written["duplicate_images"] >= 1
    assert written["blank_images"] >= 1
    assert len(written["channel_mean"]) == 3 and len(written["channel_std"]) == 3
    for key in ("source_width", "source_height"):
        assert set(written[key]) == {"min", "median", "max"}
    assert written["figures"] == [
        "raw_thumbnails.png", "raw_class_balance.png", "raw_intensity.png",
        "raw_class_means.png",
    ]
    for name in written["figures"]:
        assert (project.plots_dir / name).exists()


def test_profile_tag_and_input_flags_write_the_clean_profile(project):
    project, imageset = image_project(project)
    write_pair(imageset, project.data_clean)
    run_profile(project, ["--input", "data/clean/data.npz", "--tag", "clean"])
    written = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert written["figures"][0] == "clean_thumbnails.png"
    assert (project.plots_dir / "clean_class_means.png").exists()


def test_profile_prints_a_caption_under_each_figure(project):
    project, _imageset = image_project(project)
    result = run_profile(project)
    for kind in KINDS:
        stripped = captions.CAPTIONS[kind].replace("[[", "").replace("]]", "")
        assert result.stdout.count("How to read this: " + stripped) == 1


def test_template_captions_match_mlagent_captions():
    module = load_module("profile")
    assert module.CAPTIONS == {k: captions.CAPTIONS[k] for k in KINDS}


def test_cli_argv_ignores_kernel_launchers_but_parses_script_argv(monkeypatch):
    module = load_module("profile")
    monkeypatch.setattr(sys, "argv", ["/x/colab_kernel_launcher.py", "-f", "k.json"])
    assert module.cli_argv() == []
    monkeypatch.setattr(sys, "argv", ["profile.py", "--tag", "clean"])
    assert module.cli_argv() == ["--tag", "clean"]
```

Append to `tests/test_captions.py`:

```python
def test_the_image_caption_kinds_exist_and_read_as_guidance():
    from mlagent.captions import CAPTIONS

    for kind in ("thumbnails", "intensity", "class_means", "misclassified",
                 "clean_before_after_classes"):
        assert kind in CAPTIONS
        assert len(CAPTIONS[kind]) > 80


def test_caption_for_matches_the_new_image_figure_names():
    from mlagent.captions import CAPTIONS, caption_for

    assert caption_for("raw_thumbnails.png") == CAPTIONS["thumbnails"]
    assert caption_for("clean_class_means.png") == CAPTIONS["class_means"]
    assert caption_for("val_misclassified.png") == CAPTIONS["misclassified"]
    assert caption_for("raw_intensity.png") == CAPTIONS["intensity"]
    assert caption_for("clean_before_after_classes.png") == (
        CAPTIONS["clean_before_after_classes"]
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_template_profile_images.py tests/test_captions.py -q
```
Expected: `FileNotFoundError: no shared template at 'image_common/profile.py'` in every `test_template_profile_images.py` test, and `KeyError: 'thumbnails'` / `assert 'thumbnails' in CAPTIONS` in the two new caption tests.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/captions.py`, insert these five entries into `CAPTIONS` -- `thumbnails`, `intensity`, `class_means` after `"class_balance"` (line 26), and `misclassified` after `"per_class"` (line 59), and `clean_before_after_classes` after `"clean_before_after_missing"` (line 40):

```python
    "thumbnails": (
        "A few example images from each class, labelled with the class they belong to. "
        "Check the pictures really show what the label says, and that the size you chose "
        "still leaves the thing you care about visible."
    ),
    "intensity": (
        "How bright the red, green and blue channels are across the whole dataset, from 0 "
        "(black) to 255 (white). A spike hard against one end means many images are blown "
        "out or nearly black; three curves lying on top of each other means the pictures "
        "are effectively greyscale."
    ),
    "class_means": (
        "The average of every image in one class. A mean image that still shows a clear "
        "shape means that class looks consistent, so it is easy to learn; a formless blur "
        "means the object moves around the frame, which is harder but more realistic."
    ),
    "misclassified": (
        "The validation images the model got most confidently wrong, each labelled "
        "true -> predicted. The same confusion repeated is a fixable labelling or "
        "[[class imbalance]] problem; a scatter of unrelated one-offs is just noise."
    ),
    "clean_before_after_classes": (
        "How many images each class has before and after cleaning. Bars that shrink lost "
        "duplicate or blank images; a bar that disappears means the class has no images "
        "left, which the assistant treats as an error rather than a smaller dataset."
    ),
```

Create `mlagent/templates/image_common/profile.py`:

```python
"""Profile an image dataset and draw its four overview figures. Generated by mlagent;
safe to edit.

Run it from the project folder:

    python profile.py                                      profile the raw images
    python profile.py --input data/clean/data.npz --tag clean

Reads the `data.npz` / `manifest.csv` pair, writes `profile_{tag}.json` and PNGs into
`plots/`. Imports only numpy, pandas and matplotlib, so you can run it anywhere the
project folder exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- settings ---
SCRIPT_NAME = "profile.py"
PROJECT_DIR = Path(".")
META_FILE = "data_meta.json"
MANIFEST_FILE = "manifest.csv"
TAG = "raw"
INPUT = None  # None means "the raw .npz named in data_meta.json"
BLANK_STD_THRESHOLD = 3.0   # matches mlagent/imageset.py
THUMBS_PER_CLASS = 4
MAX_THUMB_CLASSES = 10
INTENSITY_BINS = 32

# --- captions ---
# Byte-identical to the matching entries in mlagent/captions.py
# (tests/test_template_profile_images.py checks this); kept here too since this script
# never depends on the mlagent package.
CAPTIONS = {
    "thumbnails": (
        "A few example images from each class, labelled with the class they belong to. "
        "Check the pictures really show what the label says, and that the size you chose "
        "still leaves the thing you care about visible."
    ),
    "class_balance": (
        "How many rows carry each label. A large gap between the bars is [[class imbalance]]: "
        "a model can score well just by always predicting the biggest class, so accuracy alone "
        "will flatter it."
    ),
    "intensity": (
        "How bright the red, green and blue channels are across the whole dataset, from 0 "
        "(black) to 255 (white). A spike hard against one end means many images are blown "
        "out or nearly black; three curves lying on top of each other means the pictures "
        "are effectively greyscale."
    ),
    "class_means": (
        "The average of every image in one class. A mean image that still shows a clear "
        "shape means that class looks consistent, so it is easy to learn; a formless blur "
        "means the object moves around the frame, which is harder but more realistic."
    ),
}

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)


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


def load_pair(project_dir: Path, input_path: str | None, meta: dict):
    """(images, labels, class_names, manifest, path) from the .npz plus its manifest.csv."""
    relative = input_path or meta.get("raw_path") or "data/raw/data.npz"
    path = project_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"no image file at {path}")
    with np.load(path) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]
    manifest_path = path.parent / MANIFEST_FILE
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    return images, labels, class_names, manifest, path


# --- describing the images ---
def count_duplicates(images: np.ndarray) -> int:
    """How many images are an exact copy of an earlier one (by content hash)."""
    seen: dict[str, int] = defaultdict(int)
    for image in images:
        seen[hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()] += 1
    return int(sum(count - 1 for count in seen.values() if count > 1))


def count_blanks(images: np.ndarray) -> int:
    if images.size == 0:
        return 0
    stds = images.reshape(images.shape[0], -1).astype(np.float32).std(axis=1)
    return int((stds < BLANK_STD_THRESHOLD).sum())


def size_stats(series: pd.Series) -> dict:
    if series.empty:
        return {"min": None, "median": None, "max": None}
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(values.min()),
        "median": int(round(float(values.median()))),
        "max": int(values.max()),
    }


def profile_images(images, labels, class_names, manifest) -> dict:
    counts = {name: 0 for name in class_names}
    for label in labels.tolist():
        if 0 <= int(label) < len(class_names):
            counts[class_names[int(label)]] += 1
    flat = images.reshape(-1, images.shape[3]).astype(np.float32) if images.size else None
    return {
        "n_images": int(images.shape[0]),
        "image_size": int(images.shape[1]) if images.ndim == 4 else 0,
        "n_channels": int(images.shape[3]) if images.ndim == 4 else 0,
        "n_classes": len(class_names),
        "class_counts": counts,
        "duplicate_images": count_duplicates(images),
        "blank_images": count_blanks(images),
        "channel_mean": [round(float(v), 4) for v in flat.mean(axis=0)] if flat is not None
        else [],
        "channel_std": [round(float(v), 4) for v in flat.std(axis=0)] if flat is not None
        else [],
        "source_width": size_stats(manifest.get("source_width", pd.Series(dtype=float))),
        "source_height": size_stats(manifest.get("source_height", pd.Series(dtype=float))),
    }


# --- figures ---
def show(fig, kind: str) -> None:
    """Display in a notebook if one is running, then print how to read the figure."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            display(fig)
    caption = CAPTIONS.get(kind, "")
    if caption:
        print("How to read this: " + caption.replace("[[", "").replace("]]", ""))


def save(fig, plots_dir: Path, name: str, kind: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig, kind)
    plt.close(fig)
    return f"{name}.png"


def thumbnails(images, labels, class_names):
    shown = class_names[:MAX_THUMB_CLASSES]
    nrows = max(1, len(shown))
    fig, axes = plt.subplots(nrows, THUMBS_PER_CLASS,
                             figsize=(1.6 * THUMBS_PER_CLASS, 1.7 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for row, name in enumerate(shown):
        picks = np.flatnonzero(labels == class_names.index(name))[:THUMBS_PER_CLASS]
        for col in range(THUMBS_PER_CLASS):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_color(AXIS)
            if col < len(picks):
                ax.imshow(images[int(picks[col])])
            else:
                ax.set_facecolor(SURFACE)
            if col == 0:
                ax.set_ylabel(name, color=INK_2, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle("Example images per class", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def class_balance(counts: dict):
    fig = plt.figure(figsize=(5, 0.5 * max(3, len(counts)) + 1.2), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    labels = [str(k) for k in counts]
    values = [int(v) for v in counts.values()]
    denom = sum(values) or 1
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v} ({v / denom:.0%})", va="center", color=INK_2, fontsize=8)
    style(ax, "Images per class", grid_axis="x")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3 if values else 1)
    fig.tight_layout()
    return fig


def intensity(images):
    fig = plt.figure(figsize=(6, 3.0), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    names = ("red", "green", "blue")
    for channel in range(min(3, images.shape[3] if images.ndim == 4 else 0)):
        values = images[:, :, :, channel].reshape(-1)
        ax.hist(values, bins=INTENSITY_BINS, range=(0, 255), histtype="step",
                linewidth=2, color=SERIES[channel], label=names[channel])
    style(ax, "Pixel intensity per channel")
    ax.set_xlabel("intensity (0 = black, 255 = white)", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def class_means(images, labels, class_names):
    shown = class_names[:MAX_THUMB_CLASSES]
    n = max(1, len(shown))
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 2.0 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)
    for i, name in enumerate(shown):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        picks = np.flatnonzero(labels == class_names.index(name))
        if picks.size:
            ax.imshow(images[picks].astype(np.float32).mean(axis=0).astype(np.uint8))
        ax.set_title(name, color=INK, fontsize=9, loc="left")
    fig.suptitle("Mean image per class", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def draw_all(images, labels, class_names, profile: dict, plots_dir: Path, tag: str):
    return [
        save(thumbnails(images, labels, class_names), plots_dir, f"{tag}_thumbnails",
             "thumbnails"),
        save(class_balance(profile["class_counts"]), plots_dir, f"{tag}_class_balance",
             "class_balance"),
        save(intensity(images), plots_dir, f"{tag}_intensity", "intensity"),
        save(class_means(images, labels, class_names), plots_dir, f"{tag}_class_means",
             "class_means"),
    ]


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Profile an image dataset and draw figures.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--input", default=INPUT, help=".npz path relative to --project")
    parser.add_argument("--tag", default=TAG, help="prefix for the output files")
    args = parser.parse_args(argv)

    project_dir = Path(args.project).resolve()
    meta = read_json(project_dir / META_FILE, default={}) or {}
    images, labels, class_names, manifest, path = load_pair(project_dir, args.input, meta)
    profile = profile_images(images, labels, class_names, manifest)
    profile["figures"] = draw_all(images, labels, class_names, profile,
                                  project_dir / "plots", args.tag)
    out = project_dir / f"profile_{args.tag}.json"
    out.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"profiled {path.name}: {profile['n_images']} images at "
        f"{profile['image_size']}px across {profile['n_classes']} classes "
        f"-> {out.name} and {len(profile['figures'])} figures",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_template_profile_images.py tests/test_captions.py -q
python -m pytest -W error::DeprecationWarning tests/test_template_profile_images.py -q
ruff check .
```
Expected: all pass, no deprecation warnings, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates/image_common/profile.py mlagent/captions.py tests/test_template_profile_images.py tests/test_captions.py
git commit -m "feat: the image profile template and its thumbnails, intensity and class-mean captions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 6: Image audit, image cleaning and the generated `clean.py`

**Files:**
- Create: `mlagent/audit_images.py`, `mlagent/cleaning_images.py`, `mlagent/templates/image_common/clean.py`, `tests/test_audit_images.py`, `tests/test_cleaning_images.py`
- Test: `tests/test_audit_images.py`, `tests/test_cleaning_images.py`

**Interfaces:**
- Consumes: `audit.Issue` (`mlagent/audit.py:24`), `imageset.ImageSet` / `duplicate_groups` / `blank_indices` / `write_pair` / `read_pair` (Task 2), `templates_io.shared_file` (Task 1), `synth.images.generate` (Task 3).
- Produces (used by Task 7):
  - `audit_images.MIN_IMAGES_PER_CLASS = 10`, `audit_images.MAJORITY_LIMIT = 0.9`
  - `audit_images.audit_images(imageset: ImageSet, meta: dict, skipped: Sequence[tuple[str, str]] = ()) -> list[Issue]`
  - `cleaning_images.apply_steps(imageset: ImageSet, steps: list[dict]) -> ImageSet`
  - `cleaning_images.describe_step(step: dict) -> str`
  - `cleaning_images.render_clean_py(steps: list[dict]) -> str`
  - `cleaning_images.STEPS_MARKER = 'STEPS_JSON = r"""[]"""'`
  - `mlagent/templates/image_common/clean.py` writing `data/clean/data.npz`, `data/clean/manifest.csv`, `profile_clean.json` and `plots/clean_before_after_classes.png`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_images.py`:

```python
from __future__ import annotations

import numpy as np

from mlagent.audit_images import MIN_IMAGES_PER_CLASS, audit_images
from mlagent.imageset import ImageSet, make_manifest


def build(labels, class_names, images=None, sizes=None, image_size=16):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if images is None:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(n, image_size, image_size, 3), dtype=np.uint8)
    sizes = sizes or [(image_size * 4, image_size * 4)] * n
    manifest = make_manifest(labels, class_names, [f"file{i}.png" for i in range(n)], sizes)
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


META = {"image_size": 16, "task_type": "image_classification"}


def kinds(issues):
    return [i.kind for i in issues]


def test_no_issues_on_a_balanced_varied_dataset():
    labels = [0] * 20 + [1] * 20
    assert audit_images(build(labels, ["a", "b"]), META) == []


def test_class_imbalance_fires_without_a_fix():
    labels = [0] * 95 + [1] * 5
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META)
              if i.kind == "class_imbalance"]
    assert len(issues) == 1
    assert issues[0].severity == "medium"
    assert issues[0].fix is None
    assert issues[0].evidence["majority_fraction"] >= 0.9


def test_tiny_classes_are_high_severity_with_no_automatic_fix():
    labels = [0] * 30 + [1] * 3
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META)
              if i.kind == "tiny_classes"]
    assert len(issues) == 1
    assert issues[0].severity == "high"
    assert issues[0].fix is None
    assert issues[0].evidence["classes"] == {"b": 3}
    assert str(MIN_IMAGES_PER_CLASS) in issues[0].message


def test_duplicate_images_propose_dropping_all_but_the_first_of_each_group():
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    other = np.full((16, 16, 3), 90, dtype=np.uint8)
    rng = np.random.default_rng(1)
    unique = rng.integers(0, 255, size=(16, 16, 16, 3), dtype=np.uint8)
    images = np.concatenate([np.stack([base, base, base, other, other]), unique])
    labels = [0] * 10 + [1] * 11
    issues = [i for i in audit_images(build(labels, ["a", "b"], images=images), META)
              if i.kind == "duplicate_images"]
    assert len(issues) == 1
    assert issues[0].fix["op"] == "drop_indices"
    assert issues[0].fix["params"]["indices"] == [1, 2, 4]
    assert "duplicate" in issues[0].fix["params"]["reason"]


def test_blank_images_propose_dropping_them():
    rng = np.random.default_rng(2)
    images = rng.integers(0, 255, size=(20, 16, 16, 3), dtype=np.uint8)
    images[3] = 180
    images[11] = 20
    labels = [0] * 10 + [1] * 10
    issues = [i for i in audit_images(build(labels, ["a", "b"], images=images), META)
              if i.kind == "blank_images"]
    assert len(issues) == 1
    assert issues[0].fix["params"]["indices"] == [3, 11]


def test_upscaled_tiny_sources_are_informational_only():
    labels = [0] * 10 + [1] * 10
    sizes = [(6, 6)] * 5 + [(64, 64)] * 15
    issues = [i for i in audit_images(build(labels, ["a", "b"], sizes=sizes), META)
              if i.kind == "upscaled_sources"]
    assert len(issues) == 1
    assert issues[0].severity == "low"
    assert issues[0].fix is None
    assert issues[0].evidence["count"] == 5


def test_skipped_files_become_one_informational_issue():
    labels = [0] * 10 + [1] * 10
    skipped = [("a/broken.png", "OSError: truncated"), ("stray.png", "no class folder")]
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META, skipped=skipped)
              if i.kind == "unreadable_files"]
    assert len(issues) == 1
    assert issues[0].severity == "low"
    assert issues[0].fix is None
    assert issues[0].evidence["count"] == 2
    assert "no class folder" in json_dumps(issues[0].evidence)


def json_dumps(value) -> str:
    import json

    return json.dumps(value, default=str)


def test_drop_indices_is_the_only_fix_op_ever_proposed():
    rng = np.random.default_rng(3)
    images = rng.integers(0, 255, size=(25, 16, 16, 3), dtype=np.uint8)
    images[1] = images[0]
    images[5] = 200
    labels = [0] * 22 + [1] * 3
    issues = audit_images(build(labels, ["a", "b"], images=images), META,
                          skipped=[("x.png", "bad")])
    ops = {i.fix["op"] for i in issues if i.fix}
    assert ops == {"drop_indices"}
    assert set(kinds(issues)) >= {"tiny_classes", "duplicate_images", "blank_images"}


def test_issues_are_sorted_high_severity_first():
    labels = [0] * 30 + [1] * 2
    issues = audit_images(build(labels, ["a", "b"]), META)
    severities = [i.severity for i in issues]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])
```

Create `tests/test_cleaning_images.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from mlagent.cleaning_images import (
    STEPS_MARKER,
    apply_steps,
    describe_step,
    render_clean_py,
)
from mlagent.imageset import read_pair, write_pair
from mlagent.synth.images import SynthImageConfig, generate


def make_set(n=24, size=32, n_classes=3):
    return generate(SynthImageConfig(n_images=n, image_size=size, n_classes=n_classes,
                                     seed=6, noise=0.05))


def test_apply_steps_with_no_steps_returns_an_equal_copy():
    s = make_set()
    out = apply_steps(s, [])
    assert out.n_images == s.n_images
    assert np.array_equal(out.images, s.images)


def test_apply_steps_drops_the_named_indices_and_reindexes():
    s = make_set(n=12)
    out = apply_steps(s, [{"op": "drop_indices", "params": {"indices": [0, 5, 11],
                                                            "reason": "duplicate images"}}])
    assert out.n_images == 9
    assert out.manifest["index"].tolist() == list(range(9))
    kept = [i for i in range(12) if i not in (0, 5, 11)]
    assert np.array_equal(out.images, s.images[kept])
    assert out.class_names == s.class_names


def test_apply_steps_merges_two_drop_steps_without_double_counting():
    s = make_set(n=10)
    out = apply_steps(s, [
        {"op": "drop_indices", "params": {"indices": [1, 2], "reason": "duplicates"}},
        {"op": "drop_indices", "params": {"indices": [2, 3], "reason": "blank images"}},
    ])
    assert out.n_images == 7
    assert np.array_equal(out.images, s.images[[0, 4, 5, 6, 7, 8, 9]])


def test_apply_steps_rejects_an_unknown_op():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        apply_steps(make_set(), [{"op": "drop_columns", "params": {"columns": ["a"]}}])


def test_describe_step_reads_as_a_sentence():
    text = describe_step({"op": "drop_indices",
                          "params": {"indices": [1, 2, 3], "reason": "duplicate images"}})
    assert "3" in text and "duplicate images" in text


def test_render_clean_py_substitutes_the_steps_and_keeps_the_conventions():
    steps = [{"op": "drop_indices", "params": {"indices": [0, 1], "reason": "blank images"}}]
    source = render_clean_py(steps)
    assert STEPS_MARKER not in source
    assert '"drop_indices"' in source
    assert "import mlagent" not in source and "from mlagent" not in source
    assert "# --- settings ---" in source
    assert "def cli_argv()" in source
    assert "sys.exit(0)" not in source
    assert source.count("\n# --- ") >= 5


def test_the_rendered_script_runs_for_real_and_writes_the_clean_pair(project):
    s = make_set(n=24, size=32, n_classes=3)
    write_pair(s, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "raw_path": "data/raw/data.npz", "image_size": 32, "n_channels": 3,
    })
    steps = [{"op": "drop_indices", "params": {"indices": [0, 1, 2], "reason": "duplicates"}}]
    (project.root / "clean.py").write_text(render_clean_py(steps), encoding="utf-8")

    result = subprocess.run([sys.executable, "clean.py"], cwd=str(project.root),
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr

    cleaned = read_pair(project.data_clean)
    assert cleaned.n_images == 21
    assert np.array_equal(cleaned.images, s.images[3:])
    assert cleaned.class_names == s.class_names

    profile = json.loads((project.root / "profile_clean.json").read_text(encoding="utf-8"))
    assert profile["before"]["n_images"] == 24
    assert profile["after"]["n_images"] == 21
    assert profile["steps"] == steps
    assert profile["figures"] == ["clean_before_after_classes.png"]
    assert (project.plots_dir / "clean_before_after_classes.png").exists()
    assert "How to read this:" in result.stdout


def test_the_rendered_script_leaves_the_raw_pair_untouched(project):
    s = make_set(n=16, size=32, n_classes=2)
    write_pair(s, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "raw_path": "data/raw/data.npz", "image_size": 32, "n_channels": 3,
    })
    (project.root / "clean.py").write_text(
        render_clean_py([{"op": "drop_indices",
                          "params": {"indices": [0], "reason": "blank images"}}]),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "clean.py"], cwd=str(project.root), check=True,
                   capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert read_pair(project.data_raw).n_images == 16
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_audit_images.py tests/test_cleaning_images.py -q
```
Expected: two collection errors, `ModuleNotFoundError: No module named 'mlagent.audit_images'` and `... 'mlagent.cleaning_images'`.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/audit_images.py`:

```python
"""Cleanliness audit for image datasets.

The image counterpart of `audit.py`: each check returns `audit.Issue`s with evidence and,
where a fix is possible, a proposed step. Dropping images is the only automatic fix an
image dataset supports, so every `fix` here is a `drop_indices` step.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlagent.audit import SEVERITY_ORDER, Issue
from mlagent.imageset import ImageSet, blank_indices, duplicate_groups

# Mirrors audit.py's tabular class-balance threshold.
MAJORITY_LIMIT = 0.9
MIN_IMAGES_PER_CLASS = 10
# A source image whose shorter side is below image_size * this was upscaled to fit.
UPSCALE_RATIO = 0.5
MAX_LISTED = 20


def _drop(indices: list[int], reason: str) -> dict:
    return {"op": "drop_indices", "params": {"indices": [int(i) for i in indices],
                                             "reason": reason}}


def _check_balance(imageset: ImageSet, _meta: dict) -> list[Issue]:
    counts = imageset.class_counts()
    total = sum(counts.values())
    if not total:
        return []
    out: list[Issue] = []
    majority = max(counts.values()) / total
    if majority > MAJORITY_LIMIT:
        out.append(Issue(
            "class_imbalance", "medium",
            f"The biggest class holds {majority:.0%} of the images; accuracy will look "
            "good even for a model that always guesses that class.",
            None,
            {"majority_fraction": round(float(majority), 4), "counts": dict(counts)},
        ))
    tiny = {name: n for name, n in counts.items() if n < MIN_IMAGES_PER_CLASS}
    if tiny:
        out.append(Issue(
            "tiny_classes", "high",
            f"{len(tiny)} class(es) have fewer than {MIN_IMAGES_PER_CLASS} images "
            f"({', '.join(f'{k}: {v}' for k, v in tiny.items())}). There is no automatic "
            "fix: collect more examples of these classes, or drop them from the dataset.",
            None,
            {"classes": tiny, "minimum": MIN_IMAGES_PER_CLASS},
        ))
    return out


def _check_duplicates(imageset: ImageSet, _meta: dict) -> list[Issue]:
    groups = duplicate_groups(imageset.images)
    if not groups:
        return []
    extras = sorted(i for group in groups for i in group[1:])
    return [Issue(
        "duplicate_images", "medium",
        f"{len(extras)} image(s) are an exact copy of another image. Duplicates that land "
        "in different splits leak the answer from training into validation.",
        None,
        {"groups": len(groups), "extra_copies": len(extras),
         "example_group": groups[0][:MAX_LISTED]},
        _drop(extras, "duplicate images"),
    )]


def _check_blanks(imageset: ImageSet, _meta: dict) -> list[Issue]:
    blanks = blank_indices(imageset.images)
    if not blanks:
        return []
    return [Issue(
        "blank_images", "medium",
        f"{len(blanks)} image(s) are a near-constant block of colour with nothing in them. "
        "They teach the model nothing and dilute the class they sit in.",
        None,
        {"count": len(blanks), "examples": blanks[:MAX_LISTED]},
        _drop(blanks, "blank images"),
    )]


def _check_upscaled(imageset: ImageSet, meta: dict) -> list[Issue]:
    size = int(meta.get("image_size") or imageset.image_size or 0)
    manifest = imageset.manifest
    if not size or "source_width" not in manifest or "source_height" not in manifest:
        return []
    shorter = np.minimum(
        manifest["source_width"].to_numpy(dtype=float),
        manifest["source_height"].to_numpy(dtype=float),
    )
    small = np.flatnonzero(shorter < size * UPSCALE_RATIO)
    if small.size == 0:
        return []
    return [Issue(
        "upscaled_sources", "low",
        f"{small.size} source image(s) were smaller than {int(size * UPSCALE_RATIO)}px on "
        f"their shorter side and had to be blown up to {size}px, so they carry less detail "
        "than the rest. Nothing to fix; just do not expect much from them.",
        None,
        {"count": int(small.size), "image_size": size,
         "examples": [int(i) for i in small[:MAX_LISTED].tolist()]},
    )]


def _check_skipped(skipped: Sequence[tuple[str, str]]) -> list[Issue]:
    items = list(skipped or ())
    if not items:
        return []
    return [Issue(
        "unreadable_files", "low",
        f"{len(items)} file(s) in the source folder were skipped: they were not readable "
        "images, or sat outside a class subfolder. They are not part of the dataset.",
        None,
        {"count": len(items),
         "files": [{"path": str(p), "reason": str(r)} for p, r in items[:MAX_LISTED]]},
    )]


CHECKS = (_check_balance, _check_duplicates, _check_blanks, _check_upscaled)


def audit_images(
    imageset: ImageSet, meta: dict, skipped: Sequence[tuple[str, str]] = ()
) -> list[Issue]:
    """Every problem worth telling the user about, highest severity first."""
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(imageset, meta or {}))
    issues.extend(_check_skipped(skipped))
    issues.sort(key=lambda i: SEVERITY_ORDER[i.severity])
    return issues
```

Create `mlagent/cleaning_images.py`:

```python
"""Image cleaning steps as data.

`apply_steps()` runs them; `render_clean_py()` writes a re-runnable script that calls the
same operation. The mirror of `cleaning.py`, but the only operation an image dataset
supports is dropping images by index.
"""

from __future__ import annotations

import json

from mlagent.imageset import ImageSet
from mlagent.templates_io import IMAGE_COMMON_DIRNAME, shared_file

CLEAN_TEMPLATE = f"{IMAGE_COMMON_DIRNAME}/clean.py"
STEPS_MARKER = 'STEPS_JSON = r"""[]"""'
OPS = ("drop_indices",)


def apply_steps(imageset: ImageSet, steps: list[dict]) -> ImageSet:
    """Drop every index any approved step names, once, keeping the original order."""
    if not steps:
        return imageset.take(range(imageset.n_images))
    dropped: set[int] = set()
    for step in steps:
        op = step.get("op")
        if op not in OPS:
            raise ValueError(f"unknown image cleaning op {op!r}")
        dropped.update(int(i) for i in (step.get("params") or {}).get("indices", []))
    keep = [i for i in range(imageset.n_images) if i not in dropped]
    return imageset.take(keep)


def describe_step(step: dict) -> str:
    params = step.get("params", {})
    if step.get("op") == "drop_indices":
        n = len(params.get("indices", []))
        reason = params.get("reason") or "flagged by the audit"
        return f"drop {n} image(s) ({reason})"
    return f"{step.get('op')} {params}"


def render_clean_py(steps: list[dict]) -> str:
    """The standalone image-cleaning script for this project: the shared template with the
    approved steps substituted into its one placeholder line."""
    source = shared_file(CLEAN_TEMPLATE).read_text(encoding="utf-8")
    if STEPS_MARKER not in source:
        raise ValueError(
            f"templates/{CLEAN_TEMPLATE} no longer contains the line {STEPS_MARKER!r}"
        )
    payload = json.dumps(steps, indent=2)
    return source.replace(STEPS_MARKER, f'STEPS_JSON = r"""{payload}"""', 1)
```

Create `mlagent/templates/image_common/clean.py`:

```python
"""Apply the image drops you approved. Generated by mlagent; safe to edit.

Run it from the project folder:

    python clean.py

Reads the raw `data.npz` / `manifest.csv` pair named in `data_meta.json`, drops the
approved image indices, and writes `data/clean/data.npz`, `data/clean/manifest.csv`,
`profile_clean.json` (a before/after summary) and
`plots/clean_before_after_classes.png`. Editing `STEPS` and rerunning is the supported
way to redo the cleaning differently; the raw pair is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- settings ---
SCRIPT_NAME = "clean.py"
PROJECT_DIR = Path(".")
META_FILE = "data_meta.json"
RAW_FILE = "data.npz"
MANIFEST_FILE = "manifest.csv"
CLEAN_DIR = Path("data") / "clean"
PROFILE_FILE = "profile_clean.json"
FIGURE_NAME = "clean_before_after_classes"

# --- the steps mlagent recorded ---
# Each step is {"op": "drop_indices", "params": {"indices": [...], "reason": "..."}} and
# runs against the *raw* indices. Edit and rerun freely.
STEPS_JSON = r"""[]"""
STEPS: list[dict] = json.loads(STEPS_JSON)

# --- captions ---
# Byte-identical to the matching entry in mlagent/captions.py
# (tests/test_captions.py checks this); kept here too since this script never depends on
# the mlagent package.
CAPTIONS = {
    "clean_before_after_classes": (
        "How many images each class has before and after cleaning. Bars that shrink lost "
        "duplicate or blank images; a bar that disappears means the class has no images "
        "left, which the assistant treats as an error rather than a smaller dataset."
    ),
}

# --- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)


# --- reading and writing the pair ---
def read_pair(directory: Path):
    with np.load(directory / RAW_FILE) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]
    manifest = pd.read_csv(directory / MANIFEST_FILE)
    return images, labels, class_names, manifest


def write_pair(directory: Path, images, labels, class_names, manifest) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / RAW_FILE,
        images=np.ascontiguousarray(images, dtype=np.uint8),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(class_names, dtype="U"),
    )
    manifest.to_csv(directory / MANIFEST_FILE, index=False, encoding="utf-8")


# --- cleaning operations ---
def dropped_indices(steps: list[dict] | None = None) -> list[int]:
    """Every image index the approved steps drop, de-duplicated and sorted."""
    steps = STEPS if steps is None else steps
    dropped: set[int] = set()
    for step in steps:
        op = step.get("op")
        if op != "drop_indices":
            raise ValueError(f"unknown image cleaning op {op!r}")
        dropped.update(int(i) for i in (step.get("params") or {}).get("indices", []))
    return sorted(dropped)


def clean(images, labels, manifest, steps: list[dict] | None = None):
    """Return (images, labels, manifest) with the approved indices removed."""
    drop = set(dropped_indices(steps))
    keep = [i for i in range(len(labels)) if i not in drop]
    kept_manifest = manifest.iloc[keep].copy().reset_index(drop=True)
    kept_manifest["index"] = range(len(kept_manifest))
    return images[keep], labels[keep], kept_manifest


# --- before and after summary ---
def summarise(images, labels, class_names) -> dict:
    counts = {name: 0 for name in class_names}
    for label in np.asarray(labels).tolist():
        if 0 <= int(label) < len(class_names):
            counts[class_names[int(label)]] += 1
    return {
        "n_images": int(len(labels)),
        "image_size": int(images.shape[1]) if images.ndim == 4 else 0,
        "n_channels": int(images.shape[3]) if images.ndim == 4 else 0,
        "n_classes": len(class_names),
        "class_counts": counts,
    }


# --- figure ---
def before_after_classes(before: dict, after: dict):
    names = list(before["class_counts"])
    y = np.arange(len(names))
    b = [before["class_counts"][n] for n in names]
    a = [after["class_counts"].get(n, 0) for n in names]
    fig = plt.figure(figsize=(6, 0.45 * max(3, len(names)) + 1.4), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.barh(y - 0.18, b, height=0.34, color=SERIES[0], label="before cleaning")
    ax.barh(y + 0.18, a, height=0.34, color=SERIES[1], label="after cleaning")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK_2)
    ax.set_xlabel("images", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Images per class before and after cleaning", color=INK, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def show(fig, kind: str) -> None:
    """Display in a notebook if one is running, then print how to read the figure."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            display(fig)
    caption = CAPTIONS.get(kind, "")
    if caption:
        print("How to read this: " + caption.replace("[[", "").replace("]]", ""))


def save(fig, plots_dir: Path, name: str, kind: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig, kind)
    plt.close(fig)
    return f"{name}.png"


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Apply the approved image drops.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()

    meta_path = project_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    raw_dir = (project_dir / str(meta.get("raw_path") or "data/raw/data.npz")).parent
    images, labels, class_names, manifest = read_pair(raw_dir)

    before = summarise(images, labels, class_names)
    kept_images, kept_labels, kept_manifest = clean(images, labels, manifest)
    after = summarise(kept_images, kept_labels, class_names)
    write_pair(project_dir / CLEAN_DIR, kept_images, kept_labels, class_names, kept_manifest)

    figures = [save(before_after_classes(before, after), project_dir / "plots", FIGURE_NAME,
                    FIGURE_NAME)]
    (project_dir / PROFILE_FILE).write_text(
        json.dumps({"before": before, "after": after, "steps": STEPS, "figures": figures},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"cleaned {before['n_images']} -> {after['n_images']} images after "
        f"{len(STEPS)} step(s) -> {(CLEAN_DIR / RAW_FILE).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_audit_images.py tests/test_cleaning_images.py -q && ruff check .
```
Expected: 17 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/audit_images.py mlagent/cleaning_images.py mlagent/templates/image_common/clean.py tests/test_audit_images.py tests/test_cleaning_images.py
git commit -m "feat: image audit checks, drop_indices cleaning and the generated image clean.py

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 7: The image modality record and the data and clean stages

**Files:**
- Create: none
- Modify: `mlagent/modality.py` (add the `IMAGE` record), `mlagent/stages/data.py`, `mlagent/stages/clean.py`, `tests/conftest.py`, `tests/test_modality.py`
- Test: `tests/test_data_stage.py`, `tests/test_clean_stage.py`, `tests/test_modality.py`

**Interfaces:**
- Consumes: `synth.images.generate` / `SynthImageConfig` (Task 3), `drive_images.load_folder` (Task 4), `hf_images.load_image_dataset` (Task 4), `audit_images.audit_images` (Task 6), `cleaning_images.apply_steps` / `render_clean_py` (Task 6), `imageset.write_pair` / `read_pair` / `IMAGE_SIZES` / `DEFAULT_IMAGE_SIZE` (Task 2), `templates_io.copy_shared` (Task 1).
- Produces (used by Tasks 8, 10, 13):
  - `modality.IMAGE: Modality` with `name="image"`, `task_types=("image_classification",)`, `data_file="data.npz"`, `profile_template="image_common/profile.py"`, `clean_template="image_common/clean.py"`, `template_family="image_torch"`, `teaching_material="model_choices_images"`, `profile_figures=("thumbnails", "class_balance", "intensity", "class_means")`
  - `modality.MODALITIES == (TABULAR, IMAGE)`
  - `stages/data.py`: `IMAGE_RAW_FILE = "data.npz"`, `IMAGE_TARGET = "label"`, `DataStage.__init__(..., hf_image_load=None)`; form answer keys `data.n_images`, `data.image_size`, `data.n_classes`, `data.noise`, `data.inject_quirks`, `data.drive_folder`, `data.hf_dataset`, `data.max_images`
  - `data_meta.json` for images: `target="label"`, `modality="image"`, `image_size`, `n_channels=3`, `raw_path="data/raw/data.npz"`, `raw_n_rows=N`, `raw_n_cols=H*W*3`, `class_labels`, `n_classes`, `skipped_files`
  - `stages/clean.py`: `CLEAN_IMAGE_REL_PATH = "data/clean/data.npz"`; image debrief writes `clean_path`, `clean_n_rows`, `clean_n_cols`, `dropped_columns=[]`, `feature_columns=[]`, `categorical_columns=[]`, `n_classes`, `class_labels`

- [ ] **Step 1: Write the failing test**

Add to `tests/conftest.py` (after the existing `write_clean_project` helper):

```python
def write_clean_image_project(
    project: Project, n_images: int = 60, image_size: int = 32, n_classes: int = 3
) -> Project:
    """Write data/clean/data.npz + manifest.csv, data_meta.json and spec.json as the
    image clean stage leaves them."""
    from mlagent.imageset import write_pair
    from mlagent.synth.images import SynthImageConfig, generate

    imageset = generate(SynthImageConfig(n_images=n_images, image_size=image_size,
                                         n_classes=n_classes, seed=8, noise=0.05))
    project.ensure_dirs()
    write_pair(imageset, project.data_raw)
    write_pair(imageset, project.data_clean)
    project.write_json("data_meta.json", {
        "target": "label",
        "task_type": "image_classification",
        "modality": "image",
        "source": "synthetic",
        "raw_path": "data/raw/data.npz",
        "raw_n_rows": imageset.n_images,
        "raw_n_cols": image_size * image_size * 3,
        "clean_path": "data/clean/data.npz",
        "clean_n_rows": imageset.n_images,
        "clean_n_cols": image_size * image_size * 3,
        "dropped_columns": [],
        "feature_columns": [],
        "categorical_columns": [],
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "split_seed": 42,
        "image_size": image_size,
        "n_channels": 3,
        "n_classes": n_classes,
        "class_labels": list(imageset.class_names),
    })
    project.write_json("spec.json", {
        "goal": "image fixture project",
        "task_type": "image_classification",
        "metric": "accuracy",
        "target_value": 0.8,
        "data_source": "synthetic",
        "minutes_per_run": 5,
        "max_rounds": 3,
        "gpu": "none",
        "notes": "",
    })
    return project


@pytest.fixture
def clean_image_project(project: Project) -> Project:
    return write_clean_image_project(project)
```

Add to `tests/test_modality.py`:

```python
def test_the_image_record_is_registered_and_points_at_the_image_modules():
    from mlagent import audit_images, cleaning_images
    from mlagent.datasources import drive_images, hf_images
    from mlagent.modality import IMAGE, MODALITIES
    from mlagent.synth import images as synth_images

    assert MODALITIES == (TABULAR, IMAGE)
    assert modality_for("image_classification") is IMAGE
    assert IMAGE.name == "image"
    assert IMAGE.data_file == "data.npz"
    assert IMAGE.template_family == "image_torch"
    assert IMAGE.profile_template == "image_common/profile.py"
    assert IMAGE.clean_template == "image_common/clean.py"
    assert IMAGE.teaching_material == "model_choices_images"
    assert IMAGE.profile_figures == ("thumbnails", "class_balance", "intensity", "class_means")
    assert IMAGE.generate is synth_images.generate
    assert IMAGE.load_drive is drive_images.load_folder
    assert IMAGE.load_hf is hf_images.load_image_dataset
    assert IMAGE.audit is audit_images.audit_images
    assert IMAGE.apply_steps is cleaning_images.apply_steps
    assert IMAGE.render_clean_py is cleaning_images.render_clean_py


def test_the_image_record_writes_and_reads_the_npz_pair(tmp_path):
    from mlagent.modality import IMAGE
    from mlagent.synth.images import SynthImageConfig, generate

    s = generate(SynthImageConfig(n_images=12, image_size=32, n_classes=2, seed=1))
    IMAGE.write_raw(s, tmp_path)
    assert (tmp_path / "data.npz").exists() and (tmp_path / "manifest.csv").exists()
    back = IMAGE.read(tmp_path)
    assert back.n_images == 12 and back.class_names == s.class_names
```

Add to `tests/test_data_stage.py`:

```python
def make_image_ctx(project, answers):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    shown: list[str] = []
    project.write_json("spec.json", {
        "goal": "classify shapes", "task_type": "image_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 3, "gpu": "T4", "notes": "",
    })
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner(answers, ScriptedQuestioner([])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    return ctx, shown


def test_image_synthetic_writes_the_npz_pair_and_the_image_meta(project):
    from mlagent.imageset import read_pair
    from mlagent.stages.data import DataStage

    ctx, _shown = make_image_ctx(project, {
        "data.n_images": 40, "data.image_size": 32, "data.n_classes": 3,
        "data.noise": 0.1, "data.inject_quirks": True,
    })
    handoff = DataStage().prepare(ctx)

    assert handoff.commands == [["profile.py"]]
    assert handoff.outputs == ["profile_raw.json"]
    assert (project.root / "profile.py").exists()
    imageset = read_pair(project.data_raw)
    assert imageset.n_images == 40 and imageset.image_size == 32

    meta = project.read_json("data_meta.json")
    assert meta["target"] == "label"
    assert meta["modality"] == "image"
    assert meta["task_type"] == "image_classification"
    assert meta["source"] == "synthetic"
    assert meta["raw_path"] == "data/raw/data.npz"
    assert meta["raw_n_rows"] == 40
    assert meta["raw_n_cols"] == 32 * 32 * 3
    assert meta["image_size"] == 32
    assert meta["n_channels"] == 3
    assert meta["class_labels"] == imageset.class_names
    assert meta["skipped_files"] == []


def test_the_image_profile_template_is_the_one_copied(project):
    from mlagent.stages.data import DataStage

    ctx, _shown = make_image_ctx(project, {
        "data.n_images": 20, "data.image_size": 32, "data.n_classes": 2,
        "data.noise": 0.0, "data.inject_quirks": False,
    })
    DataStage().prepare(ctx)
    source = (project.root / "profile.py").read_text(encoding="utf-8")
    assert "Profile an image dataset" in source


def test_image_drive_source_records_the_skipped_files(project, tmp_path):
    from PIL import Image

    from mlagent.stages.data import DataStage

    folder = tmp_path / "pics"
    for cls in ("cats", "dogs"):
        for i in range(6):
            path = folder / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (10 * i, 60, 90)).save(path)
    (folder / "stray.png").write_bytes(b"")

    ctx, _shown = make_image_ctx(project, {
        "data.drive_folder": str(folder), "data.image_size": 32, "data.max_images": 0,
    })
    ctx.project.write_json("spec.json", {**ctx.project.read_json("spec.json"),
                                         "data_source": "drive"})
    DataStage().prepare(ctx)
    meta = project.read_json("data_meta.json")
    assert meta["source"] == "drive"
    assert meta["source_path"] == str(folder)
    assert meta["raw_n_rows"] == 12
    assert any(reason == "no class folder" for _path, reason in meta["skipped_files"])


def test_image_huggingface_source_uses_the_injected_loader(project):
    from mlagent.stages.data import DataStage
    from mlagent.synth.images import SynthImageConfig, generate

    fake = generate(SynthImageConfig(n_images=18, image_size=32, n_classes=2, seed=2))
    calls = []

    def loader(dataset_id, image_size, split="train", max_images=None, **kwargs):
        calls.append((dataset_id, image_size, max_images))
        return fake

    ctx, _shown = make_image_ctx(project, {
        "data.hf_dataset": "acme/shapes", "data.image_size": 32, "data.max_images": 500,
    })
    ctx.project.write_json("spec.json", {**ctx.project.read_json("spec.json"),
                                         "data_source": "huggingface"})
    DataStage(hf_image_load=loader).prepare(ctx)
    assert calls == [("acme/shapes", 32, 500)]
    meta = project.read_json("data_meta.json")
    assert meta["source"] == "huggingface" and meta["hf_id"] == "acme/shapes"
    assert meta["raw_n_rows"] == 18
```

Add to `tests/test_clean_stage.py`:

```python
def image_ctx(project, answers=(), form=None):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    shown: list[str] = []
    scripted = ScriptedQuestioner(list(answers))
    questioner = FormQuestioner(form or {}, scripted) if form is not None else scripted
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=questioner,
                       explainer=None, display=shown.append,
                       display_figure=lambda path, caption="": None)
    return ctx, shown


def test_image_clean_prepare_writes_the_audit_and_an_image_clean_py(project):
    from mlagent.imageset import write_pair
    from mlagent.stages.clean import CleanStage
    from mlagent.synth.images import SynthImageConfig, generate

    imageset = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=3,
                                         duplicate_fraction=0.2, blank_fraction=0.1))
    write_pair(imageset, project.data_raw)
    project.write_json("data_meta.json", {
        "target": "label", "task_type": "image_classification", "modality": "image",
        "source": "synthetic", "raw_path": "data/raw/data.npz", "raw_n_rows": 40,
        "raw_n_cols": 32 * 32 * 3, "image_size": 32, "n_channels": 3,
        "class_labels": list(imageset.class_names), "n_classes": 2, "skipped_files": [],
    })
    project.write_json("spec.json", {
        "goal": "shapes", "task_type": "image_classification", "metric": "accuracy",
        "target_value": 0.9, "data_source": "synthetic", "minutes_per_run": 5,
        "max_rounds": 3, "gpu": "none", "notes": "",
    })
    ctx, _shown = image_ctx(project, answers=["y"] * 10,
                            form={"clean.train_fraction": 0.7, "clean.val_fraction": 0.15})
    handoff = CleanStage().prepare(ctx)

    assert handoff.commands == [["clean.py"]]
    assert handoff.outputs == ["data/clean/data.npz", "profile_clean.json"]
    audit = project.read_json("audit.json")
    assert {i["kind"] for i in audit["issues"]} & {"duplicate_images", "blank_images"}
    assert all(s["op"] == "drop_indices" for s in audit["steps"])
    source = (project.root / "clean.py").read_text(encoding="utf-8")
    assert "Apply the image drops you approved" in source
    meta = project.read_json("data_meta.json")
    assert meta["dropped_columns"] == [] and meta["split_seed"] == 42


def test_image_clean_debrief_completes_the_meta(clean_image_project):
    from mlagent.stages.clean import CleanStage

    project = clean_image_project
    project.write_json("profile_clean.json", {
        "before": {"n_images": 60, "class_counts": {"circle": 20, "cross": 20, "square": 20}},
        "after": {"n_images": 58, "class_counts": {"circle": 19, "cross": 20, "square": 19}},
        "steps": [], "figures": ["clean_before_after_classes.png"],
    })
    ctx, shown = image_ctx(project)
    CleanStage().debrief(ctx)
    meta = project.read_json("data_meta.json")
    assert meta["clean_path"] == "data/clean/data.npz"
    assert meta["clean_n_rows"] == 60
    assert meta["clean_n_cols"] == 32 * 32 * 3
    assert meta["feature_columns"] == [] and meta["categorical_columns"] == []
    assert meta["n_classes"] == 3
    assert meta["class_labels"] == ["circle", "cross", "square"]
    assert any("58" in text or "Cleaning summary" in text for text in shown)


def test_image_clean_debrief_reports_an_emptied_class_as_an_error(clean_image_project):
    from mlagent.imageset import read_pair, write_pair
    from mlagent.stages.clean import CleanStage

    project = clean_image_project
    imageset = read_pair(project.data_clean)
    keep = [i for i, label in enumerate(imageset.labels.tolist()) if label != 0]
    write_pair(imageset.take(keep), project.data_clean)
    ctx, shown = image_ctx(project)
    CleanStage().debrief(ctx)
    joined = "\n".join(shown)
    assert "no images left" in joined
    assert imageset.class_names[0] in joined
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_modality.py tests/test_data_stage.py tests/test_clean_stage.py -q
```
Expected: `ImportError: cannot import name 'IMAGE' from 'mlagent.modality'` in the two new modality tests; `ValueError: unknown task type 'image_classification'` in the four new data-stage tests; `KeyError`/`ValueError` in the three new clean-stage tests.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/modality.py`, add the image imports and the record after `TABULAR`:

```python
from mlagent import audit_images, cleaning_images
from mlagent.datasources import drive_images, hf_images
from mlagent.imageset import RAW_FILE as IMAGE_RAW_FILE
from mlagent.imageset import read_pair, write_pair
from mlagent.synth import images as synth_images
```
```python
IMAGE = Modality(
    name="image",
    task_types=("image_classification",),
    data_file=IMAGE_RAW_FILE,
    profile_template="image_common/profile.py",
    clean_template="image_common/clean.py",
    template_family="image_torch",
    teaching_material="model_choices_images",
    profile_figures=("thumbnails", "class_balance", "intensity", "class_means"),
    generate=synth_images.generate,
    load_drive=drive_images.load_folder,
    load_hf=hf_images.load_image_dataset,
    audit=audit_images.audit_images,
    apply_steps=cleaning_images.apply_steps,
    render_clean_py=cleaning_images.render_clean_py,
    write_raw=lambda imageset, directory: write_pair(imageset, directory)[0],
    read=read_pair,
)

MODALITIES: tuple[Modality, ...] = (TABULAR, IMAGE)
```

In `mlagent/stages/data.py`, add the image constants and imports:

```python
from mlagent.imageset import DEFAULT_IMAGE_SIZE, IMAGE_SIZES
from mlagent.synth.images import MAX_CLASSES as MAX_IMAGE_CLASSES
from mlagent.synth.images import SynthImageConfig
```
```python
IMAGE_RAW_FILE = "data.npz"
IMAGE_TARGET = "label"
IMAGE_N_CHANNELS = 3
```

extend `__init__` with `hf_image_load=None` (stored as `self.hf_image_load`), make `is_complete` modality-aware, and branch `prepare` on the record's name:

```python
    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        if not meta or not meta.get("target"):
            return False
        data_file = IMAGE_RAW_FILE if meta.get("modality") == "image" else RAW_FILE
        return (
            (ctx.project.data_raw / data_file).exists()
            and ctx.project.exists(PROFILE_RAW_FILE)
        )
```
```python
    def prepare(self, ctx: StageContext) -> Handoff:
        spec = ctx.spec()
        modality = modality_for(spec.task_type)
        ctx.teaching().preamble("data", {"spec": spec.to_dict()})
        if modality.name == "image":
            meta = self._image(ctx, spec, modality)
        else:
            meta = self._tabular(ctx, spec, modality)
        ctx.project.write_json(META_FILE, meta)
        copy_shared(modality.profile_template, ctx.project.root)
        return Handoff(stage=self.name, commands=[["profile.py"]],
                       outputs=[PROFILE_RAW_FILE])
```

with the existing body of `prepare` (the synthetic/drive/HuggingFace branch, the CSV write, the `meta.update` and the `ctx.display`) moved verbatim into `_tabular(self, ctx, spec, modality) -> dict` returning `meta`, and the new image path:

```python
    def _image(self, ctx: StageContext, spec, modality) -> dict:
        q = ctx.questioner
        image_size = int(q.choice(
            "How big should each image be, in pixels? (bigger sees more detail but trains "
            "more slowly)",
            [str(s) for s in IMAGE_SIZES], allow_other=False, key="data.image_size",
            default=str(DEFAULT_IMAGE_SIZE),
        ))
        if spec.data_source == "synthetic":
            imageset, source_meta = self._image_synthetic(ctx, image_size)
            skipped: list[tuple[str, str]] = []
        elif spec.data_source == "drive":
            imageset, skipped, source_meta = self._image_drive(ctx, modality, image_size)
        else:
            imageset, source_meta = self._image_huggingface(ctx, modality, image_size)
            skipped = []

        modality.write_raw(imageset, ctx.project.data_raw)
        meta = {
            **source_meta,
            "target": IMAGE_TARGET,
            "task_type": spec.task_type,
            "modality": "image",
            "raw_path": f"data/raw/{IMAGE_RAW_FILE}",
            "raw_n_rows": imageset.n_images,
            "raw_n_cols": imageset.image_size * imageset.image_size * IMAGE_N_CHANNELS,
            "image_size": imageset.image_size,
            "n_channels": IMAGE_N_CHANNELS,
            "n_classes": len(imageset.class_names),
            "class_labels": list(imageset.class_names),
            "skipped_files": [[str(p), str(r)] for p, r in skipped],
        }
        note = (
            f" I skipped {len(skipped)} file(s) that were not readable images; the audit "
            "lists them." if skipped else ""
        )
        message = (
            f"I saved {imageset.n_images} images at {imageset.image_size}x"
            f"{imageset.image_size} pixels across {len(imageset.class_names)} classes "
            f"({', '.join(imageset.class_names)}) to `data/raw/{IMAGE_RAW_FILE}`, with one "
            f"row per image in `data/raw/manifest.csv`.{note} I also wrote `profile.py`, "
            "which measures the images and draws four figures. Run it in the next cell; "
            "nothing about the raw images is changed."
        )
        ctx.display(message)
        return meta

    def _image_synthetic(self, ctx: StageContext, image_size: int):
        q = ctx.questioner
        n_images = int(q.number("How many images?", default=300, minimum=20, maximum=20000,
                                key="data.n_images"))
        n_classes = int(q.number(
            f"How many shape classes? (2 to {MAX_IMAGE_CLASSES})", default=3, minimum=2,
            maximum=MAX_IMAGE_CLASSES, key="data.n_classes",
        ))
        noise = q.number(
            "Background noise (0 = clean shapes, 0.3 = grainy and realistic)?",
            default=0.1, minimum=0.0, maximum=1.0, key="data.noise",
        )
        inject = q.confirm(
            "Inject realistic problems (duplicate images, blank images, an uneven number "
            "per class) so the cleaning stage has work to do?",
            default=True, key="data.inject_quirks",
        )
        cfg = SynthImageConfig(
            n_images=n_images, image_size=image_size, n_classes=n_classes, seed=42,
            noise=float(noise),
            class_imbalance=0.55 if inject else 0.0,
            duplicate_fraction=0.08 if inject else 0.0,
            blank_fraction=0.05 if inject else 0.0,
        )
        return generate_images(cfg), {"source": "synthetic", "synth_config": asdict(cfg)}

    def _image_drive(self, ctx: StageContext, modality, image_size: int):
        q = ctx.questioner
        answer = q.text(
            "Which folder holds your images? It needs one subfolder per class, e.g. "
            "`MyDrive/pets/cats/` and `MyDrive/pets/dogs/`",
            key="data.drive_folder",
        )
        folder = Path(answer.strip())
        if not folder.is_dir():
            raise NotADirectoryError(f"no such folder: {folder}")
        cap = int(q.number(
            "Cap the number of images? (0 means use them all; a cap keeps the first run "
            "quick)", default=0, minimum=0, maximum=100000, key="data.max_images",
        ))
        ctx.display(f"Reading images from **{folder}**; this is the slow part, once.")
        imageset, skipped = modality.load_drive(folder, image_size, cap or None)
        return imageset, skipped, {"source": "drive", "source_path": str(folder)}

    def _image_huggingface(self, ctx: StageContext, modality, image_size: int):
        q = ctx.questioner
        dataset_id = q.text(
            "Which HuggingFace image dataset? Give its id, e.g. `cifar10` or "
            "`beans`", key="data.hf_dataset",
        ).strip()
        cap = int(q.number(
            "Cap the number of images? (0 means use them all)", default=2000, minimum=0,
            maximum=100000, key="data.max_images",
        ))
        ctx.display(f"Downloading **{dataset_id}** from the HuggingFace Hub...")
        loader = self.hf_image_load or modality.load_hf
        imageset = loader(dataset_id, image_size, max_images=cap or None)
        return imageset, {"source": "huggingface", "hf_id": dataset_id}
```

with `from mlagent.synth.images import generate as generate_images` added to the imports.

In `mlagent/stages/clean.py`, add the image constants and branch:

```python
CLEAN_IMAGE_FILE = "data.npz"
CLEAN_IMAGE_REL_PATH = "data/clean/data.npz"
```

`is_complete` picks the file by modality:

```python
    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE) or {}
        image = meta.get("modality") == "image"
        clean_file = CLEAN_IMAGE_FILE if image else CLEAN_FILE
        return (
            (ctx.project.data_clean / clean_file).exists()
            and ctx.project.exists(AUDIT_FILE)
            and bool(meta.get("splits"))
            and (image or bool(meta.get("feature_columns")))
        )
```

`prepare` gains an image path before the tabular one (the audit review, `_collect_steps`, `_ask_splits`, `audit.json` write and the handoff are shared):

```python
    def prepare(self, ctx: StageContext) -> Handoff:
        meta = ctx.project.read_json(META_FILE) or {}
        target = meta.get("target")
        if not target:
            raise RuntimeError("data_meta.json has no target; run the data stage first")
        modality = modality_for(str(meta.get("task_type") or ctx.spec().task_type))
        if modality.name == "image":
            return self._prepare_images(ctx, modality, meta)
        ...  # the existing tabular body, unchanged
```
```python
    def _prepare_images(self, ctx: StageContext, modality, meta: dict) -> Handoff:
        imageset = modality.read(ctx.project.data_raw)
        ctx.teaching().preamble(
            "clean", {"target": meta.get("target"), "n_rows": imageset.n_images}
        )
        skipped = [tuple(pair) for pair in (meta.get("skipped_files") or [])]
        issues = modality.audit(imageset, meta, skipped)
        decisions = self._review_issues(ctx, issues, {
            "n_rows": imageset.n_images,
            "n_cols": len(imageset.class_names),
            "target": meta.get("target"),
        })
        steps = self._collect_steps(decisions)
        splits = self._ask_splits(ctx)

        ctx.project.write_json(
            AUDIT_FILE,
            {"issues": [i.to_dict() for i in issues], "decisions": decisions, "steps": steps},
        )
        (ctx.project.root / CLEAN_PY).write_text(
            modality.render_clean_py(steps), encoding="utf-8"
        )
        meta.update({"splits": splits, "dropped_columns": [], "split_seed": SPLIT_SEED})
        ctx.project.write_json(META_FILE, meta)

        listed = "\n".join(f"- {describe_image_step(s)}" for s in steps) or "- (no changes)"
        ctx.display(
            f"I wrote `clean.py` with {len(steps)} step(s):\n\n{listed}\n\n"
            "Run it in the next cell. It reads `data/raw/data.npz`, writes "
            "`data/clean/data.npz`, and never touches the raw images -- so if you change "
            "your mind you can edit `STEPS` in `clean.py` and run it again."
        )
        return Handoff(stage=self.name, commands=[[CLEAN_PY]],
                       outputs=[CLEAN_IMAGE_REL_PATH, PROFILE_CLEAN_FILE])
```

`_collect_steps` must not treat image steps as column drops; guard its `drop_columns` scan:

```python
        for d in decisions:
            fix = d["fix"]
            if d["approved"] and fix and fix.get("op") == "drop_columns":
                dropped_cols.update(fix["params"].get("columns", []))
```

`debrief` branches the same way:

```python
    def debrief(self, ctx: StageContext) -> None:
        meta = ctx.project.read_json(META_FILE) or {}
        if meta.get("modality") == "image":
            return self._debrief_images(ctx, meta)
        ...  # the existing tabular body, unchanged
```
```python
    def _debrief_images(self, ctx: StageContext, meta: dict) -> None:
        clean_path = ctx.project.data_clean / CLEAN_IMAGE_FILE
        if not clean_path.exists():
            ctx.display(
                "I can't see `data/clean/data.npz` yet. Run the `clean.py` cell, then run "
                "this cell again."
            )
            return
        modality = modality_for(str(meta.get("task_type")))
        cleaned = modality.read(ctx.project.data_clean)
        counts = cleaned.class_counts()
        empty = [name for name, n in counts.items() if n == 0]
        if empty:
            ctx.display(
                f"Cleaning removed every image of {', '.join(empty)}, so those classes have "
                "no images left. Edit `STEPS` in `clean.py` to keep some of them and run "
                "the cell again, or redo the clean stage and decline that fix."
            )
            return
        meta.update({
            "clean_path": CLEAN_IMAGE_REL_PATH,
            "clean_n_rows": cleaned.n_images,
            "clean_n_cols": cleaned.image_size * cleaned.image_size * cleaned.n_channels,
            "feature_columns": [],
            "categorical_columns": [],
            "n_classes": len(cleaned.class_names),
            "class_labels": list(cleaned.class_names),
        })
        ctx.project.write_json(META_FILE, meta)

        profile = ctx.project.read_json(PROFILE_CLEAN_FILE) or {}
        figures = [ctx.project.plots_dir / str(n) for n in (profile.get("figures") or [])]
        before = profile.get("before") or {}
        after = profile.get("after") or {}
        splits = meta.get("splits") or {}
        ctx.display("\n".join([
            "### Cleaning summary",
            "",
            f"- Images: {before.get('n_images', '?')} -> {after.get('n_images', '?')}",
            f"- Classes: {', '.join(f'{k} {v}' for k, v in counts.items())}",
            f"- Steps applied: {len(profile.get('steps') or [])}",
            "",
            f"Splits frozen at train {splits.get('train')}, validation {splits.get('val')}, "
            f"test {splits.get('test')} with seed {meta.get('split_seed', SPLIT_SEED)}, so "
            "every run sees the same images. Next: choosing a model and generating the "
            "[[training pipeline]].",
        ]))
        note = ctx.teaching().debrief(
            "clean_debrief",
            {"before": before, "after": after, "steps": profile.get("steps"),
             "splits": splits},
            figures, fallback="",
        )
        if note:
            ctx.display(note)
```

with `from mlagent.cleaning_images import describe_step as describe_image_step` added to the imports.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_modality.py tests/test_data_stage.py tests/test_clean_stage.py -q
python -m pytest -q
ruff check .
```
Expected: all green, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/modality.py mlagent/stages/data.py mlagent/stages/clean.py tests/conftest.py tests/test_modality.py tests/test_data_stage.py tests/test_clean_stage.py
git commit -m "feat: the image modality record and image paths through the data and clean stages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 8: `image_torch` schema, `data.py` and `model.py`

**Files:**
- Create: `mlagent/templates/image_torch/config_schema.json`, `mlagent/templates/image_torch/data.py`, `mlagent/templates/image_torch/model.py`, `tests/test_template_model_images.py`
- Test: `tests/test_template_model_images.py`

**Interfaces:**
- Consumes: `templates_io.load_schema` / `schema_for` / `default_config` / `validate_config` (`mlagent/templates_io.py:34-58`), the `clean_image_project` fixture (Task 7).
- Produces (used by Tasks 9, 10, 11):
  - `config_schema.json` with `common` keys `model_type` (choice `tiny_cnn`/`small_cnn`/`resnet18`, default `small_cnn`), `epochs` (1-50, default 10), `batch_size` (8-256, default 32), `learning_rate` (1e-5 - 1e-1, default 0.001), `weight_decay` (0 - 0.1, default 0.0001), `early_stopping_patience` (0-50, default 5), `seed` (0-1000000, default 42), `augment` (choice `none`/`basic`, default `basic`); model keys `dropout` for `tiny_cnn`/`small_cnn`, `pretrained` + `freeze_backbone` for `resnet18`
  - `data.py`: `read_meta(project_dir) -> dict`, `load_pair(path) -> tuple`, `stratified_split(labels, splits, seed) -> dict[str, np.ndarray]`, `load_data(project_dir, config) -> dict`, `make_loader(pair, batch_size, shuffle, seed) -> DataLoader`, `augment_batch(x, generator) -> Tensor`, `pick_device() -> str`
  - `load_data` returns `{"train": (x, y), "val": (x, y), "test": (x, y), "classes": list[str], "task_type": "image_classification", "image_size": int, "n_channels": int, "n_train": int, "n_val": int, "n_test": int, "channel_mean": list[float], "channel_std": list[float]}` with `x` a float32 `(N, 3, H, W)` tensor and `y` an int64 `(N,)` tensor
  - `model.py`: `MODEL_TYPES = ("tiny_cnn", "small_cnn", "resnet18")`, `TinyCNN`, `SmallCNN`, `build_model(config: dict, n_classes: int, image_size: int) -> torch.nn.Module`

- [ ] **Step 1: Write the failing test**

Create `tests/test_template_model_images.py`:

```python
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mlagent.templates_io import default_config, load_schema, schema_for, validate_config

TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py")
torch = pytest.importorskip("torch")


def install(project) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    return project.root


def load_module(root: Path, name: str):
    """Import a template file as a standalone module, with the project root on sys.path."""
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(f"img_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"img_{name}"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))


def test_the_schema_has_only_choice_integer_and_number_types():
    schema = load_schema("image_torch")
    for group in (schema["common"], *schema["models"].values()):
        for key, rule in group.items():
            assert rule["type"] in ("choice", "integer", "number"), key


def test_every_family_default_config_validates():
    schema = load_schema("image_torch")
    assert sorted(schema["models"]) == ["resnet18", "small_cnn", "tiny_cnn"]
    for family in schema["models"]:
        flat = schema_for(schema, family)
        config = {**default_config(flat), "model_type": family}
        assert validate_config(config, flat) == []


def test_the_common_keys_and_ranges_match_the_spec():
    common = load_schema("image_torch")["common"]
    assert common["model_type"]["default"] == "small_cnn"
    assert common["model_type"]["choices"] == ["tiny_cnn", "small_cnn", "resnet18"]
    assert (common["epochs"]["min"], common["epochs"]["max"]) == (1, 50)
    assert (common["batch_size"]["min"], common["batch_size"]["max"]) == (8, 256)
    assert common["learning_rate"]["min"] == pytest.approx(1e-5)
    assert common["learning_rate"]["max"] == pytest.approx(1e-1)
    assert (common["weight_decay"]["min"], common["weight_decay"]["max"]) == (0.0, 0.1)
    assert common["augment"]["choices"] == ["none", "basic"]
    assert common["augment"]["default"] == "basic"


def test_the_model_specific_keys_match_the_spec():
    models = load_schema("image_torch")["models"]
    assert set(models["tiny_cnn"]) == {"dropout"}
    assert set(models["small_cnn"]) == {"dropout"}
    assert (models["small_cnn"]["dropout"]["min"], models["small_cnn"]["dropout"]["max"]) == (
        0.0, 0.7
    )
    assert models["resnet18"]["pretrained"]["choices"] == ["imagenet", "none"]
    assert models["resnet18"]["freeze_backbone"]["choices"] == ["yes", "no"]
    assert models["resnet18"]["freeze_backbone"]["default"] == "no"


def test_data_loads_splits_normalises_and_never_overlaps(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    config = {"model_type": "tiny_cnn", "batch_size": 8, "seed": 1, "augment": "basic"}
    loaded = data.load_data(root, config)

    assert loaded["task_type"] == "image_classification"
    assert loaded["classes"] == clean_image_project.read_json("data_meta.json")["class_labels"]
    assert loaded["image_size"] == 32 and loaded["n_channels"] == 3
    total = loaded["n_train"] + loaded["n_val"] + loaded["n_test"]
    assert total == 60
    assert loaded["n_train"] > loaded["n_val"] > 0 and loaded["n_test"] > 0
    x_train, y_train = loaded["train"]
    assert x_train.shape == (loaded["n_train"], 3, 32, 32)
    assert x_train.dtype == torch.float32 and y_train.dtype == torch.int64
    assert float(x_train.min()) > -10.0 and float(x_train.max()) < 10.0
    assert len(set(y_train.tolist())) >= 2


def test_the_split_is_frozen_by_split_seed_not_by_the_model_seed(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    a = data.load_data(root, {"model_type": "tiny_cnn", "seed": 1})
    b = data.load_data(root, {"model_type": "tiny_cnn", "seed": 999})
    assert torch.equal(a["val"][1], b["val"][1])
    assert torch.equal(a["test"][0], b["test"][0])


def test_make_loader_and_augment_batch_keep_the_shape(clean_image_project):
    root = install(clean_image_project)
    data = load_module(root, "data")
    loaded = data.load_data(root, {"model_type": "tiny_cnn", "batch_size": 8})
    loader = data.make_loader(loaded["train"], batch_size=8, shuffle=True, seed=0)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1:] == (3, 32, 32) and batch_y.dtype == torch.int64
    generator = torch.Generator().manual_seed(0)
    augmented = data.augment_batch(batch_x, generator)
    assert augmented.shape == batch_x.shape
    assert augmented.dtype == batch_x.dtype


def test_pick_device_reports_cpu_without_cuda(clean_image_project, monkeypatch):
    root = install(clean_image_project)
    data = load_module(root, "data")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert data.pick_device() == "cpu"


@pytest.mark.parametrize("model_type", ["tiny_cnn", "small_cnn"])
def test_the_cnn_families_forward_to_the_right_shape(clean_image_project, model_type):
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": model_type, "dropout": 0.2, "seed": 3}, n_classes=4, image_size=32
    )
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 4)


def test_the_cnn_families_never_import_torchvision(clean_image_project):
    root = install(clean_image_project)
    source = (root / "model.py").read_text(encoding="utf-8")
    module_level = source.split("def ")[0]
    assert "torchvision" not in module_level
    assert "import torchvision" in source


def test_resnet18_builds_without_pretrained_weights(clean_image_project):
    pytest.importorskip("torchvision")
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": "resnet18", "pretrained": "none", "freeze_backbone": "no", "seed": 3},
        n_classes=3, image_size=32,
    )
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_freeze_backbone_freezes_everything_but_the_final_layer(clean_image_project):
    pytest.importorskip("torchvision")
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    model = model_module.build_model(
        {"model_type": "resnet18", "pretrained": "none", "freeze_backbone": "yes", "seed": 3},
        n_classes=3, image_size=32,
    )
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_build_model_rejects_an_unknown_family(clean_image_project):
    root = install(clean_image_project)
    model_module = load_module(root, "model")
    with pytest.raises(ValueError, match="unknown model_type"):
        model_module.build_model({"model_type": "vit"}, n_classes=3, image_size=32)


def test_the_scripts_follow_the_generated_script_conventions():
    for name in CODE_FILES:
        source = (TEMPLATE / name).read_text(encoding="utf-8")
        assert "import mlagent" not in source and "from mlagent" not in source
        assert "# --- settings ---" in source
        assert "sys.exit(0)" not in source
        assert source.count("\n# --- ") >= 3


def test_data_py_runs_as_a_script_without_arguments(clean_image_project):
    root = install(clean_image_project)
    proc = subprocess.run(
        [sys.executable, "data.py"], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=180,
        # "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
        env={**dict(__import__("os").environ), "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "train" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_template_model_images.py -q
```
Expected: `FileNotFoundError: no template named 'image_torch'` from `load_schema` in the four schema tests, and `FileNotFoundError` copying `mlagent/templates/image_torch/data.py` in the rest.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/templates/image_torch/config_schema.json`:

```json
{
  "common": {
    "model_type": {"type": "choice", "default": "small_cnn",
      "choices": ["tiny_cnn", "small_cnn", "resnet18"],
      "description": "Which network to train."},
    "epochs": {"type": "integer", "default": 10, "min": 1, "max": 50,
      "description": "Number of passes over the training images."},
    "batch_size": {"type": "integer", "default": 32, "min": 8, "max": 256,
      "description": "How many images the network looks at before each weight update."},
    "learning_rate": {"type": "number", "default": 0.001, "min": 0.00001, "max": 0.1,
      "description": "Step size for each weight update; too high and the loss jumps about."},
    "weight_decay": {"type": "number", "default": 0.0001, "min": 0.0, "max": 0.1,
      "description": "Penalty on large weights; higher values regularise."},
    "early_stopping_patience": {"type": "integer", "default": 5, "min": 0, "max": 50,
      "description": "Stop when validation has not improved for this many epochs; 0 disables."},
    "seed": {"type": "integer", "default": 42, "min": 0, "max": 1000000,
      "description": "Random seed for the model (the data split has its own frozen seed)."},
    "augment": {"type": "choice", "default": "basic", "choices": ["none", "basic"],
      "description": "Randomly flip and shift training images so the model sees more variety."}
  },
  "models": {
    "tiny_cnn": {
      "dropout": {"type": "number", "default": 0.3, "min": 0.0, "max": 0.7,
        "description": "Fraction of activations dropped during training; higher regularises."}
    },
    "small_cnn": {
      "dropout": {"type": "number", "default": 0.3, "min": 0.0, "max": 0.7,
        "description": "Fraction of activations dropped during training; higher regularises."}
    },
    "resnet18": {
      "pretrained": {"type": "choice", "default": "imagenet", "choices": ["imagenet", "none"],
        "description": "Start from ImageNet weights (needs the network) or from scratch."},
      "freeze_backbone": {"type": "choice", "default": "no", "choices": ["yes", "no"],
        "description": "Train only the final layer, keeping the pretrained features fixed."}
    }
  }
}
```

Create `mlagent/templates/image_torch/data.py`:

```python
"""Load the cleaned images and split them. Generated by mlagent; safe to edit.

Reads `data_meta.json` (class labels, split fractions, image size) and the clean
`data.npz` / `manifest.csv` pair. Pixels are scaled to float32 [0, 1] and normalised per
channel using the training split's own mean and standard deviation, so validation and
test never see a statistic computed from themselves.

The train/val/test split is seeded from `data_meta.json`'s `split_seed` (default 42),
never from `config.json`, so the held-out test split is the same for every run and a
checkpoint's validation numbers stay comparable across tuning. `config.json`'s `seed`
only seeds the model and the augmentation (see `model.py` and `augment_batch`).

Run `python data.py` on its own to print the split sizes without training anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- settings ---
SCRIPT_NAME = "data.py"
PROJECT_DIR = Path(".")
META_FILE = "data_meta.json"
DEFAULT_SPLIT_SEED = 42
MAX_SHIFT_FRACTION = 0.1   # "basic" augmentation shifts by up to this share of the width
EPS = 1e-6


# --- loading ---
def read_meta(project_dir: Path) -> dict:
    return json.loads((project_dir / META_FILE).read_text(encoding="utf-8"))


def load_pair(path: Path):
    """(images uint8 (N, H, W, 3), labels int64 (N,), class_names) from one .npz."""
    with np.load(path) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]
    return images, labels, class_names


# --- splitting ---
def stratified_split(labels: np.ndarray, splits: dict, seed: int) -> dict[str, np.ndarray]:
    """Index arrays for train/val/test, each class split in the same proportions."""
    rng = np.random.default_rng(int(seed))
    train_frac = float(splits["train"])
    val_frac = float(splits["val"])
    chosen: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        chosen["train"].extend(idx[:n_train].tolist())
        chosen["val"].extend(idx[n_train:n_train + n_val].tolist())
        chosen["test"].extend(idx[n_train + n_val:].tolist())
    return {name: np.array(sorted(values), dtype=np.int64) for name, values in chosen.items()}


# --- tensors ---
def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) in [0, 1]."""
    array = np.asarray(images, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3)).clamp_min(EPS)
    return mean, std


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def load_data(project_dir: Path | str, config: dict) -> dict:
    project_dir = Path(project_dir)
    meta = read_meta(project_dir)
    path = project_dir / str(meta.get("clean_path") or "data/clean/data.npz")
    images, labels, class_names = load_pair(path)
    class_names = [str(c) for c in (meta.get("class_labels") or class_names)]
    splits = meta["splits"]
    seed = int(meta.get("split_seed", DEFAULT_SPLIT_SEED))
    parts = stratified_split(labels, splits, seed)

    x_all = to_tensor(images)
    y_all = torch.from_numpy(labels).to(torch.int64)
    mean, std = channel_stats(x_all[parts["train"]])
    data: dict = {}
    for name in ("train", "val", "test"):
        idx = torch.from_numpy(parts[name])
        data[name] = (normalise(x_all[idx], mean, std), y_all[idx])
        data[f"n_{name}"] = int(idx.numel())
    data.update({
        "classes": class_names,
        "task_type": str(meta.get("task_type", "image_classification")),
        "image_size": int(x_all.shape[2]),
        "n_channels": int(x_all.shape[1]),
        "channel_mean": [round(float(v), 6) for v in mean.tolist()],
        "channel_std": [round(float(v), 6) for v in std.tolist()],
    })
    validate(data)
    return data


# --- loaders and augmentation ---
def make_loader(pair, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    x, y = pair
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(x, y), batch_size=max(1, int(batch_size)), shuffle=bool(shuffle),
        generator=generator if shuffle else None, drop_last=False,
    )


def augment_batch(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random horizontal flip plus a small random shift, in plain tensor ops.

    No torchvision: the two cheap CNN families must run wherever torch does.
    """
    flip = torch.rand(x.shape[0], generator=generator) < 0.5
    out = x.clone()
    out[flip] = torch.flip(out[flip], dims=[3])
    limit = max(1, int(round(MAX_SHIFT_FRACTION * x.shape[3])))
    shifts = torch.randint(-limit, limit + 1, (2,), generator=generator).tolist()
    return torch.roll(out, shifts=(int(shifts[0]), int(shifts[1])), dims=(2, 3))


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- validation ---
def validate(data: dict) -> None:
    """Raise ValueError if the splits are unusable: empty, overlapping, or one class."""
    for name in ("train", "val", "test"):
        if data[f"n_{name}"] == 0:
            raise ValueError(f"the {name} split is empty; adjust the split fractions")
    if len(torch.unique(data["train"][1])) < 2:
        raise ValueError("training split contains fewer than 2 classes")
    if len(data["classes"]) < 2:
        raise ValueError("data_meta.json lists fewer than 2 classes")


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the image splits without training.")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
    config_path = project_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    data = load_data(project_dir, config)
    print(
        f"train {data['n_train']} / val {data['n_val']} / test {data['n_test']} images at "
        f"{data['image_size']}px across {len(data['classes'])} classes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

Create `mlagent/templates/image_torch/model.py`:

```python
"""Build the network. Generated by mlagent; safe to edit.

Three families, all returning a plain `torch.nn.Module` that maps a (N, 3, H, W) batch to
(N, n_classes) logits:

- tiny_cnn:  two convolutional blocks. Fast, and enough for simple shapes.
- small_cnn: three convolutional blocks with batch norm and dropout.
- resnet18:  torchvision's ResNet-18, optionally starting from ImageNet weights.
             `torchvision` is imported *inside* the builder, never at module scope, so a
             tiny_cnn or small_cnn run never needs it installed.
"""

from __future__ import annotations

import torch
from torch import nn

# --- settings ---
MODEL_TYPES = ("tiny_cnn", "small_cnn", "resnet18")
RESNET_MIN_SIZE = 32


# --- the two hand-built CNNs ---
def _block(in_channels: int, out_channels: int, batch_norm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return layers


class TinyCNN(nn.Module):
    """Two convolutional blocks, then a linear head."""

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            *_block(3, 16, batch_norm=False),
            *_block(16, 32, batch_norm=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(float(dropout)),
                                  nn.Linear(32, int(n_classes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


class SmallCNN(nn.Module):
    """Three convolutional blocks with batch norm, then a dropout head."""

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            *_block(3, 32, batch_norm=True),
            *_block(32, 64, batch_norm=True),
            *_block(64, 128, batch_norm=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(float(dropout)),
                                  nn.Linear(128, int(n_classes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


# --- transfer learning ---
def _resnet18(config: dict, n_classes: int, image_size: int) -> nn.Module:
    """torchvision's ResNet-18 with a fresh final layer.

    The import lives here so tiny_cnn and small_cnn never require torchvision.
    """
    import torchvision

    if int(image_size) < RESNET_MIN_SIZE:
        raise ValueError(f"resnet18 needs images of at least {RESNET_MIN_SIZE}px")
    pretrained = str(config.get("pretrained", "imagenet")) == "imagenet"
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    if str(config.get("freeze_backbone", "no")) == "yes":
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, int(n_classes))
    return model


# --- building each family ---
def build_model(config: dict, n_classes: int, image_size: int) -> nn.Module:
    model_type = str(config.get("model_type", "small_cnn"))
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")
    torch.manual_seed(int(config.get("seed", 42)))
    dropout = float(config.get("dropout", 0.3))
    if model_type == "tiny_cnn":
        return TinyCNN(n_classes, dropout)
    if model_type == "small_cnn":
        return SmallCNN(n_classes, dropout)
    return _resnet18(config, n_classes, image_size)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_template_model_images.py -q && ruff check .
```
Expected: 16 passed (the two `resnet18` tests skip if `torchvision` is missing), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates/image_torch/config_schema.json mlagent/templates/image_torch/data.py mlagent/templates/image_torch/model.py tests/test_template_model_images.py
git commit -m "feat: the image_torch schema, data loader and the tiny/small CNN and ResNet-18 builders

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 9: `image_torch` `train.py` and `evaluate.py`, run for real

**Files:**
- Create: `mlagent/templates/image_torch/train.py`, `mlagent/templates/image_torch/evaluate.py`, `tests/test_template_evaluate_images.py`
- Test: `tests/test_template_evaluate_images.py`

**Interfaces:**
- Consumes: `data.load_data` / `make_loader` / `augment_batch` / `pick_device` and `model.build_model` from Task 8; `captions.CAPTIONS["misclassified"]` etc. from Task 5.
- Produces (used by Tasks 10, 11, 13):
  - `train.py`: `SCRIPT_NAME = "train.py"`, `CHECKPOINT = Path("checkpoints") / "best.pt"`, `CHECKPOINT_REL = "checkpoints/best.pt"`, `CURVES_FIGURE = Path("plots") / "training_curves.png"`, `empty_metrics() -> dict`, `train(project_dir) -> dict`, `main(argv) -> int`
  - `metrics.json` keys: every key `tabular_sklearn/train.py:86-105` writes, plus `"device"` (`"cuda"` / `"cpu"`) and `"checkpoint"` (`"checkpoints/best.pt"`, `None` on failure)
  - `evaluate.py`: `SCRIPT_NAME = "evaluate.py"`, `DEFAULT_CHECKPOINT = "checkpoints/best.pt"`, `HIGHER_IS_BETTER`, `compute_metric(name, y_true, y_pred)`, `evaluate_split(model, pair, device, metric, n_classes, batch_size) -> dict`, `eval_record(...)`, `best_checkpoint(project_dir)`, `save_figures(record, plots_dir, split) -> list[str]`
  - `eval_{split}.json` keys: `split`, `task_type`, `metric`, `value`, `loss`, `classes`, `y_true`, `y_pred`, `y_proba`, `started_at`, `checkpoint`, `run_id`, `figures`
  - figures `{split}_confusion.png`, `{split}_per_class.png`, `{split}_misclassified.png`

- [ ] **Step 1: Write the failing test**

Create `tests/test_template_evaluate_images.py`:

```python
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mlagent import captions

TEMPLATE = Path("mlagent/templates/image_torch").resolve()
CODE_FILES = ("data.py", "model.py", "train.py", "evaluate.py")
pytest.importorskip("torch")

TINY = {"model_type": "tiny_cnn", "epochs": 2, "batch_size": 16, "learning_rate": 0.01,
        "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1, "augment": "basic",
        "dropout": 0.1}
SMALL = {**TINY, "model_type": "small_cnn", "epochs": 1, "augment": "none"}
RESNET = {"model_type": "resnet18", "epochs": 1, "batch_size": 16, "learning_rate": 0.01,
          "weight_decay": 0.0001, "early_stopping_patience": 0, "seed": 1, "augment": "none",
          "pretrained": "none", "freeze_backbone": "no"}
BLOWN_UP = {**TINY, "learning_rate": 0.1, "epochs": 3}

# "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
CPU_ONLY = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}


def install(project, config) -> Path:
    for name in CODE_FILES:
        shutil.copy(TEMPLATE / name, project.root / name)
    project.write_json("config.json", config)
    return project.root


def run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=600, env=CPU_ONLY,
    )


def train(project, config) -> Path:
    root = install(project, config)
    proc = run(root, "train.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root


def load_evaluate_module(root: Path):
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("img_evaluate", root / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["img_evaluate"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))


def test_train_writes_the_full_metrics_contract(clean_image_project):
    root = train(clean_image_project, TINY)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    for key in ("status", "started_at", "model_type", "task_type", "metric",
                "higher_is_better", "config", "classes", "n_train", "n_val", "n_test",
                "epochs", "best_epoch", "best_val_metric", "stopped_early", "error",
                "seconds", "seconds_per_epoch", "device", "checkpoint"):
        assert key in metrics, key
    assert metrics["status"] == "done"
    assert metrics["model_type"] == "tiny_cnn"
    assert metrics["task_type"] == "image_classification"
    assert metrics["metric"] == "accuracy" and metrics["higher_is_better"] is True
    assert metrics["device"] == "cpu"
    assert metrics["checkpoint"] == "checkpoints/best.pt"
    assert len(metrics["epochs"]) == 2
    for row in metrics["epochs"]:
        assert set(row) >= {"epoch", "train_loss", "val_loss", "train_metric", "val_metric"}
    assert 1 <= metrics["best_epoch"] <= 2
    assert (root / "checkpoints" / "best.pt").exists()
    assert (root / "plots" / "training_curves.png").exists()


def test_evaluate_val_writes_the_record_and_all_three_figures(clean_image_project):
    root = train(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_val.json").read_text(encoding="utf-8"))
    assert record["split"] == "val" and record["task_type"] == "image_classification"
    assert record["metric"] == "accuracy" and 0.0 <= record["value"] <= 1.0
    assert record["checkpoint"] == "checkpoints/best.pt"
    assert record["run_id"] is None
    assert len(record["y_true"]) == len(record["y_pred"]) == len(record["y_proba"])
    assert len(record["y_proba"][0]) == 3
    assert set(record["figures"]) == {"val_confusion.png", "val_per_class.png",
                                      "val_misclassified.png"}
    for name in record["figures"]:
        assert (root / "plots" / name).exists()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert record["started_at"] == metrics["started_at"]


def test_evaluate_prints_a_caption_under_the_misclassified_grid(clean_image_project):
    root = train(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    stripped = captions.CAPTIONS["misclassified"].replace("[[", "").replace("]]", "")
    assert proc.stdout.count("How to read this: " + stripped) == 1


def test_template_captions_match_mlagent_captions(clean_image_project):
    root = install(clean_image_project, TINY)
    module = load_evaluate_module(root)
    kinds = ["confusion", "per_class", "misclassified"]
    assert module.CAPTIONS == {k: captions.CAPTIONS[k] for k in kinds}


def test_small_cnn_trains_and_evaluates(clean_image_project):
    root = train(clean_image_project, SMALL)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == "small_cnn"
    assert run(root, "evaluate.py").returncode == 0


def test_resnet18_without_pretrained_weights_trains_and_evaluates(clean_image_project):
    pytest.importorskip("torchvision")
    root = train(clean_image_project, RESNET)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "done" and metrics["model_type"] == "resnet18"
    assert metrics["config"]["pretrained"] == "none"
    assert run(root, "evaluate.py").returncode == 0


def test_evaluate_test_uses_the_best_runs_checkpoint(clean_image_project):
    root = train(clean_image_project, TINY)
    shutil.copy(root / "checkpoints" / "best.pt", root / "checkpoints" / "run1.pt")
    (root / "runs.jsonl").write_text(
        json.dumps({"run_id": 1, "status": "done", "best_val_metric": 0.7,
                    "checkpoint": "checkpoints/run1.pt"}) + "\n"
        + json.dumps({"run_id": 2, "status": "failed", "best_val_metric": 0.99,
                      "checkpoint": None}) + "\n",
        encoding="utf-8",
    )
    proc = run(root, "evaluate.py", "--split", "test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((root / "eval_test.json").read_text(encoding="utf-8"))
    assert record["checkpoint"] == "checkpoints/run1.pt"
    assert record["run_id"] == 1
    assert "test_confusion.png" in record["figures"]


def test_a_non_finite_loss_is_recorded_as_a_failed_run(clean_image_project):
    root = install(clean_image_project, BLOWN_UP)
    (root / "data.py").write_text(
        (root / "data.py").read_text(encoding="utf-8").replace(
            "    validate(data)\n    return data",
            "    validate(data)\n"
            "    data['train'] = (data['train'][0] * float('inf'), data['train'][1])\n"
            "    return data",
        ),
        encoding="utf-8",
    )
    proc = run(root, "train.py")
    assert proc.returncode == 1
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["error"]
    assert metrics["checkpoint"] is None


def test_evaluate_without_a_checkpoint_fails_clearly(clean_image_project):
    root = install(clean_image_project, TINY)
    proc = run(root, "evaluate.py")
    assert proc.returncode == 1
    assert "checkpoint" in (proc.stdout + proc.stderr).lower()


def test_the_scripts_follow_the_generated_script_conventions():
    for name in ("train.py", "evaluate.py"):
        source = (TEMPLATE / name).read_text(encoding="utf-8")
        assert "import mlagent" not in source and "from mlagent" not in source
        assert "# --- settings ---" in source
        assert "def cli_argv()" in source
        assert "sys.exit(0)" not in source.split("if __name__")[0]
        assert source.count("\n# --- ") >= 4
        assert "torchvision" not in source
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_template_evaluate_images.py -q
```
Expected: `FileNotFoundError: ...image_torch/train.py` from `install()` in every test.

- [ ] **Step 3: Write minimal implementation**

Create `mlagent/templates/image_torch/train.py`:

```python
"""Train the network, evaluate per epoch, write metrics.json. Generated by mlagent; safe
to edit.

Usage (run from the project folder):
  python train.py                one full training run

After training, run `evaluate.py` to score the saved checkpoint and draw its figures.
The checkpoint holds a `state_dict` plus the config and class names, so `evaluate.py` can
rebuild exactly the same network.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from data import augment_batch, load_data, make_loader, pick_device
from evaluate import (
    HIGHER_IS_BETTER,
    INK_2,
    MUTED,
    SERIES,
    SURFACE,
    evaluate_split,
    frame,
    metric_for,
)
from model import build_model
from torch import nn

# --- settings ---
SCRIPT_NAME = "train.py"
CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.json"
CHECKPOINT = Path("checkpoints") / "best.pt"
CHECKPOINT_REL = "checkpoints/best.pt"
CURVES_FIGURE = Path("plots") / "training_curves.png"

# --- captions ---
# Byte-identical to the matching entry in mlagent/captions.py; kept here too since this
# script never depends on the mlagent package.
CAPTIONS = {
    "training_curves": (
        "Left: [[loss]] per [[epoch]] for the training and validation splits. Right: the same "
        "for your chosen metric. Training loss falling while validation loss rises is "
        "[[overfitting]]; both flat is a [[plateau]]."
    ),
}


# --- small helpers ---
def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    """Atomic write so a reader polling the file never sees a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
    # Windows: os.replace can fail with PermissionError if a poller has the target open; retry.
    attempts = 20
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                Path(tmp).unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def empty_metrics() -> dict:
    """A metrics dict with all required keys and placeholder values."""
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
        "device": None,
        "checkpoint": None,
    }


# --- the live training curve ---
_HANDLE = None
_CAPTION_PRINTED = False


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
        frame(ax, title)
        ax.set_xlabel("epoch", color=INK_2, fontsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    if not epochs:
        ax_loss.text(0.5, 0.5, "no epochs yet", ha="center", va="center", color=MUTED,
                     transform=ax_loss.transAxes)
    fig.tight_layout()
    return fig


def redraw(fig) -> None:
    """Update one output area in place, so the curve animates instead of stacking up."""
    global _HANDLE, _CAPTION_PRINTED
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            if _HANDLE is None:
                _HANDLE = display(fig, display_id=True)
            else:
                _HANDLE.update(fig)
    if not _CAPTION_PRINTED:
        _CAPTION_PRINTED = True
        caption = CAPTIONS.get("training_curves", "")
        print("How to read this: " + caption.replace("[[", "").replace("]]", ""))


def draw_curves(project_dir: Path, epochs: list[dict], metric: str) -> None:
    fig = training_curves(epochs, metric)
    path = project_dir / CURVES_FIGURE
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    redraw(fig)
    plt.close(fig)


# --- training ---
def fit_epoch(model, loader, optimiser, loss_fn, device: str, augment: bool,
              generator) -> None:
    model.train()
    for batch_x, batch_y in loader:
        if augment:
            batch_x = augment_batch(batch_x, generator)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimiser.step()


def train(project_dir: Path) -> dict:
    config = read_json(project_dir / CONFIG_FILE, default={}) or {}
    data = load_data(project_dir, config)
    task_type = data["task_type"]
    metric = metric_for(project_dir, task_type)
    higher = HIGHER_IS_BETTER[metric]
    classes = data["classes"]
    n_classes = len(classes)
    epochs = int(config.get("epochs", 10))
    patience = int(config.get("early_stopping_patience", 0))
    batch_size = int(config.get("batch_size", 32))
    seed = int(config.get("seed", 42))
    device = pick_device()
    metrics_path = project_dir / METRICS_FILE

    metrics = empty_metrics()
    metrics["started_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    metrics.update({
        "model_type": str(config.get("model_type", "small_cnn")),
        "task_type": task_type,
        "metric": metric,
        "higher_is_better": higher,
        "config": config,
        "classes": classes,
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "n_test": data["n_test"],
        "device": device,
    })

    def save() -> None:
        write_json(metrics_path, metrics)

    save()
    torch.manual_seed(seed)
    model = build_model(config, n_classes, data["image_size"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    train_loader = make_loader(data["train"], batch_size, shuffle=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    augment = str(config.get("augment", "basic")) != "none"

    best_state = None
    best_value = None
    since_best = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        fit_epoch(model, train_loader, optimiser, loss_fn, device, augment, generator)
        tr = evaluate_split(model, data["train"], device, metric, n_classes, batch_size)
        va = evaluate_split(model, data["val"], device, metric, n_classes, batch_size)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_metric": tr["value"],
            "val_metric": va["value"],
            "seconds": round(time.time() - t0, 3),
        }
        metrics["epochs"].append(row)
        if not (math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"])):
            metrics["status"] = "failed"
            metrics["error"] = f"non-finite loss at epoch {epoch}"
            save()
            return metrics
        improved = best_value is None or (
            row["val_metric"] > best_value if higher else row["val_metric"] < best_value
        )
        if improved:
            best_value = row["val_metric"]
            metrics["best_epoch"] = epoch
            metrics["best_val_metric"] = best_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        save()
        draw_curves(project_dir, metrics["epochs"], metric)
        print(
            f"epoch {epoch}/{epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_{metric}={row['val_metric']:.4f}",
            flush=True,
        )
        if patience and since_best >= patience:
            metrics["stopped_early"] = True
            break

    elapsed = time.time() - started
    metrics["seconds"] = round(elapsed, 3)
    metrics["seconds_per_epoch"] = round(elapsed / max(1, len(metrics["epochs"])), 4)

    (project_dir / CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state or model.state_dict(), "config": config,
         "classes": classes, "image_size": data["image_size"]},
        project_dir / CHECKPOINT,
    )
    metrics["checkpoint"] = CHECKPOINT_REL
    metrics["status"] = "done"
    save()
    return metrics


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the network and record every epoch.")
    parser.add_argument("--project", default=".", help="project folder (default: cwd)")
    args = parser.parse_args(argv)
    project_dir = Path(args.project).resolve()
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


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

Create `mlagent/templates/image_torch/evaluate.py`:

```python
"""Score a saved checkpoint on one split and draw its evaluation figures.

Run it from the project folder:

    python evaluate.py                                  the validation split, latest checkpoint
    python evaluate.py --split test                     the test split, best run's checkpoint
    python evaluate.py --split test --checkpoint P      a specific checkpoint

Writes `eval_{split}.json` (the numbers plus the raw predictions, so the figures can be
redrawn without retraining) and PNGs into `plots/`. `train.py` imports `evaluate_split`
from here so the two scripts can never disagree about what a metric means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import load_data, make_loader, pick_device
from matplotlib.colors import LinearSegmentedColormap
from model import build_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

# --- settings ---
SCRIPT_NAME = "evaluate.py"
PROJECT_DIR = Path(".")
CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
RUNS_FILE = "runs.jsonl"
METRICS_FILE = "metrics.json"
META_FILE = "data_meta.json"
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
SPLITS = ("val", "test")
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False}
DEFAULT_METRIC = {"image_classification": "accuracy"}
EVAL_BATCH_SIZE = 64
MAX_MISCLASSIFIED = 12

# --- captions ---
# Byte-identical to the matching entries in mlagent/captions.py
# (tests/test_template_evaluate_images.py checks this); kept here too since this script
# never depends on the mlagent package.
CAPTIONS = {
    "confusion": (
        "Rows are the true label, columns are what the model predicted, so the diagonal is "
        "correct. A bright off-diagonal cell names the two classes the model keeps confusing."
    ),
    "per_class": (
        "Precision and recall for each class. Precision is how often a prediction of that "
        "class is right; recall is how much of that class the model finds. Small classes with "
        "low bars are the ones to fix."
    ),
    "misclassified": (
        "The validation images the model got most confidently wrong, each labelled "
        "true -> predicted. The same confusion repeated is a fixable labelling or "
        "[[class imbalance]] problem; a scatter of unrelated one-offs is just noise."
    ),
}

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
    return str(spec.get("metric") or DEFAULT_METRIC.get(task_type, "accuracy"))


# --- scoring ---
def compute_metric(name: str, y_true, y_pred) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"unknown metric {name!r} for image classification")


def evaluate_split(model, pair, device: str, metric: str, n_classes: int,
                   batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Loss, metric and raw predictions for one (x, y) pair, in eval mode without grads."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loader = make_loader(pair, batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    n = 0
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            total_loss += float(loss_fn(logits, batch_y).item())
            n += int(batch_y.numel())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            truths.append(batch_y.cpu().numpy())
    proba = np.concatenate(probabilities) if probabilities else np.zeros((0, n_classes))
    y_true = np.concatenate(truths) if truths else np.zeros(0, dtype=np.int64)
    y_pred = proba.argmax(axis=1) if proba.size else np.zeros(0, dtype=np.int64)
    return {
        "loss": total_loss / max(1, n),
        "value": compute_metric(metric, y_true, y_pred) if n else 0.0,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": proba.tolist(),
    }


def eval_record(split: str, task_type: str, metric: str, classes, ev: dict,
                started_at: str | None) -> dict:
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
        "started_at": started_at,
    }


# --- choosing a checkpoint ---
def best_checkpoint(project_dir: Path) -> tuple[str | None, int | None]:
    """The checkpoint of the best finished run, and its run id."""
    task_type = (read_json(project_dir / META_FILE, default={}) or {}).get(
        "task_type", "image_classification"
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


def load_checkpoint(project_dir: Path, checkpoint_path: Path, config: dict, data: dict):
    """Rebuild the network the checkpoint was saved from and load its weights."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = payload.get("config") or config
    n_classes = len(payload.get("classes") or data["classes"])
    image_size = int(payload.get("image_size") or data["image_size"])
    model = build_model(saved_config, n_classes, image_size)
    model.load_state_dict(payload["state_dict"])
    return model


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


def show(fig, kind: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            display(fig)
    caption = CAPTIONS.get(kind, "")
    if caption:
        print("How to read this: " + caption.replace("[[", "").replace("]]", ""))


def save(fig, plots_dir: Path, name: str, kind: str) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{name}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    show(fig, kind)
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


def misclassified_figure(images, y_true, y_pred, y_proba, labels: list[str]):
    """The wrong predictions the model was most confident about, worst first."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    proba = np.asarray(y_proba, dtype=float)
    wrong = np.flatnonzero(truth != predicted)
    if wrong.size:
        confidence = proba[wrong, predicted[wrong]] if proba.size else np.zeros(wrong.size)
        wrong = wrong[np.argsort(-confidence)][:MAX_MISCLASSIFIED]
    ncols = 4
    nrows = max(1, int(np.ceil(max(1, wrong.size) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)
    if wrong.size == 0:
        ax = axes[0][0]
        ax.set_visible(True)
        ax.text(0.5, 0.5, "nothing misclassified", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    for i, index in enumerate(wrong.tolist()):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        ax.imshow(images[int(index)])
        ax.set_title(f"{labels[int(truth[index])]} -> {labels[int(predicted[index])]}",
                     color=INK, fontsize=8, loc="left")
    fig.suptitle("Worst mistakes", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def displayable(x: torch.Tensor) -> np.ndarray:
    """Undo the per-channel normalisation enough to look at: (N, H, W, 3) in [0, 1]."""
    array = x.permute(0, 2, 3, 1).cpu().numpy()
    low = array.min(axis=(1, 2, 3), keepdims=True)
    high = array.max(axis=(1, 2, 3), keepdims=True)
    return np.clip((array - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def save_figures(record: dict, plots_dir: Path, split: str, images) -> list[str]:
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    labels = [str(c) for c in (record.get("classes") or sorted({*y_true, *y_pred}))]
    return [
        save(confusion_figure(y_true, y_pred, labels), plots_dir, f"{split}_confusion",
             "confusion"),
        save(per_class_figure(y_true, y_pred, labels), plots_dir, f"{split}_per_class",
             "per_class"),
        save(misclassified_figure(images, y_true, y_pred, record["y_proba"], labels),
             plots_dir, f"{split}_misclassified", "misclassified"),
    ]


# --- command line ---
def cli_argv() -> list[str]:
    """Arguments when run as a script or via `%run`; nothing under a bare kernel cell.

    A Jupyter/Colab kernel sets `sys.argv[0]` to its own launcher (e.g.
    `ipykernel_launcher.py` or Colab's `colab_kernel_launcher.py`), which also ends in
    `.py`, so checking the extension alone would treat the kernel's own
    `-f <connection-file>.json` flags as ours and crash `argparse`.
    """
    name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return sys.argv[1:] if name == SCRIPT_NAME.lower() else []


def main(argv: list[str] | None = None) -> int:
    # A caption below may contain a non-ASCII character; on Windows a piped stdout
    # otherwise defaults to the console codepage and mangles it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    n_classes = len(data["classes"])
    device = pick_device()
    model = load_checkpoint(project_dir, checkpoint_path, config, data).to(device)

    metrics = read_json(project_dir / METRICS_FILE, default=None)
    started_at = metrics.get("started_at") if isinstance(metrics, dict) else None

    pair = data[args.split]
    ev = evaluate_split(model, pair, device, metric, n_classes)
    record = eval_record(args.split, task_type, metric, data["classes"], ev, started_at)
    record["checkpoint"] = Path(checkpoint).as_posix()
    record["run_id"] = run_id
    record["figures"] = save_figures(record, project_dir / "plots", args.split,
                                     displayable(pair[0]))
    out = project_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{args.split} {metric}={record['value']:.4f} loss={record['loss']:.4f} "
        f"({len(record['y_true'])} images, checkpoint {record['checkpoint']}) -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main(cli_argv())
    if code:
        sys.exit(code)
```

Also add `"checkpoint": None` to `mlagent/templates/tabular_sklearn/train.py`'s `empty_metrics()` (after `"seconds_per_epoch": None,` at line 104), and set it just before `metrics["status"] = "done"` (line 267):

```python
    metrics["checkpoint"] = CHECKPOINT.as_posix()
```

Add the matching assertion to `tests/test_template_train.py`:

```python
def test_tabular_train_records_its_checkpoint_path(clean_project):
    root = train(clean_project)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["checkpoint"] == "checkpoints/best.joblib"
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_template_evaluate_images.py tests/test_template_train.py -q
python -m pytest -W error::DeprecationWarning tests/test_template_evaluate_images.py -q
ruff check .
```
Expected: all pass in under roughly two minutes for the image file, no deprecation warnings, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add mlagent/templates/image_torch/train.py mlagent/templates/image_torch/evaluate.py mlagent/templates/tabular_sklearn/train.py tests/test_template_evaluate_images.py tests/test_template_train.py
git commit -m "feat: image_torch train.py and evaluate.py writing the same metrics and eval contracts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 10: Codegen for images, the model-choice teaching text, and suffix-aware archiving

**Files:**
- Create: `mlagent/prompts/teaching/model_choices_images.md`
- Modify: `mlagent/stages/codegen.py:30-38`, `mlagent/stages/codegen.py:57-104`, `mlagent/stages/codegen.py:141-210`, `mlagent/runs.py:53-68`, `mlagent/runs.py:135-145`, `mlagent/stages/train.py:44-48`
- Test: `tests/test_codegen_stage.py`, `tests/test_runs.py`, `tests/test_prompts_io.py`, `tests/test_train_stage.py`

**Interfaces:**
- Consumes: `modality_for` (Task 1), `templates_io.model_types` (`mlagent/templates_io.py:38`), the `image_torch` schema (Task 8), `clean_image_project` (Task 7), `metrics["checkpoint"]` (Task 9).
- Produces (used by Tasks 11, 13):
  - `codegen.MODEL_LABELS_BY_FAMILY: dict[str, dict[str, str]]`, `codegen.labels_for(family: str) -> dict[str, str]`, `codegen.label_for(family: str, code: str) -> str`, `codegen.MODEL_LABELS` (the tabular map, kept for back-compat)
  - `codegen.fallback_model(meta: dict, family: str = "tabular_sklearn") -> str`
  - `codegen.check_data(meta: dict, project_root: Path) -> list[str]` handling `modality == "image"`
  - `runs.archive_run(project: Project, run_id: int, checkpoint: str | None = None) -> list[Path]`
  - `mlagent/prompts/teaching/model_choices_images.md`
  - `TrainStage.prepare`'s preamble text, now modality-aware: `checkpoints/model.pt` and a
    run-time device sentence for images, `checkpoints/best.joblib` and the CPU sentence for
    tabular (unchanged)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_codegen_stage.py`:

```python
def test_labels_for_each_family_and_the_back_compat_alias():
    from mlagent.stages.codegen import MODEL_LABELS, label_for, labels_for

    assert set(labels_for("image_torch").values()) == {"tiny_cnn", "small_cnn", "resnet18"}
    assert set(labels_for("tabular_sklearn").values()) == {
        "linear", "random_forest", "gradient_boosting"
    }
    assert MODEL_LABELS == labels_for("tabular_sklearn")
    assert label_for("image_torch", "resnet18") == "Pretrained ResNet-18"
    assert label_for("tabular_sklearn", "linear") == "Linear / logistic regression"


def test_the_image_labels_cover_every_model_type_in_the_schema():
    from mlagent.stages.codegen import labels_for
    from mlagent.templates_io import load_schema, model_types

    assert sorted(labels_for("image_torch").values()) == sorted(
        model_types(load_schema("image_torch"))
    )


def test_fallback_model_is_family_aware():
    from mlagent.stages.codegen import fallback_model

    assert fallback_model({"clean_n_rows": 100}, "tabular_sklearn") == "linear"
    assert fallback_model({"clean_n_rows": 5000}, "tabular_sklearn") == "gradient_boosting"
    assert fallback_model({"clean_n_rows": 100}, "image_torch") == "small_cnn"


def test_check_data_accepts_a_clean_image_project(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    assert check_data(project.read_json("data_meta.json"), project.root) == []


def test_check_data_reports_a_missing_image_file(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    (project.data_clean / "data.npz").unlink()
    problems = check_data(project.read_json("data_meta.json"), project.root)
    assert problems and "data.npz" in problems[0]


def test_check_data_reports_too_few_image_classes(clean_image_project):
    from mlagent.stages.codegen import check_data

    project = clean_image_project
    meta = {**project.read_json("data_meta.json"), "n_classes": 1}
    problems = check_data(meta, project.root)
    assert any("2 classes" in p for p in problems)


def test_codegen_on_an_image_project_writes_the_image_template_and_config(clean_image_project):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.stages.codegen import CodegenStage
    from mlagent.templates_io import load_schema, schema_for, validate_config
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    project = clean_image_project
    shown: list[str] = []
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner({"codegen.model_type": "Small CNN"},
                                  ScriptedQuestioner(["y"])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    stage = CodegenStage()
    stage.prepare(ctx)

    for name in ("data.py", "model.py", "train.py", "evaluate.py"):
        assert project.exists(name)
    source = (project.root / "model.py").read_text(encoding="utf-8")
    assert "SmallCNN" in source
    config = project.read_json("config.json")
    assert config["model_type"] == "small_cnn"
    flat = schema_for(load_schema("image_torch"), "small_cnn")
    assert validate_config(config, flat) == []
    assert stage.is_complete(ctx) is True
    assert any("CNN" in text for text in shown)


def test_the_image_model_choices_material_is_shown(clean_image_project):
    from mlagent.llm import FakeLLM
    from mlagent.stages.base import StageContext
    from mlagent.stages.codegen import CodegenStage
    from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

    project = clean_image_project
    shown: list[str] = []
    ctx = StageContext(
        project=project, llm=FakeLLM([]),
        questioner=FormQuestioner({"codegen.model_type": "Tiny CNN"},
                                  ScriptedQuestioner(["y"])),
        explainer=None, display=shown.append,
        display_figure=lambda path, caption="": None,
    )
    CodegenStage().prepare(ctx)
    joined = "\n".join(shown)
    assert "ResNet" in joined and "transfer learning" in joined.lower()
```

Add to `tests/test_prompts_io.py`:

```python
def test_the_image_model_choices_material_trims_per_level():
    from mlagent.teaching import material

    for level in ("beginner", "intermediate", "expert"):
        text = material("model_choices_images", level)
        assert text.strip()
        assert "<!--" not in text
```

Add to `tests/test_runs.py`:

```python
def test_archive_run_preserves_a_pt_checkpoint_suffix(project):
    from mlagent.runs import archive_run

    project.ensure_dirs()
    (project.checkpoints_dir / "best.pt").write_bytes(b"weights")
    (project.plots_dir / "training_curves.png").write_bytes(b"png")
    (project.plots_dir / "val_misclassified.png").write_bytes(b"png")

    figures = archive_run(project, 3, checkpoint="checkpoints/best.pt")
    assert (project.checkpoints_dir / "run3.pt").exists()
    assert not (project.checkpoints_dir / "run3.joblib").exists()
    assert {p.name for p in figures} == {"run3_training.png", "run3_val_misclassified.png"}


def test_archive_run_falls_back_to_the_legacy_joblib_name(project):
    from mlagent.runs import archive_run

    project.ensure_dirs()
    (project.checkpoints_dir / "best.joblib").write_bytes(b"model")
    archive_run(project, 1)
    assert (project.checkpoints_dir / "run1.joblib").exists()


def test_log_finished_run_records_the_pt_checkpoint(project):
    from mlagent.runs import log_finished_run

    project.ensure_dirs()
    started = "2026-09-10T12:00:00.000000+00:00"
    project.write_json("metrics.json", {
        "status": "done", "started_at": started, "model_type": "small_cnn",
        "task_type": "image_classification", "config": {"model_type": "small_cnn"},
        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_metric": 0.7,
                    "val_metric": 0.65}],
        "best_epoch": 1, "best_val_metric": 0.65, "seconds": 1.0, "error": None,
        "checkpoint": "checkpoints/best.pt",
    })
    project.write_json("eval_val.json", {"started_at": started, "value": 0.65})
    (project.checkpoints_dir / "best.pt").write_bytes(b"weights")

    logged = log_finished_run(project)
    assert logged is not None and logged.new is True
    assert logged.entry["checkpoint"] == "checkpoints/run1.pt"
    assert (project.checkpoints_dir / "run1.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_codegen_stage.py tests/test_runs.py tests/test_prompts_io.py -q
```
Expected: `ImportError: cannot import name 'labels_for'` in the codegen tests; `FileNotFoundError: ...prompts/teaching/model_choices_images.md` in the prompts test; `TypeError: archive_run() got an unexpected keyword argument 'checkpoint'` and `assert 'checkpoints/run1.joblib' == 'checkpoints/run1.pt'` in the runs tests.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/runs.py`, replace `archive_run` (lines 53-68) with:

```python
def archive_run(project: Project, run_id: int, checkpoint: str | None = None) -> list[Path]:
    """Freeze this run's figures and checkpoint under `run{N}` names.

    `checkpoint` is the run's own relative path out of `metrics.json` -- `.joblib` for a
    tabular run, `.pt` for an image run. The archived copy keeps that suffix, so nothing
    downstream has to know which family produced it. `None` means a run logged before
    `train.py` recorded the key; fall back to the tabular name.
    """
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
    relative = Path(checkpoint or f"checkpoints/{BEST_CHECKPOINT}")
    best = project.root / relative
    if best.exists():
        shutil.copy2(best, project.checkpoints_dir / f"run{run_id}{relative.suffix}")
    return archived
```

and replace `log_finished_run`'s archiving block (lines 135-144) with:

```python
    ok = metrics.get("status") == "done"
    run_id = len(runs) + 1
    relative = Path(str(metrics.get("checkpoint") or f"checkpoints/{BEST_CHECKPOINT}"))
    figures = archive_run(project, run_id, str(relative.as_posix())) if ok else []
    archive_metrics(project, run_id, metrics)
    archived = project.checkpoints_dir / f"run{run_id}{relative.suffix}"
    checkpoint = f"checkpoints/run{run_id}{relative.suffix}" if ok and archived.exists() \
        else None
```

Create `mlagent/prompts/teaching/model_choices_images.md` (level-fenced the way `model_choices.md` is):

```markdown
# Choosing a model for images

A model for images is a **neural network**: a stack of layers that each look at small
patches of the picture and pass on what they found. The early layers learn edges and
colours, the later ones learn shapes and, eventually, whole objects. Three choices, from
cheapest to strongest.

<!--level:beginner,intermediate-->
## Tiny CNN

Two convolutional layers and nothing else. It trains in seconds even on the CPU, which
makes it the right first run: it tells you whether your data and labels line up before
you spend time on anything bigger. Expect modest accuracy on anything harder than clear
shapes on a plain background.

## Small CNN

Three convolutional blocks with batch normalisation and dropout. Still trains from
scratch, still fine on the CPU for a few thousand small images, and much better at real
photographs than the tiny one. This is the default, and the one to stay with while you
are learning what the tuning loop does.

## Pretrained ResNet-18 (transfer learning)

ResNet-18 has already been trained on a million photographs, so it arrives knowing what
edges, textures and object parts look like. **Transfer learning** means keeping that
knowledge and only teaching it your classes. It is by far the most accurate option on
real photographs, especially when you have only a few hundred images per class -- but it
is roughly ten times slower per epoch than the small CNN, so switch to the GPU runtime
before you pick it, and prefer 64px over 128px images on the CPU.

`freeze_backbone` decides how much of it you retrain: `yes` trains only the final layer
(fast, and enough when your images look like ordinary photographs), `no` retrains
everything (slower, better when your images look nothing like everyday photos -- X-rays,
say, or satellite tiles).
<!--/level-->

<!--level:expert-->
- `tiny_cnn`: 2 conv blocks, ~5k parameters. Sanity check.
- `small_cnn`: 3 conv blocks with BN and dropout. Default; CPU-viable to a few thousand
  images.
- `resnet18`: torchvision ResNet-18. `pretrained=imagenet` downloads weights (needs the
  network); `freeze_backbone=yes` trains the head only. ~10x the small CNN's cost per
  epoch; use the GPU runtime, and prefer 64px on CPU.
<!--/level-->

Whatever you pick, you are not stuck with it: the tuning step can switch families, and
*Redo a stage* regenerates the whole training project.
```

In `mlagent/stages/codegen.py`, replace the label constants (lines 32-37) with:

```python
MODEL_LABELS_BY_FAMILY: dict[str, dict[str, str]] = {
    "tabular_sklearn": {
        "Linear / logistic regression": "linear",
        "Random forest": "random_forest",
        "Gradient boosting": "gradient_boosting",
    },
    "image_torch": {
        "Tiny CNN": "tiny_cnn",
        "Small CNN": "small_cnn",
        "Pretrained ResNet-18": "resnet18",
    },
}
# Kept as the tabular map so `codegen.MODEL_LABELS` still resolves for older callers.
MODEL_LABELS = MODEL_LABELS_BY_FAMILY["tabular_sklearn"]
FALLBACK_MODEL_BY_FAMILY = {"tabular_sklearn": "gradient_boosting", "image_torch": "small_cnn"}
ASK_LABEL = "Ask me after the explanation"


def labels_for(family: str) -> dict[str, str]:
    """The label -> model_type map for one template family."""
    return MODEL_LABELS_BY_FAMILY[family]


def label_for(family: str, code: str) -> str:
    """The human label for one model_type, or the code itself if it has none."""
    for label, value in MODEL_LABELS_BY_FAMILY.get(family, {}).items():
        if value == code:
            return label
    return code
```

Replace `RECOMMEND_TOOL`'s fixed enum (lines 57-72) with a builder, since the choices are
now per family:

```python
def recommend_tool(choices: list[str], handler) -> ToolSpec:
    return ToolSpec(
        name="recommend_model",
        description=(
            "Recommend one model family for this dataset. `reason` is shown to the user "
            "before they choose, so make it about their data, not about models in general."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {"type": "string", "enum": list(choices)},
                "reason": {"type": "string"},
            },
            "required": ["model_type", "reason"],
        },
        handler=handler,
    )
```

Replace `fallback_model` (lines 75-81) and `check_data` (lines 84-104) with:

```python
def fallback_model(meta: dict, family: str = "tabular_sklearn") -> str:
    """What to recommend when Claude is unreachable: flexibility needs rows."""
    default = FALLBACK_MODEL_BY_FAMILY.get(family, "gradient_boosting")
    if family != "tabular_sklearn":
        return default
    try:
        rows = int(meta.get("clean_n_rows") or 0)
    except (TypeError, ValueError):
        return default
    return "linear" if 0 < rows < SMALL_DATA_ROWS else default


def _check_splits(meta: dict) -> list[str]:
    splits = meta.get("splits") or {}
    fractions = [float(splits.get(k, 0)) for k in ("train", "val", "test")]
    if abs(sum(fractions) - 1.0) > 0.01 or any(f <= 0 for f in fractions):
        return [f"split fractions {splits} must be positive and sum to 1"]
    return []


def _check_image_data(meta: dict, project_root: Path) -> list[str]:
    problems: list[str] = []
    clean_path = project_root / str(meta.get("clean_path") or "data/clean/data.npz")
    if not clean_path.exists():
        return [f"clean image file {clean_path.name} is missing; rerun the clean stage"]
    if not (clean_path.parent / "manifest.csv").exists():
        problems.append("manifest.csv is missing next to the clean images")
    if int(meta.get("n_classes") or 0) < 2:
        problems.append("image classification needs at least 2 classes")
    if int(meta.get("clean_n_rows") or 0) < 2:
        problems.append("no images are available for training")
    return problems + _check_splits(meta)


def check_data(meta: dict, project_root: Path) -> list[str]:
    """Problems that would make training impossible; empty list means go ahead."""
    if meta.get("modality") == "image":
        return _check_image_data(meta, project_root)
    problems: list[str] = []
    clean_path = project_root / str(meta.get("clean_path") or "data/clean/data.csv")
    if not clean_path.exists():
        return [f"clean data file {clean_path.name} is missing; rerun the clean stage"]
    columns = list(pd.read_csv(clean_path, nrows=0).columns)
    target = meta.get("target")
    if not target or target not in columns:
        problems.append(f"target column {target!r} is not in the clean data")
    features = [c for c in meta.get("feature_columns") or [] if c in columns and c != target]
    if not features:
        problems.append("no feature columns are available for training")
    problems += _check_splits(meta)
    if meta.get("task_type") == "tabular_classification" and (meta.get("n_classes") or 0) < 2:
        problems.append("classification needs at least 2 classes in the target")
    return problems
```

Extend `meta_summary` (lines 107-120) with the image keys, which are `None` for tabular
runs. Full replacement, every key spelled out:

```python
def meta_summary(meta: dict) -> dict:
    features = list(meta.get("feature_columns") or [])
    labels = list(meta.get("class_labels") or [])
    modality = meta.get("modality", "tabular")
    return {
        "task_type": meta.get("task_type"),
        "target": meta.get("target"),
        "n_rows": meta.get("clean_n_rows"),
        "n_features": len(features),
        "feature_columns": features[:MAX_LISTED],
        "categorical_columns": list(meta.get("categorical_columns") or [])[:MAX_LISTED],
        "n_classes": meta.get("n_classes"),
        "class_labels": labels[:MAX_LISTED],
        "splits": meta.get("splits"),
        "modality": modality,
        "image_size": meta.get("image_size"),
        "n_images": meta.get("clean_n_rows") if modality == "image" else None,
    }
```

Make `prepare` and the two helpers family-aware. `prepare` passes `template` down; the
model pick, the recommendation and the message use `labels_for` / `label_for` and the
modality's teaching material. Full replacement for lines 141-176, every statement spelled
out (nothing left as a comment claiming to be "unchanged" inside the `message` list):

```python
    def prepare(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        modality = modality_for(spec.task_type)
        template = modality.template_family
        meta = ctx.project.read_json(META_FILE) or {}
        problems = check_data(meta, ctx.project.root)
        if problems:
            ctx.display("The data is not ready for training:\n- " + "\n- ".join(problems))
            return None

        nested = load_schema(template)
        model_type = self._choose_model(ctx, spec, meta, nested, modality)
        schema = schema_for(nested, model_type)

        proposal, rationale = self._propose(ctx, spec, meta, schema)
        proposal = {**proposal, "model_type": model_type}
        config, notes = coerce_config(proposal, schema)
        written = copy_template(template, ctx.project.root)
        ctx.project.write_json(cfg.CONFIG_FILE, config)

        files = ", ".join(f"`{p.name}`" for p in written) + ", `config.json`"
        message = [
            f"I wrote the training project into the project folder: {files}.",
            f"`train.py` trains a **{label_for(template, model_type)}** model; each "
            "[[epoch]] is one pass that records train and validation [[loss]] so we can "
            "watch for [[overfitting]]. `evaluate.py` scores one saved model on one split.",
            "",
            rationale,
            "",
            config_table(config, schema),
        ]
```

The rest of `prepare` (the `notes` append, the confirm/edit block, `self._walkthrough`)
is unchanged.
```python
    def _choose_model(self, ctx, spec, meta: dict, nested: dict, modality) -> str:
        family = modality.template_family
        labels = labels_for(family)
        recommended, reason = self._recommend(ctx, spec, meta, nested, modality)
        ctx.display(material(modality.teaching_material, ctx.learning_level()))
        ctx.display(f"**My recommendation: {label_for(family, recommended)}.** {reason}")
        answer = ctx.questioner.choice(
            f"Which model shall I set up? (I recommend {label_for(family, recommended)})",
            [ASK_LABEL, *labels], allow_other=False, key="codegen.model_type",
        )
        if answer == ASK_LABEL:
            return recommended
        return labels.get(answer, recommended)

    def _recommend(self, ctx, spec, meta: dict, nested: dict, modality) -> tuple[str, str]:
        family = modality.template_family
        choices = model_types(nested)
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["model_type"] = str(inp.get("model_type") or "")
            captured["reason"] = str(inp.get("reason") or "")
            return "recorded"

        tool = recommend_tool(choices, handler)
        prompt = json.dumps(
            {"spec": spec.to_dict(), "data": meta_summary(meta), "model_choices": choices,
             "task": "recommend one model family"},
            indent=2, default=str,
        )
        try:
            ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}], [tool],
            )
        except LLMError as exc:
            chosen = fallback_model(meta, family)
            rows = meta.get("clean_n_rows")
            size = f" ({rows} examples)" if rows is not None else ""
            return chosen, (
                f"(The assistant was unavailable: {exc}.) Going by the size of the dataset"
                f"{size}, {label_for(family, chosen)} is the safe default."
            )
        chosen = captured.get("model_type") or ""
        if chosen not in choices:
            chosen = fallback_model(meta, family)
        return chosen, captured.get("reason") or "It suits the shape of this dataset."
```

Finally, `is_complete` (line 135) already looks the family up through `modality_for` after
Task 1; leave it as is.

- [ ] **Step 4: Make the train-stage preamble modality-aware**

`mlagent/stages/train.py:44-48` currently says training always runs on the CPU and saves
`checkpoints/best.joblib`, which is only true for tabular. Add
`from mlagent.modality import modality_for` to the imports, and replace those lines with:

```python
        modality = modality_for(spec.task_type)
        if modality.name == "image":
            device_sentence = (
                "Training picks its [[device]] at run time -- the [[GPU]] if this runtime "
                "has one, otherwise the [[CPU]] -- so there is no [[compute unit]] cost gate "
                "for this run."
            )
            checkpoint_sentence = (
                f"`train.py` runs {config.get('epochs')} [[epoch]]s, redrawing the loss and "
                "metric curves as it goes, and saves the best model to "
                "`checkpoints/model.pt`."
            )
        else:
            device_sentence = (
                "Training runs on the [[CPU]] for tabular data, so there is no "
                "[[compute unit]] cost gate for this run."
            )
            checkpoint_sentence = (
                f"`train.py` runs {config.get('epochs')} [[epoch]]s, redrawing the loss and "
                "metric curves as it goes, and saves the best model to "
                "`checkpoints/best.joblib`."
            )
        ctx.display(
            f"{device_sentence} {checkpoint_sentence} `evaluate.py` then scores that model "
            f"on the [[validation set]] and draws the {spec.metric} figures. Run both cells."
        )
```

Add to `tests/test_train_stage.py`:

```python
def test_the_preamble_names_the_image_checkpoint_and_the_tabular_one(project, image_project):
    from mlagent.stages.train import TrainStage

    shown, ctx = capture_ctx(image_project)
    TrainStage().prepare(ctx)
    assert "checkpoints/model.pt" in " ".join(shown)
    assert "GPU if this runtime" in " ".join(shown)

    shown, ctx = capture_ctx(project)
    TrainStage().prepare(ctx)
    assert "checkpoints/best.joblib" in " ".join(shown)
    assert "runs on the [[CPU]] for tabular data" in " ".join(shown)
```

(`capture_ctx` and `image_project` are the fixtures already used elsewhere in this file;
adjust the call to however this file currently builds a train-ready image project.)

- [ ] **Step 5: Run test to verify it passes**

```
python -m pytest tests/test_codegen_stage.py tests/test_runs.py tests/test_prompts_io.py tests/test_train_stage.py tests/test_tune_stage.py -q
python -m pytest -q
ruff check .
```
Expected: all green, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add mlagent/stages/codegen.py mlagent/stages/train.py mlagent/runs.py mlagent/prompts/teaching/model_choices_images.md tests/test_codegen_stage.py tests/test_runs.py tests/test_prompts_io.py tests/test_train_stage.py
git commit -m "feat: family-aware codegen, the image model-choice primer and suffix-aware run archiving

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 11: The tuning loop on image runs

**Files:**
- Modify: `mlagent/diagnose.py:151-241`
- Test: `tests/test_diagnose.py`, `tests/test_templates_io.py`, `tests/test_prompts_io.py`

**Interfaces:**
- Consumes: `diagnose.Diagnosis` / `Proposal` (`mlagent/diagnose.py:37-51`), `templates_io.schema_for` / `edit_config` (`mlagent/templates_io.py:42, 202`), the `image_torch` schema (Task 8).
- Produces (used by Task 13): `heuristic_proposals(diagnosis: Diagnosis, config: dict, schema: dict) -> list[Proposal]` that only ever sets keys present in `schema`, with an image branch per the spec's table.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_diagnose.py`:

```python
IMAGE_FAMILIES = ("tiny_cnn", "small_cnn", "resnet18")
IMAGE_LABELS = ("overfitting", "underfitting", "improving", "learning_rate_too_high",
                "plateau", "failed_run")


def image_schema(family: str) -> dict:
    from mlagent.templates_io import load_schema, schema_for

    return schema_for(load_schema("image_torch"), family)


def image_config(family: str) -> dict:
    from mlagent.templates_io import default_config

    return {**default_config(image_schema(family)), "model_type": family}


@pytest.mark.parametrize("family", IMAGE_FAMILIES)
@pytest.mark.parametrize("label", IMAGE_LABELS)
def test_every_image_proposal_only_touches_keys_in_its_own_schema(family, label):
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    schema = image_schema(family)
    proposals = heuristic_proposals(Diagnosis(label), image_config(family), schema)
    assert len(proposals) == 1
    changes = proposals[0].changes
    assert changes
    assert set(changes) <= set(schema)


@pytest.mark.parametrize("family", IMAGE_FAMILIES)
def test_the_image_proposals_coerce_to_a_real_change(family):
    from mlagent.diagnose import Diagnosis, apply_proposal, heuristic_proposals
    from mlagent.templates_io import load_schema

    nested = load_schema("image_torch")
    config = image_config(family)
    for label in IMAGE_LABELS:
        proposal = heuristic_proposals(Diagnosis(label), config, image_schema(family))[0]
        new_config, diff, _notes = apply_proposal(config, proposal, nested)
        assert diff, f"{family}/{label} coerced to no change"
        assert new_config["model_type"] == family


def test_overfitting_on_a_cnn_turns_augmentation_on_and_regularises():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    config = {**image_config("small_cnn"), "augment": "none", "dropout": 0.2,
              "weight_decay": 0.0001}
    changes = heuristic_proposals(
        Diagnosis("overfitting"), config, image_schema("small_cnn")
    )[0].changes
    assert changes["augment"] == "basic"
    assert changes["dropout"] > 0.2
    assert changes["weight_decay"] > 0.0001


def test_underfitting_on_a_cnn_adds_epochs_and_raises_the_learning_rate():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    config = image_config("tiny_cnn")
    changes = heuristic_proposals(
        Diagnosis("underfitting"), config, image_schema("tiny_cnn")
    )[0].changes
    assert changes["epochs"] > config["epochs"]
    assert changes["learning_rate"] > config["learning_rate"]


def test_a_high_learning_rate_is_lowered_and_the_batch_grows():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    config = image_config("small_cnn")
    changes = heuristic_proposals(
        Diagnosis("learning_rate_too_high"), config, image_schema("small_cnn")
    )[0].changes
    assert changes["learning_rate"] < config["learning_rate"]
    assert changes["batch_size"] > config["batch_size"]


def test_a_plateau_lowers_only_the_learning_rate():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    config = image_config("resnet18")
    changes = heuristic_proposals(
        Diagnosis("plateau"), config, image_schema("resnet18")
    )[0].changes
    assert set(changes) == {"learning_rate"}
    assert changes["learning_rate"] < config["learning_rate"]


def test_a_failed_image_run_is_retried_more_gently():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    config = image_config("small_cnn")
    changes = heuristic_proposals(
        Diagnosis("failed_run"), config, image_schema("small_cnn")
    )[0].changes
    assert changes["learning_rate"] < config["learning_rate"]
    assert changes["epochs"] < config["epochs"]


def test_a_resnet_overfitting_proposal_never_sets_dropout():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    changes = heuristic_proposals(
        Diagnosis("overfitting"), image_config("resnet18"), image_schema("resnet18")
    )[0].changes
    assert "dropout" not in changes


def test_a_tabular_proposal_never_sets_an_image_key():
    from mlagent.diagnose import Diagnosis, heuristic_proposals
    from mlagent.templates_io import default_config, load_schema, schema_for

    schema = schema_for(load_schema("tabular_sklearn"), "gradient_boosting")
    config = {**default_config(schema), "model_type": "gradient_boosting"}
    for label in IMAGE_LABELS:
        changes = heuristic_proposals(Diagnosis(label), config, schema)[0].changes
        assert set(changes) <= set(schema)
        assert "augment" not in changes and "batch_size" not in changes


def test_target_met_still_proposes_nothing_for_images():
    from mlagent.diagnose import Diagnosis, heuristic_proposals

    assert heuristic_proposals(
        Diagnosis("target_met"), image_config("small_cnn"), image_schema("small_cnn")
    ) == []
```

Add to `tests/test_templates_io.py`:

```python
def test_edit_config_never_offers_the_image_choice_keys():
    from mlagent.templates_io import default_config, edit_config, load_schema, schema_for
    from mlagent.ui.questions import ScriptedQuestioner

    schema = schema_for(load_schema("image_torch"), "small_cnn")
    config = {**default_config(schema), "model_type": "small_cnn"}
    questioner = ScriptedQuestioner(["Done"])
    assert edit_config(questioner, config, schema) == config
    offered = questioner.asked[0] if questioner.asked else ""
    assert "augment" not in offered and "model_type" not in offered


def test_edit_config_changes_a_numeric_image_key():
    from mlagent.templates_io import default_config, edit_config, load_schema, schema_for
    from mlagent.ui.questions import ScriptedQuestioner

    schema = schema_for(load_schema("image_torch"), "tiny_cnn")
    config = {**default_config(schema), "model_type": "tiny_cnn"}
    questioner = ScriptedQuestioner([f"batch_size = {config['batch_size']}", "64", "Done"])
    updated = edit_config(questioner, config, schema)
    assert updated["batch_size"] == 64
    assert updated["augment"] == config["augment"]
```

Add to `tests/test_prompts_io.py`:

```python
def test_the_tune_prompt_names_no_family_specific_keys():
    from mlagent.prompts_io import load_prompt

    text = load_prompt("tune", audience="test audience")
    for key in ("min_samples_leaf", "max_leaf_nodes", "trees_per_epoch", "l2_regularization",
                "max_features", "alpha", "iters_per_epoch", "dropout", "freeze_backbone",
                "pretrained", "augment"):
        assert key not in text, key
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_diagnose.py tests/test_templates_io.py tests/test_prompts_io.py -q
```
Expected: today's `heuristic_proposals` has no image branch, so an image family
(`tiny_cnn`/`small_cnn`/`resnet18`) never matches `family == "gradient_boosting"` or
`family == "random_forest"` and always falls into the linear `else` branch inside
whichever label's `if`/`elif`. Concretely:
`test_every_image_proposal_only_touches_keys_in_its_own_schema[overfitting-*]` (3 of the
18 parametrised cases) fail on `assert set(changes) <= set(schema)` because that branch
sets `changes = {"alpha": 0.001}` and no image schema has `alpha`. The other 15
parametrised cases already pass, by coincidence: `underfitting`/`improving` only ever
scale `epochs`, `learning_rate_too_high` only ever scales `learning_rate`, and
`failed_run` scales both -- all common keys every image schema also has -- and `plateau`
sets `changes = {"model_type": "gradient_boosting"}`, and `model_type` is a common key
too, so the subset check passes even though `"gradient_boosting"` is not a valid image
model (Task 11's job is to fix the value, not just the key). Among the six named image
tests, four fail: `test_overfitting_on_a_cnn_turns_augmentation_on_and_regularises`
(`KeyError: 'augment'`), `test_underfitting_on_a_cnn_adds_epochs_and_raises_the_learning_rate`
(`KeyError: 'learning_rate'`), `test_a_high_learning_rate_is_lowered_and_the_batch_grows`
(`KeyError: 'batch_size'`), and `test_a_plateau_lowers_only_the_learning_rate`
(`AssertionError`: `set(changes) == {'model_type'}`, not `{'learning_rate'}`).
`test_a_failed_image_run_is_retried_more_gently` and
`test_a_resnet_overfitting_proposal_never_sets_dropout` already pass by the same
coincidence. `test_a_tabular_proposal_never_sets_an_image_key` and
`test_target_met_still_proposes_nothing_for_images` already pass because they exercise
only pre-existing tabular/`target_met` behaviour that Task 11 does not touch.
`test_the_tune_prompt_names_no_family_specific_keys` and the two `edit_config` tests
should PASS already -- record that in the step-4 note if so; `prompts/tune.md` already
names no family keys.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/diagnose.py`, replace the body of `heuristic_proposals` (lines 151-241) with a
presence-checked version whose helpers refuse to touch a key the schema does not have,
plus the image branch:

```python
IMAGE_FAMILIES = ("tiny_cnn", "small_cnn", "resnet18")


def heuristic_proposals(diagnosis: Diagnosis, config: dict, schema: dict) -> list[Proposal]:
    """One fixed proposal per label and family, used when Claude is unavailable.

    Every helper checks the flat `schema` before setting a key, so a proposal for one
    family can never set another family's key (an image proposal never reaches for
    `min_samples_leaf`, a tabular one never reaches for `augment`). Values are scaled from
    the current config; `apply_proposal` clamps them to the schema.
    """
    label = diagnosis.label
    family = str(config.get("model_type", ""))
    changes: dict[str, object] = {}
    expected = "better"

    def scaled(key: str, factor: float) -> None:
        if key not in schema:
            return
        current = config.get(key)
        if isinstance(current, int | float) and not isinstance(current, bool):
            changes[key] = current * factor

    def bumped(key: str, factor: float, floor: float) -> None:
        """Scale a key that may sit at zero, so it still moves off the floor."""
        if key not in schema:
            return
        current = config.get(key)
        if not isinstance(current, int | float) or isinstance(current, bool):
            return
        changes[key] = current * factor if current else floor

    def choose(key: str, value: str) -> None:
        if key in schema and config.get(key) != value:
            changes[key] = value

    if label == "target_met":
        return []

    if family in IMAGE_FAMILIES:
        expected, reason = _image_move(label, choose, scaled, bumped)
    elif label == "overfitting":
        ...  # the existing tabular branches, unchanged apart from using `scaled`/`bumped`
    if not changes:
        return []
    return [Proposal(rank=1, changes=changes, reason=reason, expected=expected)]
```

and add the image move table above it:

```python
def _image_move(label: str, choose, scaled, bumped) -> tuple[str, str]:
    """The fixed image-family move per diagnosis label; returns (expected, reason)."""
    if label == "overfitting":
        choose("augment", "basic")
        bumped("dropout", 1.4, 0.2)
        bumped("weight_decay", 10.0, 0.001)
        return "steadier", (
            "Validation [[loss]] turned upward while training loss kept falling: the "
            "network is memorising these exact pictures. Random flips and shifts, more "
            "[[dropout]] and a stronger weight penalty make that much harder."
        )
    if label in ("underfitting", "improving"):
        scaled("epochs", 1.5)
        scaled("learning_rate", 1.5)
        reason = (
            "The last change helped and validation loss was still falling when training "
            "stopped, so keep going in the same direction with more [[epoch]]s and a "
            "bigger step each time."
            if label == "improving"
            else "Validation loss was still falling at the last [[epoch]]: the network had "
                 "not finished learning. More epochs and a bigger step per update let it."
        )
        return "better", reason
    if label == "learning_rate_too_high":
        scaled("learning_rate", 0.3)
        scaled("batch_size", 2)
        return "steadier", (
            "Validation loss jumps up and down instead of settling: each update "
            "overshoots. A much smaller [[learning rate]] takes smaller steps, and a "
            "bigger batch averages out more of the noise before each one."
        )
    if label == "plateau":
        scaled("learning_rate", 0.5)
        return "better", (
            "The curve has flattened. Halving the [[learning rate]] lets the network "
            "settle into a better spot instead of bouncing over it."
        )
    scaled("learning_rate", 0.3)
    scaled("epochs", 0.5)
    return "steadier", (
        "The run failed, most often from the loss blowing up to infinity. A much smaller "
        "[[learning rate]] and fewer epochs are the gentlest retry."
    )
```

The existing tabular branches keep their `scaled(...)` calls verbatim; the one edit they
need is that `linear`'s `alpha` assignment becomes `bumped("alpha", 10.0, 0.001)` so it too
is presence-checked.

`prompts/tune.md` already names no family-specific key (it describes `changes` as "only
keys from the schema"), so it is unchanged; the new test locks that in.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_diagnose.py tests/test_templates_io.py tests/test_prompts_io.py tests/test_tune_stage.py -q
python -m pytest -q
ruff check .
```
Expected: all green, ruff clean, and the existing tabular `heuristic_proposals` tests still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add mlagent/diagnose.py tests/test_diagnose.py tests/test_templates_io.py tests/test_prompts_io.py
git commit -m "feat: presence-checked heuristic proposals with image moves for the tuning loop

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 12: The notebook -- image fields in "2. Data", the pip line and the intro cell text

**Files:**
- Modify: `scripts/build_notebook.py:29-52`, `scripts/build_notebook.py:70-73`, `scripts/build_notebook.py:100-132`, `scripts/build_notebook.py:147-194`, `notebooks/ML_Training_Agent.ipynb` (regenerated)
- Test: `tests/test_colab.py`

**Interfaces:**
- Consumes: `colab.HANDOFF_COMMANDS` / `script_cells` (`mlagent/colab.py:34-58`, unchanged), the `data.*` answer keys from Task 7.
- Produces: a rebuilt `notebooks/ML_Training_Agent.ipynb` whose "2. Data" cell passes `data.n_images`, `data.image_size`, `data.drive_folder`, `data.hf_dataset`, `data.max_images` alongside the existing tabular keys, and whose "1. Project and interview" TASK field offers `Image classification`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_colab.py`:

```python
NOTEBOOK = Path("notebooks/ML_Training_Agent.ipynb")


def notebook_sources() -> list[str]:
    import json

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def data_cell() -> str:
    return next(s for s in notebook_sources() if "#@title 2. Data" in s)


def test_the_task_field_offers_image_classification():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    assert "'Image classification'" in intake
    assert "TASK_LABELS" not in intake  # the labels live in the intake stage, not the cell


def test_every_task_label_in_the_notebook_is_one_the_intake_stage_knows():
    import re

    from mlagent.stages.intake import TASK_LABELS

    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    line = next(ln for ln in intake.splitlines() if ln.startswith("TASK = "))
    assert set(re.findall(r"'([^']+)'", line)) <= set(TASK_LABELS) | {
        "Tabular classification"
    }


def test_the_data_cell_has_the_image_fields_with_hints():
    source = data_cell()
    for field in ("N_IMAGES", "IMAGE_SIZE", "DRIVE_FOLDER", "HF_DATASET", "MAX_IMAGES"):
        assert f"{field} = " in source, field
        assert f"**{field}**" in source, field
    assert "IMAGE_SIZE = '64'" in source
    assert "['32', '64', '128']" in source


def test_the_tabular_only_hints_say_so():
    source = data_cell()
    for line in source.splitlines():
        if "**N_FEATURES**" in line or "**TARGET_COLUMN**" in line:
            assert "ignored for image tasks" in line


def test_the_n_classes_hint_covers_images_too():
    source = data_cell()
    line = next(ln for ln in source.splitlines() if "**N_CLASSES**" in ln)
    assert "for images, 2 to 5" in line


def test_the_data_cell_passes_every_image_answer_key():
    source = data_cell()
    for key in ("data.n_images", "data.image_size", "data.drive_folder", "data.hf_dataset",
                "data.max_images"):
        assert f"'{key}'" in source, key


def test_the_answer_keys_the_data_cell_sends_are_ones_the_data_stage_reads():
    import re

    source = (Path("mlagent/stages/data.py")).read_text(encoding="utf-8")
    used = set(re.findall(r'key="(data\.[a-z_]+)"', source))
    sent = set(re.findall(r"'(data\.[a-z_]+)'", data_cell()))
    assert sent <= used, sent - used


def test_the_pip_line_installs_torch():
    pip = next(s for s in notebook_sources() if s.startswith("%pip"))
    assert "torch" in pip and "torchvision" in pip
    assert "pillow" in pip


def test_the_intro_mentions_image_tasks():
    intro = notebook_sources()[0]
    assert "image" in intro.lower()


def test_the_gpu_hint_no_longer_defers_images_to_a_later_milestone():
    intake = next(s for s in notebook_sources() if "#@title 1. Project" in s)
    assert "later milestone" not in intake


def test_the_notebook_is_up_to_date_with_the_builder():
    import subprocess
    import sys

    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, "scripts/build_notebook.py"], check=True,
                   capture_output=True, text=True)
    assert NOTEBOOK.read_bytes() == before
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_colab.py -q
```
Expected: `StopIteration`/`AssertionError` -- `'Image classification'` is absent from the TASK field, `N_IMAGES = ` is absent from the data cell, `torch` is absent from the pip line, and the GPU hint still says "which come in a later milestone".

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_notebook.py`:

Replace the pip cell (lines 70-73) with:

```python
    code(
        "%pip -q install anthropic markdown pandas numpy scikit-learn matplotlib pyarrow "
        "openpyxl pillow huggingface_hub datasets torch torchvision"
    ),
```

In the intro markdown cell, extend step 2's sentence (lines 36-40) to:

```python
        "2. **Data.** *You do:* give the path, search keywords, or the size of the "
        "synthetic dataset (the source itself is chosen in step 1). *You get:* a raw "
        "dataset and a profile with two to four figures. For a table those are "
        "histograms, missing values, class balance or target spread, and correlation; "
        "for images they are example thumbnails, class balance, pixel intensity and the "
        "mean image per class.\n"
```

and add to the "Where your files are" paragraph (line 58-61), after "the raw and cleaned
data": " (`data.csv` for a table, `data.npz` plus `manifest.csv` for images)".

In the "1. Project and interview" cell, replace the TASK hint and field (lines 100-104) with:

```python
        "#@markdown **TASK** — *Tabular classification* predicts a category from a table "
        "(yes/no, which type). *Tabular regression* predicts a number from a table (a "
        "price, a temperature). *Image classification* sorts pictures into classes. If "
        "your answer is a label, choose one of the classification options.",
        "TASK = 'Tabular classification'  #@param ['Tabular classification', "
        "'Tabular regression', 'Image classification']",
```

and replace the GPU hint (lines 129-132) with:

```python
        "#@markdown **GPU** — Tables train on the CPU, so keep the no-GPU option for them. "
        "Images are much faster on a GPU: pick *T4 GPU* for image tasks, and set the "
        "runtime type to match (Runtime > Change runtime type > T4 GPU). *Pretrained "
        "ResNet-18* really wants one.",
        "GPU = 'No GPU (CPU only)'  #@param ['No GPU (CPU only)', 'T4 GPU', "
        "'Any available GPU']",
```

Also update the shared `N_CLASSES` hint (lines 158-159), since it now drives image
classification too, not just tabular:

```python
        "#@markdown **N_CLASSES** — Synthetic only, classification. How many categories "
        "the label can take; `2` for yes/no, or for images, 2 to 5.",
        "N_CLASSES = 2  #@param {type:'integer'}",
```

In the "2. Data" cell, change the two tabular hints and add the five image fields (the new
lines go after `HF_QUERY` and before the `orch.run` call):

```python
        "#@markdown **N_FEATURES** — Synthetic only, ignored for image tasks. How many "
        "input columns (things the model can look at). `8` is a good start.",
        "N_FEATURES = 8  #@param {type:'integer'}",
```
```python
        "#@markdown **TARGET_COLUMN** — Drive and HuggingFace only, ignored for image "
        "tasks. The column you want to predict. Leave blank: after loading the data the "
        "assistant shows your columns and asks you to pick, with its best guess marked.",
        "TARGET_COLUMN = ''  #@param {type:'string'}",
        "#@markdown **N_IMAGES** — Image tasks, synthetic only. How many shape pictures to "
        "invent. `300` trains in under a minute on the CPU.",
        "N_IMAGES = 300  #@param {type:'integer'}",
        "#@markdown **IMAGE_SIZE** — Image tasks only. How big each picture is made, in "
        "pixels. `32` is fastest and fine for shapes; `64` is the balanced default; `128` "
        "sees the most detail but wants the GPU runtime, especially with ResNet-18.",
        "IMAGE_SIZE = '64'  #@param ['32', '64', '128']",
        "#@markdown **DRIVE_FOLDER** — Image tasks, Drive only. The folder holding your "
        "pictures, with one subfolder per class, e.g. `MyDrive/pets` containing "
        "`cats/` and `dogs/`. Files that are not readable images are reported, never fatal.",
        "DRIVE_FOLDER = ''  #@param {type:'string'}",
        "#@markdown **HF_DATASET** — Image tasks, HuggingFace only. The dataset id to "
        "download, e.g. `cifar10` or `beans`.",
        "HF_DATASET = ''  #@param {type:'string'}",
        "#@markdown **MAX_IMAGES** — Image tasks, Drive and HuggingFace only. Cap on how "
        "many pictures to use; `0` means all of them. A cap keeps your first run quick and "
        "still keeps every class.",
        "MAX_IMAGES = 0  #@param {type:'integer'}",
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
        "    'data.n_images': N_IMAGES,",
        "    'data.image_size': IMAGE_SIZE,",
        "    'data.drive_folder': DRIVE_FOLDER,",
        "    'data.hf_dataset': HF_DATASET,",
        "    'data.max_images': MAX_IMAGES,",
        "})",
```

Also change the "2. Data" cell's opening hint (line 149-151) to name both modalities:

```python
        "#@markdown Get the data. Only the boxes for the task and data source you chose "
        "above matter; the others are ignored. When the assistant stops and names the next "
        "cell, run that cell.",
```

and the "4. Model" hint (lines 223-231) to cover both families:

```python
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
```

Then rebuild:

```
python scripts/build_notebook.py
```

- [ ] **Step 4: Run test to verify it passes**

```
python scripts/build_notebook.py
python -m pytest tests/test_colab.py -q
python -m pytest -q
ruff check .
```
Expected: the builder prints `wrote ... (21 cells)`, all tests pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebook.py notebooks/ML_Training_Agent.ipynb tests/test_colab.py
git commit -m "feat: image fields in the 2. Data cell, an image task choice and torch in the pip line

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

### Task 13: The image pipeline end to end, the report's data section, the smoke checklist and CLAUDE.md

**Files:**
- Create: `tests/test_pipeline_e2e_images.py`
- Modify: `mlagent/stages/report.py:31-76`, `docs/colab-smoke.md`, `CLAUDE.md`
- Test: `tests/test_pipeline_e2e_images.py`, `tests/test_report_stage.py`

**Interfaces:**
- Consumes: everything from Tasks 1-12; `Orchestrator` (`mlagent/orchestrator.py`), `FakeLLM` (`mlagent/llm.py`), the `advance` fixture (`tests/conftest.py:114`), `tune.STOP_LABEL` / `apply_label` (`mlagent/stages/tune.py:51, 135`).
- Wall-clock budget: `tests/test_pipeline_e2e_images.py` trains real (tiny, CPU) models via
  subprocess, same as `tests/test_pipeline_e2e.py` does for tabular. Target under 90
  seconds total for the file on CPU; `small_config` (below) is what keeps it there by
  shrinking `epochs`/`batch_size` right after codegen writes its defaults, before the
  real `train.py` subprocess runs.
- Produces: `render_report(project_name, spec, runs, best, eval_test, lessons, figures, meta=None) -> str` -- one new optional trailing parameter carrying `data_meta.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_e2e_images.py`:

```python
"""End-to-end image run of intake -> data -> clean -> codegen -> train -> tune -> report."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mlagent.imageset import read_pair
from mlagent.llm import FakeLLM
from mlagent.orchestrator import Orchestrator
from mlagent.runlog import read_runs
from mlagent.stages.base import StageContext
from mlagent.stages.clean import AUDIT_FILE, CLEAN_PY
from mlagent.stages.codegen import CodegenStage
from mlagent.stages.data import DataStage
from mlagent.stages.intake import IntakeStage
from mlagent.stages.report import ReportStage
from mlagent.stages.train import TrainStage
from mlagent.stages.tune import STOP_LABEL, TuneStage, apply_label
from mlagent.ui.questions import ScriptedQuestioner

pytest.importorskip("torch")

FORM_ANSWERS = {
    "intake.goal": "Sort pictures of shapes into their kind",
    "intake.learning_level": "Beginner - explain everything as we go",
    "intake.task_type": "Image classification",
    "intake.metric": "accuracy",
    "intake.target_value": 0.95,
    "intake.data_source": "Synthetic data",
    "intake.minutes_per_run": 5,
    "intake.max_rounds": 3,
    "intake.gpu": "No GPU (CPU only)",
    "data.n_images": 60,
    "data.image_size": "32",
    "data.n_classes": 3,
    "data.noise": 0.05,
    "data.inject_quirks": True,
    "clean.train_fraction": 0.7,
    "clean.val_fraction": 0.15,
    "codegen.model_type": "Tiny CNN",
    "tune.action": STOP_LABEL,
}
ALL_STAGES = ["intake", "data", "clean", "codegen", "train", "tune", "report"]


class AutoApproveQuestioner(ScriptedQuestioner):
    """Every confirm() is approved without consuming a scripted answer (the number of
    audit fixes varies with the data)."""

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        self.asked.append(question)
        return True


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The generated scripts run as subprocesses; keep them off the dev machine's GPU."""
    # "-1", not "": an empty value unsets the variable on Windows instead of hiding the GPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def make_orchestrator(project, questioner=None):
    ctx = StageContext(
        project=project,
        llm=FakeLLM([]),  # empty script -> every call raises LLMError -> graceful fallback
        questioner=questioner or AutoApproveQuestioner([]),
        explainer=None,
        display=lambda s: None,
        display_figure=lambda path, caption="": None,
    )
    return Orchestrator(ctx, [IntakeStage(), DataStage(), CleanStage(), CodegenStage(),
                              TrainStage(), TuneStage(), ReportStage()])


def small_config(project) -> None:
    """Keep the real training run inside a few seconds."""
    project.write_json("config.json", {**project.read_json("config.json"), "epochs": 2,
                                       "batch_size": 16})


def advance_to_codegen(orch, project, answers) -> list[str]:
    """Like the `advance` fixture, but stops the moment codegen completes, so the test can
    call `small_config` before the real `train.py` subprocess starts."""
    ran: list[str] = []
    for _ in range(12):
        ran += orch.run(until="codegen", answers=answers)
        if "codegen" in ran:
            return ran
        handoff = orch.waiting()
        assert handoff is not None, "pipeline stalled before reaching codegen"
        for command in handoff.commands:
            result = subprocess.run(
                [sys.executable, *command], cwd=str(project.root), capture_output=True,
                text=True, encoding="utf-8", timeout=300,
            )
            assert result.returncode == 0, result.stdout + result.stderr
    raise AssertionError("pipeline did not reach codegen")


def test_the_image_pipeline_runs_through_every_handoff(project, advance):
    orch = make_orchestrator(project)
    ran = advance_to_codegen(orch, project, FORM_ANSWERS)
    small_config(project)
    ran += advance(orch, project, answers=FORM_ANSWERS)
    assert ran == ALL_STAGES

    assert project.read_json("spec.json")["task_type"] == "image_classification"
    meta = project.read_json("data_meta.json")
    assert meta["modality"] == "image" and meta["target"] == "label"
    assert meta["image_size"] == 32 and meta["n_channels"] == 3
    assert meta["raw_path"] == "data/raw/data.npz"
    assert meta["clean_path"] == "data/clean/data.npz"
    assert meta["feature_columns"] == [] and meta["dropped_columns"] == []
    assert meta["n_classes"] == 3 and len(meta["class_labels"]) == 3

    for name in ("profile.py", "profile_raw.json", AUDIT_FILE, CLEAN_PY,
                 "profile_clean.json", "data.py", "model.py", "train.py", "evaluate.py",
                 "config.json", "metrics.json", "eval_val.json", "eval_test.json",
                 "runs.jsonl", "report.md", "report_meta.json"):
        assert project.exists(name), name
    assert (project.data_raw / "data.npz").exists()
    assert (project.data_raw / "manifest.csv").exists()
    assert (project.data_clean / "manifest.csv").exists()
    assert read_pair(project.data_clean).n_images <= 60
    assert (project.checkpoints_dir / "best.pt").exists()
    assert (project.checkpoints_dir / "run1.pt").exists()
    for figure in ("raw_thumbnails.png", "raw_class_means.png", "raw_intensity.png",
                   "clean_before_after_classes.png", "run1_training.png",
                   "run1_val_misclassified.png", "test_confusion.png"):
        assert (project.plots_dir / figure).exists(), figure

    runs = read_runs(project.runs_path)
    assert len(runs) == 1 and runs[0]["status"] == "done"
    assert runs[0]["checkpoint"] == "checkpoints/run1.pt"
    assert project.read_json("metrics.json")["device"] == "cpu"
    report = project.report_path.read_text(encoding="utf-8")
    assert "image_classification" in report
    assert "32" in report and "image" in report.lower()

    # Deleting state.json: every stage is complete via its artifacts, so nothing reruns.
    (project.root / "state.json").unlink()
    assert orch.run() == []


def test_one_guided_tuning_round_on_an_image_run_then_stop(project, advance):
    answers = {k: v for k, v in FORM_ANSWERS.items() if k != "tune.action"}
    answers["intake.target_value"] = 1.5   # unreachable: the loop must not end on target_met
    orch = make_orchestrator(
        project, questioner=AutoApproveQuestioner([apply_label(1), STOP_LABEL])
    )
    ran = advance_to_codegen(orch, project, answers)
    small_config(project)
    ran += advance(orch, project, answers=answers)
    assert ran == ALL_STAGES

    runs = read_runs(project.runs_path)
    assert [r["run_id"] for r in runs] == [1, 2]
    assert runs[0]["applied_diff"] is None and runs[1]["applied_diff"]
    assert set(runs[1]["applied_diff"]) <= set(project.read_json("config.json"))
    assert runs[1]["checkpoint"] == "checkpoints/run2.pt"
    assert (project.runs_dir / "run2_metrics.json").exists()
    assert (project.plots_dir / "compare_curves.png").exists()
    assert (project.plots_dir / "compare_runs.png").exists()
    state = project.read_json("tune_state.json")
    assert state["decision"] == "stopped" and state["round"] == 1
    best = max(runs, key=lambda r: r["best_val_metric"])
    assert project.read_json("report_meta.json")["best_run"] == best["run_id"]
    assert project.read_json("eval_test.json")["run_id"] == best["run_id"]


def test_the_expert_level_image_run_still_produces_every_figure(project, advance):
    answers = {**FORM_ANSWERS, "intake.learning_level": "Expert - just the numbers"}
    orch = make_orchestrator(project)
    ran = advance_to_codegen(orch, project, answers)
    small_config(project)
    ran += advance(orch, project, answers=answers)
    assert ran == ALL_STAGES
    assert project.read_json("spec.json")["learning_level"] == "expert"
    for figure in ("raw_thumbnails.png", "raw_class_balance.png", "raw_intensity.png",
                   "raw_class_means.png"):
        assert (project.plots_dir / figure).exists(), figure
```

Add to `tests/test_report_stage.py`:

```python
def test_render_report_data_section_for_a_table():
    from mlagent.stages.report import render_report

    report = render_report(
        "demo", {"goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
                 "target_value": 0.9},
        [], None, {"metric": "accuracy", "value": 0.8}, "lessons", [],
        meta={"clean_n_rows": 400, "clean_n_cols": 9, "n_classes": 2},
    )
    assert "## Data" in report
    assert "400 rows" in report and "9 columns" in report


def test_render_report_data_section_for_images():
    from mlagent.stages.report import render_report

    report = render_report(
        "demo", {"goal": "g", "task_type": "image_classification", "metric": "accuracy",
                 "target_value": 0.9},
        [], None, {"metric": "accuracy", "value": 0.8}, "lessons", [],
        meta={"modality": "image", "clean_n_rows": 240, "image_size": 64, "n_classes": 3,
              "class_labels": ["cat", "dog", "fox"]},
    )
    assert "240 images" in report
    assert "64x64" in report
    assert "3 classes" in report


def test_render_report_without_meta_omits_the_data_section():
    from mlagent.stages.report import render_report

    report = render_report(
        "demo", {"goal": "g", "task_type": "tabular_classification", "metric": "accuracy",
                 "target_value": 0.9},
        [], None, {"metric": "accuracy", "value": 0.8}, "lessons", [],
    )
    assert "## Data" not in report
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_pipeline_e2e_images.py tests/test_report_stage.py -q
```
Expected: the three report tests fail with `TypeError: render_report() got an unexpected keyword argument 'meta'`; the three e2e tests fail on `assert "32" in report and "image" in report.lower()` (no Data section) once the rest of the pipeline is green from Tasks 1-12.

- [ ] **Step 3: Write minimal implementation**

In `mlagent/stages/report.py`, add the optional `meta` parameter and the section. Change
the signature (line 31) to:

```python
def render_report(project_name: str, spec: dict, runs: list[dict], best: dict | None,
                  eval_test: dict, lessons: str, figures: list[Path],
                  meta: dict | None = None) -> str:
```

add the helper above it:

```python
def _data_section(meta: dict) -> list[str]:
    """One line describing the dataset the runs were trained on, per modality."""
    if meta.get("modality") == "image":
        size = meta.get("image_size")
        labels = ", ".join(str(c) for c in (meta.get("class_labels") or []))
        detail = (
            f"{_fmt(meta.get('clean_n_rows'))} images at {size}x{size} pixels across "
            f"{_fmt(meta.get('n_classes'))} classes"
        )
        if labels:
            detail += f" ({labels})"
    else:
        detail = (
            f"{_fmt(meta.get('clean_n_rows'))} rows and "
            f"{_fmt(meta.get('clean_n_cols'))} columns after cleaning"
        )
    return ["## Data", "", f"Trained on {detail}.", ""]
```

and split the `lines = [...]` literal (lines 35-60) so the data section can be spliced in
between the `**Task:** ...` block and `"## Run history"`, rather than appended after the
whole list is already built:

```python
    lines = [
        f"# {project_name}: training report",
        "",
        f"**Goal:** {spec.get('goal', '')}",
        "",
        f"**Task:** {spec.get('task_type', '')} | **Metric:** {metric} | "
        f"**Target:** {_fmt(spec.get('target_value'))}",
        "",
    ]
    if meta:
        lines += _data_section(meta)
    lines += [
        "## Run history",
        "",
        summarise(runs, metric),
        "",
        "## Best configuration",
        "",
        f"Run {(best or {}).get('run_id', '-')} with validation {metric} "
        f"{_fmt((best or {}).get('best_val_metric'))}:",
        "",
        "```json",
        json.dumps(best_cfg, indent=2, sort_keys=True),
        "```",
        "",
        "## Held-out test result",
        "",
        f"Test {eval_test.get('metric', metric)}: **{_fmt(eval_test.get('value'))}** "
        f"(loss {_fmt(eval_test.get('loss'))}). Evaluated once on the test split.",
    ]
```

The rest of the function (the checkpoint line and everything from `lines += [` at the old
line 64 onward) is unchanged.

In `ReportStage.debrief`, pass the meta through at the `render_report(` call (line 196):

```python
        report = render_report(
            ...,  # every existing argument, unchanged
            meta=ctx.project.read_json(META_FILE) or {},
        )
```

with `from mlagent.stages.data import META_FILE` added to the report stage's imports.

- [ ] **Step 4: Add the Milestone 6a smoke checklist**

Append to `docs/colab-smoke.md`:

```markdown
## Milestone 6a: image classification

Six runs, each on a fresh project name. Steps 1-5 replace the tabular answers at *1.
Project and interview* with **TASK = Image classification**; the cell list is unchanged.

1. **Synthetic shapes, end to end at beginner level, with one tune round.** TASK
   *Image classification*, LEARNING_LEVEL *Beginner*, DATA_SOURCE *Synthetic data*,
   N_IMAGES `300`, IMAGE_SIZE `64`, N_CLASSES `3`, INJECT_QUIRKS on. The *2. Data* cell
   writes `data/raw/data.npz` and `manifest.csv`; `profile.py` shows four figures --
   thumbnails, class balance, intensity, class means -- each with its caption. The audit
   offers duplicate and blank image drops; approve them. `clean.py` writes
   `data/clean/data.npz` and the before/after class chart. Pick *Small CNN* at *4. Model*,
   run `train.py` and `evaluate.py`, then *6. Tune*: apply proposal 1, rerun both cells,
   run *6. Tune* again, then answer *Stop tuning and write the report*. Confirm
   `runs.jsonl` has two runs, `checkpoints/run2.pt` exists (note the `.pt`, not
   `.joblib`), and `report.md`'s **Data** section says "300 images at 64x64 pixels".
2. **Your own Drive folder, including a deliberately bad file.** Make
   `MyDrive/smoke_images/` with two class subfolders of about 30 pictures each, plus one
   file that is not really an image (`echo hi > MyDrive/smoke_images/catsbroken.png`) and
   one stray picture sitting in the root with no class folder. DATA_SOURCE *Upload or
   Google Drive path*, DRIVE_FOLDER `MyDrive/smoke_images`, MAX_IMAGES `0`. The data cell
   must finish, saying how many files it skipped, and the audit's **unreadable_files**
   issue must list both the broken file and the stray one with reason "no class folder".
   Nothing crashes.
3. **A HuggingFace dataset with a cap.** DATA_SOURCE *HuggingFace Hub dataset*,
   HF_DATASET `beans`, IMAGE_SIZE `64`, MAX_IMAGES `300`. Confirm the download runs, the
   cap holds (`data_meta.json`'s `raw_n_rows` is 300), and every class survives the cap
   in the class-balance figure.
4. **ResNet-18 with ImageNet weights on the GPU runtime.** Runtime > Change runtime type >
   T4 GPU, then rerun from *1. Project and interview* with GPU *T4 GPU*. At *4. Model*
   pick *Pretrained ResNet-18* and keep `pretrained: imagenet`. Confirm the weights
   download once, `metrics.json` records `"device": "cuda"`, and one epoch is
   substantially faster than the same model on the CPU runtime.
5. **Expert level stays terse and still draws everything.** LEARNING_LEVEL *Expert*: no
   preamble, no primer, no per-figure notes -- but all four profile figures and all three
   evaluation figures (confusion, per class, misclassified) are still produced with their
   fixed captions.
6. **Train again after a tuning loop on an image run.** After step 1's project, edit
   `config.json` by hand (say `epochs: 4`), run the *Train again* cell, then the two train
   cells, then *6. Tune*. `tune_state.json` disappears and comes back at round 1, the run
   table lists every earlier run, and the new run archives as `checkpoints/run3.pt`.
```

- [ ] **Step 5: Update `CLAUDE.md`**

- In "What this is": add the Milestone 6a spec to the design-spec list and change the
  pipeline sentence to note that each stage runs for tables or for images.
- In the Pipeline bullet, after the `data` sentence insert: "For an image task the same
  stages take a different modality record (`mlagent/modality.py`): `data` writes
  `data/raw/data.npz` plus `manifest.csv` and copies `templates/image_common/profile.py`
  as `profile.py`; `clean` audits with `audit_images.py`, renders
  `templates/image_common/clean.py` and writes `data/clean/data.npz`; `codegen` copies
  `templates/image_torch/*` and offers `tiny_cnn` / `small_cnn` / `resnet18`."
- Add a bullet after the `data_meta.json` bullet: "For image runs `data_meta.json` also
  carries `modality: \"image\"`, `image_size`, `n_channels` and `skipped_files`; `target`
  is the fixed string `\"label\"`, `raw_n_rows` is the image count, `raw_n_cols` is
  H x W x 3, and `dropped_columns` / `feature_columns` / `categorical_columns` are empty
  lists. A `data_meta.json` with no `modality` key is tabular."
- Extend the generated-scripts bullet: "`mlagent/templates/image_common/{profile,clean}.py`
  and `mlagent/templates/image_torch/{data,model,train,evaluate}.py` follow the same
  conventions and additionally import torch; `torchvision` is imported lazily inside
  `model.py`'s ResNet builder so the two CNN families never need it. Image `train.py`
  writes `metrics.json[\"device\"]` and `metrics.json[\"checkpoint\"]`
  (`checkpoints/best.pt`); `runs.archive_run` preserves that suffix."
- Extend the pure-modules bullet with: "`modality.py` (the registry), `imageset.py`,
  `synth/images.py`, `audit_images.py`, `cleaning_images.py`,
  `datasources/drive_images.py`, `datasources/hf_images.py`."
- In Commands, add:
  ```
  python -m pytest tests/test_template_evaluate_images.py -v   # the image templates, run for real
  python -m pytest tests/test_pipeline_e2e_images.py -v   # the image pipeline including a real CPU training run
  ```
  and extend the deprecation-warning line to
  `python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py tests/test_template_profile_images.py tests/test_template_evaluate_images.py`.
- In Conventions, add: "`torch` lives in the `images` extra and in `dev`; `torchvision`
  lives only in the `images` extra. Every test that runs an image template sets
  `CUDA_VISIBLE_DEVICES=\"-1\"` (not `\"\"`, which unsets the variable on Windows) and every
  `resnet18` test run uses `pretrained=none`, so the suite never downloads weights."

- [ ] **Step 6: Run everything**

```
python -m pip install -e ".[dev]"
python scripts/build_notebook.py
python -m pytest -q
python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py tests/test_template_profile_images.py tests/test_template_evaluate_images.py -q
ruff check .
```
Expected: the notebook rebuild is a no-op diff, the whole suite passes, the chart tests are warning-free, and ruff is clean.

- [ ] **Step 7: Commit**

```bash
git add tests/test_pipeline_e2e_images.py tests/test_report_stage.py mlagent/stages/report.py docs/colab-smoke.md CLAUDE.md
git commit -m "feat: image pipeline e2e test, the report data section, the 6a smoke list and the docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DjW1P51vebWfHUZqQ2fW3c"
```

---

## Self-review

**Spec coverage — every section, and the task that implements it.**

| Spec section | Task |
|---|---|
| Decisions 1-6 (three sources, three models, `.npz` + manifest storage, tuning works, torch on CPU, registry not inline branches) | 1, 2, 3, 4, 8, 9, 11 |
| Section 1 — `mlagent/modality.py`, the `Modality` dataclass, `modality_for` raising on an unknown type, the tabular record wrapping existing functions | 1 |
| Section 1 — stage integration: data/clean/codegen use the record, `TEMPLATE_FOR_TASK` from the family, `DataStage`'s `NotImplementedError` guard removed | 1 (tabular wiring), 7 (image branch) |
| Section 1 — `Spec.TASK_TYPES` and intake already offer image classification, no new intake question | verified in `mlagent/spec.py:8,16` and `mlagent/stages/intake.py:25`; only the notebook's TASK field changes (Task 12) |
| Section 1 — `ImageSet`, sorted class names, the Drive layout, ingest normalisation (RGB, centre crop, resize, 32/64/128 default 64), pre-resize sizes in the manifest | 2, and the `data.image_size` question in 7 |
| Section 1 — the `data_meta.json` table, including `modality`, `image_size`, `n_channels` and source-specific keys, and tabular meta unchanged | 7 |
| Section 2 — `synth/images.py` shapes and the four quirk fields | 3 |
| Section 2 — `datasources/drive_images.py` including `skipped` with "no class folder", and `datasources/hf_images.py` with a lazy `datasets` import, feature auto-detection, `hf:{id}/{split}#{i}` sources and stratified caps | 4 |
| Section 2 — the "2. Data" cell's new fields, hints, reused `DATA_SOURCE`, "ignored for image tasks" clauses and `data.*` answer keys | 12 (fields), 7 (the keys the stage reads) |
| Section 2 — `templates/image_common/profile.py`, its JSON keys, its four figures, the caption-parity test, and `captions.py`'s new kinds | 5 |
| Section 2 — `audit_images.py`'s six checks and `drop_indices` as the only fix op | 6 |
| Section 2 — `cleaning_images.py`, the rendered `clean.py`, the unchanged clean-stage flow, the image `data_meta.json` completion and the empty-class error | 6 (modules), 7 (stage) |
| Section 3 — `config_schema.json` (no booleans; the exact common and per-model keys) | 8 |
| Section 3 — `data.py` (stratified split on `split_seed`, [0,1] scaling, per-channel normalisation, flip + shift augmentation without torchvision, loaders, `pick_device`) | 8 |
| Section 3 — `model.py` (`TinyCNN`, `SmallCNN`, lazy-`torchvision` `resnet18`, `freeze_backbone`) | 8 |
| Section 3 — `train.py`'s metrics contract plus `device`, `checkpoints/best.pt`, `metrics["checkpoint"]`, the live curve, non-finite loss handling | 9 |
| Section 3 — `evaluate.py`'s eval contract, its own `DEFAULT_CHECKPOINT`, `best_checkpoint` unchanged, the three split figures | 9 |
| Section 3 — suffix-aware `archive_run` | 10 |
| Section 4 — codegen family from the registry, `model_choices_images.md`, `recommend_model` choices from the schema, the walkthrough on the new templates | 10 |
| Section 4 — presence-checked `heuristic_proposals`, the image move table, `prompts/tune.md` naming no sklearn keys | 11 |
| Section 4 — `render_report` mentioning `image_size` and `n_images` | 13 |
| Section 4 — notebook pip line, "2. Data" fields, intro cell text, rebuild, `HANDOFF_COMMANDS` unchanged | 12 |
| Section 4 — `pyproject.toml` (`pillow` core; `images` extra; `dev`) | 1 |
| Section 4 — Testing: the six unit test files, the real-template test, the pipeline test, `-W error::DeprecationWarning`, no network | 1-9, 13 |
| Section 4 — Colab smoke, six items | 13 |
| Risks — the one-time Drive read cost, `pretrained=imagenet` never exercised in tests, 128px + ResNet on CPU, torchvision install | 4 (no caching beyond the `.npz`), 8/9 (`pretrained=none` everywhere), 10/12 (the GPU wording in the primer and the `IMAGE_SIZE` hint), 1 (the install step and its failure note) |

**Gaps found and closed while reviewing.** Three spec statements did not match the code and
became rulings 2, 5 and 10 (the image `clean.py` writes `profile_clean.json` itself;
`training_curves.png` stays in `train.py`; `render_report` gains a new Data section rather
than editing one that does not exist). Two additions the spec does not name were needed and
are rulings 3 and 9 (a `clean_before_after_classes` caption for ruling 2's figure;
`tabular_sklearn/train.py` also writing `metrics["checkpoint"]` so ruling 8's suffix-aware
archive has one source of truth for both families).

**Placeholder scan.** No "TBD", no "TODO", no "similar to Task N", no "add appropriate
error handling". Every step names the exact file and the exact command. The four places
that say "the existing body, unchanged" (`DataStage._tabular`, `CleanStage.prepare`'s
tabular half, `CleanStage.debrief`'s tabular half, `heuristic_proposals`' tabular branches)
each point at the current line range in the Files block and describe the one edit needed.

**Type and signature consistency across tasks.**
- `ImageSet(images, labels, class_names, manifest)` with `n_images` / `image_size` /
  `n_channels` / `class_counts()` / `validate()` / `take()` is defined in Task 2 and used
  with those exact names in 3, 4, 6, 7.
- `write_pair(imageset, directory) -> (Path, Path)` and `read_pair(directory) -> ImageSet`
  (Task 2) are what the `IMAGE` record's `write_raw` / `read` wrap in Task 7; `write_raw`
  returns the `.npz` path (the first element), matching the tabular `_write_table`'s
  `-> Path`.
- `stratified_indices(labels, max_images, seed=0)` (Task 2) is called with that argument
  order by both loaders in Task 4.
- `load_folder(path, image_size, max_images=None) -> (ImageSet, skipped)` and
  `load_image_dataset(dataset_id, image_size, split="train", max_images=None, ...)`
  (Task 4) are called with those signatures by `DataStage._image_drive` and
  `_image_huggingface` in Task 7, and the Task 7 test's fake loader matches
  `(dataset_id, image_size, max_images=...)`.
- `audit_images(imageset, meta, skipped=())` (Task 6) is what `CleanStage._prepare_images`
  calls through `modality.audit(imageset, meta, skipped)` in Task 7.
- `render_clean_py(steps) -> str` has the same one-argument shape in `cleaning.py` and
  `cleaning_images.py`, which is what lets `modality.render_clean_py(steps)` in Task 1's
  clean-stage edit work for both.
- `shared_file(relpath)` / `copy_shared(relpath, project_root, name=None)` (Task 1) are
  what `Modality.profile_template` / `clean_template` feed, and what
  `cleaning_images.render_clean_py` reads (Task 6).
- `build_model(config, n_classes, image_size)` (Task 8) is called with that order by
  `train.py` and by `evaluate.load_checkpoint` (Task 9).
- `evaluate_split(model, pair, device, metric, n_classes, batch_size=64)` (Task 9) is
  imported and called by `image_torch/train.py` with `(model, data["train"], device,
  metric, n_classes, batch_size)` — positionally identical.
- `make_loader(pair, batch_size, shuffle, seed=0)` (Task 8) is called by `train.py` with
  keywords and by `evaluate_split` with `(pair, batch_size, shuffle=False)`; the default
  `seed=0` covers the shuffle-free call.
- `archive_run(project, run_id, checkpoint=None)` (Task 10) is called by
  `log_finished_run` with a positional third argument; the tabular fallback keeps
  `runs.BEST_CHECKPOINT` meaningful for legacy runs.
- `heuristic_proposals(diagnosis, config, schema)` (Task 11) keeps its Milestone 5
  signature, so `mlagent/stages/tune.py:321` needs no change.
- `render_report(..., figures, meta=None)` (Task 13) adds only a trailing optional
  parameter, so `tests/test_report_stage.py:205`'s existing positional call still works.
- `labels_for(family)` / `label_for(family, code)` / `fallback_model(meta, family)`
  (Task 10) are used with those orders by `CodegenStage._choose_model` and `_recommend` in
  the same task.
