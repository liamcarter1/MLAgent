# Milestone 6a: Image classification tasks

Date: 2026-09-10. The original roadmap listed Milestone 6 as "Image tasks and cost gate" (design spec `2026-09-06-ml-training-agent-design.md` line 96; sketched in the Milestone 4 spec, `2026-09-08-milestone-4-learning-mode-design.md` lines 119-121: `synth/images.py`, image audit checks, `templates/image_torch/{data,model,train,evaluate}.py`, thumbnail-grid and misclassified-grid captions, an image `profile.py` variant, and a cost gate). On 2026-09-10 the user split that outline in two: 6a is image tasks (this spec), 6b is the Colab cost gate (GPU detection, `rates.json`, `train.py --dry-run` timing, estimate-and-confirm in `TrainStage.prepare`). Milestones 1-5 are on `main` (`3de7572`). Branch: `milestone-6a-image-tasks` from `main`.

## Goal

A beginner can pick "image classification" at intake and go through the whole existing pipeline (`intake -> data -> clean -> codegen -> train -> tune -> report`) on image data, with the same two-phase stages, notebook cells, teaching, captions and tuning loop that tabular tasks already have, using synthetic shapes, their own Drive folder of images, or a HuggingFace image dataset. The user's stated motive: after learning on synthetic shapes they want to upload their own images and train on them.

## Non-goals

- The cost gate and `--dry-run` (Milestone 6b).
- Image regression, detection, segmentation, multi-label classification.
- A `tabular_torch` family, or any change to how tabular tasks train.
- New notebook cells. The cell list stays as it is; only the "2. Data" cell's form fields grow.

## Decisions the user made (2026-09-10)

1. All three data sources: synthetic shapes, a Drive folder with one subfolder per class, and a HuggingFace image dataset.
2. Three model choices: tiny CNN, small CNN, pretrained ResNet-18.
3. Storage as one `.npz` plus `manifest.csv` under `data/raw` and `data/clean`, mirroring the tabular `data.csv` pair.
4. The Milestone 5 tuning loop must fully work on image runs — no separate image tuning stage.
5. `torch` on CPU is an acceptable new dev dependency.
6. Integration is via a modality registry (`mlagent/modality.py`), not inline `if task_type == "image_classification"` branches scattered through the stages, and not separate image-specific stages.

## Section 1 — Modality registry

### `mlagent/modality.py`

A frozen dataclass `Modality` describes everything a stage needs to treat tabular and image data uniformly:

```python
@dataclass(frozen=True)
class Modality:
    name: str                      # "tabular" | "image"
    task_types: tuple[str, ...]    # e.g. ("tabular_classification", "tabular_regression")
    data_file: str                 # "data.csv" | "data.npz"
    profile_template: str          # source path under mlagent/templates, e.g. "common/profile.py"
    clean_template: str            # e.g. "common/clean.py" | "image_common/clean.py"
    template_family: str           # "tabular_sklearn" | "image_torch"
    profile_figures: tuple[str, ...]   # ordered figure kinds the profile draws
    generate: Callable[..., object]
    load_drive: Callable[..., object]
    load_hf: Callable[..., object]
    audit: Callable[..., object]
    apply_steps: Callable[..., object]
    render_clean_py: Callable[..., str]
    write_raw: Callable[..., object]
    read: Callable[[Path], object]
```

`profile_template` and `clean_template` name the *source* file the modality copies (`common/profile.py` for tabular, `image_common/profile.py` for image); the destination file the stage writes into the project is always `profile.py` and `clean.py` regardless of modality, so notebook cells and `HANDOFF_COMMANDS` never change.

`modality_for(task_type: str) -> Modality` is the only lookup function; passing an unknown task type raises a `ValueError` naming the task type and the valid ones. Two module-level records are registered at import time:

- The tabular record wraps the existing modules unchanged: `generate` = `synth.tabular.generate`, `load_drive` = `datasources.drive.load_table`, `load_hf` = `datasources.hf.load_tabular`, `audit` = `audit.audit_tabular`, `apply_steps` = `cleaning.apply_steps`, `render_clean_py` = `cleaning.render_clean_py`. `data_file` = `"data.csv"`, `template_family` = `"tabular_sklearn"`.
- The image record wraps the new modules from Sections 2-3 below. `data_file` = `"data.npz"`, `template_family` = `"image_torch"`.

### Stage integration

`DataStage`, `CleanStage` and `CodegenStage` call `modality_for(spec.task_type)` and use the record's callables instead of importing `synth.tabular`, `datasources.drive`, `datasources.hf`, `audit` or `cleaning` directly. `codegen.TEMPLATE_FOR_TASK` (currently `{"tabular_classification": "tabular_sklearn", "tabular_regression": "tabular_sklearn"}` in `mlagent/templates_io.py`) is replaced by `modality_for(spec.task_type).template_family`; `TEMPLATE_FOR_TASK` itself stays as a thin dict built from the registry so any code that still imports it keeps working.

`DataStage.prepare`'s current guard —

```python
if spec.task_type not in TABULAR_TASKS:
    raise NotImplementedError(
        f"{spec.task_type} data is not supported yet (image tasks arrive in Milestone 6)"
    )
```

— is removed; `TABULAR_TASKS` becomes redundant for that purpose and the stage instead branches on `modality_for(spec.task_type).name` where it needs modality-specific behaviour (choosing which synth config fields to ask for, etc.).

`Spec.TASK_TYPES` already includes `"image_classification"` (`mlagent/spec.py`) and `METRICS_FOR_TASK["image_classification"] = ["accuracy", "f1"]`; intake already offers "Image classification" as a task-type choice (`mlagent/stages/intake.py`). No new intake question is added; the GPU/notes prose intake shows may mention images, but the question flow is unchanged.

## Section 1 — Image data contract

### In memory: `ImageSet`

```python
@dataclass
class ImageSet:
    images: np.ndarray       # uint8, shape (N, H, W, 3)
    labels: np.ndarray       # int, shape (N,)
    class_names: list[str]   # sorted folder/feature names; labels index into this list
    manifest: pd.DataFrame   # columns: index, label, class_name, source, source_width, source_height
```

Class names are always sorted (folder names for Drive, feature names for HuggingFace, the fixed shape list for synthetic) so `labels[i]` is a stable index into `class_names`.

### On Drive

- Raw: `data/raw/data.npz` (arrays `images`, `labels`) plus `data/raw/manifest.csv`.
- Clean: `data/clean/data.npz` plus `data/clean/manifest.csv`, written by the clean stage's generated `clean.py`.
- Raw is never modified, exactly as for tabular data.

### Ingest normalisation

Every source converts images to RGB, centre-crops to a square, and resizes to `image_size` — a choice of 32, 64 or 128 pixels, default 64, asked as a form field in the "2. Data" cell. The manifest records each source image's width and height *before* resizing, so the profile and audit can report on upscaling.

### `data_meta.json`

Every key the Milestone 3 plan froze for tabular data still applies with an image-appropriate meaning documented here, plus new keys for image runs:

| Key | Tabular meaning | Image meaning |
|---|---|---|
| `target` | target column name | fixed value `"label"` |
| `task_type` | e.g. `tabular_classification` | `image_classification` |
| `source` | `synthetic` \| `drive` \| `huggingface` | same |
| `raw_path` | `data/raw/data.csv` | `data/raw/data.npz` |
| `raw_n_rows` | row count | number of images, N |
| `raw_n_cols` | column count | H × W × 3 pixel values per image |
| `clean_path`, `clean_n_rows`, `clean_n_cols` | as above, post-clean | as above, post-clean |
| `dropped_columns` | dropped column names | `[]` (cleaning drops rows, not columns) |
| `feature_columns` | model input columns | `[]` (not meaningful for images) |
| `categorical_columns` | categorical column names | `[]` |
| `splits`, `split_seed` | train/val/test fractions and the frozen split seed | unchanged in meaning |
| `n_classes`, `class_labels` | classification only | always present |
| `modality` | absent | `"image"` |
| `image_size` | absent | the chosen pixel size (32/64/128) |
| `n_channels` | absent | `3` |

Source-specific keys (`synth_config`, `source_path`, `hf_id`) mirror the tabular contract's fields of the same purpose. Tabular `data_meta.json` is unchanged: it gains no `modality` key, and every reader of `data_meta.json` defaults to tabular behaviour when `modality` is absent, so existing tabular projects and tests are unaffected.

## Section 2 — Sources

### `mlagent/synth/images.py`

`generate(cfg) -> ImageSet` draws 2-5 shape classes (circle, square, triangle, star, cross) with PIL: random size, position, rotation and colour on a noisy background, deterministic given a seed. `cfg` fields: `n_images`, `image_size`, `n_classes`, `seed`, `noise`, `class_imbalance`, `duplicate_fraction`, `blank_fraction` (near-constant images). The last four injected-quirk fields mirror `synth/tabular.py`'s `SynthTabularConfig` quirks (missing values, duplicates, outliers, etc.) so the same class-imbalance, duplicate and blank-image checks that the audit performs have something to find.

### `mlagent/datasources/drive_images.py`

`load_folder(path, image_size, max_images=None) -> (ImageSet, skipped)` walks `<class>/<file>` for `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` (case-insensitive). Unreadable files, non-image files, and files placed directly in the root folder (no class subfolder) are never raised on; they are collected in `skipped` as `(path, reason)` pairs, with the root-level case reported as `"no class folder"`. The class name is the immediate subfolder name. `max_images` caps the total with stratified sampling across classes.

### `mlagent/datasources/hf_images.py`

`load_image_dataset(dataset_id, image_size, split="train", max_images=None, image_column=None, label_column=None, loader=None) -> ImageSet`. The `datasets` import is lazy (as `datasources/hf.py` already does for tabular) so importing `mlagent` never requires it installed. It auto-detects the image feature and a `ClassLabel` feature when `image_column`/`label_column` are not given, taking class names from the `ClassLabel` feature. `max_images` applies the same stratified cap as the Drive loader. `loader` is an injectable callable so tests never hit the network; the default calls `datasets.load_dataset`. Each manifest `source` value is `f"hf:{dataset_id}/{split}#{i}"`.

### Notebook "2. Data" cell fields

New `#@param` fields are added to the existing "2. Data" cell alongside the tabular ones (`N_FEATURES`, `TARGET_COLUMN`, `DRIVE_PATH`, `HF_QUERY`, etc.): `N_IMAGES` (int, synthetic only), `IMAGE_SIZE` (choice `32`/`64`/`128`, default `64`), `DRIVE_FOLDER` (string, Drive only — the class-subfolder root), `HF_DATASET` (string, HuggingFace only — the dataset id). The existing `DATA_SOURCE` field is reused rather than adding a second source selector. Every new field gets a hint comment under it in the beginner-first style the existing fields use (see `beginner-first-notebook-ux` project note: a hint under every form box). The hints on `N_FEATURES` and `TARGET_COLUMN` gain a trailing clause — "ignored for image tasks" — so a user who switches task type mid-notebook is not confused by boxes that no longer apply; those two fields are passed through as before but the data stage does not read them when `modality_for(spec.task_type).name == "image"`. The cell passes the new fields as answers keyed `data.n_images`, `data.image_size`, `data.drive_folder`, `data.hf_dataset`, following the existing `data.*` key convention.

## Section 2 — Profile

`mlagent/templates/image_common/profile.py` is copied into the project as `profile.py` (the same destination name tabular uses), following the same conventions as `mlagent/templates/common/profile.py`: a `# --- settings ---` block with a `SCRIPT_NAME` constant, `# --- Title ---` section markers for `codewalk.split_sections`, a `cli_argv()` guard, and a `__main__` guard that never calls `sys.exit(0)`. It imports only numpy, pandas, matplotlib and PIL — never `mlagent`, and IPython only inside `try/except`.

It reads the `.npz` and `manifest.csv` pair and writes `profile_raw.json` with: `n_images`, `image_size`, `n_channels`, `n_classes`, `class_counts`, `duplicate_images` (count found by exact content hash), `blank_images` (count with per-image pixel standard deviation below a fixed threshold), per-channel mean and standard deviation intensity, and source width/height min/median/max (from the manifest, before resizing). Its `figures` list names four PNGs:

| Figure | Caption kind | Content |
|---|---|---|
| `thumbnails.png` | `thumbnails` | grid of up to 4 example images per class, labelled |
| `class_balance.png` | `class_balance` | reuses the existing tabular kind and caption |
| `intensity.png` | `intensity` | per-channel intensity histograms |
| `class_means.png` | `class_means` | mean image per class |

The clean stage reuses the same script as `profile_clean.json`, exactly as the tabular clean stage reuses `common/profile.py` today (same script, run again against the cleaned pair).

`mlagent/captions.py` gains three new kinds: `thumbnails`, `intensity`, `class_means` (the fourth figure, `class_balance`, reuses the existing kind and caption verbatim). A fourth new kind, `misclassified`, is added for the evaluate template's figure (Section 3). `image_common/profile.py`'s own `CAPTIONS` dict mirrors the `class_balance`, `thumbnails`, `intensity` and `class_means` entries from `mlagent/captions.py`, and the existing caption-parity test pattern (`test_template_captions_match_mlagent_captions` in `tests/test_template_profile.py`, which asserts `module.CAPTIONS == {k: captions.CAPTIONS[k] for k in kinds}`) is extended to cover the new template module.

## Section 2 — Audit and clean

### `mlagent/audit_images.py`

`audit_images(imageset, meta, skipped=()) -> list[Issue]`, reusing `audit.Issue` from `mlagent/audit.py`. Checks:

- Class imbalance, using the same thresholds `audit.py` already applies to tabular class balance.
- Exact duplicate images, by hashing raw image bytes.
- Blank (near-constant) images, by per-image pixel standard deviation below a fixed threshold.
- Classes with fewer than 10 images: high severity, no automatic fix — the issue advises collecting more data.
- Upscaled tiny sources: informational, flagged when a source image's shorter side is less than `image_size / 2`.
- Unreadable files, informational, one issue summarising the `skipped` list passed in from the loader.

The only fix operation these checks propose is `drop_indices`, with params `{indices: [...], reason: str}` — dropping rows (images) from the dataset, mirroring how tabular fixes drop columns or rows via `audit.py`'s existing fix-dict shape.

### `mlagent/cleaning_images.py`

`apply_steps(imageset, steps) -> ImageSet` applies approved `drop_indices` steps. `render_clean_py(steps, ...)` writes a re-runnable `clean.py` with the same conventions as `mlagent/templates/common/clean.py` (settings block, section markers, `cli_argv()`, `__main__` guard), importing only numpy and pandas: it loads the raw `.npz`/`manifest.csv` pair, drops the approved indices, writes the clean pair, and prints a summary of what was dropped and why.

The clean stage's overall flow — writing `audit.json`, the `prompts/clean.md` / `prompts/clean_debrief.md` prompts, choosing `splits` and `split_seed`, handing off `[["clean.py"]]` — is unchanged; only the modality record's `audit`, `apply_steps` and `render_clean_py` callables differ. The clean stage's debrief completes `data_meta.json` with the image-specific keys listed in Section 1's table. If cleaning empties a class entirely (every image of some label dropped), the debrief surfaces this as an error rather than silently producing a dataset with fewer classes than `class_labels` claims.

## Section 3 — `mlagent/templates/image_torch/`

Four scripts (`data.py`, `model.py`, `train.py`, `evaluate.py`) plus `config_schema.json`, following the same file layout `tabular_sklearn/` uses.

### `config_schema.json`

Mirrors the tabular schema's `{"common": {...}, "models": {...}}` shape (see `mlagent/templates/tabular_sklearn/config_schema.json`). All types are `choice`, `integer` or `number` — never `boolean` — because `templates_io.edit_config` only handles numeric and choice keys, and `templates_io.coerce_config` only clamps numeric ranges or restricts to a choice list.

Common keys:

| Key | Type | Default | Range/choices |
|---|---|---|---|
| `model_type` | choice | `small_cnn` | `tiny_cnn`, `small_cnn`, `resnet18` |
| `epochs` | integer | `10` | 1-50 |
| `batch_size` | integer | `32` | 8-256 |
| `learning_rate` | number | `0.001` | 1e-5 - 1e-1 |
| `weight_decay` | number | `0.0001` | 0 - 0.1 |
| `early_stopping_patience` | integer | `5` | as tabular (0 disables) |
| `seed` | integer | `42` | as tabular |
| `augment` | choice | `basic` | `none`, `basic` |

Model-specific keys:

- `tiny_cnn`, `small_cnn`: `dropout` (number, default e.g. `0.3`, range 0-0.7).
- `resnet18`: `pretrained` (choice, default `imagenet`, choices `imagenet`/`none`), `freeze_backbone` (choice, default `no`, choices `yes`/`no`).

### `data.py`

Loads the clean `.npz` and `data_meta.json`, performs a stratified split using `split_seed` and the `splits` fractions from `data_meta.json` (defaulting `split_seed` to `42` when absent, exactly as the tabular `data.py` does), scales pixel values to float32 `[0, 1]` and normalises per channel, and applies augmentation (random horizontal flip, small random shift) in plain tensor ops — `tiny_cnn` and `small_cnn` never import `torchvision`, only `resnet18`'s builder does. Builds `DataLoader`s. `pick_device()` returns `"cuda"` if `torch.cuda.is_available()` else `"cpu"`.

### `model.py`

`build_model(config, n_classes, image_size)` returns one of:

- `TinyCNN`: two convolutional blocks.
- `SmallCNN`: three convolutional blocks with batch norm and dropout (`dropout` from config).
- `resnet18`: built via a lazy `import torchvision` inside the builder function (never at module scope), so `tiny_cnn`/`small_cnn` runs and tests never require `torchvision` installed. Pretrained ImageNet weights are downloaded only when `pretrained == "imagenet"`. `freeze_backbone == "yes"` freezes every parameter except the final fully-connected layer.

### `train.py`

Writes the identical `metrics.json` contract tabular's `train.py` writes — `status`, `started_at`, `model_type`, `task_type`, `metric`, `higher_is_better`, `config`, `classes`, `n_train`/`n_val`/`n_test`, `epochs[]` (each with `epoch`, `train_loss`, `val_loss`, `train_metric`, `val_metric`), `best_epoch`, `best_val_metric`, `stopped_early`, `error`, `seconds`, `seconds_per_epoch` — plus one new key, `device` (`"cuda"` or `"cpu"`), reserved for Milestone 6b's cost gate. It writes its running-best checkpoint to `checkpoints/best.pt` (a `state_dict` plus the config and class names) and records that relative path in `metrics.json["checkpoint"]`, the same key tabular's `train.py` fills with `checkpoints/best.joblib`. `DEFAULT_CHECKPOINT` in `evaluate.py` (see below) is a per-template constant, so each family's template names its own default without shared code needing to know both extensions. A live curve is drawn via the same guarded `try/except` IPython import tabular's template uses. Non-finite loss (`nan`/`inf`) sets `status = "failed"` with the error recorded, exactly as tabular's `train.py` does. Same settings block, section markers, `cli_argv()` and `__main__` guard conventions. Imports numpy, pandas, matplotlib and torch only; `torchvision` stays inside `model.py`'s lazy import.

### `evaluate.py`

Writes the identical `eval_{split}.json` contract — `metric`, `value`, `loss`, `checkpoint`, `predictions`, `y_true`, and a `started_at` stamp copied from `metrics.json` — that tabular's `evaluate.py` writes. Its own `DEFAULT_CHECKPOINT` constant is `"checkpoints/best.pt"` (tabular's is `"checkpoints/best.joblib"`, in `mlagent/templates/tabular_sklearn/evaluate.py`); each template hardcodes only its own family's extension, so no shared code needs to branch on it. `best_checkpoint()` (used for `--split test`) already reads the checkpoint path out of `runs.jsonl` rather than assuming a name, so it needs no change for images.

Figures: `training_curves.png` (matching tabular's `CURVES_FIGURE = "training_curves.png"` in `mlagent/runs.py`) plus split-prefixed `{split}_confusion.png`, `{split}_per_class.png` (matching tabular's `f"{split}_confusion"`/`f"{split}_per_class"` naming in `mlagent/templates/tabular_sklearn/evaluate.py`), and a new `{split}_misclassified.png` — a grid of the worst mistakes with true/predicted labels, captioned by the new `misclassified` kind. Fixed captions are printed under each figure via the template's own `CAPTIONS` dict, as tabular's `evaluate.py` already does.

One shared-code change is needed for archiving: `mlagent/runs.py::archive_run` currently assumes a fixed `BEST_CHECKPOINT = "best.joblib"` filename and always copies `checkpoints/run{N}.joblib`. Both become suffix-aware: `archive_run` reads the checkpoint's actual path from the just-logged `metrics.json["checkpoint"]` (already in scope wherever `archive_run` is called) instead of the hardcoded `BEST_CHECKPOINT` name, and copies it to `checkpoints/run{N}<suffix>` — `.joblib` for tabular runs, `.pt` for image runs — preserving whatever extension the family's `train.py` used. `archive_run`'s figure globbing (`plots_dir.glob("val_*.png")`) is already extension-agnostic for PNGs and needs no change; `val_misclassified.png` archives the same way `val_confusion.png` does today.

`--dry-run` is deferred to Milestone 6b; `evaluate.py`'s `--split`/`--checkpoint` flags are otherwise identical to tabular's.

## Section 4 — Codegen, tune, report, notebook, dependencies

### Codegen

The template family comes from `modality_for(spec.task_type).template_family`. A new `mlagent/prompts/teaching/model_choices_images.md` explains tiny CNN vs. small CNN vs. transfer learning with ResNet-18 at each learning level, alongside the existing `model_choices.md` for tabular. The `recommend_model` tool's choice list is read from the schema's `model_type` choices (already family-generic, since `templates_io.model_types(schema)` reads them from whichever schema is loaded), so no tool-schema change is needed. The walkthrough continues to split the new templates by their `# --- ... ---` section markers via `codewalk.split_sections`, unchanged.

### Tune

`diagnose.heuristic_proposals` becomes presence-checked: a heuristic proposal only sets keys that exist in the flat schema passed to it, so an image-family proposal never tries to set `min_samples_leaf` on a CNN config or vice versa. New image moves, added alongside the existing tabular ones in the same function:

| Diagnosis label | Move |
|---|---|
| `overfitting` | `augment` `none` -> `basic`; `dropout` up; `weight_decay` up |
| `underfitting` / `improving` | `epochs` up; `learning_rate` up |
| `learning_rate_too_high` | `learning_rate` down; `batch_size` up |
| `plateau` | `learning_rate` down |
| `failed_run` | `learning_rate` down; `epochs` down |

`prompts/tune.md` (the `propose_diffs` prompt) must not name sklearn-specific keys such as `min_samples_leaf` in its instructions, since it is shared across families; it already receives the flat schema plus the current config as JSON, so the LLM sees the right keys for whichever family is active without a prompt change. Comparison figures (`plots.compare_curves`, `plots.compare_runs`) and `runs.log_finished_run` are unchanged, except that `archive_run`'s checkpoint copy now preserves the checkpoint's actual file suffix (`.joblib` or `.pt`) instead of assuming `.joblib`, as described in the `evaluate.py` section above.

### Report

`render_report` is already generic and needs no change beyond mentioning `image_size` and `n_images` in the data summary section, where it currently prints tabular row/column counts from `data_meta.json`.

### Notebook

`scripts/build_notebook.py`: the pip install line gains `torch torchvision` (both are preinstalled in Colab, so the added install cost there is low); the "2. Data" cell gains the fields listed in Section 2; the roadmap cell's text mentions image tasks. The notebook is rebuilt with `python scripts/build_notebook.py`. `mlagent/colab.py`'s `HANDOFF_COMMANDS` is unchanged — image runs use exactly the same cell names (`profile.py`, `clean.py`, `train.py`, `evaluate.py`) as tabular runs, because the copied files always use those destination names regardless of modality.

### Dependencies

`pyproject.toml`: `pillow` is added to the core `dependencies` list (needed by the image synth and profile modules unconditionally). `torch` and `torchvision` are added under a new `[project.optional-dependencies]` extra named `images`, and also added to the existing `dev` extra so the test suite can run the image templates for real. The dev machine has Python 3.14, `torch` 2.9.1 already importable, `torchvision` not yet installed (install needed), and a CUDA GPU present.

## Section 4 — Testing

### Unit

- `tests/test_modality.py`: registry lookup by task type, unknown task type raises, the tabular record's callables delegate to the existing `synth.tabular`, `datasources.drive`, `datasources.hf`, `audit`, `cleaning` functions unchanged.
- `tests/test_synth_images.py`: shape generation, determinism given a seed, injected duplicate/blank/imbalance quirks are present and detectable.
- `tests/test_datasources_images.py`: temp class folders including one unreadable file and one root-level stray file (both land in `skipped`, never raise); a fake HF loader injected via `loader=`; stratified `max_images` caps.
- `tests/test_audit_images.py`: each check fires on a crafted `ImageSet`; `drop_indices` is the only fix op produced.
- `tests/test_cleaning_images.py`: `apply_steps` drops the right indices; the rendered `clean.py` is run for real (subprocess) and produces the expected clean pair.
- `tests/test_template_profile_images.py`: `profile.py` run for real, producing all four figures and the `profile_raw.json` keys listed in Section 2; caption parity against `mlagent/captions.py` as tabular's `test_template_profile.py` does today.

### Templates for real

`tests/test_template_evaluate_images.py` runs `data.py` -> `model.py` (imported, not standalone) -> `train.py` -> `evaluate.py` via subprocess on the same pattern as `tests/test_template_evaluate.py`, against roughly 60 synthetic images at 32px: `tiny_cnn` for 2 epochs, `small_cnn` for 1 epoch, `resnet18` with `pretrained=none` for 1 epoch (so no network download happens in tests). Asserts the `metrics.json`/`eval_val.json` contracts and that every expected figure file exists. The subprocess environment sets `CUDA_VISIBLE_DEVICES=""` so results never depend on the dev machine's GPU. Budget: under roughly two minutes added to the full suite.

### Pipeline

`tests/test_pipeline_e2e_images.py` runs `intake -> report` with `FakeLLM`/`ScriptedQuestioner` on synthetic shapes, including one tune round (apply proposal 1) and a stop decision, following the pattern of `tests/test_pipeline_e2e.py`. The suite is run with `-W error::DeprecationWarning` to keep the new chart code warning-free, matching the project's existing `python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py` convention (this new template test file is added to that command).

Nothing in the test suite hits the network: the HF loader is always injected, and every `resnet18` test run uses `pretrained=none`.

## Section 4 — Colab smoke

A new "Milestone 6a" section is added to `docs/colab-smoke.md`, in the same step-by-step style as the existing Milestone 5 section:

1. Synthetic shapes end to end at beginner level, including one tune round.
2. A Drive folder of the user's own images, with a deliberately unreadable file and a root-level stray file present, and confirm both are reported rather than crashing the run.
3. A HuggingFace dataset given by name, with a cap on the number of images.
4. `resnet18` with `pretrained=imagenet` on the GPU runtime.
5. Expert level, terse output, all four figures still produced.
6. "Train again" after a tune loop on an image run.

## Risks / open points

- A Drive folder with thousands of files is slow to read once (many small file opens), but after the first `data.npz` write, every later stage reads the single packed file, so the slow path is a one-time cost the user accepts when they first point at their own folder. No caching beyond the written `.npz` is added.
- `resnet18` with `pretrained=imagenet` needs network access to download weights inside Colab; that is fine there, but it is never exercised in the test suite (`pretrained=none` is used throughout tests, per the Testing section above).
- `image_size=128` with `resnet18` on CPU is slow; the "2. Data" cell's `IMAGE_SIZE` hint and the model-choice teaching text both say to prefer the GPU runtime for the larger sizes and for ResNet-18.
- Python 3.14 with `torch` 2.9.1 is verified importable on the dev machine; `torchvision` still needs installing there before the image templates can be run locally.

Plan follows: `docs/superpowers/plans/2026-09-10-milestone-6a-image-tasks.md`, written via writing-plans after the user reviews this spec.
