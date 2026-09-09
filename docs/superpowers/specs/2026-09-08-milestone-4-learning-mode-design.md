# Milestone 4 design: learning mode and notebook-first pipeline

Date: 2026-09-08. Status: approved by the user after the first Colab smoke test of Milestones 1-3.

## Why

The first real Colab run worked end to end, but the user's feedback was about learning, not correctness:

- The generated code never appears in the notebook. Codegen copies three template files to Drive and shows only their names. Profiling, plots, audit and evaluation are internal library calls the user cannot read.
- Explanations are uniform and capped. There is no "what we're about to do and why" before a step, no "what this output means" after it, and no plot captions.
- Only one model exists (scikit-learn HistGradientBoosting) and the message about it is one hardcoded sentence, so the user could not tell which kind of model was in use.
- The `input()` question UI is clunky and occasionally fails to render in Colab.

## Decisions

1. **Notebook-first.** One notebook cell per stage. The agent writes real scripts; the user loads and runs them in their own cell (`%load`-style helper isolated behind one small function so the mechanism can be swapped).
2. **Every analysis step is a script the user runs:** `profile.py`, `clean.py`, `data.py`/`model.py`/`train.py`, `evaluate.py`. Scripts import only numpy, pandas, scikit-learn, matplotlib and joblib, never `mlagent`. IPython is a soft, optional dependency (guarded import) used only for the live training curve.
3. **Three model choices, agent recommends one:** linear/logistic regression, random forest, gradient boosting.
4. **Learning level asked once at intake** (`beginner` / `intermediate` / `expert`) and stored in `spec.json`. Expert is the current terse style.
5. **Colab form fields** (`#@param`) for every fixed question; confirmations and agent-invented follow-ups keep `input()`.
6. **Every figure gets a caption:** a fixed "how to read this chart" text per figure kind plus, below expert level, an LLM sentence about this data.
7. **Sequencing:** this is Milestone 4. The tuning loop becomes Milestone 5; image tasks and the cost gate become Milestone 6.
8. **Rulings:** the clean stage does not apply fixes in-process; cleaned data exists only after the user runs `clean.py`. Templates duplicate profiling/plotting code rather than importing `mlagent`.

## Architecture

### Two-phase stages and a handoff

```python
@dataclass
class Handoff:
    stage: str
    commands: list[list[str]]   # e.g. [["train.py"], ["evaluate.py"]]
    outputs: list[str]          # project-relative files that must exist afterwards

class Stage(Protocol):
    name: str
    def prepare(self, ctx) -> Handoff | None   # interview, write scripts, explain them
    def outputs_ready(self, ctx) -> bool       # default: outputs exist and are newer than the scripts (1 s tolerance)
    def debrief(self, ctx) -> None             # read outputs, show figures + captions, narrate, write completion artifact
    def is_complete(self, ctx) -> bool
```

`ScriptStageBase` implements `outputs_ready` from the stored handoff. Intake and codegen return `None` from `prepare` and have a no-op `debrief`.

`state.json` gains `prepared: [names]` and `handoff: dict | null`. `Orchestrator.run(until=None, answers=None)` per stage: skip if complete; if not prepared, banner + `prepare()`; if no handoff, `debrief()` and mark complete; if handoff and outputs not ready, display "Run the next cell(s) ... then run this cell again" and return; if ready, `debrief()`, mark complete, clear handoff. New methods: `waiting() -> Handoff | None`, `debrief(name)` (force when mtimes lie), and `reset()` also trims `prepared`. `answers` wraps the questioner in a `FormQuestioner` for that call only.

Re-runs: re-running `train.py` produces a new `metrics.json["started_at"]`; `TrainStage.is_complete` means "at least one done run and the current `metrics.json` run is logged in `runs.jsonl`", so the next `orch.run()` logs run N+1 and the report stage re-triggers when the best run changes. Editing a script makes `outputs_ready` false. A runtime reset resumes at the right phase because `prepared` and `handoff` persist.

### Notebook layout

| # | cell | content |
|---|---|---|
| 0 | md | How it works: assistant cell, then script cell "run once to load, once to run"; click terms |
| 1-2 | code | `%pip` install; mount, `sys.path`, `colab.setup()` |
| 3 | code | `#@title 1. Project and interview` form: `PROJECT_NAME`, `LEARNING_LEVEL`, `GOAL`, `TASK`, `METRIC`, `TARGET_VALUE`, `DATA_SOURCE`, `MINUTES_PER_RUN`, `MAX_ROUNDS`, `GPU`; `orch = colab.start(PROJECT_NAME)`; `orch.run(until="intake", answers={...})` |
| 4 | code | `#@title 2. Data` form: `N_ROWS`, `N_FEATURES`, `N_CLASSES`, `CLASS_BALANCE`, `NOISE`, `INJECT_QUIRKS`, `DRIVE_PATH`, `HF_QUERY`, `TARGET_COLUMN`; `orch.run(until="data", answers=...)` |
| 5 | code | `%load profile.py` |
| 6 | code | `#@title 3. Clean` form: `DROP_COLUMNS`, `TRAIN_FRACTION`, `VAL_FRACTION`; `orch.run(until="clean", answers=...)` (per-fix confirms stay `input()`) |
| 7 | code | `%load clean.py` |
| 8 | code | `#@title 4. Model` form: `MODEL` in ["Ask me after the explanation", "Linear / logistic regression", "Random forest", "Gradient boosting"]; `orch.run(until="train", answers=...)` |
| 9-10 | code | `%load train.py`, `%load evaluate.py` |
| 11 | code | `#@title 5. Report`; `orch.run(until="report")` (confirms the test-set touch) |
| 12 | code | `%run evaluate.py --split test` |
| 13 | code | `orch.run()` (report debrief) |
| 14-16 | md/code | Train again (edit `config.json`, re-run 9-10, `orch.run()`); `colab.explain(...)`; `# orch.reset('intake')` |

`mlagent/colab.py`: `start()` also `os.chdir(project.root)` and registers a `pre_run_cell` hook that purges cached `data`/`model`/`evaluate` modules. `SCRIPT_CELL_FORMATS = {"load": "%load {script}", "run": "%run {script} {args}"}` is the single place the cell mechanism lives. `cell_source(command)` and `script_cells(stage_name)` come from a static `HANDOFF_COMMANDS` table shared with `scripts/build_notebook.py`. The module stays importable outside Colab.

### Scripts (templates)

`mlagent/templates/common/{profile.py, clean.py}` and `mlagent/templates/tabular_sklearn/{data,model,train,evaluate}.py`. All have a `# --- settings ---` constants block overridable by argparse, a `cli_argv()` that returns `[]` under ipykernel, a `__main__` guard doing `code = main(); if code: sys.exit(code)`, and sections marked `# --- Title ---` for the walkthrough.

| script | inputs | outputs |
|---|---|---|
| `profile.py` (`--input --tag`, default raw) | csv, `data_meta.json` | `profile_{tag}.json` (today's shape plus `figures`), `plots/{tag}_histograms.png`, `{tag}_missing.png`, `{tag}_class_balance.png` or `{tag}_target_distribution.png`, `{tag}_correlation.png` |
| `clean.py` (self-contained: op functions plus embedded `STEPS`) | `data/raw/data.csv` | `data/clean/data.csv`, `profile_clean.json` (before/after plus figures), `plots/clean_before_after_missing.png` |
| `train.py` | `config.json`, `data_meta.json`, `spec.json` | `metrics.json` (plus `started_at`, `model_type`), `checkpoints/best.joblib`, `plots/training_curves.png` per epoch; live redraw when IPython is present; `--dry-run` unchanged; `--eval-test` removed |
| `evaluate.py` (`--split val|test --checkpoint`) | checkpoint, data | `eval_{split}.json` (today's shape plus `checkpoint`, `run_id`, `figures`), `plots/{split}_confusion.png`, `{split}_roc_pr.png`, `{split}_per_class.png` or `{split}_pred_vs_actual.png`, `{split}_residuals.png` |

`evaluate.py` owns `compute_metric`, `full_proba` and `evaluate_split`; `train.py` imports them. `evaluate.py --split test` picks the best run's checkpoint from `runs.jsonl` unless `--checkpoint` is given. The train debrief archives `plots/training_curves.png` to `plots/run{N}_training.png`, `checkpoints/best.joblib` to `checkpoints/run{N}.joblib` and `val_*.png` to `run{N}_val_*.png`.

Duplicate, don't share: templates carry their own profiling, plotting and cleaning code. `mlagent/plots.py` loses the data and evaluation figure functions and keeps palette constants plus `present()` for Milestone 5's agent-side comparison plots. `mlagent/profile.py` keeps `profile_markdown` and `profile_dataframe`. `cleaning.apply_steps` stays as the reference implementation the template embeds and tests compare against.

### Model choice

`model.py`: `build_model(config, task_type, categorical_mask) -> EpochModel` with `fit_epoch`, `predict`, `predict_proba`, `classes_`, `estimator`.

| `model_type` | estimator | per epoch |
|---|---|---|
| `gradient_boosting` | `HistGradientBoosting*` warm start | `max_iter += iters_per_epoch` |
| `random_forest` | `RandomForest*` warm start | `n_estimators += trees_per_epoch` |
| `linear` | `SGDClassifier(loss="log_loss")` / `SGDRegressor` with median imputer and `StandardScaler` | `partial_fit`, one pass |

`config_schema.json` becomes `{"common": {model_type (choice), epochs, early_stopping_patience, seed}, "models": {gradient_boosting: {...today's keys}, random_forest: {trees_per_epoch, max_depth, min_samples_leaf, max_features}, linear: {learning_rate, alpha}}}`. `config.json` stays flat (common plus the chosen model's keys). `templates_io.schema_for(schema, model_type)` returns flat rules; `validate_config`, `coerce_config` and `default_config` take the flat form. Milestone 5's tuner is bounded by `schema_for`.

Codegen flow: display `prompts/teaching/model_choices.md` (three sections, `[[term]]`s, level-trimmed); LLM tool `recommend_model(model_type, reason)` with a fallback heuristic (`linear` if `clean_n_rows < 300`, else `gradient_boosting`); `q.choice("Which model?", ..., key="codegen.model_type")` naming the recommendation; `propose_config` bounded by `schema_for`; copy templates, write `config.json`, code walkthrough.

### Learning level, captions, code walkthrough

- `Spec.learning_level` (enum, default `intermediate`); intake asks it (`key="intake.learning_level"`).
- `prompts_io.load_prompt(name, **params)` formats only when params are given. Level guidance lives in `prompts/levels/{beginner,intermediate,expert}.md`; every stage prompt gains an `Audience: {audience}` line. New prompts `preamble.md` and `walkthrough.md`; debrief prompts gain a `write_debrief(narrative, figure_notes)` tool.
- `mlagent/teaching.py`: `Teaching(level, llm, display, display_figure)` with `preamble(stage, payload)` (beginner and intermediate), `walkthrough(paths)` (beginner: every section, one call per script; intermediate: one paragraph per file; expert: file names and section titles), `debrief(prompt_name, payload, figures)` (one call; figure notes below expert) and `show_figures(paths, notes)`. LLM failure falls back to fixed text. Calls per stage: expert 1, intermediate 3, beginner 3-4.
- `mlagent/codewalk.py`: `split_sections(source)` on `^# --- (.+) ---$`; `render_walkthrough(sections, explanations)` (fenced code, explanation under each).
- `mlagent/captions.py`: `CAPTIONS: dict[kind, str]` for every figure kind; `caption_for(path)` by filename suffix. `ui/render.display_figure(path, caption)` shows the PNG with the caption beneath, reusing `to_html` for `[[term]]`s. `StageContext.display_figure` is added.

### Questioner

`Questioner` methods gain `key: str | None = None` (ignored by `ConsoleQuestioner` and `ScriptedQuestioner`). `FormQuestioner(answers, fallback)`: `choice` matches labels case-insensitively, `number` validates bounds, `confirm` accepts bools, `text` accepts any non-empty string; `None`, `""` or an invalid value falls back with a one-line note. Keys: `intake.*`, `data.*`, `clean.drop_columns/train_fraction/val_fraction`, `codegen.model_type`.

## Testing

Existing stage tests move to `prepare -> run_handoff -> debrief`; `tests/conftest.py` gains `run_handoff(project, handoff)` (subprocess `[sys.executable, *command]`, cwd=root, asserts exit 0). The end-to-end test loops `orch.run()` / `orch.waiting()` / `run_handoff` until all stages complete, re-runs `train.py` and checks run 2 is logged, touches `profile.py` and checks `outputs_ready` is false, and runs beginner and expert levels with `FakeLLM`. Data and evaluation figure tests move from `test_plots*` to the template tests.

## Milestone 5 outline: tuning loop

Insert `tune` between `train` and `report` as a script stage: diagnose from `runs.jsonl` and `metrics.json` (overfitting, underfitting, learning rate too high, plateau, target met) via a `propose_diffs` tool bounded by `schema_for(model_type)`; show 1-3 ranked diffs with a level-appropriate explanation; the user applies, edits, skips or stops (`max_rounds`); apply writes `config.json` and `applied_diff` on the next run entry; the handoff is the same `train.py`/`evaluate.py` cells; a `compare.py` template plots overlaid curves and a run table. A model-type change routes back through codegen. Traceback-to-fix loop on failed runs (log tail plus script to a unified diff shown via the walkthrough machinery, max 3 attempts).

## Milestone 6 outline: image tasks and cost gate

`synth/images.py`, image audit checks, `templates/image_torch/{data,model,train,evaluate}.py`, thumbnail-grid and misclassified-grid captions, an image `profile.py` variant. `cost.py`: GPU detection, `rates.json`, `train.py --dry-run` timing, estimate shown by `TrainStage.prepare` as a preamble payload with a confirm (form field for the compute-unit rate). Reuses the three-way model choice pattern (CNN sizes) and `schema_for`.

## Risks

- The `%load` two-click flow may confuse beginners or misbehave in Colab; the mechanism is isolated in `SCRIPT_CELL_FORMATS` with `%run` as the fallback. The first smoke-test item verifies it.
- Drive mtimes may lag; 1 s tolerance plus the `orch.debrief(name)` escape hatch.
- Kernel-side execution: cached modules (purge hook), `sys.argv` (guard), `SystemExit` (guard), the inline backend needs `plt.show()`.
- Beginner mode costs 3-4 LLM calls per stage; the debrief tool must degrade to fixed captions on `LLMError`.
