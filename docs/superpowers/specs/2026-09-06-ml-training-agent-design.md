# ML Training Agent for Google Colab — Design Spec

Date: 2026-09-06. Status: approved.

## Purpose

An assistant that acts like an AI/data engineer inside a Google Colab notebook. It interviews the user about their goal, obtains or synthesises data, audits and cleans it, generates a working training pipeline as real files, runs training, plots results, estimates Colab cost before GPU runs, coaches the user through tuning with approved config changes, and explains any technical term on click.

## Decisions

| Decision | Choice |
|---|---|
| Runtime | Inside Colab; notebook calls the Claude API (key in Colab Secrets as `ANTHROPIC_API_KEY`) |
| Model | `claude-opus-5`, adaptive thinking, server-side refusal fallbacks on; configurable in `config.py` |
| Tasks v1 | Tabular classification, tabular regression, image classification |
| Tuning | Assistant proposes config diffs with reasons; user approves each |
| Real data | Upload or Google Drive path; HuggingFace Hub |
| Audience | Single user, one notebook copy per project |
| Code delivery | Assistant writes real project files to Drive; notebook runs them in a subprocess |
| Agents | One orchestrator running a fixed stage pipeline; each stage is a specialised prompt with its own restricted tool set |
| Questions | v1 uses Colab's inline `input()` (numbered options); ipywidgets deferred because widget clicks cannot block a running cell |

## Architecture

Deliverables in this repo: `mlagent/` package (unit-tested locally with a fake LLM) and `notebooks/ML_Training_Agent.ipynb` (thin template). In Colab the setup cell mounts Drive, adds `MyDrive/ml_agent` to `sys.path`, installs deps, and reads the API key from `google.colab.userdata`.

Package modules and responsibilities:

- `config.py` paths, model id, rate-table defaults.
- `llm.py` Anthropic wrapper: manual tool-use loop, retries, refusal handling; `FakeLLM` for tests.
- `project.py` project folder, JSON artifact helpers.
- `spec.py` `Spec` dataclass (intake output) with validation.
- `ui/questions.py` `Questioner` protocol: console (`input()`), scripted (tests).
- `ui/render.py` markdown + `[[term]]` → HTML with clickable spans; Colab JS hook.
- `ui/explain.py` glossary cache, Claude explanation with project context, Colab callback.
- `orchestrator.py` stage state machine; `state.json` checkpoint/resume.
- `stages/` `base.py`, `intake.py`, `data.py`, `clean.py`, `codegen.py`, `train.py`, `tune.py`, `report.py`.
- `runner.py` subprocess execution of `train.py`, streaming, `metrics.json` polling, live plot.
- `runlog.py` `runs.jsonl` append/read/summarise.
- `cost.py` GPU detect, dry-run timing, compute-unit and money estimate.
- `synth/tabular.py`, `synth/images.py` synthetic data with injected quirks.
- `datasources/drive.py`, `datasources/hf.py`.
- `profile.py` dataset profiling. `audit.py` cleanliness checks → `Issue` list. `plots.py` all matplotlib figures.
- `templates/` reference `data.py`/`model.py`/`train.py`/`config.json` per task family.
- `prompts/` one markdown system prompt per stage plus `explain.md`.

Per-project layout on Drive:

```
MyDrive/ml_agent/
  mlagent/  rates.json
  projects/<name>/
    spec.json  state.json  glossary.json  audit.json
    data/raw/  data/clean/
    clean.py  data.py  model.py  train.py  config.json
    runs.jsonl  metrics.json  checkpoints/  plots/  report.md
```

## Stage pipeline

1. **Intake → `spec.json`.** Goal, task type, success metric and target, data source, compute budget (minutes per run, max tuning rounds), GPU preference. Fixed question set, then Claude may ask up to two follow-ups and writes the spec via a tool.
2. **Data → `data/raw/` + profile + plots.** Synthetic tabular (sklearn with quirks), synthetic images (PIL), Drive/upload pick, HuggingFace search/download.
3. **Clean → `clean.py`, `data/clean/`, `audit.json`.** Audit on every source. Tabular checks: missing values, exact/near duplicates, constant columns, mixed types, outliers, inconsistent categoricals, label issues, leakage candidates, suspicious ranges. Image checks: corrupt files, format/size inconsistencies, tiny/huge images, class imbalance, near-duplicates, stray files. Each `Issue` has kind, severity, evidence, recommended fix. Claude explains each with `[[term]]` links; user approves/edits/skips; approved steps become pure functions in `clean.py`; before/after profile shown. Then prep dialogue: target column, drops, splits, image resize/augmentation.
4. **Codegen → `data.py`, `model.py`, `train.py`, `config.json`.** Template adapted by Claude. `train.py` is self-contained, writes `metrics.json` per epoch, saves best checkpoint, supports `--dry-run`.
5. **Train.** Cost gate (GPU only), subprocess run with live loss plot, `runs.jsonl` append, evaluation plots.
6. **Tune.** Diagnosis (overfitting, underfitting, lr-too-high, plateau, data issue, target met) and 1–3 ranked config diffs bounded by the template's config schema; architecture changes route back through codegen. Run-comparison plot. User applies/edits/skips/stops.
7. **Report.** Single held-out test evaluation, `report.md` with run table, best config, lessons, figures.

## Plots

`plots.py` displays and saves to `plots/`: feature histograms, class balance, missing-value matrix, correlation heatmap, outlier box plots, thumbnail grid, before/after cleaning; live train/val loss and metric; confusion matrix, ROC/PR, per-class bars, misclassified grid; predicted vs actual, residuals; run comparison and overlaid loss curves.

## Cost gate

Estimate only (no pricing API, no balance API). Detect GPU; rate from `rates.json` (defaults T4 2.0, L4 4.8, A100 13 units/hour, $9.99 per 100 units), with a first-session prompt to enter the figure from Colab's Resources panel; 3-batch dry run; `hours = batches_per_epoch × epochs × sec_per_batch × 1.2 / 3600`; show this run and remaining rounds; free tier shows time and session-limit warning; user confirms or adjusts.

## Click-to-explain

Stage prompts wrap technical terms as `[[term]]`. Renderer emits clickable spans; in Colab a click calls a registered Python callback that asks Claude for: what it is, why it matters for this task now, why the current value was chosen, what changes if altered, one further-reading pointer. Cached in `glossary.json`; refresh re-asks; free-text explain function for unlinked terms.

## Error handling

Failing generated script → traceback to codegen → fix diff for approval, max 3 attempts. NaN loss aborts early. Data validated before codegen. API errors retried with backoff. Refusal stop reason surfaced. Runtime reset resumes from `state.json`.

## Testing

pytest with `FakeLLM`; templates trained on tiny synthetic data in tests; `audit.py` against planted-issue fixtures; `plots.py` with Agg backend; manual Colab smoke checklist per milestone in `docs/colab-smoke.md`.

## Milestones

1. Foundation: scaffold, llm, questions, render, explain, orchestrator, intake, notebook, docs.
2. Tabular data + cleaning.
3. Tabular training end-to-end with plots and report.
4. Tuning loop.
5. Image tasks + cost gate.
