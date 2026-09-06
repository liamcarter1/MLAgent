# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlagent` is an AI/data-engineer assistant that runs inside a Google Colab notebook. It interviews the user, obtains or synthesises data, audits and cleans it, generates real training files onto Google Drive, runs them in a subprocess, plots results, estimates Colab cost before GPU runs, coaches tuning with approved config diffs, and explains any clicked technical term. Pipeline: `intake -> data -> clean -> codegen -> train -> report`. Design spec: `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md`. Plans: `docs/superpowers/plans/`.

## Commands

```
python -m pip install -e ".[dev]"      # install with dev tools
python -m pytest                       # all tests
python -m pytest tests/test_intake.py -v            # one file
python -m pytest tests/test_intake.py::test_name -v # one test
ruff check .                           # lint
python scripts/build_notebook.py       # regenerate notebooks/ML_Training_Agent.ipynb
python -m pytest -W error::DeprecationWarning tests/test_plots.py   # keep chart output warning-free
python -m pytest tests/test_pipeline_e2e.py -v   # full pipeline including a real CPU training run
```

Tests never hit the network: anything that talks to Claude takes an `LLM` and tests pass `FakeLLM` (`mlagent/llm.py`).

## Architecture

- `mlagent/orchestrator.py` runs `stages/*` in order and checkpoints to `state.json` in the project folder so a Colab runtime reset resumes. A stage is complete when its artifact exists and validates (`Stage.is_complete`).
- Pipeline: `intake` -> `data` -> `clean` -> `codegen` -> `train` -> `report` (`mlagent/stages/`). `data` writes `data/raw/data.csv`, `profile_raw.json`, `data_meta.json`. `clean` writes `data/clean/data.csv`, `clean.py`, `audit.json`, `profile_clean.json` and completes `data_meta.json`. `codegen` copies `mlagent/templates/<family>/{data,model,train}.py` into the project (they import only numpy/pandas/scikit-learn/joblib, never `mlagent`) and writes `config.json`, whose keys are exactly those in the template's `config_schema.json`. `train` runs `train.py` in a subprocess (`runner.py`), polls `metrics.json` for the live plot, appends to `runs.jsonl` (`runlog.py`), and shows evaluation figures from `eval_val.json`. `report` runs `train.py --eval-test` once and writes `report.md`. Raw data is never modified; the test split is evaluated only by the report stage.
- `data_meta.json` contract (frozen in the Milestone 3 plan): `target`, `task_type`, `source`, `raw_path`, `raw_n_rows`, `raw_n_cols` (data stage); `clean_path`, `clean_n_rows`, `clean_n_cols`, `dropped_columns`, `feature_columns`, `categorical_columns`, `splits` `{train, val, test}` fractions, and for classification `n_classes`, `class_labels` (clean stage). Generated `data.py` reads exactly these keys.
- Pure modules do the work and are unit-tested without stages: `synth/tabular.py`, `profile.py`, `audit.py` (checks → `Issue` with a proposed fix dict), `cleaning.py` (fix dicts → `apply_steps`; `render_clean_py` writes a re-runnable script that calls the same function), `plots.py` (all matplotlib figures; `present()` saves to `plots/` and displays under IPython), `datasources/drive.py`, `datasources/hf.py` (HuggingFace calls are lazy imports and injectable; tests never hit the network).
- `mlagent/llm.py` is the only place that calls the Anthropic SDK. It runs a manual tool-use loop: stages hand it `ToolSpec`s whose handlers do the real work (ask the user, write files). Structured results come back through tools, never by parsing prose.
- `mlagent/stages/base.py` defines `StageContext` (project, llm, questioner, explainer, display). Stages take everything from the context so they are testable with `ScriptedQuestioner` and `FakeLLM`.
- User questions go through the `Questioner` protocol (`ui/questions.py`). In Colab this is `input()`-based because widget clicks cannot block a running cell.
- Assistant text uses `[[term]]` markup. `ui/render.py` turns it into clickable spans; in Colab a click invokes the registered `mlagent.explain` callback, and `ui/explain.py` returns a cached, project-contextual explanation stored in `glossary.json`.
- System prompts live in `mlagent/prompts/*.md` and are loaded with `prompts_io.load_prompt`; never inline prompts in Python.
- `mlagent/colab.py` is the notebook's entry point (`setup`, `make_context`, `start`, `explain`). It must stay importable outside Colab.
- Per-project files live under `<drive_root>/projects/<name>/` (see `project.py` for the layout).

## Conventions

- Default model `claude-opus-5` with adaptive thinking; override with `MLAGENT_MODEL`.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere (Windows dev machine, Linux in Colab).
- Commit messages end with the Claude co-author trailer used in `docs/superpowers/plans/`.
