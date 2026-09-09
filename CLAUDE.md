# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlagent` is an AI/data-engineer assistant that runs inside a Google Colab notebook. It interviews the user, obtains or synthesises data, audits and cleans it, generates real training files onto Google Drive, plots results, estimates Colab cost before GPU runs, coaches tuning with approved config diffs, and explains any clicked technical term. Pipeline: `intake -> data -> clean -> codegen -> train -> report`, each stage running in two phases with the user running the generated scripts in their own notebook cells. Design specs: `docs/superpowers/specs/2026-09-06-ml-training-agent-design.md` and `docs/superpowers/specs/2026-09-08-milestone-4-learning-mode-design.md`. Plans: `docs/superpowers/plans/`.

## Commands

```
python -m pip install -e ".[dev]"      # install with dev tools
python -m pytest                       # all tests
python -m pytest tests/test_intake.py -v            # one file
python -m pytest tests/test_intake.py::test_name -v # one test
ruff check .                           # lint
python scripts/build_notebook.py       # regenerate notebooks/ML_Training_Agent.ipynb
python -m pytest -W error::DeprecationWarning tests/test_plots.py tests/test_template_profile.py tests/test_template_evaluate.py   # keep chart output warning-free
python -m pytest tests/test_pipeline_e2e.py -v   # full pipeline including a real CPU training run
python -m pytest tests/test_template_evaluate.py -v   # the generated scripts, run for real
```

Tests never hit the network: anything that talks to Claude takes an `LLM` and tests pass `FakeLLM` (`mlagent/llm.py`).

## Architecture

- `mlagent/orchestrator.py` runs `stages/*` in order and checkpoints to `state.json` (`completed`, `current`, `forced`, `prepared`, `handoff`) so a Colab runtime reset resumes at the right phase. Each stage runs in two phases: `prepare(ctx)` interviews the user, writes scripts and explains them, and returns a `Handoff` (`stage`, `commands`, `outputs`) naming the cells the user must run; `debrief(ctx)` reads those outputs, shows figures with captions, narrates and completes the artifact. `orch.run(until=None, answers=None)` wraps the questioner in a `FormQuestioner` for that call; `orch.waiting()` returns the pending handoff; `orch.debrief(name)` forces a debrief when Drive's mtimes lag. A stage is complete when `Stage.is_complete` says its artifact exists and validates.
- Pipeline: `intake` -> `data` -> `clean` -> `codegen` -> `train` -> `report` (`mlagent/stages/`). `data` writes `data/raw/data.csv` and `data_meta.json`, copies `profile.py` and hands off `[["profile.py"]]` -> `profile_raw.json` plus four figures. `clean` audits, writes `audit.json` and a rendered `clean.py`, records `splits`/`split_seed`/`dropped_columns`, and hands off `[["clean.py"]]` -> `data/clean/data.csv` and `profile_clean.json`; its debrief completes `data_meta.json`. `codegen` shows `prompts/teaching/model_choices.md`, asks Claude for a `recommend_model`, lets the user pick one of `linear` / `random_forest` / `gradient_boosting`, copies `mlagent/templates/<family>/{data,model,train,evaluate}.py` and writes a flat `config.json` bounded by `templates_io.schema_for(schema, model_type)`. `train` hands off `[["train.py"], ["evaluate.py"]]` -> `metrics.json` and `eval_val.json`; its debrief logs the run by `metrics.json["started_at"]`, archives `run{N}` figures and checkpoints, and narrates. `report` hands off `[["evaluate.py", "--split", "test"]]` -> `eval_test.json` and writes `report.md`. Raw data is never modified; the test split is evaluated only by the report stage, once per best run.
- `data_meta.json` contract (frozen in the Milestone 3 plan): `target`, `task_type`, `source`, `raw_path`, `raw_n_rows`, `raw_n_cols` (data stage); `clean_path`, `clean_n_rows`, `clean_n_cols`, `dropped_columns`, `feature_columns`, `categorical_columns`, `splits` `{train, val, test}` fractions, and for classification `n_classes`, `class_labels` (clean stage). `split_seed` (int) freezes the train/val/test split independently of the tunable model `seed`; the clean stage now writes it. Generated `data.py` reads exactly these keys, defaulting `split_seed` to `42` if it is ever absent.
- Generated scripts (`mlagent/templates/common/{profile,clean}.py`, `mlagent/templates/tabular_sklearn/{data,model,train,evaluate}.py`) import only numpy/pandas/scikit-learn/matplotlib/joblib, never `mlagent`; IPython is imported only inside `try/except`. Each has a `# --- settings ---` block, `# --- Title ---` section markers that `codewalk.split_sections` splits for the walkthrough, a `cli_argv()` that returns `[]` under ipykernel, and a `__main__` guard that never calls `sys.exit(0)`. `mlagent/plots.py` keeps only the palette, `style_axes`, `save_figure` and `present`; every user-facing figure is drawn by a template.
- Teaching: `Spec.learning_level` (`beginner` / `intermediate` / `expert`, asked once at intake) drives `mlagent/teaching.py`. `Teaching.preamble` (skipped at expert), `Teaching.walkthrough` (per-section at beginner, per-file at intermediate, titles only at expert) and `Teaching.debrief` (one `write_debrief` call returning a narrative plus per-figure notes) are the only places stages call the LLM for explanation. `prompts_io.load_prompt(name, **params)` formats only when params are given; every stage prompt ends with `Audience: {audience}` filled from `prompts/levels/<level>.md`. `mlagent/captions.py` holds the fixed "how to read this chart" text per figure kind; `ui/render.display_figure` shows a PNG with its caption.
- Pure modules do the work and are unit-tested without stages: `synth/tabular.py`, `profile.py`, `audit.py` (checks → `Issue` with a proposed fix dict), `cleaning.py` (fix dicts → `apply_steps`; `render_clean_py` writes a re-runnable script that calls the same function), `datasources/drive.py`, `datasources/hf.py` (HuggingFace calls are lazy imports and injectable; tests never hit the network).
- `mlagent/llm.py` is the only place that calls the Anthropic SDK. It runs a manual tool-use loop: stages hand it `ToolSpec`s whose handlers do the real work (ask the user, write files). Structured results come back through tools, never by parsing prose.
- `mlagent/stages/base.py` defines `StageContext` (project, llm, questioner, explainer, display, `display_figure`, `stage`) and its `teaching()` accessor. Stages take everything from the context so they are testable with `ScriptedQuestioner` and `FakeLLM`.
- User questions go through the `Questioner` protocol (`ui/questions.py`). In Colab this is `input()`-based because widget clicks cannot block a running cell.
- Assistant text uses `[[term]]` markup. `ui/render.py` turns it into clickable spans; in Colab a click invokes the registered `mlagent.explain` callback, and `ui/explain.py` returns a cached, project-contextual explanation stored in `glossary.json`.
- System prompts live in `mlagent/prompts/*.md` and are loaded with `prompts_io.load_prompt`; never inline prompts in Python.
- `mlagent/colab.py` is the notebook's entry point (`setup`, `make_context`, `start`, `explain`). It must stay importable outside Colab.
- Per-project files live under `<drive_root>/projects/<name>/` (see `project.py` for the layout).

## Conventions

- Default model `claude-opus-5` with adaptive thinking; override with `MLAGENT_MODEL`.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere (Windows dev machine, Linux in Colab).
- Commit messages end with the Claude co-author trailer used in `docs/superpowers/plans/`.
