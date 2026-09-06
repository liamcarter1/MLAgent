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
- `mlagent/colab.py` is the notebook's entry point (`setup`, `make_context`, `start`, `explain`). It must stay importable outside Colab.
- Per-project files live under `<drive_root>/projects/<name>/` (see `project.py` for the layout).

## Conventions

- Default model `claude-opus-5` with adaptive thinking; override with `MLAGENT_MODEL`.
- Write text files with `encoding="utf-8"`; use `pathlib` everywhere (Windows dev machine, Linux in Colab).
- Commit messages end with the Claude co-author trailer used in `docs/superpowers/plans/`.
