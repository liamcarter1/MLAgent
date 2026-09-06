# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This is a greenfield project. As of 2026-09-06 it contains only a PyCharm template `main.py` (a placeholder `print_hi` function, not real code) and `.idea/` IDE metadata. There is no packaging config, dependency file, test suite, linter config, README, or git repository yet.

The directory name indicates the intended purpose: an agent that orchestrates machine-learning training. No architecture, module layout, or conventions have been established, so nothing below should be assumed beyond what is written here.

## Commands

None are defined yet. There is no `pyproject.toml`, `requirements.txt`, `Makefile`, or test runner configuration. When these are added, record the install, run, lint, type-check, and single-test commands in this section.

## Architecture

Not yet defined. Once real modules exist, document here the big-picture flow (for example config -> data -> model -> training loop -> evaluation -> outputs), where checkpoints/logs are written, any LLM or external-service integrations, and non-obvious conventions such as config format, environment variables, seeding, and device handling.

## Keeping this file current

Update this file when the first of each of the following lands: packaging/dependency setup, entry point or CLI, test layout, and the core training/agent loop.
