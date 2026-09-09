# Milestone 5: the tuning loop

Date: 2026-09-09. Builds on the Milestone 4 design (`2026-09-08-milestone-4-learning-mode-design.md`), whose "Milestone 5 outline" this spec refines. Decisions that changed from that outline are marked **(changed)** and collected under "Decisions to confirm" at the end.

## Goal

After the first training run the assistant reads the run history, diagnoses what the curves show, proposes one to three ranked configuration changes with a level-appropriate explanation, and lets the user apply one, edit it, or stop. Applying a change re-runs the same `train.py` and `evaluate.py` cells; the next run is logged with the change that produced it; a comparison figure shows every run so far. The loop ends when the user stops, the target is met, or `max_rounds` is reached; the report stage then evaluates the best run on the test set as it does today.

What the user sees per round, in one `orch.run()` cell: a comparison figure with a caption and note, a diagnosis in plain words, the ranked proposals, and one question.

## Non-goals

- Patching generated scripts. Templates are fixed and tested; a failed run is handled by proposing a configuration change, not a code diff. The traceback-to-fix loop from the Milestone 4 outline is dropped **(changed)**.
- Hyperparameter search. One change per round, approved by the user; no grid or Bayesian search.
- Re-running codegen to switch model family. For tabular tasks the family is a `config.json` key that the existing templates already honour, so a family switch is one of the proposals the tuner can make **(changed)**.
- The cost gate (Milestone 6).

## Pipeline

`intake -> data -> clean -> codegen -> train -> tune -> report`. The `tune` stage is a script stage in the Milestone 4 sense: `prepare` interviews and hands off cells, `debrief` reads the outputs. Its handoff reuses the train cells: `Handoff("tune", [["train.py"], ["evaluate.py"]], ["metrics.json", "eval_val.json"])`.

### One round

1. `TuneStage.prepare(ctx)`:
   - Reads `runs.jsonl`, the archived per-run metrics (`runs/run{N}_metrics.json`, see below), `metrics.json`, `eval_val.json` and `tune_state.json`.
   - If the loop is over (`decision` is not `continue`), displays why and returns `None`. If the best run already meets `spec.target_value`, records `target_met`, says so, and returns `None`; the user can still edit `config.json` and use the "Train again" cell.
   - Computes a `Diagnosis` with `mlagent/diagnose.py` (pure, deterministic): the trend of validation loss over the last third of the epochs, the train/val loss gap at the best epoch, whether early stopping fired, whether the best epoch is the last epoch, whether the latest run beat the previous best, and the number of failed runs with their error class. It yields one label from `overfitting`, `underfitting`, `learning_rate_too_high`, `plateau`, `failed_run`, `improving`, `target_met`, plus the evidence numbers.
   - On the first round only, shows a preamble (below expert) and the teaching material `prompts/teaching/tuning.md` trimmed to the level: what each family's knobs do and what the four curve diagnoses look like.
   - Calls the LLM once with `prompts/tune.md` and the `propose_diffs` tool. Input: the spec, the diagnosis, the run table, the current config and `schema_for(model_type)`. Output: one to three `{rank, changes: {key: value}, reason, expected: "faster" | "better" | "steadier"}` proposals. Every value is coerced by `coerce_config` against the family schema (or the target family's schema when `changes` includes `model_type`); a proposal that coerces to no change is dropped. On `LLMError` a fixed heuristic table in `diagnose.py` supplies one proposal per label (for example `overfitting` on gradient boosting raises `min_samples_leaf` and lowers `learning_rate`).
   - Displays the diagnosis and the proposals as a table (`config_table` over each proposal's changed keys, old and new), with the level-appropriate explanation from the tool's `reason`.
   - Asks one `choice` question (key `tune.action`): `Apply proposal 1` ... `Apply proposal N`, `Edit a proposal first`, `Stop tuning and write the report`. `Edit` asks which proposal, then reuses the config edit menu (moved from `stages/codegen.py` into `templates_io.edit_config(questioner, config, schema)`), then applies. There is no separate `skip` **(changed)**.
   - On apply: writes the new `config.json`, writes `tune_state.json` with `pending = {round, applied_diff, reason}`, names the cells to run, and returns the handoff. On stop: writes `decision = "stopped"` and returns `None`.
2. The user runs the `train.py` and `evaluate.py` cells.
3. `TuneStage.debrief(ctx)`:
   - Logs the new run exactly as `TrainStage.debrief` does today. The run-logging code (the `started_at` match, the `eval_val.json` staleness check, `build_run_entry`, `archive_run`) moves from `stages/train.py` into `mlagent/runs.py` as `log_finished_run(project, applied_diff=None) -> (entry, figures) | None`, which both stages call. The entry's `applied_diff` is the pending diff from `tune_state.json`.
   - Archives `metrics.json` as `runs/run{N}_metrics.json` (added to `archive_run`, so run 1 from the train stage is archived too).
   - Draws the comparison figures agent-side with `plots.compare_curves` and `plots.compare_runs` (see Figures) and shows them with the run's evaluation figures through `Teaching.debrief("tune_debrief", ...)`, which returns the narrative: what changed, what happened, whether it helped.
   - Updates `tune_state.json`: `round += 1`, `pending = None`, a `history` entry appended; `decision` becomes `rounds_exhausted` when `round >= max_rounds`, `target_met` when the new best meets the target, otherwise stays `continue`.
   - Returns `True` when the loop should go round again, `None` otherwise.
4. `Orchestrator._run` treats a truthy debrief return as "round complete, prepare again": it removes the stage from `prepared`, clears `handoff`, and falls through to `prepare` in the same call, so the next proposal appears right under the debrief of the run it follows. Any stage may use this; today only `tune` does.

### Completion

`TuneStage.is_complete` is true when `tune_state.json` exists and `decision != "continue"`. `reset("train")` also removes `tune_state.json` (a fresh run history starts a fresh loop); `reset("tune")` resets the loop only.

Report is unchanged: it re-prepares for the best run, which the tuning loop may have moved.

## Files and contracts

- `tune_state.json` (project root): `{"round": int, "max_rounds": int, "decision": "continue" | "stopped" | "target_met" | "rounds_exhausted", "pending": null | {"round": int, "applied_diff": {key: {"from": old, "to": new}}, "reason": str, "proposed_at": iso}, "history": [{"round": int, "diagnosis": str, "applied_diff": {...}, "run_id": int}]}`.
- `runs.jsonl` entry `applied_diff`: `null` for a run the tuner did not produce, otherwise the `{key: {"from", "to"}}` dict above. `checkpoint` and the other Milestone 3 keys are unchanged.
- `runs/run{N}_metrics.json`: a verbatim copy of `metrics.json` at debrief time. `archive_run` writes it; `runs/` is created by `project.ensure_dirs()`.
- `Diagnosis` (`mlagent/diagnose.py`): `label`, `evidence: dict[str, float | bool | str]`, `latest_run_id`, `best_run_id`, `improved: bool`. Pure functions `diagnose(runs, metrics_by_run, spec) -> Diagnosis` and `heuristic_proposals(diagnosis, config, schema) -> list[Proposal]`.
- `Proposal`: `rank: int`, `changes: dict[str, object]`, `reason: str`, `expected: str`. `apply_proposal(config, proposal, nested_schema) -> (new_config, applied_diff, notes)` coerces and computes the diff; a `model_type` change resolves the target family's schema and fills its defaults.
- `propose_diffs` tool schema: `{"proposals": [{"rank": int, "changes": object, "reason": str, "expected": enum}]}`, `maxItems: 3`.
- Prompts: `prompts/tune.md` (diagnosis in, proposals out, audience line), `prompts/tune_debrief.md` (`write_debrief` with the comparison figures), `prompts/teaching/tuning.md` (level-fenced primer on knobs and curves). `prompts/train.md` keeps its last-line hint about the tuning stage.
- Captions: `compare_curves` and `compare_runs` kinds in `captions.py`.

## Figures

Drawn agent-side in `mlagent/plots.py`, which Milestone 4 reserved for exactly this. This replaces the outline's `compare.py` template so a round needs no third cell **(changed)**.

- `compare_curves(metrics_by_run, metric) -> Figure`: validation loss per epoch for every run, one series per run in the house palette, legend labels such as `run 2 (learning_rate 0.05)` built from the applied diff, best epoch marked on each.
- `compare_runs(runs, metric, target) -> Figure`: a horizontal bar per run of `best_val_metric`, the target as a dashed line, failed runs drawn hollow.
- Both saved by `present()` as `plots/compare_curves.png` and `plots/compare_runs.png`, overwritten each round; the per-run archive is the run table itself.

## Notebook and Colab

- A new markdown plus code cell after the evaluate cell: "Tune", running `orch.run(until="tune")`. It has no form field: the choice depends on proposals the user has not seen yet, so it uses the console questioner, which the Milestone 4 spec allows for follow-ups. `max_rounds` comes from intake (`Spec.max_rounds`).
- The train and evaluate cells are reused unchanged; the tune cell's output names them.
- The "Train again" cell stays for manual edits; its text says the tuner is the guided alternative.
- `HANDOFF_COMMANDS["tune"] = [["train.py"], ["evaluate.py"]]`; `PURGED_MODULES` unchanged.
- The smoke checklist gains a Milestone 5 section: one guided round, a stop, a target-met exit, a `reset("tune")`.

## Learning levels

- Expert: no preamble, no primer, proposals as a table with one-line reasons, comparison figures with fixed captions only.
- Intermediate: preamble on round 1, the primer trimmed to intermediate, reasons in a paragraph, figure notes.
- Beginner: the above plus the primer's beginner blocks and a sentence per proposal on what the knob does.
- Every LLM path degrades: heuristic proposals, fixed narrative, fixed captions.

## Orchestrator changes

- Stage list: `tune` inserted before `report` in `colab.make_context` and the notebook.
- `_run`: honour a truthy `debrief` return as described. `reset(name)` removes `tune_state.json` when `name` is `train` or `tune`.
- State compatibility: a `state.json` written before this milestone has no `tune` in `completed`; the stage simply runs after train. Nothing else changes shape.

## Testing

- `tests/test_diagnose.py`: each label from hand-written run histories; heuristic proposals stay inside the schema; `apply_proposal` with a family switch fills defaults.
- `tests/test_plots.py`: the two comparison figures render for one run, three runs and a failed run without warnings.
- `tests/test_tune_stage.py`: a scripted round (`FakeLLM` proposals, apply, `run_handoff`, debrief logs run 2 with `applied_diff`, archives `run2_metrics.json`, returns `True`); stop; target met; rounds exhausted; LLM unavailable uses the heuristic; the edit path; a family switch produces a valid config and a run.
- `tests/test_train_stage.py`: unchanged behaviour through `runs.log_finished_run`.
- `tests/test_orchestrator.py`: a stage whose debrief returns `True` is re-prepared in the same `run()`; `reset("train")` clears the loop.
- `tests/test_pipeline_e2e.py`: one guided round then stop; the report evaluates the new best run.

## Risks

- A proposal that makes the run much slower (more trees, more epochs). The tool prompt asks for changes that keep the run within `spec.minutes_per_run`; the cost gate in Milestone 6 will enforce it.
- Two LLM calls per round (proposals, debrief) plus the round-1 preamble; expert level is one call per round.
- `metrics.json` must be archived before the next run overwrites it; `archive_run` does it in the same debrief that logs the run.

## Decisions to confirm

1. A family switch is a tuner proposal, not a codegen re-run.
2. No script patching for failed runs; failed runs get configuration proposals only.
3. Comparison figures are drawn agent-side from the archived metrics, not by a `compare.py` cell.
4. `Edit` replaces `skip`: the actions are apply, edit-then-apply, stop.
5. The tune cell uses the console questioner for its one choice; `max_rounds` comes from intake.
6. A debrief may return `True` to re-prepare its stage within the same `orch.run()`.
