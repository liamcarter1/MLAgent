# Milestone 6b: The Colab cost gate

Date: 2026-09-11. Milestone 6 was split on 2026-09-10 into 6a (image tasks) and 6b (this
spec) — see `docs/superpowers/specs/2026-09-10-milestone-6a-image-tasks-design.md`, line 3.
6a merged to `main` as `eebfce5` on 2026-09-11. The original design spec
(`docs/superpowers/specs/2026-09-06-ml-training-agent-design.md`, "Cost gate" section,
lines 73-75) set the numbers this spec implements:

> Estimate only (no pricing API, no balance API). Detect GPU; rate from `rates.json`
> (defaults T4 2.0, L4 4.8, A100 13 units/hour, $9.99 per 100 units), with a first-session
> prompt to enter the figure from Colab's Resources panel; 3-batch dry run;
> `hours = batches_per_epoch × epochs × sec_per_batch × 1.2 / 3600`; show this run and
> remaining rounds; free tier shows time and session-limit warning; user confirms or
> adjusts.

Since 6a, `metrics.json` carries a `device` key and both train templates record `seconds`
and `seconds_per_epoch` (`mlagent/templates/tabular_sklearn/train.py`,
`mlagent/templates/image_torch/train.py`); `TrainStage.prepare`
(`mlagent/stages/train.py`) prints a placeholder sentence saying there is no cost gate for
this run, once for images and once for tabular data. Milestones 1-5 and 6a are on `main`.
Branch: `milestone-6b-cost-gate` from `main`.

## Goal

Before any training run in Colab — the train stage's first run and every tune round — the
user sees how long the run should take, how many Colab compute units it should consume and
what that costs at their price, compared with the `minutes_per_run` budget they gave at
intake, and confirms, edits the config, or stops, so a beginner never starts an expensive
GPU run by accident. After the run, the debrief compares the estimate with the actual time.

## Non-goals

- Reading the Colab compute-unit balance or the Resources panel automatically.
- Cross-project or cross-session spending totals.
- Gating the report stage's `evaluate.py --split test` run (it is short: one forward pass
  over the test split, not a training run).
- Changing the report stage.
- A GPU-selection UI. The runtime type is chosen in Colab's own "Change runtime type"
  menu; intake's `gpu` preference (`mlagent/spec.py` `GPU_CHOICES = ("none", "T4", "any")`)
  stays a preference recorded in `spec.json`, not something this milestone can act on.
- Per-GPU price lookup from the web. The price is a form field.

## Decisions the user made (2026-09-11)

1. The gate applies to every training run: the train stage's first run and each tune
   round — not the report stage.
2. Estimates come from a dry run first (the project's first training run), then from run
   history for later runs on the same project.
3. The estimate shows minutes, compute units and money. The user sets the price per unit
   (defaulting to Colab Pro's $9.99 per 100 units, i.e. `0.0999` per unit) and a currency
   label.
4. When the estimate exceeds `minutes_per_run`, the gate warns, names concrete cuts to the
   config, and still asks rather than refusing to run.
5. The dry run is executed by the agent inside `prepare` as a subprocess in the same Colab
   runtime — real GPU detection, no new notebook cell.

## Section 1 — `mlagent/cost.py` (pure, unit-tested without stages)

### `RuntimeInfo`

A frozen dataclass: `device` (`"cuda"` | `"cpu"`), `gpu_name` (`str | None`), `gpu_type`
(one of the rates keys, `"unknown"`, or `None` when there is no GPU), `source`
(`"torch"` | `"nvidia-smi"` | `"none"`).

### `detect_runtime(probes=None) -> RuntimeInfo`

Tries, in order: a torch probe (a lazy `import torch` inside the function, exactly as
`mlagent/templates/image_torch/data.py`'s `pick_device()` already does — `torch` is not
imported at module scope anywhere in `mlagent/`); `torch.cuda.is_available()` plus
`torch.cuda.get_device_name(0)`; then an `nvidia-smi --query-gpu=name --format=csv,noheader`
subprocess; then `"none"`. `probes` is an injectable sequence of zero-argument callables
each returning a GPU name string or `None`, tried in order before the two built-in probes,
so unit tests never touch real hardware or shell out. `gpu_type` is derived by matching the
known keys (`"T4"`, `"L4"`, `"A100"`) case-insensitively as a substring of `gpu_name`;
anything else present is `"unknown"`.

### `mlagent/rates.json`

A packaged data file, loaded the same way `mlagent/prompts_io.py` loads
`prompts/*.md` — `Path(__file__).parent / "rates.json"`, read directly, no
`importlib.resources` indirection (nothing in this codebase uses one):

```json
{
  "units_per_hour": {"T4": 2.0, "L4": 4.8, "A100": 13.0},
  "unknown_gpu_rate": 2.0,
  "default_price_per_unit": 0.0999,
  "default_currency": "$"
}
```

`mlagent/config.py`'s `DEFAULT_RATES` (`{"T4": 2.0, "L4": 4.8, "A100": 13.0}`, line 26) and
`PRICE_PER_100_UNITS_USD` (`9.99`, line 27) are removed; both values move into
`rates.json`. Their only current references — `tests/test_config.py` lines 8-9
(`assert set(config.DEFAULT_RATES) == {"T4", "L4", "A100"}` and
`assert config.PRICE_PER_100_UNITS_USD > 0`) — are deleted, since `tests/test_cost.py`
takes over asserting the rates (see Section 4).

`load_rates() -> dict` returns the parsed file. `rate_for(runtime: RuntimeInfo, rates: dict)
-> tuple[float, str | None]` returns `(units_per_hour, note)`: `0.0` with no note on a CPU
runtime; the matched rate with no note for a known `gpu_type`; `unknown_gpu_rate` with a
note such as `"'{gpu_name}' is not one of the known GPU types; using the T4 rate as an
estimate."` when `gpu_type == "unknown"`.

### `DryRunResult`

A dataclass: `device`, `gpu_name`, `batches_per_epoch` (int), `seconds_per_batch` (float),
`seconds_per_epoch` (float), `n_train` (int), `script` (`"train.py"`), `error`
(`str | None`). `read_dry_run(path: Path) -> DryRunResult` parses `dry_run.json` (Section 2);
a missing or unparsable file yields a `DryRunResult` with every numeric field `0`/`0.0` and
`error` set.

### `run_dry_run(project_dir, python=sys.executable, timeout=600, env=None) -> DryRunResult`

Runs `[python, "train.py", "--dry-run"]` with `cwd=project_dir`, waits up to `timeout`
seconds, then calls `read_dry_run(project_dir / "dry_run.json")`. A non-zero exit, a
timeout, or a missing `dry_run.json` after the process ends all produce a `DryRunResult`
with `error` set to a one-line summary plus the last ~20 lines of stderr — never an
exception raised out of `run_dry_run`. `env` defaults to the current process environment
merged with nothing extra; the gate (Section 3) passes `CUDA_VISIBLE_DEVICES` through
untouched, so tests that want a CPU-only dry run set it themselves before calling.

### `Estimate`

A dataclass: `seconds_per_epoch`, `epochs`, `minutes` (this run, dry-run seconds already
carry the 1.2 safety factor — see `estimate_run` below), `units` (float), `cost` (float),
`price_per_unit`, `currency`, `rate_units_per_hour`, `rounds_remaining` (int),
`minutes_all_rounds`, `units_all_rounds`, `cost_all_rounds`, `basis`
(`"dry_run"` | `"history"`), `runtime: RuntimeInfo`, `note` (`str | None`, from `rate_for`).

### `estimate_run(seconds_per_epoch, epochs, runtime, rates, price_per_unit, currency, rounds_remaining, basis) -> Estimate`

Implements the 2026-09-06 spec's formula directly:

```
hours = seconds_per_epoch * epochs * 1.2 / 3600
units = hours * rate_units_per_hour
cost  = units * price_per_unit
```

(`seconds_per_epoch * epochs` is the same quantity as `batches_per_epoch * epochs *
sec_per_batch` from the original paragraph — this codebase's dry run already reduces to a
per-epoch time, so the formula is expressed in those terms instead of re-multiplying by
`batches_per_epoch`.) All-rounds figures multiply the this-run figures by
`(1 + rounds_remaining)` — this run plus every remaining tune round.

### `estimate_from_dry_run(dry: DryRunResult, epochs, runtime, rates, price_per_unit, currency, rounds_remaining) -> Estimate`

A thin wrapper passing `dry.seconds_per_epoch` and `basis="dry_run"` to `estimate_run`.

### `estimate_from_history(seconds_per_epoch: float, epochs, runtime, rates, price_per_unit, currency, rounds_remaining) -> Estimate`

A thin wrapper with `basis="history"`. The caller (Section 3) supplies
`seconds_per_epoch` from the latest finished run of the same model family, read from that
run's `runs/run{N}_metrics.json` (kept by `mlagent/runs.py::archive_run`'s figure/checkpoint
archive today; `seconds_per_epoch` already lives in `metrics.json`, hence in the archived
copy) — `estimate_from_history` itself does no file I/O, so it is as easy to unit-test as
`estimate_from_dry_run`.

### `budget_check(estimate, minutes_per_run, config, schema, modality) -> BudgetAdvice`

A dataclass: `over` (bool, `estimate.minutes > minutes_per_run`), `minutes_over`, and
`suggestions`, a list of `(key, current, proposed, reason)` tuples computed against the
flat schema (`mlagent/templates_io.py::schema_for`'s return shape) already in scope
wherever `budget_check` is called:

- The largest `epochs` value that would bring `estimate.minutes` back under
  `minutes_per_run`, computed from the same linear relationship `estimate_run` uses
  (`minutes` is proportional to `epochs`), floored, minimum `1`. Always offered when
  `epochs` is a schema key (it is, in both `tabular_sklearn/config_schema.json` and
  `image_torch/config_schema.json` per the 6a spec's Section 3 table).
- For `modality.name == "image"`: `image_size` is fixed at ingest, not a config key
  (6a spec, Section 1's `data_meta.json` table), so it is never suggested. Instead
  `budget_check` suggests a larger `batch_size` within the schema's range (fewer, bigger
  batches reduce per-epoch overhead) and `render_estimate` (below) adds one fixed sentence
  noting that re-ingesting the data at a smaller `IMAGE_SIZE` is the other lever, run from
  the "2. Data" cell.
- For `modality.name == "tabular"` and `config["model_type"] == "gradient_boosting"`:
  only the `epochs` suggestion (gradient boosting's `epochs` key maps to boosting rounds;
  no other schema key changes wall-clock time predictably).
- A suggestion is only emitted for a key actually present in `schema`, so a family that
  drops a key never gets a suggestion naming it.

### `render_estimate(estimate, advice, spec_gpu) -> str`

Fixed markdown built from f-strings — no LLM call. In order:

1. One runtime line: `"This runtime has a {gpu_name} [[GPU]]."` or `"This runtime has no
   [[GPU]]."`.
2. A mismatch line when `spec_gpu` (`mlagent.spec.Spec.gpu`, one of `GPU_CHOICES`) asked
   for a GPU (`"T4"` or `"any"`) and `estimate.runtime.device == "cpu"`, or vice versa
   (`spec_gpu == "none"` but a GPU is present): names the Colab menu path, "Runtime ->
   Change runtime type", to switch.
3. A table with minutes / [[compute units]] / cost, one row for this run and one row for
   all `estimate.rounds_remaining` remaining rounds combined.
4. The basis line: `"Timed with a 3-batch dry run."` or `"From run {N}'s measured time."`
   (`N` is the source run's `run_id`, threaded through from the caller in Section 3).
5. The CPU line, only when `estimate.runtime.device == "cpu"`: `"A CPU runtime uses no
   compute units, so this run is free."`.
6. The tabular-on-GPU line, only when `modality.name == "tabular"` and
   `estimate.runtime.device == "cuda"`: `"This tabular model runs on the CPU, but a GPU
   runtime still spends units; switch to a CPU runtime to train for free."` (tabular
   templates never move data to `cuda`; a GPU runtime is idle compute cost for them).
7. When `advice.over`: a block naming each suggestion as a bullet — `"lower {key} from
   {current} to {proposed} — {reason}"` — plus, for images over budget, the fixed
   re-ingest-at-a-smaller-`IMAGE_SIZE` sentence from the `image_size` bullet above.

Every line uses `[[term]]` markup for click-to-explain, matching every other
stage-authored (non-LLM) sentence in the codebase (e.g. `TrainStage.prepare`'s
`device_sentence` today).

## Section 2 — `--dry-run` in both train templates

### Contract

`python train.py --dry-run` loads config and data exactly as a real run, runs one warm-up
batch (not timed, so first-batch compilation/caching cost is excluded) then three timed
training batches. For `tabular_sklearn`, one boosting/fit chunk counts as one "batch" and
`batches_per_epoch = 1`, so the dry run times up to three such chunks on the training
split. It writes `dry_run.json` in the project directory with keys `device`, `gpu_name`
(`torch.cuda.get_device_name(0)` when `device == "cuda"` for `image_torch`, else `null`;
always `null` for `tabular_sklearn`), `batches_per_epoch`, `seconds_per_batch`,
`seconds_per_epoch` (`= batches_per_epoch * seconds_per_batch`), `n_train`, `script`. It
prints one human-readable line, e.g. `"Dry run: 3 batches in 0.42 s on cuda (T4); about
0.14 s per batch, 38 batches per epoch"`, and never writes `metrics.json`,
`training_curves.png` or a checkpoint. Exit code `0` on success; on failure it prints the
error to stderr and exits non-zero — the one place a generated script is allowed a
non-zero exit (`sys.exit(0)` stays forbidden everywhere, per the project conventions in
`CLAUDE.md`).

### `tabular_sklearn/train.py`

The existing `--dry-run` path (`dry_run_timing`, `mlagent/templates/tabular_sklearn/train.py`
lines 165-181 today) times one real `model.fit_epoch` call and prints a JSON object with
`seconds_per_epoch` and `n_train` to stdout, writing nothing to disk. It is reworked to the
contract above: three timed chunks instead of one, a `dry_run.json` file instead of stdout
JSON (the human line above replaces the current `print(json.dumps(...))`), and the new keys
(`device` is always `"cpu"`, `gpu_name` always `null`, `batches_per_epoch` always `1`). The
existing docstring line `"python train.py --dry-run      one round; prints timing and
n_train, writes nothing"` (line 5) is updated to describe the new behaviour.

### `image_torch/train.py`

Gains the `--dry-run` flag via the same `argparse` shape `main(argv)` already uses for
`--project` (lines 319-323 today: `parser.add_argument(...)`, `args = parser.parse_args(argv)`).
`pick_device()` (imported from `data.py`, line 27) is reused unchanged for GPU detection.
`cli_argv()` (lines 306-317) is unchanged — it already returns `[]` under any kernel
launcher.

### Section markers

Both templates gain a `# --- Dry run ---` section (alongside the existing `# ---
training ---` section in `tabular_sklearn/train.py`, line 164, and the equivalent section
in `image_torch/train.py`) so `mlagent/codewalk.py`'s `split_sections` shows it in the
walkthrough like every other section.

### Not a handoff output

`dry_run.json` is never named in a `Handoff.outputs` list and is never waited on by
`mlagent/stages/base.py::outputs_ready`; it exists purely for `cost.run_dry_run` to read
back inside the same `prepare` call that wrote it.

## Section 3 — the gate (`mlagent/stages/cost_gate.py`) and stage wiring

### `gate(ctx, *, rounds_remaining: int) -> str`

Returns `"run"` or `"stop"`. Steps, in order:

1. **Load context.** `config = ctx.project.read_json(cfg.CONFIG_FILE)` for `epochs`;
   `spec = ctx.spec()` for `minutes_per_run` and `gpu`; `modality =
   modality_for(spec.task_type)` (`mlagent/modality.py`, from the 6a spec) for `name` and
   `template_family`; `flat_schema = schema_for(load_schema(modality.template_family),
   config["model_type"])` (`mlagent/templates_io.py`'s `load_schema`/`schema_for`, the same
   two calls `TuneStage.prepare` already makes) for the bounds `budget_check`'s suggestions
   and the "Edit the config first" path (step 7) stay within.
2. **Price and currency.** Read `cost.json` in the project if present. Otherwise ask via
   `ctx.questioner.number("Price per compute unit?", default=rates["default_price_per_unit"],
   minimum=0, key="train.price_per_unit")` and `ctx.questioner.text("Currency symbol or
   code?", default=rates["default_currency"], key="train.currency")`
   (`mlagent/ui/questions.py`'s `Questioner.number`/`text` signatures), then write both to
   `cost.json`. The notebook's Train form (Section 4) supplies these two keys through
   `FormQuestioner`, so a beginner using the notebook never sees a console `input()` for
   them; the console/scripted questioners are only reached on a fresh project run outside
   the form flow, or in tests.
3. **Runtime.** `runtime = cost.detect_runtime()`.
4. **Basis.** Read `runs.jsonl` (`mlagent.runlog.read_runs`, already used by
   `TrainStage.is_complete` and `TuneStage.prepare`) for a finished run
   (`status == "done"`) whose `config.get("model_type")` shares the current
   `config["model_type"]`'s family. If one exists, use its archived
   `runs/run{N}_metrics.json` `seconds_per_epoch` with `cost.estimate_from_history` (basis
   `"history"`, run id `N` threaded into `render_estimate`'s basis line). Otherwise display
   `"Timing a short dry run…"` and call `cost.run_dry_run(ctx.project.root)`; on
   `DryRunResult.error` being set, display the error's last lines and the line `"No
   estimate is available; the run can still proceed."`, then skip straight to step 7's
   confirm with no table rendered (there is nothing to estimate).
5. **Budget.** `advice = cost.budget_check(estimate, spec.minutes_per_run, config, flat_schema,
   modality)`.
6. **Display.** `ctx.display(cost.render_estimate(estimate, advice, spec.gpu))`.
7. **Decision.** If `runtime.device == "cpu"` and not `advice.over`: display the CPU-is-free
   line (already part of `render_estimate`'s output) and return `"run"` without asking —
   a beginner on a free CPU runtime is never interrupted. Otherwise ask:

   ```python
   answer = ctx.questioner.choice(
       "Run it?", ["Run it", "Edit the config first", "Stop"],
       allow_other=False, key="train.cost_decision", default="Run it",
   )
   ```

   - `"Run it"` -> `"run"`.
   - `"Edit the config first"` -> `templates_io.edit_config(ctx.questioner, config,
     flat_schema, ctx.display)` (the same helper `TuneStage._choose` already uses,
     `mlagent/templates_io.py` lines 220 onward), write the edited config to
     `config.json`, recompute the estimate from the *same* measured
     `seconds_per_batch * batches_per_epoch` (no second dry run — only `epochs` or other
     schema keys changed, not the per-batch cost), re-display via `render_estimate`, then
     ask again. This loop runs at most 3 times; on the 3rd repeat the question drops the
     "Edit the config first" option, asking only `["Run it", "Stop"]`, so a user cannot
     loop forever.
   - `"Stop"` -> display `"Change the config and run this cell again, or switch runtime,
     then run this cell again."` and return `"stop"`.

### Estimate persistence

`gate` writes the just-computed `Estimate` (as a plain dict) to `cost.json` under
`last_estimate`, alongside the persisted `price_per_unit`/`currency`. `cost.json`'s shape:

```json
{
  "price_per_unit": 0.0999,
  "currency": "$",
  "last_estimate": {"minutes": 3.1, "units": 0.2, "cost": 0.02, "basis": "dry_run", "...": "..."}
}
```

### `TrainStage.prepare` and `TuneStage.prepare`

`TrainStage.prepare` (`mlagent/stages/train.py`, currently declared `-> Handoff`, line 38)
changes its return type to `Handoff | None`. Immediately before building its `Handoff`
(after the existing `ctx.display(...)` / `ctx.teaching().preamble(...)` calls, before the
`(project.root / cfg.METRICS_FILE).unlink(...)` lines that commit to a fresh run), it calls
`decision = cost_gate.gate(ctx, rounds_remaining=spec.max_rounds)`; on `"stop"` it returns
`None` without touching `metrics.json`/`eval_val.json` and without building a `Handoff`.
The 6a placeholder sentences — `device_sentence`'s `"...so there is no [[compute unit]]
cost gate for this run."` for both the image and tabular branches (lines 46-50 and 57-60
today) — are removed; `device_sentence` keeps only the device-choice half of the sentence
(`"Training picks its [[device]] at run time..."` / `"Training runs on the [[CPU]] for
tabular data."`).

`TuneStage.prepare` (`mlagent/stages/tune.py`, already declared `-> Handoff | None`) calls
the same `gate(ctx, rounds_remaining=state["max_rounds"] - round_no)` right before building
its `Handoff` at the end (lines 312-313 today, after `ctx.display(f"Applied to
config.json:...")`); on `"stop"` it displays the gate's own stop message and returns `None`
exactly as its three existing early-`None`-return branches already do (lines 240, 244, 256,
281, 295) — it does **not** additionally set `state["decision"] = "stopped"`, because a
cost-gate stop is not a tuning decision: the proposal is still pending, the user is
expected to edit the config and re-run the same cell, and `tune_state.json`'s `pending`
key stays set so the next `prepare` call re-offers the same proposal rather than asking
Claude again.

### The orchestrator's existing handling of `prepare` returning `None`

`mlagent/orchestrator.py::Orchestrator._run` (lines 117-169) already treats a stage whose
`prepare` returns `None` as "no cells to wait for, this round is over":

```python
handoff = stage_prepare(stage, self.ctx)
state = self._state()
state["handoff"] = handoff.to_dict() if handoff is not None else None
if handoff is not None:
    state["prepared"] = [*state["prepared"], stage.name]
self._save(state)
...
if handoff is not None and not stage_outputs_ready(stage, self.ctx, handoff):
    self.ctx.display(waiting_message(handoff))
    return ran
if not stage_debrief(stage, self.ctx):
    break
```

Three stages already return `None` from `prepare` today: `IntakeStage` and `CodegenStage`
always do (both are single-phase and define `debrief` as a no-op returning `None`,
`mlagent/stages/intake.py` line ~206, `mlagent/stages/codegen.py` line ~234), and
`CodegenStage.prepare` also returns `None` when the data is not ready
(`mlagent/stages/codegen.py`, the `if problems: ... return None` branch) *without* writing
`config.json` — after that, `stage.is_complete` is `False`, so the loop falls through to
`"Stage {stage.name} did not finish; rerun orch.run() to continue."` and returns, leaving
the stage `current` and un-marked. `TuneStage.prepare`'s existing `None`-returns rely on the
same mechanism, but land in a state where `is_complete` (lines 224-227, checking
`state["decision"]`) is already `True` by the time `prepare` returned, so the stage
*is* marked complete — that is tune's intended "the loop is over" behaviour, unrelated to
the cost gate.

This mechanism is correct for the cost gate's "stop, stage stays current" requirement
*mechanically* — a `None` from `prepare` with `is_complete` still `False` already ends
`orch.run()` with the stage current and nothing marked complete — **but there is one gap**:
the loop still calls `stage_debrief(stage, self.ctx)` immediately after a `None`-returning
`prepare`, even though no handoff was ever created this round. For `IntakeStage` and
`CodegenStage` this is harmless because their `debrief` is a hard-coded no-op. It is *not*
harmless for `TrainStage`/`TuneStage`: their `debrief` methods call
`mlagent.runs.run_problem(project)`, which reads whatever `metrics.json` happens to be on
disk (from an *earlier* run, or absent) and, finding no run matching a `started_at` it
expects, displays `"I can't see a finished run in metrics.json yet. Run the train.py cell,
then run this cell again."` — a confusing, wrong message to show directly under the cost
gate's own `"Stop: ..."` message in the same `orch.run()` call.

**The orchestrator change**: skip the `stage_debrief` call entirely when `prepare` returned
no handoff this round, since there is nothing for a debrief to read that this round
produced:

```python
if handoff is None or not stage_debrief(stage, self.ctx):
    break
```

This one-line change to the condition on the line quoted above is behaviour-preserving for
every existing caller: `IntakeStage`/`CodegenStage`'s `debrief` already returns `None`
(falsy) in every case, so skipping the call changes nothing observable; `TuneStage`'s
`None`-returns already land with `is_complete` `True`, so whether `debrief` runs or not,
the very next check (`if stage.is_complete(ctx): self.mark_complete(...)`) does the same
thing. It fixes the new case cleanly: `TrainStage`/`TuneStage.debrief` are simply never
called on a cost-gate stop, so only the gate's own stop message is shown, and the stage
stays `current` with no handoff recorded — the next `orch.run()` calls `prepare` again,
re-running the gate (from the same measured `seconds_per_batch` if the user chose "Edit
the config first" and then still exits via "Stop", or a fresh dry run only if run history
still has nothing, per Section 3's basis step).

### `runs.py` and the estimate-vs-actual line

`runs.build_run_entry` (`mlagent/runs.py`, lines 30-50) gains two optional parameters,
`estimated_minutes: float | None = None` and `estimated_units: float | None = None`, added
to the returned dict as `"estimated_minutes"` and `"estimated_units"`. Both callers of
`build_run_entry` (inside `runs.py`'s own `log_finished_run`, called from
`TrainStage.debrief` and `TuneStage.debrief` via `log_finished_run`) read `cost.json`'s
`last_estimate` (Section 3) and pass its `minutes`/`units` through; when `cost.json` or
`last_estimate` is absent (the dry run failed and no estimate exists, per Section 3 step 4)
both stay `None`.

`TrainStage.debrief`'s `_narrative` and `TuneStage.debrief`'s payload each print one fixed
line, built without an LLM call, when both `estimated_minutes` and the run's actual minutes
(`entry["seconds"] / 60`) are present: `"Estimated {est:.1f} min, actual {actual:.1f} min
({pct}% {over_or_under})."`, e.g. `"Estimated 3.1 min, actual 2.7 min (13% under)."`. When
`estimated_minutes` is `None` the line is omitted entirely — no placeholder text.

`tune_state.json` and the rest of the tune loop (`applied_diff`, `history`, `decision`,
`max_rounds`) are unchanged; the cost gate adds no new key to `tune_state.json`.

### Level behaviour

`render_estimate`'s output is identical at every learning level — it is fixed text, not a
`Teaching` call, exactly like `TrainStage.prepare`'s existing `device_sentence` /
`checkpoint_sentence`. Beginners get no second LLM call: the existing
`ctx.teaching().preamble("train", ...)` / `ctx.teaching().preamble("tune", ...)` calls
(`mlagent/teaching.py` `Teaching.preamble`, which already no-ops at `EXPERT` level,
`mlagent/stages/train.py` line 70, `mlagent/stages/tune.py` lines 260-267) are unchanged
and continue to run once per stage entry, before or after the gate's own display calls;
the gate itself never calls `ctx.teaching()`.

## Section 4 — notebook, dependencies, tests, smoke, docs

### `scripts/build_notebook.py`

There is no separate "Train" form cell today — the "4. Model" cell
(`#@title 4. Model`, line 253) is the one whose `orch.run(until='train',
answers={'codegen.model_type': MODEL})` call (line 265) carries the pipeline through
codegen and into the train stage; `script_cell("train", 0)` / `script_cell("train", 1)`
(lines 267-268) are the generated `train.py`/`evaluate.py` cells with no form of their own.
This cell gains two `#@param` fields, `PRICE_PER_UNIT` (number, default `0.0999`) and
`CURRENCY` (string, default `"$"`), each with a `#@markdown` hint in the existing
beginner-first style (a hint under every form box): `"Colab Pro is $9.99 for 100 compute
units, so 0.0999 per unit; check your plan in the Resources panel."` for `PRICE_PER_UNIT`,
and `"Shown next to the cost estimate, e.g. '$' or 'GBP'."` for `CURRENCY`. Both are added
to the cell's existing `answers` dict: `orch.run(until='train', answers={
'codegen.model_type': MODEL, 'train.price_per_unit': PRICE_PER_UNIT, 'train.currency':
CURRENCY})`. This changes an existing cell's body only — no cell is added or removed, so
the notebook stays at 21 cells (verified against `notebooks/ML_Training_Agent.ipynb`
today). The "6. Tune" cell (line 270) keeps its plain `orch.run(until='tune')` call with no
new form fields — by the time a tune round runs, `cost.json` already holds the price and
currency the user set in "4. Model", and `gate`'s price/currency step (Section 3, step 2)
finds `cost.json` and never asks again. The "1. Project and interview" cell's
`MINUTES_PER_RUN` hint (line 127) gains a trailing clause: `"...the cost gate compares each
run's estimate against this before it starts."`. The notebook is rebuilt with `python
scripts/build_notebook.py`.

### Dependencies

None added. `nvidia-smi` is invoked as a subprocess, never imported; `torch` is already an
optional import everywhere it is used (`cost.detect_runtime`'s torch probe follows the same
lazy-import pattern as `image_torch/data.py::pick_device`).

### Tests

- `tests/test_cost.py`: the `estimate_run` formula against hand-computed numbers;
  `load_rates()`/`rate_for()` including the unknown-GPU note; `detect_runtime()` with
  injected `probes` covering torch-found, `nvidia-smi`-found (a fake subprocess result) and
  none-found; `estimate_from_history`; `budget_check`'s suggestions computed against both
  the `tabular_sklearn` and `image_torch` flat schemas (per the 6a spec's schema tables);
  every `render_estimate` line — CPU, GPU, GPU/CPU mismatch, over-budget suggestions,
  tabular-on-GPU; `read_dry_run`; `run_dry_run`'s failure path against a fake script that
  exits non-zero and one that times out.
- `tests/test_template_dry_run.py`: both real templates run with `--dry-run` as
  subprocesses on CPU (`CUDA_VISIBLE_DEVICES="-1"`), asserting every `dry_run.json` key
  from Section 2's contract, that no `metrics.json`, checkpoint or figure is written, and
  exit code `0`.
- `tests/test_cost_gate.py`, with `FakeLLM` plus `ScriptedQuestioner`/`FormQuestioner`:
  CPU-under-budget (no question asked); GPU-confirm (an injected `RuntimeInfo` via
  `probes`); over-budget with suggestions shown; the edit path (re-estimate reuses the
  measured `seconds_per_batch`, confirmed by asserting `run_dry_run` is called at most
  once across the whole test); the stop path (no `Handoff` returned, `orch.waiting()` is
  `None`, the stage's `state["current"]` is unchanged, no `metrics.json` written); the
  history basis on a project's second run (asserting no dry run is attempted); a failed
  dry run (the error-tail message, then still reaching the confirm); price and currency
  persisted to `cost.json` and reused on the next `gate()` call without re-asking. The
  existing `tests/test_pipeline_e2e.py` (tabular) and `tests/test_pipeline_e2e_images.py`
  (images, from 6a) gain the two new `train.*` answers and an assertion that the debrief
  text contains the estimate-vs-actual line.

Nothing in the new or extended tests hits the network or a real GPU; `probes=` and
`CUDA_VISIBLE_DEVICES="-1"`/`"-1"`-equivalent env overrides keep every test hardware
independent, matching the project's existing rule that tests never hit the network
(`CLAUDE.md`) and 6a's rule that GPU-dependent code paths are always exercised on CPU in
CI.

### `docs/colab-smoke.md`

A new "Milestone 6b" section, seven items, in the existing step-by-step style used for the
Milestone 5 and 6a sections:

1. Tabular, on a CPU runtime: the estimate line appears with no question asked, and the
   debrief shows the estimate-vs-actual line.
2. Images, on a T4 runtime: `"Timing a short dry run…"`, the table with units and cost, the
   confirm question, and the estimate-vs-actual line after the run.
3. Set `MINUTES_PER_RUN` to `1` at intake on a fresh project: the over-budget block names
   the largest `epochs` value that fits.
4. Choose "Edit the config first", lower `epochs`, and see the recomputed estimate with no
   second dry run (confirm by watching the cell's output — no second `"Timing a short dry
   run…"` line appears).
5. Choose "Stop": the cell ends with the stop instructions, `orch.waiting()` is `None`;
   editing `config.json` by hand and running the cell again asks the gate's question again.
6. A tune round on the same project: the basis line says `"From run 1's measured time."`
   and no dry run runs.
7. Change `PRICE_PER_UNIT` and `CURRENCY` in the "4. Model" cell on a fresh project: the
   cost line in the table uses them, and `cost.json` holds the values afterwards.

### `CLAUDE.md`

The pipeline description gains one clause on the gate (train and each tune round estimate
cost and confirm before the handoff), the new modules (`mlagent/cost.py`,
`mlagent/stages/cost_gate.py`, `mlagent/rates.json`) and the new test commands (`python -m
pytest tests/test_cost.py tests/test_cost_gate.py tests/test_template_dry_run.py -v`). The
templates paragraph gains a mention of `--dry-run` and `dry_run.json` alongside the
existing description of the templates' shared conventions.

## Risks (with the decisions that answer them)

- **Timing three batches under-estimates a run whose first epochs include one-time
  compilation or data-caching cost** (torch's first CUDA kernel launch, first-epoch file
  reads). The 1.2 safety factor absorbs some of this, and the estimate-vs-actual line lets
  users calibrate their own expectations run over run rather than trusting a number that
  is silently wrong forever.
- **`nvidia-smi` may be absent on a CPU-only runtime.** That is exactly the `"none"` path
  `detect_runtime` already returns for, with no error raised.
- **Colab's rates and prices change over time.** `rates.json` is data, not a constant
  baked into `mlagent/config.py`, and the price is a form field the user can update any
  time by editing `cost.json` or reanswering the "4. Model" cell's fields on a fresh
  project.
- **The run-history basis assumes the next run is the same model family.** A tune
  proposal that switches `model_type` to a different family (codegen's job, not tune's, per
  the existing architecture) would leave no matching history entry, so `gate` falls back to
  a fresh dry run automatically (Section 3, step 4) rather than estimating from an
  unrelated family's timing.

Plan follows: `docs/superpowers/plans/2026-09-11-milestone-6b-cost-gate.md`, written via
writing-plans after the user reviews this spec.
