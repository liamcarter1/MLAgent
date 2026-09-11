"""Show what a training run will cost, then ask: run it, edit the config, or stop.

Called from `TrainStage.prepare` and `TuneStage.prepare` just before each builds its
`Handoff`. Everything numeric lives in `mlagent/cost.py`; this module only gathers the
inputs, asks the questions and persists the answers to `cost.json`.

`probes` and `dry_runner` are injectable for the same reason `DataStage` takes `hf_load`:
tests must never read real hardware or shell out to a real training script.
"""

from __future__ import annotations

from mlagent import config as cfg
from mlagent import cost
from mlagent.modality import modality_for
from mlagent.runlog import read_runs
from mlagent.runs import read_run_metrics
from mlagent.templates_io import edit_config, load_schema, schema_for

RUN_LABEL = "Run it"
EDIT_LABEL = "Edit the config first"
STOP_LABEL = "Stop"
DECISION_KEY = "train.cost_decision"
MAX_EDITS = 3
TIMING_TEXT = "Timing a short dry run..."
NO_ESTIMATE_TEXT = "No estimate is available; the run can still proceed."
STOP_TEXT = ("Change the config and run this cell again, or switch runtime, then run this "
             "cell again.")


def price_and_currency(ctx, rates: dict) -> tuple[float, str]:
    """From `cost.json` when it is there, otherwise asked once and written there.

    The notebook's "4. Model" form supplies both keys through `FormQuestioner`, so a
    beginner never meets a console `input()` for them; the console and scripted
    questioners are only reached outside the form flow and in tests.
    """
    stored = ctx.project.read_json(cfg.COST_FILE)
    stored = dict(stored) if isinstance(stored, dict) else {}
    if isinstance(stored.get("price_per_unit"), int | float) and stored.get("currency"):
        return float(stored["price_per_unit"]), str(stored["currency"])
    default_price = float(rates["default_price_per_unit"])
    default_currency = str(rates["default_currency"])
    price = float(ctx.questioner.number(
        "Price per compute unit?", default=default_price, minimum=0,
        key="train.price_per_unit"))
    currency = str(ctx.questioner.text(
        "Currency symbol or code?", default=default_currency,
        key="train.currency")).strip() or default_currency
    stored.update({"price_per_unit": price, "currency": currency})
    ctx.project.write_json(cfg.COST_FILE, stored)
    return price, currency


def history_basis(project, model_type: str) -> tuple[float, int] | None:
    """`(seconds_per_epoch, run_id)` of the latest finished run of the same family.

    A tune proposal that switched family leaves no matching entry, so the caller falls
    back to a fresh dry run rather than estimating from unrelated timing.
    """
    runs = read_runs(project.runs_path)
    metrics_by_run = read_run_metrics(project, runs)
    for run in reversed(runs):
        if run.get("status") != "done":
            continue
        if str((run.get("config") or {}).get("model_type")) != model_type:
            continue
        seconds = (metrics_by_run.get(run.get("run_id")) or {}).get("seconds_per_epoch")
        if isinstance(seconds, int | float) and seconds > 0:
            return float(seconds), int(run["run_id"])
    return None


def _write_cost(project, **changes) -> None:
    record = project.read_json(cfg.COST_FILE)
    record = dict(record) if isinstance(record, dict) else {}
    record.update(changes)
    project.write_json(cfg.COST_FILE, record)


def _ask(ctx, *, allow_edit: bool, key: str | None) -> str:
    options = [RUN_LABEL, EDIT_LABEL, STOP_LABEL] if allow_edit else [RUN_LABEL, STOP_LABEL]
    answer = ctx.questioner.choice("Run it?", options, allow_other=False, key=key,
                                   default=RUN_LABEL)
    if answer == STOP_LABEL:
        ctx.display(STOP_TEXT)
        return "stop"
    return "edit" if answer == EDIT_LABEL else "run"


def gate(ctx, *, rounds_remaining: int, probes=None, dry_runner=cost.run_dry_run) -> str:
    """`"run"` or `"stop"`. Displays the estimate and persists it to `cost.json`."""
    project = ctx.project
    config = project.read_json(cfg.CONFIG_FILE)
    if not isinstance(config, dict) or not config.get("model_type"):
        raise RuntimeError("config.json not found; run the codegen stage first")
    spec = ctx.spec()
    modality = modality_for(spec.task_type)
    flat_schema = schema_for(load_schema(modality.template_family),
                             str(config["model_type"]))
    rates = cost.load_rates()
    price, currency = price_and_currency(ctx, rates)
    runtime = cost.detect_runtime(probes)
    rounds = max(0, int(rounds_remaining))
    epochs = int(config.get("epochs", 10))

    found = history_basis(project, str(config["model_type"]))
    dry: cost.DryRunResult | None = None
    basis_run_id: int | None = None
    if found is not None:
        seconds_per_epoch, basis_run_id = found
        estimate = cost.estimate_from_history(seconds_per_epoch, epochs, runtime, rates,
                                              price, currency, rounds)
    else:
        ctx.display(TIMING_TEXT)
        dry = dry_runner(project.root)
        if dry.error:
            ctx.display(dry.error)
            ctx.display(NO_ESTIMATE_TEXT)
            _write_cost(project, last_estimate=None)
            return _ask(ctx, allow_edit=False, key=DECISION_KEY)
        estimate = cost.estimate_from_dry_run(dry, epochs, runtime, rates, price, currency,
                                              rounds)

    # A Colab form answer for `train.cost_decision` is fixed for the whole cell run, so
    # once an edit sends us round again the re-ask must reach the fallback questioner.
    key: str | None = DECISION_KEY
    for edits in range(MAX_EDITS + 1):
        advice = cost.budget_check(estimate, spec.minutes_per_run, config, flat_schema,
                                   modality)
        ctx.display(cost.render_estimate(estimate, advice, spec.gpu, modality,
                                         basis_run_id=basis_run_id))
        _write_cost(project, last_estimate=estimate.to_dict())
        if runtime.device == "cpu" and not advice.over:
            return "run"
        answer = _ask(ctx, allow_edit=edits < MAX_EDITS, key=key)
        if answer != "edit":
            return answer
        key = None
        config = edit_config(ctx.questioner, config, flat_schema, ctx.display)
        project.write_json(cfg.CONFIG_FILE, config)
        epochs = int(config.get("epochs", epochs))
        # The per-batch cost did not change -- only schema keys did -- so the same
        # measured seconds_per_epoch is re-multiplied; no second dry run is ever launched.
        estimate = (cost.estimate_from_dry_run(dry, epochs, runtime, rates, price,
                                               currency, rounds)
                    if dry is not None
                    else cost.estimate_from_history(estimate.seconds_per_epoch, epochs,
                                                    runtime, rates, price, currency,
                                                    rounds))
    return "run"   # unreachable: the last pass always returns from _ask
