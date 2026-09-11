"""Tune stage: diagnose the run history, apply one approved change, rerun the train cells.

One round is `prepare` (diagnosis, proposals, one question, new config.json, handoff) then
`debrief` (log the run with its diff, draw the comparison figures, narrate). `debrief`
returns True to have the orchestrator prepare the next round in the same call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mlagent import config as cfg
from mlagent import plots
from mlagent.diagnose import (
    EXPECTATIONS,
    Diagnosis,
    Proposal,
    apply_proposal,
    diagnose,
    diff_config,
    heuristic_proposals,
    meets_target,
)
from mlagent.llm import LLMError, ToolSpec
from mlagent.modality import modality_for
from mlagent.prompts_io import audience, load_prompt
from mlagent.runlog import best_run, is_better, read_runs, summarise
from mlagent.runs import (
    EVAL_VAL_FILE,
    archive_metrics,
    log_finished_run,
    read_run_metrics,
    run_problem,
)
from mlagent.spec import Spec
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.teaching import EXPERT, material
from mlagent.templates_io import (
    config_table,
    edit_config,
    load_schema,
    model_types,
    schema_for,
)

TUNE_COMMANDS = [["train.py"], ["evaluate.py"]]
TUNE_OUTPUTS = [cfg.METRICS_FILE, EVAL_VAL_FILE]
MAX_PROPOSALS = 3
EDIT_LABEL = "Edit a proposal first"
STOP_LABEL = "Stop tuning and write the report"
COMPARE_CURVES = "compare_curves"
COMPARE_RUNS = "compare_runs"
CONTINUE = "continue"
DECISIONS = ("continue", "stopped", "target_met", "rounds_exhausted")

DECISION_TEXT = {
    "stopped": (
        "Tuning is stopped at your request. The report stage evaluates the best run on the "
        "[[test set]]; to try another change by hand, edit `config.json` and use the "
        "*Train again* cell."
    ),
    "target_met": (
        "The best run meets your target, so tuning is done. The report stage scores it once "
        "on the [[test set]]. You can still edit `config.json` and use the *Train again* "
        "cell if you want to keep experimenting."
    ),
    "rounds_exhausted": (
        "That was the last tuning round you allowed at intake (`max_rounds`), so tuning is "
        "done. The report stage evaluates the best run; `orch.reset('tune')` starts a fresh "
        "loop if you want more rounds."
    ),
}

DIAGNOSIS_TEXT = {
    "overfitting": (
        "**Diagnosis: [[overfitting]].** Validation [[loss]] turned upward while training "
        "loss kept falling, so the model is learning noise that does not carry over to new "
        "rows."
    ),
    "underfitting": (
        "**Diagnosis: [[underfitting]].** Validation loss was still falling at the last "
        "[[epoch]]: the model had not finished learning when training stopped."
    ),
    "learning_rate_too_high": (
        "**Diagnosis: the [[learning rate]] looks too high.** Validation loss jumps up and "
        "down from epoch to epoch instead of settling."
    ),
    "plateau": (
        "**Diagnosis: a [[plateau]].** Validation loss has flattened; more of the same will "
        "not help, so the model needs a different shape or more signal."
    ),
    "failed_run": (
        "**Diagnosis: the last run failed** ({error}). A gentler configuration usually gets "
        "past this."
    ),
    "improving": (
        "**Diagnosis: still improving.** The last change helped and the curve was still "
        "falling at the end, so there is more to gain in the same direction."
    ),
    "target_met": "**Diagnosis: target met.**",
}

PROPOSE_TOOL = ToolSpec(
    name="propose_diffs",
    description=(
        "Propose one to three configuration changes, best first. Each `changes` object maps "
        "config keys to new values; `reason` is shown to the user; `expected` says what the "
        "change should improve."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "maxItems": MAX_PROPOSALS,
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "changes": {"type": "object"},
                        "reason": {"type": "string"},
                        "expected": {"type": "string", "enum": list(EXPECTATIONS)},
                    },
                    "required": ["rank", "changes", "reason", "expected"],
                },
            }
        },
        "required": ["proposals"],
    },
    handler=lambda inp: "recorded",
)


def apply_label(n: int) -> str:
    return f"Apply proposal {n}"


def _fmt(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def backfill_latest_metrics(project, runs: list[dict], metrics_by_run: dict[int, dict]) -> dict:
    """Archive `metrics.json` under the latest run's id if it was logged before this branch.

    `read_run_metrics` only finds runs that have a `runs/run{N}_metrics.json` archive; a
    project whose run 1 predates the tuning loop has none. If the current `metrics.json`
    still matches the latest run (same `started_at`), archive it now so `diagnose` and the
    comparison figures see its curve instead of treating it as missing.
    """
    if not runs:
        return metrics_by_run
    latest = runs[-1]
    run_id = latest.get("run_id")
    if not isinstance(run_id, int) or run_id in metrics_by_run:
        return metrics_by_run
    metrics = project.read_json(cfg.METRICS_FILE)
    if isinstance(metrics, dict) and metrics.get("started_at") == latest.get("started_at"):
        archive_metrics(project, run_id, metrics)
        metrics_by_run = dict(metrics_by_run)
        metrics_by_run[run_id] = metrics
    return metrics_by_run


def empty_tune_state(max_rounds: int) -> dict:
    return {"round": 0, "max_rounds": int(max_rounds), "decision": CONTINUE, "pending": None,
            "history": []}


def load_tune_state(project, max_rounds: int) -> dict:
    """tune_state.json with every key present; a missing or corrupt file is a fresh loop."""
    state = project.read_json(cfg.TUNE_STATE_FILE)
    base = empty_tune_state(max_rounds)
    if isinstance(state, dict) and isinstance(state.get("history"), list):
        base.update(state)
    if base["decision"] not in DECISIONS:
        base["decision"] = CONTINUE
    return base


def describe(diagnosis: Diagnosis) -> str:
    """The diagnosis in plain words, with the numbers that back it."""
    ev = diagnosis.evidence
    if ev.get("no_epoch_data"):
        text = (
            "**Diagnosis: no per-epoch curve is archived for this run** (it was logged "
            "before tuning existed), so I am judging it by its best validation score alone. "
            "The next run will be archived in full."
        )
    else:
        text = DIAGNOSIS_TEXT.get(diagnosis.label, DIAGNOSIS_TEXT["plateau"]).format(
            error=ev.get("error", "unknown error")
        )
    parts = []
    if isinstance(ev.get("val_trend"), int | float):
        parts.append(f"validation loss changed {ev['val_trend']:+.1%} over the last third")
    if isinstance(ev.get("gap"), int | float):
        parts.append(f"train/validation gap {ev['gap']:.1%} at the best epoch")
    if isinstance(ev.get("oscillation"), int | float):
        parts.append(f"{ev['oscillation']:.0%} of epoch-to-epoch changes flipped direction")
    if ev.get("stopped_early"):
        parts.append("training stopped early")
    return text + (" (" + "; ".join(parts) + ")" if parts else "")


def proposal_table(diff: dict, schema: dict) -> str:
    rows = ["| key | now | proposed | what it does |", "|---|---|---|---|"]
    for key, change in diff.items():
        desc = schema.get(key, {}).get("description", "")
        rows.append(f"| {key} | {_fmt(change.get('from'))} | {_fmt(change.get('to'))} | {desc} |")
    return "\n".join(rows)


class TuneStage(ScriptStageBase):
    name = "tune"

    # --- lifecycle -----------------------------------------------------------------------
    def is_complete(self, ctx: StageContext) -> bool:
        state = ctx.project.read_json(cfg.TUNE_STATE_FILE)
        if not isinstance(state, dict):
            return False
        decision = state.get("decision")
        return decision in DECISIONS and decision != CONTINUE

    def on_reset(self, ctx: StageContext) -> None:
        (ctx.project.root / cfg.TUNE_STATE_FILE).unlink(missing_ok=True)

    # --- phase 1 -------------------------------------------------------------------------
    def prepare(self, ctx: StageContext) -> Handoff | None:
        project = ctx.project
        spec = ctx.spec()
        runs = read_runs(project.runs_path)
        if not any(r.get("status") == "done" for r in runs):
            ctx.display("There is no successful training run to tune yet; finish the train "
                        "stage first.")
            return None
        state = load_tune_state(project, spec.max_rounds)
        if state["decision"] != CONTINUE:
            ctx.display(DECISION_TEXT.get(state["decision"], DECISION_TEXT["stopped"]))
            return None
        config = project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict) or not config.get("model_type"):
            raise RuntimeError("config.json not found; run the codegen stage first")
        nested = load_schema(modality_for(spec.task_type).template_family)

        metrics_by_run = backfill_latest_metrics(project, runs, read_run_metrics(project, runs))
        diagnosis = diagnose(runs, metrics_by_run, spec)
        if diagnosis.label == "target_met":
            state["decision"] = "target_met"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display(DECISION_TEXT["target_met"])
            return None

        round_no = state["round"] + 1
        level = ctx.learning_level()
        if state["round"] == 0:
            ctx.teaching().preamble("tune", {
                "max_rounds": spec.max_rounds, "metric": spec.metric,
                "target_value": spec.target_value, "model_type": config.get("model_type"),
                "best_val_metric": diagnosis.evidence.get("best_val_metric"),
            })
            if level != EXPERT:
                ctx.display(material("tuning", level))
        ctx.display(f"**Tuning round {round_no} of {state['max_rounds']}.**\n\n"
                    + summarise(runs, spec.metric))
        ctx.display(describe(diagnosis))

        applied = self._propose(ctx, spec, diagnosis, runs, config, nested)
        if not applied:
            state["decision"] = "stopped"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display(
                "I have no change to propose for this run, so tuning stops here. The report "
                "stage evaluates the best run on the [[test set]]; to try a change by hand, "
                "edit `config.json` and use the *Train again* cell."
            )
            return None
        for proposal, _new_config, diff, notes in applied:
            flat = schema_for(nested, str(_new_config["model_type"]))
            text = (f"**Proposal {proposal.rank}** (expected: {proposal.expected})\n\n"
                    f"{proposal.reason}\n\n{proposal_table(diff, flat)}")
            if notes:
                text += "\n\n" + " ".join(notes)
            ctx.display(text)

        chosen = self._choose(ctx, applied, config, nested)
        if chosen is None:
            state["decision"] = "stopped"
            project.write_json(cfg.TUNE_STATE_FILE, state)
            ctx.display(DECISION_TEXT["stopped"])
            return None
        new_config, diff, reason = chosen
        project.write_json(cfg.CONFIG_FILE, new_config)
        state["pending"] = {
            "round": round_no,
            "applied_diff": diff,
            "reason": reason,
            "diagnosis": diagnosis.label,
            "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        project.write_json(cfg.TUNE_STATE_FILE, state)
        # The cells regenerate these; a stale copy must not satisfy outputs_ready.
        (project.root / cfg.METRICS_FILE).unlink(missing_ok=True)
        (project.root / EVAL_VAL_FILE).unlink(missing_ok=True)
        changed = ", ".join(f"`{k}` {_fmt(v['from'])} -> {_fmt(v['to'])}" for k, v in diff.items())
        ctx.display(f"Applied to `config.json`: {changed}. Run the `train.py` and "
                    "`evaluate.py` cells again, then run this cell to see whether it helped.")
        return Handoff(stage=self.name, commands=[list(c) for c in TUNE_COMMANDS],
                       outputs=list(TUNE_OUTPUTS))

    def _propose(self, ctx: StageContext, spec: Spec, diagnosis: Diagnosis, runs: list[dict],
                 config: dict, nested: dict) -> list[tuple[Proposal, dict, dict, list[str]]]:
        """Proposals from Claude, else the heuristic; each coerced, no-ops dropped, re-ranked."""
        raw = self._ask_llm(ctx, spec, diagnosis, runs, config, nested)
        if not raw:
            flat = schema_for(nested, str(config["model_type"]))
            raw = heuristic_proposals(diagnosis, config, flat)
        applied: list[tuple[Proposal, dict, dict, list[str]]] = []
        for proposal in raw:
            try:
                new_config, diff, notes = apply_proposal(config, proposal, nested)
            except ValueError:
                continue
            if not diff:
                continue
            applied.append((proposal, new_config, diff, notes))
            if len(applied) == MAX_PROPOSALS:
                break
        for rank, (proposal, _c, _d, _n) in enumerate(applied, start=1):
            proposal.rank = rank
        return applied

    def _ask_llm(self, ctx: StageContext, spec: Spec, diagnosis: Diagnosis, runs: list[dict],
                 config: dict, nested: dict) -> list[Proposal]:
        store: dict = {}

        def record(inp: dict) -> str:
            store["proposals"] = inp.get("proposals")
            return "recorded"

        tool = ToolSpec(name=PROPOSE_TOOL.name, description=PROPOSE_TOOL.description,
                        input_schema=PROPOSE_TOOL.input_schema, handler=record)
        payload = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value, "minutes_per_run": spec.minutes_per_run,
                     "max_rounds": spec.max_rounds},
            "diagnosis": {"label": diagnosis.label, "evidence": diagnosis.evidence,
                          "improved": diagnosis.improved,
                          "latest_run_id": diagnosis.latest_run_id,
                          "best_run_id": diagnosis.best_run_id},
            "runs": [{k: r.get(k) for k in ("run_id", "status", "best_val_metric", "best_epoch",
                                            "epochs_run", "seconds", "applied_diff", "error")}
                     for r in runs],
            "config": config,
            "schema": schema_for(nested, str(config["model_type"])),
            "model_types": model_types(nested),
        }
        try:
            ctx.llm.run(
                load_prompt("tune", audience=audience(ctx.learning_level())),
                [{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
                [tool],
            )
        except LLMError:
            return []
        proposals: list[Proposal] = []
        for i, item in enumerate(store.get("proposals") or [], start=1):
            if not isinstance(item, dict) or not isinstance(item.get("changes"), dict):
                continue
            expected = item.get("expected")
            rank = item.get("rank")
            proposals.append(Proposal(
                rank=int(rank) if isinstance(rank, int | float) else i,
                changes=dict(item["changes"]),
                reason=str(item.get("reason") or ""),
                expected=expected if expected in EXPECTATIONS else "better",
            ))
        proposals.sort(key=lambda p: p.rank)
        return proposals

    def _choose(self, ctx: StageContext, applied: list, config: dict, nested: dict):
        """Apply / edit / stop. Returns (new_config, diff, reason) or None to stop."""
        # A Colab form answer for "tune.action" is fixed for the whole cell run, so once
        # a no-op edit sends us round again, the re-ask must go to the fallback (console
        # or scripted) questioner instead of returning that same form answer forever.
        key = "tune.action"
        while True:
            options = [apply_label(i) for i in range(1, len(applied) + 1)]
            options += [EDIT_LABEL, STOP_LABEL]
            answer = ctx.questioner.choice("What shall we do?", options, allow_other=False,
                                           key=key)
            if answer == STOP_LABEL:
                return None
            if answer == EDIT_LABEL:
                idx = 0
                if len(applied) > 1:
                    pick = ctx.questioner.choice(
                        "Which proposal do you want to edit?",
                        [f"Proposal {i}" for i in range(1, len(applied) + 1)],
                        allow_other=False, key="tune.edit_which",
                    )
                    idx = self._index(pick, len(applied))
                proposal, new_config, _diff, _notes = applied[idx]
                flat = schema_for(nested, str(new_config["model_type"]))
                edited = edit_config(ctx.questioner, new_config, flat, ctx.display)
                diff = diff_config(config, edited)
                if not diff:
                    ctx.display("That leaves the configuration unchanged; pick a proposal or stop.")
                    key = None
                    continue
                ctx.display("Edited configuration:\n\n" + config_table(edited, flat))
                return edited, diff, f"{proposal.reason} (edited by you)"
            idx = self._index(answer, len(applied))
            proposal, new_config, diff, _notes = applied[idx]
            return new_config, diff, proposal.reason

    @staticmethod
    def _index(answer: str, count: int) -> int:
        try:
            idx = int(str(answer).rsplit(" ", 1)[-1]) - 1
        except ValueError:
            idx = 0
        return min(max(idx, 0), count - 1)

    # --- phase 2 -------------------------------------------------------------------------
    def debrief(self, ctx: StageContext) -> bool | None:
        project = ctx.project
        spec = ctx.spec()
        state = load_tune_state(project, spec.max_rounds)
        pending = state.get("pending")
        if not isinstance(pending, dict):
            # A Ctrl-C between the state write and the orchestrator forgetting the handoff
            # can strand `pending` at None while the round is actually over (history has
            # the finished run and the decision is still "continue"); ask to re-prepare
            # instead of reporting "did not finish" forever.
            if state["decision"] == CONTINUE and state["history"]:
                return True
            return None
        problem = run_problem(project)
        if problem:
            ctx.display(problem)
            return None
        logged = log_finished_run(project, applied_diff=pending.get("applied_diff"))
        if logged is None:
            return None
        entry = logged.entry
        runs = read_runs(project.runs_path)
        metrics_by_run = read_run_metrics(project, runs)
        previous = [r for r in runs if r.get("run_id") != entry["run_id"]]
        previous_best = best_run(previous, spec.metric)
        best = best_run(runs, spec.metric)
        improved = bool(
            entry["status"] == "done" and previous_best is not None
            and is_better(spec.metric, entry.get("best_val_metric"),
                          previous_best.get("best_val_metric"))
        )

        figures = [
            plots.save_and_close(plots.compare_curves(runs, metrics_by_run), project.plots_dir,
                                 COMPARE_CURVES),
            plots.save_and_close(plots.compare_runs(runs, spec.metric, spec.target_value),
                                 project.plots_dir, COMPARE_RUNS),
            *logged.figures,
        ]
        payload = {
            "spec": {"task_type": spec.task_type, "metric": spec.metric,
                     "target_value": spec.target_value},
            "round": pending.get("round"),
            "applied_diff": pending.get("applied_diff"),
            "reason": pending.get("reason"),
            "diagnosis_before": pending.get("diagnosis"),
            "run": {k: entry.get(k) for k in ("run_id", "status", "best_val_metric", "best_epoch",
                                              "epochs_run", "seconds", "error")},
            "stopped_early": bool((metrics_by_run.get(entry["run_id"]) or {}).get("stopped_early")),
            "previous_best": ({k: previous_best.get(k) for k in ("run_id", "best_val_metric")}
                              if previous_best else None),
            "improved": improved,
            "runs": [{k: r.get(k) for k in ("run_id", "status", "best_val_metric", "applied_diff")}
                     for r in runs],
        }
        fallback = (
            "Compare the new run's line with the earlier ones in the first figure, and its "
            "bar with the others in the second: the change helped if its bar moved towards "
            "the dashed target line."
        )
        narrative = ctx.teaching().debrief("tune_debrief", payload, figures, fallback=fallback)
        round_after = int(pending.get("round") or state["round"] + 1)
        more_rounds = round_after < state["max_rounds"]
        ctx.display(self._headline(spec, entry, previous_best, improved, more_rounds=more_rounds)
                    + "\n\n" + narrative)

        state["round"] = round_after
        # `log_finished_run` returns the existing entry (`new=False`) when this run was
        # already logged; only append a new history entry the first time this round is
        # debriefed, so calling debrief twice for the same run stays idempotent (ruling 3).
        if not any(h.get("run_id") == entry["run_id"] for h in state["history"]):
            state["history"].append({
                "round": state["round"], "diagnosis": pending.get("diagnosis"),
                "applied_diff": pending.get("applied_diff"), "run_id": entry["run_id"],
                "improved": improved,
            })
        state["pending"] = None
        best_value = best.get("best_val_metric") if best else None
        if meets_target(spec.metric, best_value, spec.target_value):
            state["decision"] = "target_met"
        elif state["round"] >= state["max_rounds"]:
            state["decision"] = "rounds_exhausted"
        else:
            state["decision"] = CONTINUE
        project.write_json(cfg.TUNE_STATE_FILE, state)
        if state["decision"] != CONTINUE:
            ctx.display(DECISION_TEXT.get(state["decision"], DECISION_TEXT["stopped"]))
            return None
        return True

    @staticmethod
    def _headline(spec: Spec, entry: dict, previous_best: dict | None, improved: bool, *,
                 more_rounds: bool) -> str:
        run_id = entry["run_id"]
        if entry["status"] != "done":
            ending = ("I will propose a gentler configuration next." if more_rounds
                      else "That was the last allowed round.")
            return f"Run {run_id} failed: {entry.get('error')}. {ending}"
        value = _fmt(entry.get("best_val_metric"))
        if previous_best is None:
            return f"Run {run_id} finished: best validation {spec.metric} {value}."
        prev = _fmt(previous_best.get("best_val_metric"))
        verdict = "better than" if improved else "not better than"
        return (f"Run {run_id} finished: best validation {spec.metric} {value}, {verdict} the "
                f"previous best {prev} (run {previous_best['run_id']}).")
