"""The cost gate: price, basis, budget, the question, and what lands in cost.json."""

from __future__ import annotations

from functools import partial

import pytest

from mlagent import cost
from mlagent.llm import FakeLLM
from mlagent.stages import cost_gate
from mlagent.stages.base import StageContext
from mlagent.stages.codegen import CodegenStage
from mlagent.ui.questions import FormQuestioner, ScriptedQuestioner

T4_PROBES = [("torch", lambda: "Tesla T4")]
CPU_PROBES = ()
SMALL = {"epochs": 3, "iters_per_epoch": 3, "early_stopping_patience": 0}


class Runner:
    """A stand-in for cost.run_dry_run that counts how often it is asked to shell out."""

    def __init__(self, seconds_per_epoch=6.0, error=None, batches_per_epoch=1):
        self.calls = 0
        self.result = cost.DryRunResult(
            device="cpu", gpu_name=None, batches_per_epoch=batches_per_epoch,
            seconds_per_batch=seconds_per_epoch / max(1, batches_per_epoch),
            seconds_per_epoch=seconds_per_epoch, n_train=168, script="train.py",
            error=error,
        )

    def __call__(self, project_dir, **kwargs):
        self.calls += 1
        return self.result


def make_ctx(project, answers=(), form=None):
    shown: list[str] = []
    questioner = ScriptedQuestioner(list(answers))
    if form is not None:
        questioner = FormQuestioner(form, fallback=questioner, note=shown.append)
    ctx = StageContext(project=project, llm=FakeLLM([]), questioner=questioner,
                       explainer=None, display=shown.append,
                       display_figure=lambda path, caption="": None)
    return ctx, shown


def codegen(project, model_label="Gradient boosting", config_updates=None):
    """A project with data.py/model.py/train.py/evaluate.py and a config.json."""
    ctx, _shown = make_ctx(project, answers=[model_label, "y"])
    CodegenStage().prepare(ctx)
    cfg = project.read_json("config.json")
    cfg.update(config_updates or SMALL)
    project.write_json("config.json", cfg)
    return project


def priced(project, price=0.0999, currency="$"):
    project.write_json("cost.json", {"price_per_unit": price, "currency": currency})
    return project


def test_a_cpu_run_under_budget_never_asks_and_never_stops(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner(seconds_per_epoch=6.0)     # 3 epochs -> 0.36 minutes, budget is 5
    ctx, shown = make_ctx(project)             # ScriptedQuestioner with no answers at all
    decision = cost_gate.gate(ctx, rounds_remaining=3, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"
    assert runner.calls == 1
    text = "\n".join(shown)
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text
    assert "Timed with a 3-batch dry run." in text
    assert ctx.questioner.asked == []           # a beginner on CPU is never interrupted


def test_a_gpu_run_asks_and_run_it_means_run(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=2, probes=T4_PROBES, dry_runner=runner)
    assert decision == "run"
    text = "\n".join(shown)
    assert "This runtime has a Tesla T4 [[GPU]]." in text
    assert "| minutes | [[compute units]] | cost |" in text
    assert "this run plus 2 remaining rounds" in text
    assert ctx.questioner.asked == ["Run it?"]


def test_stop_returns_stop_and_says_what_to_do_next(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.STOP_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES, dry_runner=runner)
    assert decision == "stop"
    assert cost_gate.STOP_TEXT in shown


def test_over_budget_shows_the_cuts_and_still_asks(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)    # 20 epochs -> 24 minutes against 5
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"                   # the gate warns, it never refuses
    text = "\n".join(shown)
    assert "**Over budget:**" in text
    assert "- lower `epochs` from 20 to 4 --" in text
    assert ctx.questioner.asked == ["Run it?"]  # asked even though the runtime is CPU


def test_editing_the_config_reuses_the_measured_timing_and_re_estimates(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)
    ctx, shown = make_ctx(project, answers=[
        cost_gate.EDIT_LABEL,        # "Run it?"
        "epochs = 20",               # edit_config: which value
        "4",                         # edit_config: new value
        "Done",                      # edit_config: finished
        cost_gate.RUN_LABEL,         # "Run it?" again
    ])
    decision = cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert decision == "run"
    assert runner.calls == 1                   # no second dry run, ever
    assert project.read_json("config.json")["epochs"] == 4
    text = "\n".join(shown)
    assert text.count("Timed with a 3-batch dry run.") == 2
    assert "**Over budget:**" in text          # the first estimate was
    assert text.rstrip().endswith("free.")     # the second one is not


def test_the_edit_option_disappears_after_three_edits(clean_project):
    project = priced(codegen(clean_project, config_updates={**SMALL, "epochs": 20}))
    runner = Runner(seconds_per_epoch=60.0)
    seen: list[list[str]] = []

    class Watcher(ScriptedQuestioner):
        def choice(self, question, options, allow_other=True, key=None, default=None):
            if question == "Run it?":
                seen.append(list(options))
            return super().choice(question, options, allow_other, key, default)

    ctx, shown = make_ctx(project)
    ctx.questioner = Watcher([
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.EDIT_LABEL, "Done",
        cost_gate.RUN_LABEL,
    ])
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    assert len(seen) == 4
    assert all(cost_gate.EDIT_LABEL in options for options in seen[:3])
    assert seen[3] == [cost_gate.RUN_LABEL, cost_gate.STOP_LABEL]


def test_the_second_run_estimates_from_history_and_never_shells_out(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("runs/run1_metrics.json", {"seconds_per_epoch": 4.0})
    project.runs_path.write_text(
        '{"run_id": 1, "status": "done", "seconds_per_epoch": 4.0, '
        '"config": {"model_type": "gradient_boosting"}}\n', encoding="utf-8")
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    assert cost_gate.gate(ctx, rounds_remaining=1, probes=T4_PROBES,
                          dry_runner=runner) == "run"
    assert runner.calls == 0
    assert "From run 1's measured time." in "\n".join(shown)


def test_a_run_of_another_family_is_not_used_as_history(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("runs/run1_metrics.json", {"seconds_per_epoch": 4.0})
    project.runs_path.write_text(
        '{"run_id": 1, "status": "done", "config": {"model_type": "random_forest"}}\n',
        encoding="utf-8")
    runner = Runner()
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES, dry_runner=runner)
    assert runner.calls == 1
    assert "Timed with a 3-batch dry run." in "\n".join(shown)


def test_a_failed_dry_run_says_so_and_still_reaches_the_question(clean_project):
    project = priced(codegen(clean_project))
    runner = Runner(error="The dry run failed (exit 1).\n\n```\nValueError: boom\n```")
    ctx, shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    text = "\n".join(shown)
    assert "ValueError: boom" in text
    assert cost_gate.NO_ESTIMATE_TEXT in text
    assert "| minutes |" not in text            # nothing to tabulate
    assert ctx.questioner.asked == ["Run it?"]


def test_a_failed_dry_run_clears_a_stale_estimate(clean_project):
    project = priced(codegen(clean_project))
    project.write_json("cost.json", {"price_per_unit": 0.0999, "currency": "$",
                                     "last_estimate": {"minutes": 9.9, "units": 1.0}})
    runner = Runner(error="The dry run failed (exit 1).")
    ctx, _shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=runner)
    assert project.read_json("cost.json").get("last_estimate") is None


def test_the_price_and_currency_are_asked_once_then_reused(clean_project):
    project = codegen(clean_project)            # no cost.json yet
    runner = Runner()
    ctx, shown = make_ctx(project, form={"train.price_per_unit": 0.05,
                                         "train.currency": "GBP"})
    assert cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=runner) == "run"
    record = project.read_json("cost.json")
    assert record["price_per_unit"] == pytest.approx(0.05)
    assert record["currency"] == "GBP"
    assert ctx.questioner.used == ["train.price_per_unit", "train.currency"]

    # A second gate call on the same project reads cost.json and asks nothing.
    ctx2, _shown2 = make_ctx(project)            # no answers of any kind
    assert cost_gate.gate(ctx2, rounds_remaining=0, probes=CPU_PROBES,
                          dry_runner=Runner()) == "run"
    assert ctx2.questioner.asked == []


def test_the_default_price_comes_from_rates_json(clean_project):
    project = codegen(clean_project)
    ctx, _shown = make_ctx(project, answers=["", ""])   # blank -> take the defaults
    cost_gate.gate(ctx, rounds_remaining=0, probes=CPU_PROBES, dry_runner=Runner())
    record = project.read_json("cost.json")
    rates = cost.load_rates()
    assert record["price_per_unit"] == pytest.approx(rates["default_price_per_unit"])
    assert record["currency"] == rates["default_currency"]


def test_the_estimate_is_persisted_for_the_debrief(clean_project):
    project = priced(codegen(clean_project))
    ctx, _shown = make_ctx(project, answers=[cost_gate.RUN_LABEL])
    cost_gate.gate(ctx, rounds_remaining=0, probes=T4_PROBES,
                   dry_runner=Runner(seconds_per_epoch=6.0))
    stored = project.read_json("cost.json")["last_estimate"]
    assert stored["basis"] == "dry_run" and stored["device"] == "cuda"
    assert stored["minutes"] == pytest.approx(0.36)       # 6 * 3 * 1.2 / 60
    assert stored["units"] == pytest.approx(0.012, abs=5e-4)
    assert stored["gpu_type"] == "T4"


def test_gate_is_usable_as_a_partial_the_way_the_stages_take_it(clean_project):
    project = priced(codegen(clean_project))
    bound = partial(cost_gate.gate, probes=CPU_PROBES, dry_runner=Runner())
    ctx, _shown = make_ctx(project)
    assert bound(ctx, rounds_remaining=0) == "run"
