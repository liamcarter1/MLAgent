"""The pure cost module: rates, runtime detection and the dry-run record."""

from __future__ import annotations

import json
import subprocess

import pytest

from mlagent import config as cfg
from mlagent import cost


def test_rates_json_holds_the_three_known_gpus_and_the_colab_pro_price():
    rates = cost.load_rates()
    assert rates["units_per_hour"] == {"T4": 2.0, "L4": 4.8, "A100": 13.0}
    assert rates["unknown_gpu_rate"] == 2.0
    assert rates["default_price_per_unit"] == pytest.approx(0.0999)
    assert rates["default_currency"] == "$"


def test_the_rates_file_sits_next_to_the_package_like_the_prompts_do():
    assert cost.RATES_PATH.name == "rates.json"
    assert cost.RATES_PATH.parent.name == "mlagent"
    assert json.loads(cost.RATES_PATH.read_text(encoding="utf-8")) == cost.load_rates()


def test_config_no_longer_carries_the_rates():
    assert not hasattr(cfg, "DEFAULT_RATES")
    assert not hasattr(cfg, "PRICE_PER_100_UNITS_USD")
    assert cfg.COST_FILE == "cost.json"
    assert cfg.DRY_RUN_FILE == "dry_run.json"


def test_a_torch_probe_that_finds_a_t4_gives_a_cuda_runtime():
    runtime = cost.detect_runtime([("torch", lambda: "Tesla T4")])
    assert runtime == cost.RuntimeInfo(device="cuda", gpu_name="Tesla T4",
                                       gpu_type="T4", source="torch")


def test_the_nvidia_smi_probe_is_tried_when_the_torch_probe_finds_nothing():
    runtime = cost.detect_runtime([("torch", lambda: None),
                                   ("nvidia-smi", lambda: "NVIDIA L4")])
    assert runtime.device == "cuda" and runtime.gpu_type == "L4"
    assert runtime.source == "nvidia-smi" and runtime.gpu_name == "NVIDIA L4"


def test_no_probe_finding_anything_is_a_cpu_runtime():
    runtime = cost.detect_runtime(())
    assert runtime == cost.RuntimeInfo(device="cpu", gpu_name=None, gpu_type=None,
                                       source="none")


def test_an_unrecognised_gpu_name_is_typed_unknown():
    runtime = cost.detect_runtime([("torch", lambda: "NVIDIA H200")])
    assert runtime.device == "cuda" and runtime.gpu_type == "unknown"


def test_gpu_types_match_case_insensitively_as_a_substring():
    rates = cost.load_rates()
    assert cost.gpu_type_for("NVIDIA A100-SXM4-40GB", rates) == "A100"
    assert cost.gpu_type_for("nvidia a100", rates) == "A100"
    assert cost.gpu_type_for(None, rates) is None
    assert cost.gpu_type_for("Radeon Pro", rates) == "unknown"


def test_a_probe_that_raises_is_skipped_not_fatal():
    def boom() -> str:
        raise RuntimeError("no driver")

    runtime = cost.detect_runtime([("torch", boom), ("nvidia-smi", lambda: "Tesla T4")])
    assert runtime.gpu_type == "T4" and runtime.source == "nvidia-smi"


def test_the_nvidia_smi_probe_parses_the_first_line_of_a_fake_subprocess():
    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(command, 0, stdout="Tesla T4\nTesla T4\n",
                                           stderr="")

    assert cost.nvidia_smi_gpu_name(runner) == "Tesla T4"


def test_the_nvidia_smi_probe_returns_none_when_the_binary_is_absent():
    def runner(command, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert cost.nvidia_smi_gpu_name(runner) is None


def test_the_nvidia_smi_probe_returns_none_on_a_non_zero_exit():
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom")

    assert cost.nvidia_smi_gpu_name(runner) is None


def test_rate_for_a_cpu_runtime_is_zero_with_no_note():
    rates = cost.load_rates()
    units, note = cost.rate_for(cost.detect_runtime(()), rates)
    assert units == 0.0 and note is None


def test_rate_for_a_known_gpu_is_the_table_value_with_no_note():
    rates = cost.load_rates()
    units, note = cost.rate_for(cost.detect_runtime([("torch", lambda: "Tesla T4")]), rates)
    assert units == 2.0 and note is None


def test_rate_for_an_unknown_gpu_falls_back_and_says_so():
    rates = cost.load_rates()
    runtime = cost.detect_runtime([("torch", lambda: "NVIDIA H200")])
    units, note = cost.rate_for(runtime, rates)
    assert units == 2.0
    assert note is not None and "NVIDIA H200" in note and "T4" in note


def test_read_dry_run_parses_every_key(tmp_path):
    (tmp_path / "dry_run.json").write_text(json.dumps({
        "device": "cuda", "gpu_name": "Tesla T4", "batches_per_epoch": 38,
        "seconds_per_batch": 0.14, "seconds_per_epoch": 5.32, "n_train": 1200,
        "script": "train.py",
    }), encoding="utf-8")
    dry = cost.read_dry_run(tmp_path / "dry_run.json")
    assert dry == cost.DryRunResult(device="cuda", gpu_name="Tesla T4", batches_per_epoch=38,
                                    seconds_per_batch=0.14, seconds_per_epoch=5.32,
                                    n_train=1200, script="train.py", error=None)


def test_read_dry_run_of_a_missing_file_is_zeroed_with_an_error(tmp_path):
    dry = cost.read_dry_run(tmp_path / "nope.json")
    assert dry.seconds_per_epoch == 0.0 and dry.batches_per_epoch == 0
    assert dry.error is not None and "nope.json" in dry.error


def test_read_dry_run_of_broken_json_is_zeroed_with_an_error(tmp_path):
    path = tmp_path / "dry_run.json"
    path.write_text("{not json", encoding="utf-8")
    dry = cost.read_dry_run(path)
    assert dry.seconds_per_epoch == 0.0
    assert dry.error is not None and "dry_run.json" in dry.error


import sys  # noqa: E402

from mlagent.modality import modality_for  # noqa: E402
from mlagent.templates_io import load_schema, schema_for  # noqa: E402

TABULAR = modality_for("tabular_classification")
IMAGE = modality_for("image_classification")
GB_SCHEMA = schema_for(load_schema("tabular_sklearn"), "gradient_boosting")
CNN_SCHEMA = schema_for(load_schema("image_torch"), "small_cnn")

T4 = cost.RuntimeInfo(device="cuda", gpu_name="Tesla T4", gpu_type="T4", source="torch")
CPU = cost.RuntimeInfo(device="cpu", gpu_name=None, gpu_type=None, source="none")
H200 = cost.RuntimeInfo(device="cuda", gpu_name="NVIDIA H200", gpu_type="unknown",
                        source="torch")


def make_estimate(seconds_per_epoch=6.0, epochs=10, runtime=T4, rounds_remaining=3,
                  basis="dry_run", price=0.0999, currency="$"):
    return cost.estimate_run(seconds_per_epoch, epochs, runtime, cost.load_rates(),
                             price, currency, rounds_remaining, basis)


def test_the_formula_matches_the_2026_09_06_spec_by_hand():
    # hours = 6 * 10 * 1.2 / 3600 = 0.02; units = 0.02 * 2.0 = 0.04;
    # cost = 0.04 * 0.0999 = 0.003996
    est = make_estimate()
    assert est.minutes == pytest.approx(1.2)
    assert est.units == pytest.approx(0.04)
    assert est.cost == pytest.approx(0.004, abs=5e-4)
    assert est.rate_units_per_hour == 2.0
    assert est.basis == "dry_run" and est.runtime is T4
    assert est.seconds_per_epoch == 6.0 and est.epochs == 10


def test_all_rounds_is_this_run_times_one_plus_the_remaining_rounds():
    est = make_estimate(rounds_remaining=3)
    assert est.minutes_all_rounds == pytest.approx(4.8)
    assert est.units_all_rounds == pytest.approx(0.16)
    assert est.cost_all_rounds == pytest.approx(est.cost * 4, rel=1e-3)


def test_no_remaining_rounds_makes_the_totals_equal_this_run():
    est = make_estimate(rounds_remaining=0)
    assert est.minutes_all_rounds == est.minutes
    assert est.units_all_rounds == est.units


def test_a_cpu_run_spends_no_units_and_no_money():
    est = make_estimate(runtime=CPU)
    assert est.rate_units_per_hour == 0.0
    assert est.units == 0.0 and est.cost == 0.0
    assert est.minutes == pytest.approx(1.2)


def test_an_unknown_gpu_carries_the_note_onto_the_estimate():
    est = make_estimate(runtime=H200)
    assert est.rate_units_per_hour == 2.0
    assert est.note is not None and "NVIDIA H200" in est.note


def test_estimate_from_dry_run_uses_the_measured_seconds_per_epoch():
    dry = cost.DryRunResult(device="cuda", gpu_name="Tesla T4", batches_per_epoch=38,
                            seconds_per_batch=0.2, seconds_per_epoch=7.6, n_train=1200)
    est = cost.estimate_from_dry_run(dry, 5, T4, cost.load_rates(), 0.0999, "$", 0)
    assert est.basis == "dry_run" and est.seconds_per_epoch == 7.6
    assert est.minutes == pytest.approx(7.6 * 5 * 1.2 / 60)


def test_estimate_from_history_is_the_same_arithmetic_with_a_history_basis():
    est = cost.estimate_from_history(7.6, 5, T4, cost.load_rates(), 0.0999, "$", 0)
    same = cost.estimate_from_dry_run(
        cost.DryRunResult(seconds_per_epoch=7.6), 5, T4, cost.load_rates(), 0.0999, "$", 0)
    assert est.basis == "history" and same.basis == "dry_run"
    assert est.minutes == same.minutes and est.units == same.units


def test_estimate_to_dict_is_json_ready_and_flattens_the_runtime():
    payload = make_estimate().to_dict()
    assert json.loads(json.dumps(payload))["basis"] == "dry_run"
    assert payload["device"] == "cuda" and payload["gpu_type"] == "T4"
    assert payload["minutes"] == pytest.approx(1.2)
    assert payload["units"] == pytest.approx(0.04)


def test_under_budget_means_no_advice_and_no_suggestions():
    est = make_estimate(seconds_per_epoch=6.0, epochs=10)      # 1.2 minutes
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    assert advice.over is False and advice.suggestions == []
    assert advice.minutes_over == 0.0


def test_over_budget_names_the_largest_epochs_that_fits():
    # 60 s/epoch * 20 epochs * 1.2 = 1440 s = 24 minutes against a 5 minute budget.
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    config = {"epochs": 20, "model_type": "gradient_boosting"}
    advice = cost.budget_check(est, 5, config, GB_SCHEMA, TABULAR)
    assert advice.over is True
    assert advice.minutes_over == pytest.approx(19.0)
    assert advice.suggestions[0][0] == "epochs"
    assert advice.suggestions[0][1] == 20 and advice.suggestions[0][2] == 4
    # The proposal really does fit: 60 * 4 * 1.2 / 60 = 4.8 minutes.
    assert 60.0 * 4 * cost.SAFETY_FACTOR / 60 <= 5


def test_the_epochs_proposal_never_drops_below_one():
    est = make_estimate(seconds_per_epoch=600.0, epochs=2, runtime=CPU)   # 24 minutes
    advice = cost.budget_check(est, 1, {"epochs": 2, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    assert advice.suggestions[0][2] == 1


def test_an_image_run_over_budget_also_offers_a_bigger_batch_size():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    config = {"epochs": 20, "batch_size": 32, "model_type": "small_cnn"}
    advice = cost.budget_check(est, 5, config, CNN_SCHEMA, IMAGE)
    keys = [s[0] for s in advice.suggestions]
    assert keys == ["epochs", "batch_size"]
    batch = next(s for s in advice.suggestions if s[0] == "batch_size")
    assert batch[1] == 32 and batch[2] == 64
    assert "image_size" not in keys       # fixed at ingest, never a config key


def test_a_batch_size_already_at_the_schema_maximum_is_not_suggested():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    config = {"epochs": 20, "batch_size": 256, "model_type": "small_cnn"}
    advice = cost.budget_check(est, 5, config, CNN_SCHEMA, IMAGE)
    assert [s[0] for s in advice.suggestions] == ["epochs"]


def test_a_suggestion_is_never_made_for_a_key_the_schema_does_not_have():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    stripped = {k: v for k, v in GB_SCHEMA.items() if k != "epochs"}
    advice = cost.budget_check(est, 5, {"epochs": 20, "model_type": "gradient_boosting"},
                               stripped, TABULAR)
    assert advice.over is True and advice.suggestions == []


def test_money_puts_a_one_character_symbol_in_front_and_a_code_behind():
    assert cost.money(1.5, "$") == "$1.50"
    assert cost.money(1.5, "GBP") == "1.50 GBP"


def test_render_names_the_gpu_the_table_and_the_dry_run_basis():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "T4", TABULAR)
    assert "This runtime has a Tesla T4 [[GPU]]." in text
    assert "| minutes | [[compute units]] | cost |" in text
    assert "| this run | 1.2 |" in text
    assert "this run plus 3 remaining rounds" in text
    assert "Timed with a 3-batch dry run." in text
    assert "Change runtime type" not in text       # spec asked for T4 and got one


def test_render_names_the_source_run_for_a_history_estimate():
    est = make_estimate(runtime=CPU, basis="history")
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR, basis_run_id=2)
    assert "From run 2's measured time." in text
    assert "dry run" not in text


def test_render_says_a_cpu_run_is_free():
    est = make_estimate(runtime=CPU)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR)
    assert "This runtime has no [[GPU]]." in text
    assert "A [[CPU]] runtime uses no [[compute unit]]s, so this run is free." in text


def test_render_warns_when_intake_asked_for_a_gpu_and_there_is_none():
    est = make_estimate(runtime=CPU)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "T4", TABULAR)
    assert "Runtime -> Change runtime type" in text
    assert "asked for a [[GPU]] at intake" in text


def test_render_warns_when_intake_asked_for_no_gpu_and_there_is_one():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "small_cnn"},
                               CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "none", IMAGE)
    assert "asked for no [[GPU]] at intake" in text


def test_render_tells_a_tabular_run_on_a_gpu_it_is_paying_for_nothing():
    est = make_estimate(runtime=T4)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "any", TABULAR)
    assert "switch to a CPU runtime to train for free" in text


def test_render_lists_the_cuts_when_over_budget():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=CPU)
    advice = cost.budget_check(est, 5, {"epochs": 20, "model_type": "gradient_boosting"},
                               GB_SCHEMA, TABULAR)
    text = cost.render_estimate(est, advice, "none", TABULAR)
    assert "**Over budget:**" in text
    assert "5-minute limit" in text
    assert "- lower `epochs` from 20 to 4 --" in text
    assert "IMAGE_SIZE" not in text


def test_render_adds_the_reingest_sentence_for_an_over_budget_image_run():
    est = make_estimate(seconds_per_epoch=60.0, epochs=20, runtime=T4)
    advice = cost.budget_check(est, 5, {"epochs": 20, "batch_size": 32,
                                        "model_type": "small_cnn"}, CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "T4", IMAGE)
    assert "- raise `batch_size` from 32 to 64 --" in text
    assert "re-ingesting the data at a smaller `IMAGE_SIZE`" in text


def test_render_shows_the_unknown_gpu_note():
    est = make_estimate(runtime=H200)
    advice = cost.budget_check(est, 10, {"epochs": 10, "model_type": "small_cnn"},
                               CNN_SCHEMA, IMAGE)
    text = cost.render_estimate(est, advice, "any", IMAGE)
    assert "not one of the known [[GPU]] types" in text


def write_fake_train(root, body: str) -> None:
    (root / "train.py").write_text(body, encoding="utf-8")


def test_run_dry_run_reads_back_what_the_script_wrote(tmp_path):
    write_fake_train(tmp_path, (
        "import json, pathlib, sys\n"
        "assert sys.argv[1] == '--dry-run'\n"
        "pathlib.Path('dry_run.json').write_text(json.dumps({\n"
        "    'device': 'cpu', 'gpu_name': None, 'batches_per_epoch': 4,\n"
        "    'seconds_per_batch': 0.25, 'seconds_per_epoch': 1.0, 'n_train': 80,\n"
        "    'script': 'train.py'}), encoding='utf-8')\n"
    ))
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is None
    assert dry.batches_per_epoch == 4 and dry.seconds_per_epoch == 1.0


def test_run_dry_run_reports_a_non_zero_exit_with_the_stderr_tail(tmp_path):
    write_fake_train(tmp_path, (
        "import sys\n"
        "print('ValueError: the training split is empty', file=sys.stderr)\n"
        "sys.exit(3)\n"
    ))
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None
    assert "exit 3" in dry.error
    assert "the training split is empty" in dry.error
    assert dry.seconds_per_epoch == 0.0


def test_run_dry_run_reports_a_script_that_wrote_nothing(tmp_path):
    write_fake_train(tmp_path, "print('nothing to see')\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None and "dry_run.json" in dry.error


def test_run_dry_run_reports_a_timeout_instead_of_raising(tmp_path):
    write_fake_train(tmp_path, "import time\ntime.sleep(30)\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=1)
    assert dry.error is not None and "did not finish" in dry.error


def test_run_dry_run_deletes_a_stale_record_before_running(tmp_path):
    (tmp_path / "dry_run.json").write_text('{"seconds_per_epoch": 99.0}', encoding="utf-8")
    write_fake_train(tmp_path, "import sys\nsys.exit(1)\n")
    dry = cost.run_dry_run(tmp_path, python=sys.executable, timeout=60)
    assert dry.error is not None
    assert not (tmp_path / "dry_run.json").exists()
