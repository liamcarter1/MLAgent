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
