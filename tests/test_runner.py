from __future__ import annotations

import textwrap

from mlagent import runner

EPOCH_SCRIPT = textwrap.dedent(
    """
    import json, time
    for epoch in range(1, 4):
        time.sleep(0.15)
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump({"status": "running" if epoch < 3 else "done",
                       "epochs": [{"epoch": e} for e in range(1, epoch + 1)]}, f)
        print(f"epoch {epoch}", flush=True)
    """
)


def test_streams_lines_and_polls_metrics(tmp_path):
    (tmp_path / "train.py").write_text(EPOCH_SCRIPT, encoding="utf-8")
    lines, updates = [], []
    result = runner.run_training(
        tmp_path, on_line=lines.append, on_metrics=updates.append, poll_seconds=0.05
    )
    assert result.returncode == 0 and result.ok and not result.timed_out
    assert [line.strip() for line in lines] == ["epoch 1", "epoch 2", "epoch 3"]
    assert result.log_tail[-1].strip() == "epoch 3"
    assert result.metrics["status"] == "done"
    assert len(updates) >= 2
    counts = [len(u["epochs"]) for u in updates]
    assert counts == sorted(counts)
    assert result.seconds > 0


def test_failing_script_reports_returncode_and_tail(tmp_path):
    (tmp_path / "bad.py").write_text("print('starting')\nraise SystemExit(3)\n", encoding="utf-8")
    result = runner.run_script(tmp_path, ["bad.py"])
    assert result.returncode == 3 and not result.ok
    assert result.metrics is None
    assert "starting" in "".join(result.log_tail)


def test_timeout_kills_process(tmp_path):
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    result = runner.run_script(tmp_path, ["slow.py"], timeout=0.5, poll_seconds=0.05)
    assert result.timed_out and not result.ok
    assert result.seconds < 10


def test_poll_tolerates_unreadable_metrics(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text(EPOCH_SCRIPT, encoding="utf-8")
    real_read_json_file = runner.read_json_file
    calls = {"n": 0}

    def flaky_read_json_file(path, default=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("simulated Windows replace-while-open race")
        return real_read_json_file(path, default=default)

    monkeypatch.setattr(runner, "read_json_file", flaky_read_json_file)
    updates = []
    result = runner.run_training(tmp_path, on_metrics=updates.append, poll_seconds=0.05)
    assert result.ok
    assert len(updates) >= 1


def test_ok_requires_done_status_when_metrics_expected(tmp_path):
    (tmp_path / "train.py").write_text(
        "import json\n"
        "with open('metrics.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump({'status': 'failed', 'epochs': []}, f)\n",
        encoding="utf-8",
    )
    result = runner.run_training(tmp_path, poll_seconds=0.05)
    assert result.returncode == 0 and not result.ok
    assert result.metrics["status"] == "failed"
