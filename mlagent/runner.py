"""Run a generated script in the project folder, streaming output and polling metrics.json."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mlagent import config
from mlagent.project import read_json_file

LOG_TAIL_LINES = 60


@dataclass
class RunResult:
    returncode: int
    metrics: dict | None
    log_tail: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    expects_metrics: bool = False

    @property
    def ok(self) -> bool:
        if self.timed_out or self.returncode != 0:
            return False
        if self.expects_metrics:
            return bool(self.metrics) and self.metrics.get("status") == "done"
        return True


def _pump(stream, out: queue.Queue) -> None:
    for line in iter(stream.readline, ""):
        out.put(line)
    out.put(None)


def run_script(
    project_root: Path,
    args: list[str],
    *,
    on_line: Callable[[str], None] | None = None,
    on_metrics: Callable[[dict], None] | None = None,
    metrics_path: Path | None = None,
    poll_seconds: float = 0.5,
    python: str = sys.executable,
    timeout: float | None = None,
) -> RunResult:
    """Run `python <args>` with cwd=project_root.

    Every stdout/stderr line goes to `on_line`. If `metrics_path` is given it is re-read
    every `poll_seconds` and `on_metrics` is called whenever the number of recorded epochs
    changes. `timeout` (seconds) kills the process and marks the result timed out.
    """
    started = time.time()
    proc = subprocess.Popen(
        [python, *args],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True)
    reader.start()

    tail: list[str] = []
    seen_epochs = -1
    timed_out = False
    eof = False

    def poll_metrics() -> None:
        nonlocal seen_epochs
        if metrics_path is None:
            return
        try:
            data = read_json_file(metrics_path, default=None)
        except OSError:
            # Windows: os.replace can conflict with an open read; treat as not readable yet.
            return
        if not isinstance(data, dict):
            return
        n = len(data.get("epochs") or [])
        if n != seen_epochs:
            seen_epochs = n
            if on_metrics is not None:
                on_metrics(data)

    while True:
        try:
            while True:
                item = lines.get_nowait()
                if item is None:
                    eof = True
                    break
                tail.append(item)
                del tail[:-LOG_TAIL_LINES]
                if on_line is not None:
                    on_line(item)
        except queue.Empty:
            pass
        poll_metrics()
        if eof and proc.poll() is not None:
            break
        if timeout is not None and time.time() - started > timeout:
            proc.kill()
            proc.wait()
            timed_out = True
            break
        time.sleep(poll_seconds)

    proc.wait()
    poll_metrics()
    try:
        # Windows: os.replace can conflict with an open read; fall back to None.
        metrics = read_json_file(metrics_path, default=None) if metrics_path is not None else None
    except OSError:
        metrics = None
    return RunResult(
        returncode=proc.returncode,
        metrics=metrics if isinstance(metrics, dict) else None,
        log_tail=tail,
        seconds=round(time.time() - started, 3),
        timed_out=timed_out,
        expects_metrics=metrics_path is not None,
    )


def run_training(project_root: Path, **kwargs) -> RunResult:
    """Run `train.py` in the project folder, polling `metrics.json`."""
    kwargs.setdefault("metrics_path", Path(project_root) / config.METRICS_FILE)
    return run_script(Path(project_root), ["train.py"], **kwargs)
