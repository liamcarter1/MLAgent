"""The house chart style, plus the agent-side run-comparison figures.

Every per-run figure the user sees is drawn by a template script that carries its own copy
of this palette. What lives here is the palette, the axis style, and the two figures that
compare runs, which only the assistant can draw because only it has the run history."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#f3a17f", "#d95926"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
DIV_CMAP = LinearSegmentedColormap.from_list("mlagent_div", DIVERGING)


def style_axes(ax, title: str | None = None, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    return path


def _show(fig: Figure) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
    except Exception:  # noqa: BLE001 - display is best effort, never fatal
        pass


def present(fig: Figure, plots_dir: Path, name: str) -> Path:
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    _show(fig)
    plt.close(fig)
    return path


def save_and_close(fig: Figure, plots_dir: Path, name: str) -> Path:
    """Save without displaying: the teaching layer shows the file with its caption."""
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    plt.close(fig)
    return path


def _fmt_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_label(run: dict) -> str:
    """`run 2 (learning_rate 0.05)`: the run number plus the change that produced it."""
    run_id = run.get("run_id", "?")
    diff = run.get("applied_diff") or {}
    if not isinstance(diff, dict) or not diff:
        return f"run {run_id}"
    keys = list(diff)
    change = diff[keys[0]]
    to = change.get("to") if isinstance(change, dict) else change
    label = f"run {run_id} ({keys[0]} {_fmt_value(to)}"
    if len(keys) > 1:
        label += f", +{len(keys) - 1} more"
    return label + ")"


def compare_curves(runs: list[dict], metrics_by_run: dict[int, dict]) -> Figure:
    """Validation loss per epoch for every run, best epoch marked, one colour per run."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    drawn = 0
    for i, run in enumerate(runs):
        metrics = metrics_by_run.get(run.get("run_id")) or {}
        epochs = [e for e in (metrics.get("epochs") or [])
                  if isinstance(e, dict) and e.get("val_loss") is not None]
        if not epochs:
            continue
        colour = SERIES[i % len(SERIES)]
        xs = [e["epoch"] for e in epochs]
        ys = [e["val_loss"] for e in epochs]
        ax.plot(xs, ys, color=colour, linewidth=2, label=run_label(run))
        best = run.get("best_epoch")
        if best in xs:
            ax.plot([best], [ys[xs.index(best)]], marker="o", markersize=6, color=colour,
                    linestyle="none")
        drawn += 1
    if drawn:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no epochs recorded yet", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("validation loss", color=INK_2, fontsize=9)
    style_axes(ax, "Validation loss per epoch, every run")
    fig.tight_layout()
    return fig


def compare_runs(runs: list[dict], metric: str, target: float | None = None) -> Figure:
    """A horizontal bar per run of its best validation metric; failed runs hollow."""
    fig, ax = plt.subplots(figsize=(7.5, max(2.4, 0.55 * len(runs) + 1.4)))
    values = [r.get("best_val_metric") for r in runs
              if r.get("status") == "done" and r.get("best_val_metric") is not None]
    span = max([abs(v) for v in values] + [abs(target) if target is not None else 0.0, 1e-9])
    ys = list(range(len(runs)))
    for y, run in zip(ys, runs, strict=True):
        colour = SERIES[y % len(SERIES)]
        value = run.get("best_val_metric")
        if run.get("status") == "done" and value is not None:
            ax.barh(y, value, color=colour, height=0.6)
            ax.text(value, y, f" {value:.4g}", va="center", fontsize=8, color=INK_2)
        else:
            ax.barh(y, span, height=0.6, fill=False, edgecolor=colour, linestyle="--")
            ax.text(span / 2, y, "failed", ha="center", va="center", fontsize=8, color=MUTED)
    if target is not None:
        ax.axvline(target, color=INK_2, linestyle="--", linewidth=1, label=f"target {target:g}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(ys)
    ax.set_yticklabels([run_label(r) for r in runs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"best validation {metric}", color=INK_2, fontsize=9)
    style_axes(ax, "Best validation score per run", grid_axis="x")
    fig.tight_layout()
    return fig
