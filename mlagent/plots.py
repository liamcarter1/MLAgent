"""The house chart style. Every figure the *user* sees is drawn by a template script that
carries its own copy of this palette; what stays here is the palette itself plus the two
helpers Milestone 5's agent-side run-comparison plots will use."""

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
