"""All matplotlib figures. Static charts: thin marks, recessive grid, text in ink colours."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pandas.api import types as ptypes

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


def _style(ax, title: str | None = None, grid_axis: str = "y") -> None:
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


def _numeric_columns(df: pd.DataFrame, columns=None) -> list[str]:
    cols = list(columns) if columns is not None else list(df.columns)
    return [
        c for c in cols
        if c in df.columns
        and ptypes.is_numeric_dtype(df[c])
        and not ptypes.is_bool_dtype(df[c])
    ]


def feature_histograms(
    df: pd.DataFrame, columns=None, max_cols: int = 12
) -> Figure:
    numeric = _numeric_columns(df, columns)[:max_cols]
    n = max(1, len(numeric))
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
        facecolor=SURFACE, squeeze=False
    )
    for ax, col in zip(axes.flat, numeric, strict=False):
        ax.hist(
            df[col].dropna(), bins=30, color=SERIES[0], edgecolor=SURFACE,
            linewidth=0.5
        )
        _style(ax, col)
    for ax in list(axes.flat)[len(numeric):]:
        ax.set_visible(False)
    if not numeric:
        axes[0][0].set_visible(True)
        axes[0][0].text(0.5, 0.5, "no numeric columns", ha="center", color=MUTED)
    fig.suptitle("Feature distributions", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def class_balance(counts: dict[str, int], title: str = "Target class balance") -> Figure:
    fig = plt.figure(
        figsize=(5, 0.5 * max(3, len(counts)) + 1.2), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    labels = [str(k) for k in counts]
    values = [int(v) for v in counts.values()]
    total = sum(values) or 1
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v} ({v / total:.0%})", va="center", color=INK_2, fontsize=8)
    _style(ax, title, grid_axis="x")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3 if values else 1)
    fig.tight_layout()
    return fig


def target_distribution(series: pd.Series, title: str = "Target distribution") -> Figure:
    fig = plt.figure(figsize=(5, 2.8), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.hist(
        series.dropna(), bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5
    )
    _style(ax, title)
    fig.tight_layout()
    return fig


def missing_matrix(df: pd.DataFrame, max_rows: int = 500) -> Figure:
    sample = (
        df if len(df) <= max_rows
        else df.sample(max_rows, random_state=0).sort_index()
    )
    mat = sample.isna().to_numpy().T.astype(float)
    fig = plt.figure(
        figsize=(7, 0.28 * len(df.columns) + 1.5), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    ax.imshow(
        mat, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest", vmin=0, vmax=1
    )
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels([str(c) for c in df.columns], fontsize=8, color=INK_2)
    ax.set_xlabel(
        f"rows (showing {len(sample)} of {len(df)})", color=MUTED, fontsize=8
    )
    _style(ax, "Missing values (dark = missing)", grid_axis="none")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> Figure:
    numeric = df[_numeric_columns(df)[:max_cols]]
    fig = plt.figure(figsize=(6, 5), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    if numeric.shape[1] < 2:
        ax.text(0.5, 0.5, "need at least two numeric columns", ha="center", color=MUTED)
        ax.set_axis_off()
        return fig
    corr = numeric.corr().to_numpy()
    im = ax.imshow(corr, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ticks = range(numeric.shape[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(numeric.columns, rotation=60, ha="right", fontsize=8, color=INK_2)
    ax.set_yticklabels(numeric.columns, fontsize=8, color=INK_2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        colors=MUTED, labelsize=8
    )
    _style(ax, "Correlation between numeric columns", grid_axis="none")
    fig.tight_layout()
    return fig


def outlier_boxplots(df: pd.DataFrame, columns, max_cols: int = 8) -> Figure:
    cols = _numeric_columns(df, columns)[:max_cols]
    fig = plt.figure(
        figsize=(6, 0.5 * max(2, len(cols)) + 1.2), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    data = [df[c].dropna().to_numpy() for c in cols]
    if data:
        box = ax.boxplot(
            data, orientation='horizontal', tick_labels=cols, patch_artist=True,
            widths=0.5, flierprops={
                "marker": ".", "markersize": 4, "markerfacecolor": SERIES[1],
                "markeredgecolor": SERIES[1]
            }
        )
        for patch in box["boxes"]:
            patch.set_facecolor(SEQUENTIAL[1])
            patch.set_edgecolor(SERIES[0])
        for key in ("whiskers", "caps", "medians"):
            for line in box[key]:
                line.set_color(SERIES[0])
    _style(ax, "Value ranges and outliers", grid_axis="x")
    fig.tight_layout()
    return fig


def before_after_missing(before: dict, after: dict) -> Figure:
    names = [c["name"] for c in before["columns"]]
    after_pct = {c["name"]: c["missing_pct"] for c in after["columns"]}
    b = [c["missing_pct"] for c in before["columns"]]
    a = [after_pct.get(n, 0.0) for n in names]
    y = np.arange(len(names))
    fig = plt.figure(
        figsize=(6, 0.35 * max(3, len(names)) + 1.4), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    ax.barh(y - 0.18, b, height=0.34, color=SERIES[0], label="before cleaning")
    ax.barh(y + 0.18, a, height=0.34, color=SERIES[1], label="after cleaning")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK_2)
    ax.set_xlabel("% missing", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Missing values before and after cleaning", grid_axis="x")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return fig


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
    except Exception:
        pass


def present(fig: Figure, plots_dir: Path, name: str) -> Path:
    path = save_figure(fig, Path(plots_dir) / f"{name}.png")
    _show(fig)
    plt.close(fig)
    return path
