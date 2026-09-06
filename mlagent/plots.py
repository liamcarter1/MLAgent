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


def class_balance(
    counts: dict[str, int], total: int | None = None, title: str = "Target class balance"
) -> Figure:
    fig = plt.figure(
        figsize=(5, 0.5 * max(3, len(counts)) + 1.2), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    labels = [str(k) for k in counts]
    values = [int(v) for v in counts.values()]
    denom = total if total else (sum(values) or 1)
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v} ({v / denom:.0%})", va="center", color=INK_2, fontsize=8)
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


def missing_matrix(df: pd.DataFrame, max_rows: int = 500, max_cols: int = 60) -> Figure:
    sample = (
        df if len(df) <= max_rows
        else df.sample(max_rows, random_state=0).sort_index()
    )
    columns = list(df.columns)[:max_cols]
    shown_cols = sample[columns]
    mat = shown_cols.isna().to_numpy().T.astype(float)
    fig = plt.figure(
        figsize=(7, 0.28 * len(columns) + 1.5), facecolor=SURFACE
    )
    ax = fig.add_subplot(111)
    ax.imshow(
        mat, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest", vmin=0, vmax=1
    )
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels([str(c) for c in columns], fontsize=8, color=INK_2)
    xlabel = f"rows (showing {len(sample)} of {len(df)})"
    title = "Missing values (dark = missing)"
    if len(columns) < len(df.columns):
        title += f" - showing {len(columns)} of {len(df.columns)} columns"
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    _style(ax, title, grid_axis="none")
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


def before_after_missing(before: dict, after: dict, max_cols: int = 60) -> Figure:
    all_names = [c["name"] for c in before["columns"]]
    names = all_names[:max_cols]
    after_pct = {c["name"]: c["missing_pct"] for c in after["columns"]}
    before_pct = {c["name"]: c["missing_pct"] for c in before["columns"]}
    b = [before_pct[n] for n in names]
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
    title = "Missing values before and after cleaning"
    if len(names) < len(all_names):
        title += f" - showing {len(names)} of {len(all_names)} columns"
    _style(ax, title, grid_axis="x")
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


# --- Milestone 3: training and evaluation figures -------------------------------------


def _frame(ax, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)


def training_curves(epochs: list[dict], metric: str) -> Figure:
    """Loss (left) and the target metric (right) per epoch, train vs validation."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(9, 3.2))
    xs = [e.get("epoch") for e in epochs]
    panels = (
        (ax_loss, ("train_loss", "val_loss"), "Loss per epoch"),
        (ax_metric, ("train_metric", "val_metric"), f"{metric} per epoch"),
    )
    for ax, keys, title in panels:
        for key, colour, label in zip(keys, SERIES[:2], ("train", "validation"), strict=True):
            ax.plot(xs, [e.get(key) for e in epochs], color=colour, linewidth=2,
                    marker="o", markersize=4, label=label)
        _frame(ax, title)
        ax.set_xlabel("epoch", color=INK_2, fontsize=8)
        _legend(ax)
    if not epochs:
        ax_loss.text(0.5, 0.5, "no epochs yet", ha="center", va="center", color=MUTED,
                     transform=ax_loss.transAxes)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(cm, labels: list[str]) -> Figure:
    m = np.asarray(cm)
    n = len(labels)
    size = min(2.5 + 0.35 * n, 9)
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = LinearSegmentedColormap.from_list("mlagent_seq", SEQUENTIAL)
    ax.imshow(m, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK_2, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=INK_2, fontsize=8)
    ax.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax.set_ylabel("actual", color=INK_2, fontsize=8)
    if n <= 20 and m.size:
        threshold = m.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color=SURFACE if m[i, j] > threshold else INK)
    ax.set_title("Confusion matrix", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def roc_pr_curves(y_true, y_proba, labels: list[str]) -> Figure:
    """ROC (left) and precision-recall (right). Binary: one curve for the positive class.
    Multiclass: one-vs-rest per class, at most len(SERIES) classes shown."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.6))
    n = p.shape[1] if p.ndim == 2 else 1
    if n == 2:
        curves = [(1, p[:, 1], labels[1])]
    else:
        curves = [(k, p[:, k], labels[k]) for k in range(min(n, len(SERIES)))]
    for (k, score, label), colour in zip(curves, SERIES, strict=False):
        positive = (y == k).astype(int)
        if positive.sum() in (0, len(positive)):
            continue
        fpr, tpr, _ = roc_curve(positive, score)
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, label=label)
        precision, recall, _ = precision_recall_curve(positive, score)
        ax_pr.plot(recall, precision, color=colour, linewidth=2, label=label)
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    suffix = f" (showing {len(curves)} of {n} classes)" if n > len(curves) else ""
    _frame(ax_roc, "ROC curve" + suffix)
    ax_roc.set_xlabel("false positive rate", color=INK_2, fontsize=8)
    ax_roc.set_ylabel("true positive rate", color=INK_2, fontsize=8)
    _frame(ax_pr, "Precision-recall curve")
    ax_pr.set_xlabel("recall", color=INK_2, fontsize=8)
    ax_pr.set_ylabel("precision", color=INK_2, fontsize=8)
    if len(curves) > 1:
        _legend(ax_roc)
        _legend(ax_pr)
    fig.tight_layout()
    return fig


def per_class_bars(y_true, y_pred, labels: list[str]) -> Figure:
    from sklearn.metrics import precision_recall_fscore_support

    n = len(labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * n + 2), 3.2))
    x = np.arange(n)
    width = 0.38
    ax.bar(x - width / 2, precision, width=width, color=SERIES[0], label="precision")
    ax.bar(x + width / 2, recall, width=width, color=SERIES[1], label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if n > 6 else 0, ha="right" if n > 6 else "center",
                       color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.05)
    _frame(ax, "Precision and recall per class")
    _legend(ax)
    fig.tight_layout()
    return fig


def predicted_vs_actual(y_true, y_pred) -> Figure:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(y, p, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linestyle="--", linewidth=1)
    _frame(ax, "Predicted vs actual")
    ax.set_xlabel("actual", color=INK_2, fontsize=8)
    ax.set_ylabel("predicted", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def residual_plots(y_true, y_pred) -> Figure:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    residuals = y - p
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax_hist.hist(residuals, bins=min(30, max(5, len(residuals) // 5)), color=SERIES[0])
    _frame(ax_hist, "Residual distribution")
    ax_hist.set_xlabel("actual - predicted", color=INK_2, fontsize=8)
    ax_scatter.scatter(p, residuals, s=14, color=SERIES[0], alpha=0.7, edgecolors="none")
    ax_scatter.axhline(0, color=AXIS, linestyle="--", linewidth=1)
    _frame(ax_scatter, "Residuals vs predicted")
    ax_scatter.set_xlabel("predicted", color=INK_2, fontsize=8)
    ax_scatter.set_ylabel("residual", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


def present_evaluation(eval_data: dict, plots_dir: Path, prefix: str) -> list[Path]:
    """Save and show the evaluation figures appropriate to the task; return saved paths."""
    y_true = list(eval_data.get("y_true") or [])
    y_pred = list(eval_data.get("y_pred") or [])
    saved: list[Path] = []
    if eval_data.get("task_type") == "tabular_classification":
        from sklearn.metrics import confusion_matrix

        classes = eval_data.get("classes") or sorted({*y_true, *y_pred})
        labels = [str(c) for c in classes]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        saved.append(present(confusion_matrix_plot(cm, labels), plots_dir, f"{prefix}_confusion"))
        if eval_data.get("y_proba"):
            fig = roc_pr_curves(y_true, eval_data["y_proba"], labels)
            saved.append(present(fig, plots_dir, f"{prefix}_roc_pr"))
        saved.append(present(per_class_bars(y_true, y_pred, labels), plots_dir,
                             f"{prefix}_per_class"))
    else:
        saved.append(present(predicted_vs_actual(y_true, y_pred), plots_dir,
                             f"{prefix}_pred_vs_actual"))
        saved.append(present(residual_plots(y_true, y_pred), plots_dir, f"{prefix}_residuals"))
    return saved
