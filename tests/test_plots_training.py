from __future__ import annotations

import warnings

import numpy as np
from matplotlib.figure import Figure

from mlagent import plots

EPOCHS = [
    {"epoch": 1, "train_loss": 0.9, "val_loss": 0.95, "train_metric": 0.6, "val_metric": 0.55},
    {"epoch": 2, "train_loss": 0.6, "val_loss": 0.7, "train_metric": 0.75, "val_metric": 0.7},
    {"epoch": 3, "train_loss": 0.4, "val_loss": 0.65, "train_metric": 0.85, "val_metric": 0.72},
]


def test_training_curves_two_panels_with_legends():
    fig = plots.training_curves(EPOCHS, "accuracy")
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_legend() is not None
        assert len(ax.lines) == 2
    assert "accuracy" in fig.axes[1].get_title(loc="left")
    plots.plt.close(fig)


def test_training_curves_empty_does_not_crash():
    fig = plots.training_curves([], "rmse")
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_confusion_matrix_annotates_cells():
    fig = plots.confusion_matrix_plot([[5, 1], [2, 7]], ["no", "yes"])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert sorted(texts) == ["1", "2", "5", "7"]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["no", "yes"]
    plots.plt.close(fig)


def test_roc_pr_binary_has_no_legend_multiclass_has_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    p = rng.random((40, 2))
    p = p / p.sum(axis=1, keepdims=True)
    fig = plots.roc_pr_curves(y.tolist(), p.tolist(), ["a", "b"])
    assert len(fig.axes) == 2 and fig.axes[0].get_legend() is None
    # F6: binary ROC title must not claim classes were truncated.
    assert "showing" not in fig.axes[0].get_title(loc="left")
    plots.plt.close(fig)
    y3 = rng.integers(0, 3, size=60)
    p3 = rng.random((60, 3))
    fig = plots.roc_pr_curves(y3.tolist(), p3.tolist(), ["a", "b", "c"])
    assert fig.axes[0].get_legend() is not None
    assert len(fig.axes[0].lines) >= 3
    plots.plt.close(fig)


def test_roc_pr_degenerate_labels_draw_no_legend():
    y_true = [0, 0, 0, 0]
    y_proba = [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig = plots.roc_pr_curves(y_true, y_proba, ["a", "b", "c"])
        assert fig.axes[0].get_legend() is None
        assert len(w) == 0
    plots.plt.close(fig)


def test_per_class_bars_two_series_legend():
    fig = plots.per_class_bars([0, 1, 1, 0], [0, 1, 0, 0], ["a", "b"])
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.patches) == 4
    plots.plt.close(fig)


def test_regression_figures():
    y = [1.0, 2.0, 3.0, 4.0]
    p = [1.1, 1.9, 3.3, 3.8]
    fig = plots.predicted_vs_actual(y, p)
    assert len(fig.axes) == 1 and fig.axes[0].get_legend() is None
    plots.plt.close(fig)
    fig = plots.residual_plots(y, p)
    assert len(fig.axes) == 2
    plots.plt.close(fig)


def test_present_evaluation_classification_and_regression(tmp_path):
    cls = {"task_type": "tabular_classification", "classes": ["a", "b"],
           "y_true": [0, 1, 1, 0, 1], "y_pred": [0, 1, 0, 0, 1],
           "y_proba": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]]}
    saved = plots.present_evaluation(cls, tmp_path, "run1_val")
    assert [p.name for p in saved] == [
        "run1_val_confusion.png", "run1_val_roc_pr.png", "run1_val_per_class.png"
    ]
    assert all(p.exists() for p in saved)
    reg = {"task_type": "tabular_regression", "classes": None,
           "y_true": [1.0, 2.0, 3.0], "y_pred": [1.2, 1.8, 3.1], "y_proba": None}
    saved = plots.present_evaluation(reg, tmp_path, "test")
    assert [p.name for p in saved] == ["test_pred_vs_actual.png", "test_residuals.png"]
    assert plots.plt.get_fignums() == []
