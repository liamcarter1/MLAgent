from __future__ import annotations

import matplotlib.pyplot as plt

from mlagent import plots


def test_palette_constants_are_present():
    assert plots.SERIES[0] == "#2a78d6"
    assert len(plots.SERIES) == 8 and len(plots.SEQUENTIAL) == 7
    assert len(plots.DIVERGING) == 5
    assert plots.SEQ_CMAP.name == "mlagent_seq" and plots.DIV_CMAP.name == "mlagent_div"


def test_style_axes_hides_the_top_and_right_spines_and_sets_the_title():
    fig, ax = plt.subplots()
    plots.style_axes(ax, "A title")
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.get_title(loc="left") == "A title"
    plt.close(fig)


def test_present_saves_and_closes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    path = plots.present(fig, tmp_path, "demo")
    assert path == tmp_path / "demo.png" and path.exists()
    assert not plt.fignum_exists(fig.number)


def test_the_figure_functions_have_moved_to_the_templates():
    for gone in ("feature_histograms", "missing_matrix", "correlation_heatmap",
                 "before_after_missing", "training_curves", "confusion_matrix_plot",
                 "roc_pr_curves", "per_class_bars", "predicted_vs_actual",
                 "residual_plots", "present_evaluation"):
        assert not hasattr(plots, gone), gone
