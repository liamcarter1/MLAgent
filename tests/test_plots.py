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


import warnings  # noqa: E402


def _run(run_id, val, status="done", best_val=0.7, diff=None):
    epochs = [{"epoch": i + 1, "val_loss": v, "train_loss": v * 0.9} for i, v in enumerate(val)]
    entry = {"run_id": run_id, "status": status, "best_epoch": 1 + val.index(min(val)) if val
             else None, "best_val_metric": best_val if status == "done" else None,
             "applied_diff": diff}
    return entry, {"epochs": epochs}


def _history(*pairs):
    return [e for e, _m in pairs], {e["run_id"]: m for e, m in pairs}


def test_run_label_names_the_change_that_produced_the_run():
    assert plots.run_label({"run_id": 1, "applied_diff": None}) == "run 1"
    assert plots.run_label({"run_id": 2, "applied_diff": {
        "learning_rate": {"from": 0.1, "to": 0.05}}}) == "run 2 (learning_rate 0.05)"
    assert plots.run_label({"run_id": 3, "applied_diff": {
        "model_type": {"from": "linear", "to": "random_forest"},
        "alpha": {"from": 0.0001, "to": None},
        "trees_per_epoch": {"from": None, "to": 20}}}) == (
        "run 3 (model_type random_forest, +2 more)")


def test_comparison_figures_render_for_one_three_and_failed_runs(tmp_path):
    one = _history(_run(1, [1.0, 0.8, 0.7]))
    three = _history(_run(1, [1.0, 0.8, 0.7]),
                     _run(2, [1.0, 0.7, 0.5, 0.45], best_val=0.8,
                          diff={"epochs": {"from": 3, "to": 4}}),
                     _run(3, [1.0, 0.9], best_val=0.6, diff={"epochs": {"from": 4, "to": 2}}))
    failed = _history(_run(1, [1.0, 0.8, 0.7]),
                      _run(2, [1.0], status="failed", diff={"learning_rate": {"from": 0.1,
                                                                             "to": 0.9}}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i, (runs, by_run) in enumerate((one, three, failed)):
            curves = plots.compare_curves(runs, by_run)
            bars = plots.compare_runs(runs, "accuracy", target=0.9)
            a = plots.save_and_close(curves, tmp_path, f"curves{i}")
            b = plots.save_and_close(bars, tmp_path, f"bars{i}")
            assert a.exists() and b.exists()
            assert not plt.fignum_exists(curves.number)
    # a history with no archived metrics still draws (an empty-state message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.save_and_close(plots.compare_curves(one[0], {}), tmp_path, "empty")


def test_compare_curves_has_one_legend_entry_per_run_with_epochs():
    runs, by_run = _history(_run(1, [1.0, 0.8]), _run(2, [1.0, 0.7], diff={"epochs": {
        "from": 2, "to": 3}}))
    fig = plots.compare_curves(runs, by_run)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["run 1", "run 2 (epochs 3)"]
    plt.close(fig)


def test_compare_runs_draws_a_target_line_and_labels_every_run():
    runs, _by_run = _history(_run(1, [1.0]), _run(2, [1.0], status="failed"))
    fig = plots.compare_runs(runs, "rmse", target=5.0)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["run 1", "run 2"]
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    assert "rmse" in ax.get_xlabel()
    plt.close(fig)
