import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mlagent import plots
from mlagent.profile import profile_dataframe


def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
        "c": rng.integers(0, 5, size=60),
        "cat": rng.choice(["x", "y"], size=60),
        "target": rng.integers(0, 2, size=60),
    }).mask(rng.random((60, 5)) < 0.05)


def test_every_builder_returns_a_figure_with_axes():
    frame = df()
    before = profile_dataframe(frame, "target")
    after = profile_dataframe(frame.dropna(), "target")
    figs = [
        plots.feature_histograms(frame),
        plots.class_balance({"0": 30, "1": 25}),
        plots.target_distribution(frame["a"]),
        plots.missing_matrix(frame),
        plots.correlation_heatmap(frame),
        plots.outlier_boxplots(frame, ["a", "b"]),
        plots.before_after_missing(before, after),
    ]
    for fig in figs:
        assert isinstance(fig, Figure) and fig.axes
    assert plots.before_after_missing(before, after).axes[0].get_legend() is not None
    assert plots.class_balance({"0": 1}).axes[0].get_legend() is None


def test_correlation_with_one_numeric_column_does_not_crash():
    fig = plots.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]}))
    assert isinstance(fig, Figure)


def test_present_saves_png_and_closes(tmp_path):
    import matplotlib.pyplot as plt

    plt.close("all")
    fig = plots.class_balance({"0": 3, "1": 4})
    path = plots.present(fig, tmp_path / "plots", "balance")
    assert path == tmp_path / "plots" / "balance.png" and path.stat().st_size > 1000
    import matplotlib.pyplot as plt

    assert not plt.get_fignums()
