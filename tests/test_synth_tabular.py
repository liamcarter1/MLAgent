import pandas as pd
import pytest

from mlagent.synth.tabular import QUIRKS, TARGET, SynthTabularConfig, generate


def test_classification_shape_target_and_determinism():
    cfg = SynthTabularConfig(n_samples=200, n_features=5, seed=1)
    df = generate(cfg)
    assert df.shape == (200, 6)
    assert list(df.columns) == ["f1", "f2", "f3", "f4", "f5", TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    pd.testing.assert_frame_equal(df, generate(cfg))


def test_regression_target_is_continuous():
    df = generate(SynthTabularConfig(task="regression", n_samples=100, n_features=3, seed=2))
    assert df[TARGET].dtype.kind == "f"
    assert df[TARGET].nunique() > 50


def test_class_balance_is_respected():
    df = generate(SynthTabularConfig(n_samples=1000, class_balance=0.9, noise=0.0, seed=3))
    assert df[TARGET].value_counts(normalize=True).iloc[0] > 0.8


def test_all_quirks_are_injected():
    df = generate(SynthTabularConfig(n_samples=200, n_features=4, seed=4, quirks=QUIRKS))
    assert df.columns[0] == "row_id"
    assert df["constant"].nunique() == 1
    assert df["category"].nunique() > 3
    assert df["category"].str.strip().str.lower().nunique() == 3
    assert df["f1"].isna().sum() > 0
    assert df.duplicated().sum() >= 5
    assert df["f1"].max() > 50


def test_whitespace_quirk_implies_categorical():
    df = generate(SynthTabularConfig(n_samples=100, seed=5, quirks=("whitespace",)))
    assert "category" in df.columns


def test_validate_rejects_bad_configs():
    with pytest.raises(ValueError):
        SynthTabularConfig(task="clustering").validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(quirks=("glitter",)).validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(n_features=2, n_classes=10).validate()
    with pytest.raises(ValueError):
        SynthTabularConfig(noise=2.0).validate()
