"""The generated data.py/model.py are plain scripts; load them from the template folder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"tpl_{name}", TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tpl_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_load_data_classification_splits_and_encoding(clean_project):
    data = load_module("data")
    out = data.load_data(clean_project.root, {"seed": 1})
    n = 240
    assert len(out["X_train"]) + len(out["X_val"]) + len(out["X_test"]) == n
    assert abs(len(out["X_test"]) - 0.15 * n) <= 2
    assert out["feature_names"][-1] == "colour"
    assert out["categorical_mask"] == [False] * 5 + [True]
    assert out["classes"] == ["0", "1"]
    assert set(np.unique(out["y_train"])) == {0, 1}
    # categorical encoded as small non-negative codes (or NaN), all columns float
    codes = out["X_train"]["colour"].dropna().unique()
    assert set(codes) <= {0.0, 1.0, 2.0}
    assert all(np.issubdtype(dtype, np.floating) for dtype in out["X_train"].dtypes)
    # deterministic for a seed, different for another
    again = data.load_data(clean_project.root, {"seed": 1})
    pd.testing.assert_frame_equal(out["X_test"], again["X_test"])
    other = data.load_data(clean_project.root, {"seed": 2})
    assert not out["X_test"].index.equals(other["X_test"].index)


def test_load_data_regression(regression_project):
    data = load_module("data")
    out = data.load_data(regression_project.root, {"seed": 1})
    assert out["classes"] is None
    assert out["y_train"].dtype == float
    assert out["task_type"] == "tabular_regression"


def test_unseen_category_becomes_nan(clean_project):
    data = load_module("data")
    df = pd.DataFrame({"colour": ["red", "blue"]})
    encoded, cats, mask = data.encode_features(df, ["colour"], {"colour": ["blue"]})
    assert encoded["colour"].tolist()[1] == 0.0
    assert np.isnan(encoded["colour"].tolist()[0])
    assert mask == [True]


def test_validate_rejects_overlap_and_single_class():
    data = load_module("data")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    good = {"X_train": X.iloc[:2], "X_val": X.iloc[2:], "X_test": X.iloc[2:3],
            "y_train": np.array([0, 1]), "classes": ["0", "1"]}
    with pytest.raises(ValueError, match="overlap"):
        data.validate(good)
    bad = {"X_train": X.iloc[:2], "X_val": X.iloc[2:3], "X_test": X.iloc[1:2],
           "y_train": np.array([0, 0]), "classes": ["0", "1"]}
    with pytest.raises(ValueError):
        data.validate(bad)


def test_build_model_and_grow(clean_project):
    data = load_module("data")
    model = load_module("model")
    out = data.load_data(clean_project.root, {"seed": 1})
    est = model.build_model({"learning_rate": 0.2, "iters_per_epoch": 3, "seed": 1},
                            "tabular_classification", out["categorical_mask"])
    est.fit(out["X_train"], out["y_train"])
    assert est.n_iter_ == 3
    model.grow(est, 2)
    est.fit(out["X_train"], out["y_train"])
    assert est.n_iter_ == 5
    reg = model.build_model({}, "tabular_regression", [False] * 6)
    assert type(reg).__name__ == "HistGradientBoostingRegressor"
    with pytest.raises(ValueError):
        model.build_model({}, "image_classification", [])
