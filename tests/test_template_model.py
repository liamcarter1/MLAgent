from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE = Path("mlagent/templates/tabular_sklearn").resolve()


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_module():
    return load_module("model")


def classification_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X[0, 1] = np.nan  # every family must tolerate a missing value
    return X, y


def regression_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.2, size=n)
    return X, y


CONFIGS = {
    "gradient_boosting": {"model_type": "gradient_boosting", "seed": 1, "iters_per_epoch": 3,
                          "learning_rate": 0.2, "max_leaf_nodes": 8, "max_depth": None,
                          "min_samples_leaf": 5, "l2_regularization": 0.0},
    "random_forest": {"model_type": "random_forest", "seed": 1, "trees_per_epoch": 4,
                      "max_depth": 4, "min_samples_leaf": 2, "max_features": 0.8},
    "linear": {"model_type": "linear", "seed": 1, "learning_rate": 0.01, "alpha": 0.0001},
}


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "linear"])
def test_every_family_trains_epoch_by_epoch_on_classification(model_module, model_type):
    X, y = classification_data()
    model = model_module.build_model(CONFIGS[model_type], "tabular_classification",
                                     [False] * 4)
    for epoch in range(1, 4):
        model.fit_epoch(X, y)
        assert model.epochs_fitted == epoch
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert list(model.classes_) == [0, 1]


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "linear"])
def test_every_family_trains_on_regression(model_module, model_type):
    X, y = regression_data()
    model = model_module.build_model(CONFIGS[model_type], "tabular_regression", [False] * 4)
    model.fit_epoch(X, y)
    model.fit_epoch(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    assert model.classes_ is None


def test_capacity_grows_with_each_epoch(model_module):
    X, y = classification_data()
    gb = model_module.build_model(CONFIGS["gradient_boosting"], "tabular_classification",
                                  [False] * 4)
    gb.fit_epoch(X, y)
    first = gb.estimator.max_iter
    first_fitted = gb.estimator.n_iter_
    gb.fit_epoch(X, y)
    assert gb.estimator.max_iter == first + CONFIGS["gradient_boosting"]["iters_per_epoch"]
    # not just the parameter: warm_start must actually have added boosting rounds
    assert gb.estimator.n_iter_ == first_fitted + CONFIGS["gradient_boosting"]["iters_per_epoch"]

    rf = model_module.build_model(CONFIGS["random_forest"], "tabular_classification",
                                  [False] * 4)
    rf.fit_epoch(X, y)
    trees = rf.estimator.n_estimators
    first_fitted_trees = len(rf.estimator.estimators_)
    rf.fit_epoch(X, y)
    assert rf.estimator.n_estimators == trees + CONFIGS["random_forest"]["trees_per_epoch"]
    # not just the parameter: warm_start must actually have added fitted trees
    assert len(rf.estimator.estimators_) == (
        first_fitted_trees + CONFIGS["random_forest"]["trees_per_epoch"]
    )

    lin = model_module.build_model(CONFIGS["linear"], "tabular_classification", [False] * 4)
    lin.fit_epoch(X, y)
    lin.fit_epoch(X, y)
    assert lin.epochs_fitted == 2  # one partial_fit pass per epoch, no capacity change


def test_predict_proba_on_regression_raises(model_module):
    X, y = regression_data()
    model = model_module.build_model(CONFIGS["gradient_boosting"], "tabular_regression",
                                     [False] * 4)
    model.fit_epoch(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_categorical_mask_is_passed_to_gradient_boosting(model_module):
    model = model_module.build_model(CONFIGS["gradient_boosting"], "tabular_classification",
                                     [True, False, False, False])
    assert list(model.estimator.categorical_features) == [True, False, False, False]
    none_mask = model_module.build_model(CONFIGS["gradient_boosting"],
                                         "tabular_classification", [False] * 4)
    assert none_mask.estimator.categorical_features is None


def test_unknown_model_type_and_task_type_raise(model_module):
    with pytest.raises(ValueError):
        model_module.build_model({"model_type": "quantum"}, "tabular_classification", [])
    with pytest.raises(ValueError):
        model_module.build_model(CONFIGS["linear"], "image_classification", [])


def test_model_types_constant(model_module):
    assert model_module.MODEL_TYPES == ("gradient_boosting", "random_forest", "linear")
