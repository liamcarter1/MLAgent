"""Synthetic tabular datasets with optional realistic quirks for the cleaning stage to find."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

TASKS = ("classification", "regression")
QUIRKS = (
    "missing",
    "duplicates",
    "id_column",
    "constant_column",
    "outliers",
    "categorical",
    "whitespace",
)
TARGET = "target"
CATEGORIES = ("red", "green", "blue")
MESSY_VARIANTS = (" red", "Red", "green ", "GREEN", "Blue", " blue")


@dataclass
class SynthTabularConfig:
    task: str = "classification"
    n_samples: int = 1000
    n_features: int = 8
    n_classes: int = 2
    class_balance: float = 0.5
    noise: float = 0.1
    seed: int = 42
    quirks: tuple[str, ...] = ()

    def n_informative(self) -> int:
        needed = int(math.ceil(math.log2(2 * self.n_classes)))
        return min(self.n_features, max(2, self.n_features // 2, needed))

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.n_samples < 20:
            raise ValueError("n_samples must be at least 20")
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must be between 0 and 1")
        if not 0.05 <= self.class_balance <= 0.95:
            raise ValueError("class_balance must be between 0.05 and 0.95")
        unknown = set(self.quirks) - set(QUIRKS)
        if unknown:
            raise ValueError(f"unknown quirks: {sorted(unknown)}")
        if self.task == "classification" and 2 ** self.n_informative() < 2 * self.n_classes:
            msg = "too many classes for this many features; add features or reduce classes"
            raise ValueError(msg)


def _base(cfg: SynthTabularConfig) -> pd.DataFrame:
    if cfg.task == "classification":
        n_inf = cfg.n_informative()
        weights = [cfg.class_balance, 1 - cfg.class_balance] if cfg.n_classes == 2 else None
        x, y = make_classification(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            n_informative=n_inf,
            n_redundant=min(2, cfg.n_features - n_inf),
            n_classes=cfg.n_classes,
            weights=weights,
            flip_y=cfg.noise,
            random_state=cfg.seed,
        )
    else:
        x, y = make_regression(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            noise=cfg.noise * 10,
            random_state=cfg.seed,
        )
    df = pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(cfg.n_features)])
    df[TARGET] = y
    return df


def _apply_quirks(df: pd.DataFrame, cfg: SynthTabularConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    quirks = set(cfg.quirks)
    if "whitespace" in quirks:
        quirks.add("categorical")
    features = [c for c in df.columns if c != TARGET]
    n = len(df)
    if "categorical" in quirks:
        df["category"] = rng.choice(CATEGORIES, size=n)
    if "whitespace" in quirks:
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[df.index[idx], "category"] = rng.choice(MESSY_VARIANTS, size=len(idx))
    if "outliers" in quirks:
        idx = rng.choice(n, size=max(1, n // 100), replace=False)
        col = features[0]
        df.loc[df.index[idx], col] = df.loc[df.index[idx], col].abs() * 50 + 100
    if "missing" in quirks:
        for col in features[:3]:
            idx = rng.choice(n, size=max(1, n // 20), replace=False)
            df.loc[df.index[idx], col] = np.nan
    if "constant_column" in quirks:
        df["constant"] = 1
    if "id_column" in quirks:
        df.insert(0, "row_id", np.arange(1000, 1000 + n))
    if "duplicates" in quirks:
        n_dup = max(5, n // 50)
        idx = rng.choice(n, size=n_dup, replace=False)
        df = pd.concat([df, df.iloc[idx]], ignore_index=True)
    return df


def generate(cfg: SynthTabularConfig) -> pd.DataFrame:
    cfg.validate()
    return _apply_quirks(_base(cfg), cfg)
