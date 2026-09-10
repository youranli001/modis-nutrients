"""Preprocessing: log10 for the ocean-colour products that span orders of
magnitude (CHL, APH, PIC, POC), median imputation, standardisation. Everything is
fitted on the training rows only (see ``models.FeatureModel.fit``). Targets are
standardised too, so the three nutrients weigh equally in a multi-output loss."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LOG_FLOOR = 1e-6


def _log10_floor(x):
    return np.log10(np.maximum(x, LOG_FLOOR))


def log10_transformer() -> FunctionTransformer:
    return FunctionTransformer(_log10_floor, feature_names_out="one-to-one", validate=False)


def build_preprocessor(features: list[str], log_features: list[str]) -> ColumnTransformer:
    """ColumnTransformer over ``features``; ``lat``/``lon`` and any other column are dropped."""
    log_cols = [f for f in features if f in log_features]
    lin_cols = [f for f in features if f not in log_features]

    log_branch = Pipeline(
        [
            ("log10", log10_transformer()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    lin_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers = []
    if log_cols:
        transformers.append(("log", log_branch, log_cols))
    if lin_cols:
        transformers.append(("linear", lin_branch, lin_cols))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="default")
    return pre


def build_target_scaler() -> StandardScaler:
    return StandardScaler()

