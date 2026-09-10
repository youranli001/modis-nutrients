"""Models behind one ``fit(X, Y, X_val, Y_val)`` / ``predict(X)`` interface.

``X`` is a DataFrame with ``lat``, ``lon`` and the feature columns; ``Y`` is an
``(n, 3)`` array in physical units. The two location-only baselines never see
ocean colour; they are there to detect leakage. GBM and MLP see the seven MODIS
products and nothing else.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from .features import build_preprocessor, build_target_scaler


class ZonalMeanBaseline:
    """Training-set mean of each target within a latitude band."""

    name = "zonal_mean"

    def __init__(self, band_deg: float = 5):
        self.band_deg = band_deg

    def _band(self, X):
        return (X["lat"] // self.band_deg).astype(int).to_numpy()

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.means_ = pd.DataFrame(np.asarray(Y, float)).groupby(self._band(X)).mean()
        return self

    def predict(self, X):
        return self.means_.reindex(self._band(X)).fillna(self.means_.mean()).to_numpy()


class LatLonKNNBaseline:
    """k nearest training cells by great-circle distance: the 'memorise the map' model."""

    name = "latlon_knn"

    def __init__(self, n_neighbors: int = 10):
        self.knn_ = KNeighborsRegressor(n_neighbors=n_neighbors, metric="haversine",
                                        algorithm="ball_tree", weights="distance")

    @staticmethod
    def _coords(X):
        return np.deg2rad(X[["lat", "lon"]].to_numpy(float))

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.knn_.fit(self._coords(X), np.asarray(Y, float))
        return self

    def predict(self, X):
        return self.knn_.predict(self._coords(X))


class FeatureModel:
    """Preprocessing and target scaling shared by GBM and MLP. Both are fitted on
    the training rows only, inside :meth:`fit`; predictions come back in physical units."""

    name = "feature_model"

    def __init__(self, features: list[str], log_features: list[str]):
        self.pre_ = build_preprocessor(features, log_features)
        self.y_scaler_ = build_target_scaler()

    def fit(self, X, Y, X_val=None, Y_val=None):
        Xt = self.pre_.fit_transform(X)
        Yt = self.y_scaler_.fit_transform(np.asarray(Y, float))
        val = None
        if X_val is not None:
            val = (self.pre_.transform(X_val), self.y_scaler_.transform(np.asarray(Y_val, float)))
        self._fit_core(Xt, Yt, val)
        return self

    def predict(self, X):
        Yt = self._predict_core(self.pre_.transform(X))
        Yt = np.asarray(Yt).reshape(len(X), -1)
        return self.y_scaler_.inverse_transform(Yt)

    def transform(self, X) -> np.ndarray:
        return self.pre_.transform(X)


class GBMModel(FeatureModel):
    """One ``HistGradientBoostingRegressor`` per target."""

    name = "gbm"

    def __init__(self, features, log_features, *, max_iter=500, learning_rate=0.05,
                 max_leaf_nodes=31, l2_regularization=1.0, seed=42):
        super().__init__(features, log_features)
        self.est_ = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization, early_stopping=False, random_state=seed))

    def _fit_core(self, Xt, Yt, val):
        self.est_.fit(Xt, Yt)

    def _predict_core(self, Xt):
        return self.est_.predict(Xt)


class KerasMLPModel(FeatureModel):
    """Fully connected network (default 7 -> 64 -> 32 -> 3, TensorFlow/Keras) with
    early stopping on the validation partition."""

    name = "mlp"

    def __init__(self, features, log_features, *, hidden=(64, 32), epochs=300, batch_size=256,
                 learning_rate=1e-3, patience=25, seed=42, verbose=0):
        super().__init__(features, log_features)
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.history_ = None

    def _fit_core(self, Xt, Yt, val):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras

        keras.utils.set_random_seed(self.seed)
        layers = [keras.Input(shape=(Xt.shape[1],))]
        for n_units in self.hidden:
            layers.append(keras.layers.Dense(n_units, activation="relu"))
        layers.append(keras.layers.Dense(Yt.shape[1]))
        self.net_ = keras.Sequential(layers)
        self.net_.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        callbacks = []
        if val is not None:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                           restore_best_weights=True))
        hist = self.net_.fit(Xt, Yt, validation_data=val, epochs=self.epochs, batch_size=self.batch_size,
                             callbacks=callbacks, verbose=self.verbose)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    def _predict_core(self, Xt):
        return self.net_.predict(Xt, verbose=0)


def make_model(name: str, cfg: dict[str, Any]):
    d = cfg["data"]
    params = dict(cfg["models"].get(name, {}))
    if name == "zonal_mean":
        return ZonalMeanBaseline(**params)
    if name == "latlon_knn":
        return LatLonKNNBaseline(**params)
    params.setdefault("seed", cfg["split"]["seed"])
    if name == "gbm":
        return GBMModel(d["features"], d["log_features"], **params)
    if name == "mlp":
        return KerasMLPModel(d["features"], d["log_features"], **params)
    raise ValueError(f"unknown model '{name}'")
