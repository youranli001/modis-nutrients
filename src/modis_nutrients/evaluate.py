"""Metrics and the split x model experiment grid."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .dataset import TableSpec, load_table, spec_from_config
from .models import make_model
from .spatial import SplitResult, make_splitter


def regression_metrics(Y_true, Y_pred, targets: list[str]) -> pd.DataFrame:
    """RMSE / MAE / R² per target plus a ``mean`` row."""
    out = pd.DataFrame({
        "rmse": root_mean_squared_error(Y_true, Y_pred, multioutput="raw_values"),
        "mae": mean_absolute_error(Y_true, Y_pred, multioutput="raw_values"),
        "r2": r2_score(Y_true, Y_pred, multioutput="raw_values"),
    }, index=pd.Index(targets, name="target"))
    out.loc["mean"] = [np.sqrt((out["rmse"] ** 2).mean()), out["mae"].mean(), out["r2"].mean()]
    return out


@dataclass
class RunResult:
    split_name: str
    model_name: str
    split: SplitResult
    metrics: dict[str, pd.DataFrame]      # {"val": ..., "test": ...}
    predictions: dict[str, np.ndarray]    # {"val": ..., "test": ...}, physical units


def split_table(splitter, df: pd.DataFrame) -> SplitResult:
    """Apply a splitter to a table, passing the month column when present."""
    month = df["month"].to_numpy() if "month" in df.columns else None
    return splitter.split(df["lat"].to_numpy(), df["lon"].to_numpy(), month=month)


def run_one(df: pd.DataFrame, spec: TableSpec, split: SplitResult, model,
            split_name: str | None = None) -> RunResult:
    """Fit on train (early-stopping on val), score on val and test."""
    X = df[["lat", "lon", *spec.features]]
    Y = df[spec.targets].to_numpy(float)
    model.fit(X.iloc[split.train], Y[split.train], X.iloc[split.val], Y[split.val])
    predictions = {"val": model.predict(X.iloc[split.val]), "test": model.predict(X.iloc[split.test])}
    metrics = {part: regression_metrics(Y[getattr(split, part)], predictions[part], spec.targets)
               for part in ("val", "test")}
    return RunResult(split_name or split.name, model.name, split, metrics, predictions)


def run_grid(cfg: dict[str, Any], split_names: list[str], model_names: list[str],
             df: pd.DataFrame | None = None, verbose: bool = True) -> tuple[pd.DataFrame, list[RunResult]]:
    """Every split x every model on the same table. Returns a long table with one
    row per (split, model, partition, target) and the individual results."""
    spec = spec_from_config(cfg)
    if df is None:
        df = load_table(cfg["data"]["path"], spec)

    tables = []
    results = []
    for split_name in split_names:
        split = split_table(make_splitter(split_name, cfg), df)
        for model_name in model_names:
            res = run_one(df, spec, split, make_model(model_name, cfg), split_name=split_name)
            results.append(res)
            for part, metrics in res.metrics.items():
                tables.append(metrics.assign(split=split_name, model=model_name, partition=part,
                                             n_train=len(split.train), n_val=len(split.val),
                                             n_test=len(split.test), n_dropped=len(split.dropped)))
            if verbose:
                r2 = res.metrics["test"].loc["mean", "r2"]
                print(f"[{split_name:>13s}] {model_name:<12s} test R2={r2:6.3f}")
    return pd.concat(tables).reset_index(), results


def headline_table(long: pd.DataFrame, partition: str = "test", metric: str = "r2",
                   target: str = "mean") -> pd.DataFrame:
    """Wide table: rows = split, columns = model, plus partition sizes."""
    sub = long[(long["partition"] == partition) & (long["target"] == target)]
    wide = sub.pivot(index="split", columns="model", values=metric)
    sizes = sub.groupby("split")[["n_train", "n_val", "n_test", "n_dropped"]].first()
    return pd.concat([wide, sizes], axis=1)


def per_target_table(long: pd.DataFrame, partition: str = "test", metric: str = "r2") -> pd.DataFrame:
    sub = long[(long["partition"] == partition) & (long["target"] != "mean")]
    return sub.pivot_table(index=["split", "model"], columns="target", values=metric)


DEFAULT_ABLATION = {
    "sst_only": ["SST"],
    "sst_par": ["SST", "PAR"],
    "ocean_colour": ["CHL", "APH", "FLU", "PIC", "POC"],
    "all": ["CHL", "APH", "FLU", "PIC", "POC", "PAR", "SST"],
}


def feature_ablation(cfg: dict[str, Any], split_names: list[str], feature_sets: dict[str, list[str]],
                     model_name: str = "gbm", df: pd.DataFrame | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Same model, same split, different input sets. Returns a wide table:
    rows = feature set, columns = split, values = mean test R²."""
    full = spec_from_config(cfg)
    if df is None:
        df = load_table(cfg["data"]["path"], full)
    scores = {}
    for split_name in split_names:
        split = split_table(make_splitter(split_name, cfg), df)
        for set_name, features in feature_sets.items():
            sub_cfg = deepcopy(cfg)
            sub_cfg["data"]["features"] = list(features)
            spec = TableSpec(list(features), full.targets, full.max_missing_features, full.months)
            res = run_one(df, spec, split, make_model(model_name, sub_cfg), split_name=split_name)
            scores[(set_name, split_name)] = res.metrics["test"].loc["mean", "r2"]
            if verbose:
                print(f"[{split_name:>13s}] {set_name:<14s} test R2={scores[(set_name, split_name)]:.3f}")
    table = pd.Series(scores).unstack()
    return table.loc[list(feature_sets), split_names]
