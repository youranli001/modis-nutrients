"""Hyper-parameter search that respects the spatial structure.

Optuna proposes candidates; :func:`blocked_cv_score` scores each one with
``GroupKFold`` inside the training partition, grouped by spatial block, so the
tuner cannot favour the candidate that interpolates best. The test partition
plays no part in the selection. The SageMaker tuning job (``aws/``) uses the
same scoring function.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .dataset import load_table, spec_from_config
from .evaluate import RunResult, regression_metrics, run_one, split_table
from .models import make_model
from .spatial import block_group_kfold, make_splitter


def hidden_layers(n_layers: int, width: int) -> list[int]:
    """``n_layers=3, width=128`` -> ``[128, 64, 32]``: each layer half the previous one."""
    return [width // 2 ** i for i in range(n_layers)]


def gbm_space(trial) -> dict[str, Any]:
    return {"max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0)}


def mlp_space(trial) -> dict[str, Any]:
    return {"hidden": hidden_layers(trial.suggest_int("n_layers", 1, 3),
                                    trial.suggest_categorical("width", [32, 64, 128, 256])),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512])}


SPACES = {"gbm": gbm_space, "mlp": mlp_space}


def blocked_cv_score(cfg: dict[str, Any], split_name: str, model_name: str, params: dict[str, Any],
                     df: pd.DataFrame | None = None, n_splits: int = 3) -> tuple[float, float]:
    """Mean and std of fold R² for one candidate, from GroupKFold by block inside
    the training partition of ``split_name``."""
    spec = spec_from_config(cfg)
    if df is None:
        df = load_table(cfg["data"]["path"], spec)
    split = split_table(make_splitter(split_name, cfg), df)
    X = df[["lat", "lon", *spec.features]]
    Y = df[spec.targets].to_numpy(float)
    tr = split.train
    groups, cv = block_group_kfold(df["lat"].to_numpy()[tr], df["lon"].to_numpy()[tr],
                                   cfg["split"]["block_deg"], n_splits)
    sub_cfg = deepcopy(cfg)
    sub_cfg["models"][model_name].update(params)
    scores = []
    for fold_tr, fold_va in cv.split(X.iloc[tr], groups=groups):
        itr, iva = tr[fold_tr], tr[fold_va]
        m = make_model(model_name, sub_cfg).fit(X.iloc[itr], Y[itr], X.iloc[iva], Y[iva])
        scores.append(regression_metrics(Y[iva], m.predict(X.iloc[iva]), spec.targets).loc["mean", "r2"])
    return float(np.mean(scores)), float(np.std(scores))


def tune_optuna(cfg: dict[str, Any], split_name: str, model_name: str, n_trials: int = 20,
                df: pd.DataFrame | None = None, seed: int = 0, verbose: bool = True
                ) -> tuple[pd.DataFrame, dict[str, Any], RunResult]:
    """TPE search over ``SPACES[model_name]``; the best candidate is refitted on the
    full training partition and scored once on the test partition.

    Returns ``(trials_table, best_params, final_run)``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = spec_from_config(cfg)
    if df is None:
        df = load_table(cfg["data"]["path"], spec)
    best_params = {}

    def objective(trial):
        params = SPACES[model_name](trial)
        mean, std = blocked_cv_score(cfg, split_name, model_name, params, df=df)
        trial.set_user_attr("cv_r2_std", std)
        trial.set_user_attr("params", params)
        if verbose:
            print(f"[{model_name}] trial {trial.number:>3d} {params}  cv R2 = {mean:.3f} ± {std:.3f}")
        return mean

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.user_attrs["params"]

    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    table = table.rename(columns={"value": "cv_r2_mean"}).sort_values("cv_r2_mean", ascending=False)

    sub_cfg = deepcopy(cfg)
    sub_cfg["models"][model_name].update(best_params)
    split = split_table(make_splitter(split_name, cfg), df)
    final = run_one(df, spec, split, make_model(model_name, sub_cfg), split_name=split_name)
    if verbose:
        print(f"best {best_params}  ->  val R2 {final.metrics['val'].loc['mean', 'r2']:.3f}, "
              f"test R2 {final.metrics['test'].loc['mean', 'r2']:.3f}")
    return table.reset_index(drop=True), best_params, final
