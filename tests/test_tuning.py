import numpy as np
import pandas as pd

from modis_nutrients.config import load_config
from modis_nutrients.tuning import hidden_layers, tune_optuna


def _table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"lat": rng.uniform(-70, 50, n), "lon": rng.uniform(-180, 180, n)})
    for c in ["CHL", "APH", "FLU", "PIC", "POC", "PAR"]:
        df[c] = rng.uniform(0.1, 1, n)
    df["SST"] = rng.uniform(-2, 30, n)
    df["nitrate"] = 30 * (1 - df["SST"] / 32) + rng.normal(0, 1, n)
    df["phosphate"] = df["nitrate"] / 16
    df["silicate"] = df["nitrate"] * 2
    return df


def test_hidden_layers():
    assert hidden_layers(3, 128) == [128, 64, 32]
    assert hidden_layers(1, 64) == [64]


def test_tune_optuna_gbm_small():
    cfg = load_config()
    cfg["data"]["months"] = None
    cfg["models"]["gbm"]["max_iter"] = 50
    table, best, final = tune_optuna(cfg, "block", "gbm", n_trials=3, df=_table(), verbose=False)
    assert len(table) == 3 and "cv_r2_mean" in table.columns
    assert 7 <= best["max_leaf_nodes"] <= 127
    assert final.metrics["test"].loc["mean", "r2"] > 0.8
