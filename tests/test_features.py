"""Tests for preprocessing and the model interface."""

import numpy as np
import pandas as pd
import pytest

from modis_nutrients.dataset import TableSpec, grid_to_table
from modis_nutrients.features import build_preprocessor
from modis_nutrients.models import GBMModel, LatLonKNNBaseline, ZonalMeanBaseline

FEATURES = ["CHL", "APH", "FLU", "PIC", "POC", "PAR", "SST"]
LOG = ["CHL", "APH", "PIC", "POC"]
TARGETS = ["nitrate", "phosphate", "silicate"]


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "lat": rng.uniform(-70, 50, n),
            "lon": rng.uniform(-180, 180, n),
            "CHL": 10 ** rng.uniform(-2, 1, n),
            "APH": 10 ** rng.uniform(-3, 0, n),
            "FLU": rng.normal(0.1, 0.1, n),
            "PIC": 10 ** rng.uniform(-5, -1, n),
            "POC": 10 ** rng.uniform(1, 3, n),
            "PAR": rng.uniform(0, 60, n),
            "SST": rng.uniform(-2, 30, n),
        }
    )
    df["nitrate"] = 30 * (1 - df["SST"] / 32) + rng.normal(0, 1, n)
    df["phosphate"] = df["nitrate"] / 16 + rng.normal(0, 0.05, n)
    df["silicate"] = df["nitrate"] * 2 + rng.normal(0, 2, n)
    # a few missing features
    df.loc[df.index[:5], "CHL"] = np.nan
    return df


def test_preprocessor_statistics_come_from_train_only(table):
    train, test = table.iloc[:300], table.iloc[300:]
    pre = build_preprocessor(FEATURES, LOG)
    Xt = pre.fit_transform(train)
    # training columns are standardised...
    assert np.allclose(np.nanmean(Xt, axis=0), 0, atol=1e-6)
    assert np.allclose(np.nanstd(Xt, axis=0), 1, atol=1e-6)
    # ...and the test set is transformed with the *same* statistics, so it is not.
    Xs = pre.transform(test)
    assert not np.allclose(Xs.mean(axis=0), 0, atol=0.05)
    # changing the test set must not change how train is transformed
    Xt2 = pre.transform(train)
    assert np.allclose(Xt, Xt2)


def test_preprocessor_imputes_and_logs(table):
    pre = build_preprocessor(FEATURES, LOG)
    Xt = pre.fit_transform(table)
    assert not np.isnan(Xt).any()
    names = list(pre.get_feature_names_out())
    assert set(names) == set(FEATURES)
    # log branch comes first, and its columns are the log features
    assert names[: len(LOG)] == [f for f in FEATURES if f in LOG]


def test_preprocessor_drops_location(table):
    pre = build_preprocessor(FEATURES, LOG)
    Xt = pre.fit_transform(table)
    assert Xt.shape[1] == len(FEATURES)
    assert "lat" not in pre.get_feature_names_out()


def test_grid_to_table_masks_land_and_never_imputes_targets():
    import xarray as xr

    lat = np.array([-0.5, 0.5])
    lon = np.array([-0.5, 0.5, 1.5])
    shape = (2, 3)
    data = {f: (("lat", "lon"), np.ones(shape)) for f in FEATURES}
    data.update({t: (("lat", "lon"), np.ones(shape)) for t in TARGETS})
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon})
    ds["nitrate"][0, 0] = np.nan  # a "land" cell for the target
    ds["CHL"][1, 2] = np.nan      # one missing feature elsewhere

    spec0 = TableSpec(FEATURES, TARGETS, max_missing_features=0)
    df0 = grid_to_table(ds, spec0)
    assert len(df0) == 4  # 6 cells - 1 missing target - 1 missing feature
    assert df0[TARGETS].notna().all().all()

    spec1 = TableSpec(FEATURES, TARGETS, max_missing_features=1)
    df1 = grid_to_table(ds, spec1)
    assert len(df1) == 5
    assert df1["CHL"].isna().sum() == 1  # left for the pipeline to impute


@pytest.mark.parametrize("cls", [ZonalMeanBaseline, LatLonKNNBaseline])
def test_location_baselines_ignore_features(table, cls):
    Y = table[TARGETS].to_numpy()
    m = cls().fit(table, Y)
    p1 = m.predict(table)
    shuffled = table.copy()
    shuffled[FEATURES] = shuffled[FEATURES].sample(frac=1, random_state=0).to_numpy()
    p2 = m.predict(shuffled)
    assert np.allclose(p1, p2)


def test_feature_models_return_physical_units(table):
    Y = table[TARGETS].to_numpy()
    train, test = table.iloc[:400], table.iloc[400:]
    m = GBMModel(FEATURES, LOG).fit(train, Y[:400], test, Y[400:])
    pred = m.predict(test)
    assert pred.shape == (100, 3)
    # predictions should be on the nitrate scale (tens), not standardised
    assert pred[:, 0].std() > 1
    assert np.corrcoef(pred[:, 0], Y[400:, 0])[0, 1] > 0.8
