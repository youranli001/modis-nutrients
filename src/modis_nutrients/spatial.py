"""Spatial structure: variogram, block + buffer splitting, grouped CV.

Both the inputs (MODIS climatology) and the targets (WOA13 analysed fields)
are smooth in space, so a random split of 1° cells lets a model score well by
interpolating between neighbours. The splitters here assign whole tiles to
train / validation / test and optionally remove training cells within a buffer
of the held-out cells. Distances are great-circle (sklearn BallTree, haversine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def nearest_distance_km(lat_q, lon_q, lat_ref, lon_ref) -> np.ndarray:
    """Great-circle distance from each query point to its nearest reference point."""
    if len(lat_ref) == 0:
        return np.full(len(lat_q), np.inf)
    ref = np.deg2rad(np.column_stack([lat_ref, lon_ref]))
    query = np.deg2rad(np.column_stack([lat_q, lon_q]))
    tree = BallTree(ref, metric="haversine")
    dist, _ = tree.query(query, k=1)
    return dist[:, 0] * EARTH_RADIUS_KM


def block_ids(lat, lon, block_deg: float) -> np.ndarray:
    """Integer id of the ``block_deg`` x ``block_deg`` tile containing each point."""
    n_cols = int(np.ceil(360 / block_deg))
    row = np.floor((np.asarray(lat, float) + 90) / block_deg).astype(int)
    col = np.floor((np.asarray(lon, float) + 180) / block_deg).astype(int) % n_cols
    return row * n_cols + col


# --------------------------------------------------------------------------- #
# variogram (GSTools)
# --------------------------------------------------------------------------- #
def zonal_detrend(lat, values, band_deg: float = 5) -> np.ndarray:
    """Remove the mean of each latitude band. Nutrients depend strongly on latitude;
    removing that trend exposes the local correlation scale that matters for the block size."""
    band = np.floor((np.asarray(lat, float) + 90) / band_deg).astype(int)
    values = pd.Series(np.asarray(values, float))
    return (values - values.groupby(band).transform("mean")).to_numpy()


def variogram(lat, lon, values, *, max_km=4000, bin_km=100, detrend_band_deg=5,
              sampling_size=3000, seed=0):
    """Empirical semivariogram on the sphere plus a fitted spherical model.

    Returns ``(bin_centers_km, gamma, model)``. ``model.len_scale`` is the
    correlation range in km and ``model.correlation(d)`` the correlation at ``d`` km.
    """
    import gstools as gs

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if detrend_band_deg:
        z = zonal_detrend(lat, values, detrend_band_deg)
    else:
        z = np.asarray(values, float)

    bins = np.arange(0, max_km + bin_km, bin_km)
    centers, gamma = gs.vario_estimate((lat, lon), z, bin_edges=bins, latlon=True,
                                       geo_scale=gs.KM_SCALE, sampling_size=sampling_size,
                                       sampling_seed=seed)
    model = gs.Spherical(latlon=True, geo_scale=gs.KM_SCALE)
    model.fit_variogram(centers, gamma, nugget=True)
    return centers, gamma, model


# --------------------------------------------------------------------------- #
# split result
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    """Row indices of each partition. ``dropped`` are training/validation rows
    removed by the buffer; ``block_id`` is the tile of every row (spatial splits only)."""

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    dropped: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    block_id: np.ndarray | None = None

    def labels(self) -> np.ndarray:
        """Label of every row: train / val / test / buffer."""
        n = len(self.train) + len(self.val) + len(self.test) + len(self.dropped)
        labels = np.full(n, "buffer", dtype=object)
        labels[self.train] = "train"
        labels[self.val] = "val"
        labels[self.test] = "test"
        return labels


# --------------------------------------------------------------------------- #
# splitters
# --------------------------------------------------------------------------- #
class RandomSplit:
    """Random 60 / 20 / 20 split by row (5 shuffled folds: one test, one validation,
    three training). The baseline to beat, not a recommendation."""

    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(self, lat, lon, month=None) -> SplitResult:
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        folds = [held_out for _, held_out in kfold.split(np.zeros(len(lat)))]
        test = np.sort(folds[0])
        val = np.sort(folds[1])
        train = np.sort(np.concatenate(folds[2:]))
        return SplitResult("random", train, val, test)


class SpatialBlockSplit:
    """Whole ``block_deg`` tiles go to train / val / test (60 / 20 / 20); training
    cells within ``buffer_km`` of any held-out cell are removed.

    Tiles are assigned with sklearn's ``StratifiedGroupKFold`` (group = tile,
    stratum = latitude row, 5 folds: one fold is test, one is validation, three
    are training), which keeps a tile in one partition and spreads every latitude
    band over all three. Otherwise one partition could hold an entire ocean regime
    and the test score would measure extrapolation instead of generalisation.
    The test set is never changed by the buffer.
    """

    def __init__(self, block_deg: float = 20, buffer_km: float = 0, seed: int = 42):
        if block_deg <= 0 or buffer_km < 0:
            raise ValueError("block_deg must be > 0 and buffer_km >= 0")
        self.block_deg = float(block_deg)
        self.buffer_km = float(buffer_km)
        self.seed = seed

    @property
    def name(self) -> str:
        if self.buffer_km:
            return f"block{self.block_deg:g}deg_buffer{self.buffer_km:g}km"
        return f"block{self.block_deg:g}deg"

    def split(self, lat, lon, month=None) -> SplitResult:
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        blocks = block_ids(lat, lon, self.block_deg)
        lat_row = np.floor((lat + 90) / self.block_deg).astype(int)

        # 5 stratified group folds; fold 0 -> test, fold 1 -> val, the rest -> train
        kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
        folds = [held_out for _, held_out in kfold.split(lat, lat_row, groups=blocks)]
        test = np.sort(folds[0])
        val = np.sort(folds[1])
        train = np.sort(np.concatenate(folds[2:]))
        dropped = np.array([], dtype=int)

        if self.buffer_km > 0:
            # training cells too close to any validation or test cell
            holdout = np.concatenate([val, test])
            d = nearest_distance_km(lat[train], lon[train], lat[holdout], lon[holdout])
            dropped_train = train[d < self.buffer_km]
            train = train[d >= self.buffer_km]
            # validation cells too close to any test cell
            d = nearest_distance_km(lat[val], lon[val], lat[test], lon[test])
            dropped_val = val[d < self.buffer_km]
            val = val[d >= self.buffer_km]
            dropped = np.sort(np.concatenate([dropped_train, dropped_val]))

        return SplitResult(self.name, train, val, test, dropped, blocks)


class MonthHoldoutSplit:
    """Whole months go to train / val / test (temporal hold-out on the monthly file).

    This split alone does not block space: a cell's other months stay in the
    training set, so a location-only model can still interpolate in time.
    """

    name = "month"

    def __init__(self, test_months=(1, 5, 9), val_months=(3, 7, 11)):
        if set(test_months) & set(val_months):
            raise ValueError("test_months and val_months overlap")
        self.test_months = tuple(test_months)
        self.val_months = tuple(val_months)

    def split(self, lat, lon, month=None) -> SplitResult:
        if month is None:
            raise ValueError("MonthHoldoutSplit needs the month of every row")
        month = np.asarray(month)
        is_test = np.isin(month, self.test_months)
        is_val = np.isin(month, self.val_months)
        train = np.nonzero(~is_test & ~is_val)[0]
        val = np.nonzero(is_val)[0]
        test = np.nonzero(is_test)[0]
        return SplitResult("month", train, val, test)


def block_group_kfold(lat, lon, block_deg: float, n_splits: int = 5):
    """Groups (block ids) and a ``GroupKFold`` for tuning inside the training set."""
    return block_ids(lat, lon, block_deg), GroupKFold(n_splits=n_splits)


def make_splitter(name: str, cfg: dict[str, Any]):
    """``random``, ``block``, ``block_buffer`` or ``month`` with the settings in ``cfg``."""
    s = cfg["split"]
    splitters = {
        "random": lambda: RandomSplit(s["seed"]),
        "block": lambda: SpatialBlockSplit(s["block_deg"], 0, s["seed"]),
        "block_buffer": lambda: SpatialBlockSplit(s["block_deg"], s["buffer_km"], s["seed"]),
        "month": lambda: MonthHoldoutSplit(**s.get("month_holdout", {})),
    }
    if name not in splitters:
        raise ValueError(f"unknown split '{name}'; choose from {list(splitters)}")
    return splitters[name]()
