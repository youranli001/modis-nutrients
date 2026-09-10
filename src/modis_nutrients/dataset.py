"""Gridded netCDF (MODIS at 1° + WOA13 surface climatology) to a table with one
row per ocean cell (and month, if present). Land is excluded by requiring the WOA
targets; rows with a missing target are dropped, never imputed. ``cell_id``
identifies the location, so all months of a cell share one spatial block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class TableSpec:
    features: list[str]
    targets: list[str]
    max_missing_features: int = 0
    months: tuple[int, ...] | None = None   # None = all months in the file


def spec_from_config(cfg) -> TableSpec:
    d = cfg["data"]
    months = tuple(d["months"]) if d.get("months") else None
    return TableSpec(list(d["features"]), list(d["targets"]), d.get("max_missing_features", 0), months)


def load_grid(path: str) -> xr.Dataset:
    """Open the combined 1° dataset (``decode_times=False``: WOA's climatological time axis)."""
    return xr.open_dataset(path, decode_times=False)


def grid_to_table(ds: xr.Dataset, spec: TableSpec) -> pd.DataFrame:
    """One row per usable cell: ``cell_id, [month,] lat, lon, features, targets``."""
    sub = ds[list(spec.features) + list(spec.targets)]
    has_month = "month" in sub.dims
    if has_month:
        if spec.months:
            sub = sub.sel(month=list(spec.months))
        sub = sub.transpose("month", "lat", "lon")
    else:
        sub = sub.transpose("lat", "lon")
    df = sub.to_dataframe().reset_index()

    # cell_id: position of the cell on the (lat, lon) grid, the same for every month
    n_lon = sub.sizes["lon"]
    i_lat = np.searchsorted(sub["lat"].values, df["lat"].values)
    i_lon = np.searchsorted(sub["lon"].values, df["lon"].values)
    df["cell_id"] = i_lat * n_lon + i_lon

    # keep cells with all targets and at most `max_missing_features` missing inputs
    has_targets = df[spec.targets].notna().all(axis=1)
    n_missing = df[spec.features].isna().sum(axis=1)
    keep = has_targets & (n_missing <= spec.max_missing_features)

    columns = ["cell_id", "lat", "lon", *spec.features, *spec.targets]
    if has_month:
        columns.insert(1, "month")
    out = df.loc[keep, columns].reset_index(drop=True)
    out.attrs["n_grid_cells"] = int(sub.sizes["lat"] * n_lon)
    out.attrs["n_rows_total"] = len(df)
    out.attrs["n_target_rows"] = int(has_targets.sum())
    out.attrs["months"] = sorted(df["month"].unique().tolist()) if has_month else None
    return out


def load_table(path: str, spec: TableSpec) -> pd.DataFrame:
    return grid_to_table(load_grid(path), spec)
