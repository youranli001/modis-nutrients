"""Build the 1° monthly dataset from the raw files.

The 9 km MODIS grid divides the 1° WOA grid exactly (12 x 12 pixels per cell), so
regridding is an area mean over each block, geometric for the log-normal products
(CHL, APH, PIC, POC). WOA13 files contribute the surface level: the analysed field
plus the per-cell observed mean, count and standard error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .download import MODIS_PRODUCTS, WOA_VARS, month_from_filename

WOA_FIELDS = ("an", "mn", "dd", "se")


# --------------------------------------------------------------------------- #
# MODIS
# --------------------------------------------------------------------------- #
def open_modis(path: str | Path, product: str) -> xr.DataArray:
    """Open one L3m file and return the data variable (ascending latitude)."""
    _, var = MODIS_PRODUCTS[product]
    ds = xr.open_dataset(path)
    if var not in ds:
        raise KeyError(f"{Path(path).name}: expected variable '{var}', found {list(ds.data_vars)}")
    return ds[var].sortby("lat").sortby("lon")


def coarsen_to_1deg(da: xr.DataArray, geometric: bool = False, min_frac: float = 0.25) -> xr.DataArray:
    """Area-mean a 9 km (1/12 degree) field onto 1-degree cells.

    Cells with fewer than ``min_frac`` of their 144 sub-pixels valid become NaN,
    so a single clear pixel in an otherwise cloudy cell does not masquerade as
    a cell mean.
    """
    ny, nx = da.sizes["lat"], da.sizes["lon"]
    if ny % 12 or nx % 12:
        raise ValueError(f"grid {ny}x{nx} is not a multiple of 12; is this the 9 km product?")
    n_valid = da.notnull().coarsen(lat=12, lon=12, boundary="exact").sum()
    if geometric:
        # mean in log space, then back: geometric mean
        out = 10 ** np.log10(da.where(da > 0)).coarsen(lat=12, lon=12, boundary="exact").mean()
    else:
        out = da.coarsen(lat=12, lon=12, boundary="exact").mean()
    out = out.where(n_valid >= min_frac * 144)
    # snap coordinates to the WOA cell centres (-89.5 .. 89.5, -179.5 .. 179.5)
    lat_c = np.round(out["lat"].values * 2) / 2
    lon_c = np.round(out["lon"].values * 2) / 2
    return out.assign_coords(lat=lat_c, lon=lon_c).astype("float32")


def build_modis_stack(raw_dir: str | Path, products=tuple(MODIS_PRODUCTS), geometric_products=(),
                      months=range(1, 13), verbose=True) -> xr.Dataset:
    """(month, lat, lon) Dataset with one variable per product, 1 degree."""
    raw_dir = Path(raw_dir)
    data_vars = {}
    for prod in products:
        # one file per month; the month is read from the file name
        file_of_month = {}
        for path in (raw_dir / prod).glob("*.nc"):
            file_of_month[month_from_filename(path.name)] = path
        missing = [m for m in months if m not in file_of_month]
        if missing:
            raise FileNotFoundError(f"{prod}: no file for months {missing} in {raw_dir / prod}")

        layers = []
        for m in months:
            da = coarsen_to_1deg(open_modis(file_of_month[m], prod), geometric=prod in geometric_products)
            layers.append(da.expand_dims(month=[m]))
            if verbose:
                print(f"regrid {prod} month {m:02d}  valid cells: {int(da.notnull().sum())}")
        data_vars[prod] = xr.concat(layers, dim="month")
    ds = xr.Dataset(data_vars)
    ds.attrs["source"] = "Aqua-MODIS L3m monthly climatology, 9 km, area-mean to 1 degree"
    ds.attrs["geometric_mean_products"] = ",".join(geometric_products)
    return ds


# --------------------------------------------------------------------------- #
# WOA
# --------------------------------------------------------------------------- #
def open_woa_surface(path: str | Path, nutrient: str, fields=WOA_FIELDS) -> xr.Dataset:
    """Surface level of one WOA13 monthly file, renamed to ``<nutrient>``,
    ``<nutrient>_mn``, ``<nutrient>_dd``, ``<nutrient>_se``."""
    code = WOA_VARS[nutrient]           # "n" for nitrate, "p" for phosphate, "i" for silicate
    ds = xr.open_dataset(path, decode_times=False)
    out = {}
    for f in fields:
        var = f"{code}_{f}"             # e.g. n_an, n_mn, n_dd, n_se
        if var not in ds:
            continue
        name = nutrient if f == "an" else f"{nutrient}_{f}"
        out[name] = ds[var].isel(time=0, depth=0, drop=True)
    return xr.Dataset(out).sortby("lat").sortby("lon")


def build_woa_stack(raw_dir: str | Path, nutrients=tuple(WOA_VARS), months=range(1, 13), verbose=True) -> xr.Dataset:
    raw_dir = Path(raw_dir)
    layers = []
    for m in months:
        parts = []
        for n in nutrients:
            f = raw_dir / n / f"woa13_all_{WOA_VARS[n]}{m:02d}_01.nc"
            if not f.exists():
                raise FileNotFoundError(f)
            parts.append(open_woa_surface(f, n))
        layers.append(xr.merge(parts).expand_dims(month=[m]))
        if verbose:
            print(f"woa month {m:02d}")
    ds = xr.concat(layers, dim="month")
    ds.attrs["source"] = "WOA13 v2 objectively analysed climatology, surface, 1 degree"
    return ds


# --------------------------------------------------------------------------- #
# Combine
# --------------------------------------------------------------------------- #
def build_dataset(raw_modis: str | Path, raw_woa: str | Path, out_path: str | Path,
                  products=tuple(MODIS_PRODUCTS), geometric_products=("CHL", "APH", "PIC", "POC"),
                  months=range(1, 13), verbose=True) -> xr.Dataset:
    """Regrid MODIS, extract WOA surface, align on the 1-degree grid, write netCDF."""
    modis = build_modis_stack(raw_modis, products, geometric_products, months, verbose)
    woa = build_woa_stack(raw_woa, months=months, verbose=verbose)
    # exact coordinate alignment (both are cell centres on the same 1-degree grid)
    woa = woa.assign_coords(lat=modis["lat"].values, lon=modis["lon"].values)
    ds = xr.merge([modis, woa], join="exact")
    ds.attrs["description"] = (
        "MODIS ocean-colour products (area-mean to 1 degree) and WOA13 surface nutrient "
        "climatology, monthly, for the modis-nutrients project"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    if verbose:
        print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return ds
