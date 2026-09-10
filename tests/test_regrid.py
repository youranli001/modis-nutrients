"""Tests for the raw -> 1-degree build step, on synthetic files shaped like the
real ones (9 km MODIS L3m grid, WOA13 monthly netCDF)."""

import numpy as np
import xarray as xr

from modis_nutrients.dataset import TableSpec, grid_to_table
from modis_nutrients.download import modis_filename, month_from_filename, woa_filename
from modis_nutrients.regrid import build_dataset, coarsen_to_1deg
from modis_nutrients.spatial import SpatialBlockSplit


def _modis_9km(values_fn, var="chlor_a"):
    lat = 90 - (np.arange(2160) + 0.5) / 12   # descending, like OB.DAAC files
    lon = -180 + (np.arange(4320) + 0.5) / 12
    lon2, lat2 = np.meshgrid(lon, lat)
    return xr.Dataset({var: (("lat", "lon"), values_fn(lat2, lon2).astype("float32"))},
                      coords={"lat": lat, "lon": lon})


def test_coarsen_aligns_with_woa_grid():
    ds = _modis_9km(lambda la, lo: np.floor(la) + 0.5)   # value = 1-degree cell centre latitude
    out = coarsen_to_1deg(ds["chlor_a"].sortby("lat"))
    assert out.shape == (180, 360)
    assert np.allclose(out["lat"].values[[0, -1]], [-89.5, 89.5])
    assert np.allclose(out["lon"].values[[0, -1]], [-179.5, 179.5])
    # each 1-degree cell holds exactly its own centre latitude
    assert np.allclose(out.values, out["lat"].values[:, None])


def test_coarsen_geometric_mean_and_min_fraction():
    def f(la, lo):
        v = np.where((lo % 1) < 0.5, 1.0, 100.0)   # half the sub-pixels 1, half 100
        v = np.where(la > 80, np.nan, v)           # no data north of 80
        return v
    ds = _modis_9km(f)
    arith = coarsen_to_1deg(ds["chlor_a"].sortby("lat"), geometric=False)
    geo = coarsen_to_1deg(ds["chlor_a"].sortby("lat"), geometric=True)
    row = arith.sel(lat=0.5)
    assert np.allclose(row.values, 50.5)
    assert np.allclose(geo.sel(lat=0.5).values, 10.0)     # sqrt(1*100)
    assert arith.sel(lat=slice(80, 90)).isnull().all()


def test_filename_helpers():
    assert modis_filename("CHL", 7) == "AQUA_MODIS.20020701_20210731.L3m.MC.CHL.chlor_a.9km.nc"
    assert month_from_filename(modis_filename("SST", 11)) == 11
    assert woa_filename("silicate", 3) == "woa13_all_i03_01.nc"


def test_build_dataset_end_to_end(tmp_path):
    products = ["CHL", "SST"]
    raw_modis, raw_woa = tmp_path / "modis", tmp_path / "woa13"
    for prod, var in [("CHL", "chlor_a"), ("SST", "sst")]:
        for m in (1, 2):
            ds = _modis_9km(lambda la, lo, m=m: (np.cos(np.deg2rad(la)) + m), var=var)
            (raw_modis / prod).mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(raw_modis / prod / modis_filename(prod, m))
    lat = np.arange(-89.5, 90, 1.0)
    lon = np.arange(-179.5, 180, 1.0)
    for n, code in [("nitrate", "n"), ("phosphate", "p"), ("silicate", "i")]:
        for m in (1, 2):
            an = np.abs(lat)[None, None, :, None] / 3 + m + 0 * lon[None, None, None, :]
            an = np.where((np.abs(lon) < 20)[None, None, None, :] & (lat > 0)[None, None, :, None], np.nan, an)
            ds = xr.Dataset(
                {f"{code}_an": (("time", "depth", "lat", "lon"), an.astype("float32")),
                 f"{code}_dd": (("time", "depth", "lat", "lon"), np.ones_like(an))},
                coords={"time": [0.0], "depth": [0.0, 5.0][:1], "lat": lat, "lon": lon})
            (raw_woa / n).mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(raw_woa / n / woa_filename(n, m))

    out = build_dataset(raw_modis, raw_woa, tmp_path / "built.nc", products=products,
                        geometric_products=("CHL",), months=(1, 2), verbose=False)
    assert set(out.data_vars) >= {"CHL", "SST", "nitrate", "nitrate_dd", "phosphate", "silicate"}
    assert out.sizes == {"month": 2, "lat": 180, "lon": 360}

    # monthly table: same cell_id across months, so a spatial split keeps them together
    spec = TableSpec(products, ["nitrate", "phosphate", "silicate"], months=(1, 2))
    df = grid_to_table(out, spec)
    assert set(df["month"]) == {1, 2}
    assert df.groupby("cell_id")["lat"].nunique().max() == 1
    res = SpatialBlockSplit(block_deg=20, buffer_km=0).split(df["lat"], df["lon"], month=df["month"])
    lab = res.labels()
    per_cell = df.assign(lab=lab).groupby("cell_id")["lab"].nunique()
    assert per_cell.max() == 1
