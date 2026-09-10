import numpy as np
import pytest

from modis_nutrients.spatial import (
    RandomSplit,
    SpatialBlockSplit,
    block_group_kfold,
    block_ids,
    make_splitter,
    nearest_distance_km,
    variogram,
)


@pytest.fixture(scope="module")
def grid():
    lat = np.arange(-79, 80, 2.0)
    lon = np.arange(-179, 180, 2.0)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2.ravel(), lon2.ravel()


def _disjoint(res, n):
    allidx = np.concatenate([res.train, res.val, res.test, res.dropped])
    assert len(allidx) == len(np.unique(allidx)) == n


def test_nearest_distance_known_value():
    d = nearest_distance_km([51.5074], [-0.1278], [48.8566], [2.3522])   # London -> Paris
    assert 335 < d[0] < 350


def test_block_ids_wrap_at_dateline():
    assert block_ids([0.0], [180.0], 10) == block_ids([0.0], [-180.0], 10)


def test_random_split_fractions(grid):
    lat, lon = grid
    res = RandomSplit(seed=1).split(lat, lon)
    _disjoint(res, len(lat))
    assert abs(len(res.test) / len(lat) - 0.2) < 0.01


def test_blocks_are_pure(grid):
    lat, lon = grid
    res = SpatialBlockSplit(block_deg=10, seed=3).split(lat, lon)
    _disjoint(res, len(lat))
    labels = res.labels()
    for b in np.unique(res.block_id):
        assert len(set(labels[res.block_id == b])) == 1


def test_buffer_is_respected_and_test_untouched(grid):
    lat, lon = grid
    a = SpatialBlockSplit(block_deg=20, buffer_km=0, seed=3).split(lat, lon)
    b = SpatialBlockSplit(block_deg=20, buffer_km=400, seed=3).split(lat, lon)
    _disjoint(b, len(lat))
    holdout = np.concatenate([b.val, b.test])
    assert nearest_distance_km(lat[b.train], lon[b.train], lat[holdout], lon[holdout]).min() >= 400
    assert nearest_distance_km(lat[b.val], lon[b.val], lat[b.test], lon[b.test]).min() >= 400
    assert np.array_equal(a.test, b.test)
    assert len(b.dropped) > 0


def test_every_latitude_row_in_every_partition(grid):
    lat, lon = grid
    res = SpatialBlockSplit(block_deg=20, seed=5).split(lat, lon)
    rows = np.floor((lat + 90) / 20).astype(int)
    labels = res.labels()
    for r in np.unique(rows):
        assert {"train", "val", "test"} <= set(labels[rows == r])


def test_split_is_reproducible(grid):
    lat, lon = grid
    a = SpatialBlockSplit(block_deg=10, buffer_km=300, seed=7).split(lat, lon)
    b = SpatialBlockSplit(block_deg=10, buffer_km=300, seed=7).split(lat, lon)
    c = SpatialBlockSplit(block_deg=10, buffer_km=300, seed=8).split(lat, lon)
    assert np.array_equal(a.train, b.train) and not np.array_equal(a.test, c.test)


def test_block_split_fractions_roughly_60_20_20(grid):
    lat, lon = grid
    res = SpatialBlockSplit(block_deg=10, seed=3).split(lat, lon)
    n = len(lat)
    assert abs(len(res.train) / n - 0.6) < 0.05
    assert abs(len(res.test) / n - 0.2) < 0.05


def test_make_splitter_names():
    cfg = {"split": {"block_deg": 20, "buffer_km": 500, "seed": 0}}
    assert make_splitter("block_buffer", cfg).name == "block20deg_buffer500km"
    assert make_splitter("block", cfg).buffer_km == 0
    with pytest.raises(ValueError):
        make_splitter("nonsense", cfg)


def test_group_kfold_keeps_blocks_together(grid):
    lat, lon = grid
    groups, cv = block_group_kfold(lat, lon, block_deg=10, n_splits=4)
    for tr, te in cv.split(np.zeros((len(lat), 1)), groups=groups):
        assert not set(groups[tr]) & set(groups[te])


def test_variogram_detects_correlation_scale(grid):
    """Smooth field: short-range gamma far below the sill; white noise: not."""
    lat, lon = grid
    rng = np.random.default_rng(0)
    smooth = np.sin(np.deg2rad(lat) * 4) * np.cos(np.deg2rad(lon) * 3)
    noise = rng.normal(size=len(lat))
    _, g_s, m_s = variogram(lat, lon, smooth, max_km=3000, bin_km=150, detrend_band_deg=None, sampling_size=400)
    _, g_n, m_n = variogram(lat, lon, noise, max_km=3000, bin_km=150, detrend_band_deg=None, sampling_size=400)
    assert g_s[0] < 0.2 * (m_s.var + m_s.nugget)
    assert m_s.correlation(200) > 0.7
    # white noise: gamma already at the sill (~1) beyond the first bin
    assert g_n[1:4].mean() > 0.8
