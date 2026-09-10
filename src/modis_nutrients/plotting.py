"""Map figures. Each function returns the figure and saves it when ``path`` is given."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .spatial import SplitResult

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868", "buffer": "#BBBBBB"}
SPLIT_ORDER = ["train", "val", "test", "buffer"]


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        # plt.close(fig)
    return fig


def _map_axes(figsize=(11, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return fig, ax


def _to_grid(lat, lon, values):
    """Put 1° samples back on a 180 x 360 grid (NaN where there is no sample)."""
    grid = np.full((180, 360), np.nan)
    i = np.floor(np.asarray(lat) + 90).astype(int).clip(0, 179)
    j = np.floor(np.asarray(lon) + 180).astype(int).clip(0, 359)
    grid[i, j] = values
    lat_edges = np.arange(-90, 91, 1.0)
    lon_edges = np.arange(-180, 181, 1.0)
    return lon_edges, lat_edges, grid


def plot_split_map(split: SplitResult, lat, lon, title=None, path=None):
    """Map of which partition each cell belongs to."""
    codes = np.array([SPLIT_ORDER.index(label) for label in split.labels()], float)
    lon_edges, lat_edges, grid = _to_grid(lat, lon, codes)

    fig, ax = _map_axes()
    cmap = ListedColormap([SPLIT_COLORS[k] for k in SPLIT_ORDER])
    ax.pcolormesh(lon_edges, lat_edges, grid, cmap=cmap, vmin=-0.5, vmax=3.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLORS[k]) for k in SPLIT_ORDER]
    ax.legend(handles, ["train", "validation", "test", "buffer (dropped)"], loc="lower left",
              fontsize=8, framealpha=0.9)
    ax.set_title(title or f"Split: {split.name}")
    return _save(fig, path)


def plot_variogram(centers_km, gamma, model, title="Semivariogram", path=None, marks_km=(250, 500, 1000)):
    """Empirical semivariogram with the fitted GSTools model."""
    sill = model.var + model.nugget
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers_km, gamma, "o", ms=3, label="empirical (GSTools vario_estimate)")
    h = np.linspace(0, centers_km[-1], 200)
    ax.plot(h, model.variogram(h), "k-", lw=1,
            label=f"spherical fit: range {model.len_scale:.0f} km, sill {sill:.2f}")
    for d in marks_km:
        shared = model.var * model.correlation(d) / sill
        ax.plot([d], [model.variogram(d)], "rs", ms=4)
        ax.annotate(f"{shared:.0%} shared\nvariance", (d, model.variogram(d)),
                    textcoords="offset points", xytext=(6, -28), fontsize=7, color="0.3")
    ax.set_xlabel("separation h (km)")
    ax.set_ylabel("γ(h), zonal mean removed")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_prediction_maps(lat, lon, Y_true, Y_pred, targets: list[str], title="", path=None):
    """Rows: truth / prediction / residual. Columns: targets."""
    n = len(targets)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 8))
    for j, target in enumerate(targets):
        truth = Y_true[:, j]
        pred = Y_pred[:, j]
        vmax = np.nanpercentile(truth, 99)
        panels = [
            (truth, "WOA13 truth", "viridis", 0, vmax),
            (pred, "prediction", "viridis", 0, vmax),
            (pred - truth, "prediction − truth", "RdBu_r", -vmax / 2, vmax / 2),
        ]
        for row, (values, label, cmap, vmin, vmx) in enumerate(panels):
            ax = axes[row, j]
            lon_edges, lat_edges, grid = _to_grid(lat, lon, values)
            im = ax.pcolormesh(lon_edges, lat_edges, grid, cmap=cmap, vmin=vmin, vmax=vmx)
            ax.set_title(f"{target}: {label}", fontsize=9)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)
