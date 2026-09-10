"""Predicting surface nutrients (nitrate, phosphate, silicate) from MODIS ocean-colour
products, with leakage-aware spatial evaluation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("modis-nutrients")
except PackageNotFoundError:  # editable/uninstalled checkout
    __version__ = "0.0.0"
