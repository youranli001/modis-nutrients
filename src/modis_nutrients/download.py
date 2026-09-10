"""Raw inputs: WOA13 v2 nutrients (NOAA NCEI, plain HTTPS) and Aqua-MODIS L3m
monthly climatologies (NASA OB.DAAC, Earthdata login via ``~/.netrc`` or an appkey).
MODIS file names carry the climatology dates and change when OB.DAAC reprocesses;
the names below were valid in 2024 and :func:`search_modis_files` fetches current ones.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

WOA_BASE = "https://www.ncei.noaa.gov/data/oceans/woa/WOA13/DATAv2"
WOA_VARS = {"nitrate": "n", "phosphate": "p", "silicate": "i"}

OBDAAC_GETFILE = "https://oceandata.sci.gsfc.nasa.gov/ob/getfile"
OBDAAC_SEARCH = "https://oceandata.sci.gsfc.nasa.gov/api/file_search"

# product code used in this project -> (suite, variable) in the L3m file name
MODIS_PRODUCTS = {
    "CHL": ("CHL", "chlor_a"),
    "APH": ("IOP", "aph_443"),
    "FLU": ("FLH", "nflh"),
    "PIC": ("PIC", "pic"),
    "POC": ("POC", "poc"),
    "PAR": ("PAR", "par"),
    "SST": ("SST", "sst"),
}

# start_end date stamps of the 9 km monthly-climatology files, per product, months 1..12,
# as served by OB.DAAC in 2024. Refresh with search_modis_files() if a download 404s.
_MODIS_STAMPS = {
    "CHL": ["20030101_20230131", "20030201_20230228", "20030301_20210331", "20030401_20220430",
            "20030501_20220531", "20030601_20220630", "20020701_20210731", "20020801_20210831",
            "20020901_20230930", "20021001_20211031", "20021101_20211130", "20021201_20221231"],
    "APH": ["20030101_20230131", "20030201_20220228", "20030301_20230331", "20030401_20220430",
            "20030501_20230531", "20030601_20230630", "20020701_20210731", "20020801_20230831",
            "20020901_20230930", "20021001_20221031", "20021101_20211130", "20021201_20221231"],
    "FLU": ["20030101_20230131", "20030201_20220228", "20030301_20210331", "20030401_20220430",
            "20030501_20220531", "20030601_20210630", "20020701_20220731", "20020801_20210831",
            "20020901_20210930", "20021001_20211031", "20021101_20211130", "20021201_20221231"],
    "PIC": ["20030101_20230131", "20030201_20230228", "20030301_20230331", "20030401_20220430",
            "20030501_20230531", "20030601_20220630", "20020701_20230731", "20020801_20220831",
            "20020901_20230930", "20021001_20221031", "20021101_20221130", "20021201_20221231"],
    "POC": ["20030101_20230131", "20030201_20230228", "20030301_20230331", "20030401_20230430",
            "20030501_20230531", "20030601_20230630", "20020701_20230731", "20020801_20220831",
            "20020901_20220930", "20021001_20221031", "20021101_20221130", "20021201_20221231"],
    "PAR": ["20030101_20220131", "20030201_20220228", "20030301_20230331", "20030401_20220430",
            "20030501_20210531", "20030601_20210630", "20020701_20220731", "20020801_20230831",
            "20020901_20220930", "20021001_20211031", "20021101_20221130", "20021201_20221231"],
    "SST": ["20030101_20230131", "20030201_20240229", "20030301_20220331", "20030401_20220430",
            "20030501_20220531", "20030601_20220630", "20020701_20220731", "20020801_20220831",
            "20020901_20220930", "20021001_20221031", "20021101_20231130", "20021201_20231231"],
}


def modis_filename(product: str, month: int, stamp: str | None = None, res: str = "9km") -> str:
    suite, var = MODIS_PRODUCTS[product]
    stamp = stamp or _MODIS_STAMPS[product][month - 1]
    return f"AQUA_MODIS.{stamp}.L3m.MC.{suite}.{var}.{res}.nc"


def month_from_filename(name: str) -> int:
    """Month of a climatology file from its start date (``AQUA_MODIS.YYYYMMDD_...``)."""
    m = re.search(r"\.(\d{4})(\d{2})\d{2}_\d{8}\.", name)
    if not m:
        raise ValueError(f"no date stamp in {name}")
    return int(m.group(2))


def woa_filename(nutrient: str, month: int) -> str:
    return f"woa13_all_{WOA_VARS[nutrient]}{month:02d}_01.nc"


def woa_url(nutrient: str, month: int) -> str:
    return f"{WOA_BASE}/{nutrient}/netcdf/all/1.00/{woa_filename(nutrient, month)}"


# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, session: requests.Session, params=None, chunk=1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, params=params, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(
                f"{url} returned HTML instead of data: most likely an Earthdata login page. "
                "Check ~/.netrc or pass appkey=."
            )
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
    tmp.rename(dest)
    return dest


def download_woa(dest_dir: str | Path, nutrients=("nitrate", "phosphate", "silicate"),
                 months=range(1, 13), verbose=True) -> list[Path]:
    """Monthly WOA13 v2 files, 1 degree, all depths. Skips files already present."""
    dest_dir = Path(dest_dir)
    out = []
    with requests.Session() as s:
        for n in nutrients:
            for m in months:
                p = _download(woa_url(n, m), dest_dir / n / woa_filename(n, m), s)
                if verbose:
                    print(f"woa13 {n:<9s} month {m:02d}  {p.name}")
                out.append(p)
    return out


def download_modis(dest_dir: str | Path, products=tuple(MODIS_PRODUCTS), months=range(1, 13),
                   appkey: str | None = None, filenames: dict[str, list[str]] | None = None,
                   verbose=True) -> list[Path]:
    """9 km L3m monthly-climatology files from OB.DAAC.

    ``filenames`` (product -> list of 12 names) overrides the built-in stamps;
    get a fresh list with :func:`search_modis_files`.
    """
    dest_dir = Path(dest_dir)
    out = []
    params = {"appkey": appkey} if appkey else None
    with requests.Session() as s:
        s.headers["User-Agent"] = "modis-nutrients/0.2"
        for prod in products:
            for m in months:
                name = modis_filename(prod, m)
                if filenames and prod in filenames:
                    for candidate in filenames[prod]:
                        if month_from_filename(candidate) == m:
                            name = candidate
                p = _download(f"{OBDAAC_GETFILE}/{name}", dest_dir / prod / name, s, params=params)
                if verbose:
                    print(f"modis {prod:<4s} month {m:02d}  {p.name}")
                out.append(p)
    return out


def search_modis_files(products=tuple(MODIS_PRODUCTS), res: str = "9km") -> dict[str, list[str]]:
    """Ask the OB.DAAC file-search API for the current names of the monthly
    climatology files. Returns product -> sorted list of 12 file names.

    The API is documented at https://oceandata.sci.gsfc.nasa.gov/api/file_search_help;
    field names may need adjusting if it changes."""
    found: dict[str, list[str]] = {}
    with requests.Session() as s:
        for prod in products:
            suite, var = MODIS_PRODUCTS[prod]
            r = s.post(
                OBDAAC_SEARCH,
                data={"sensor": "aqua", "dtype": "L3m", "period": "MC",
                      "search": f"*L3m.MC.{suite}.{var}.{res}.nc", "results_as_file": 1},
                timeout=60,
            )
            r.raise_for_status()
            names = set()
            for line in r.text.splitlines():
                line = line.strip()
                if line.endswith(".nc"):
                    names.add(line.split("/")[-1])
            found[prod] = sorted(names, key=month_from_filename)
    return found
