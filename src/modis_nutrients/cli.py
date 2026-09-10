"""Command line for the two long-running data steps:

    modis-nutrients download     # raw WOA13 + MODIS files -> data/raw/
    modis-nutrients build        # regrid + combine -> data/satellite_and_WOA13_1deg_monthly.nc

Everything else (splits, models, evaluation, figures) lives in notebooks/02_results.ipynb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="modis-nutrients", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="raw WOA13 and MODIS files -> data/raw/")
    d.add_argument("--what", choices=["woa", "modis", "all"], default="all")
    d.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    d.add_argument("--appkey", default=None, help="OB.DAAC appkey (alternative to ~/.netrc)")
    d.add_argument("--search", action="store_true", help="query OB.DAAC for current file names")

    b = sub.add_parser("build", help="regrid MODIS to 1 degree and combine with WOA")
    b.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    raw = Path(cfg["data"]["raw_dir"])

    if args.cmd == "download":
        from .download import download_modis, download_woa, search_modis_files

        if args.what in ("woa", "all"):
            download_woa(raw / "woa13", months=args.months)
        if args.what in ("modis", "all"):
            names = search_modis_files() if args.search else None
            download_modis(raw / "modis", months=args.months, appkey=args.appkey, filenames=names)

    if args.cmd == "build":
        from .regrid import build_dataset

        build_dataset(raw / "modis", raw / "woa13", cfg["data"]["built_path"],
                      products=cfg["data"]["features"], geometric_products=cfg["data"]["log_features"],
                      months=args.months)


if __name__ == "__main__":
    main()
