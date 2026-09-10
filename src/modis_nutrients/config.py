"""Load a YAML config and resolve relative paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p if (REPO_ROOT / p).exists() else REPO_ROOT / "configs" / p
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    for section, key in (("data", "path"), ("data", "raw_dir"), ("data", "built_path")):
        if key in cfg.get(section, {}) and not Path(cfg[section][key]).is_absolute():
            cfg[section][key] = str(REPO_ROOT / cfg[section][key])
    cfg["_config_path"] = str(p)
    return cfg
