"""SageMaker entry point for *one* hyper-parameter candidate.

SageMaker Automatic Model Tuning launches this script once per trial and passes
the candidate as command-line arguments (``--hidden 128-64-32 --learning_rate
0.0007 ...``). The script scores the candidate with the same blocked GroupKFold
objective the local grid search uses (:func:`modis_nutrients.evaluate.blocked_cv_score`)
and prints one line::

    blocked_cv_r2=0.8973

which the tuner captures through the ``metric_definitions`` regex in
``aws/run_tuning.py``. The test partition is never touched here.

The MLP architecture is given as ``--n_layers`` and ``--width`` (scalars, as
SageMaker requires) and turned into the layer list by ``tuning.hidden_layers``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlp")
    p.add_argument("--split", default="block_buffer")
    p.add_argument("--config", default="default.yaml")
    p.add_argument("--n_splits", type=int, default=None)
    # candidate hyper-parameters (any subset may be present)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_leaf_nodes", type=int, default=None)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--l2_regularization", type=float, default=None)
    p.add_argument("--data-channel", default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args, _ = p.parse_known_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    (repo / "data").mkdir(exist_ok=True)
    for f in Path(args.data_channel).glob("*.nc"):
        shutil.copy(f, repo / "data" / f.name)

    from modis_nutrients.config import load_config
    from modis_nutrients.tuning import blocked_cv_score, hidden_layers

    cfg = load_config(f"configs/{args.config}")
    params = {}
    if args.n_layers and args.width:
        params["hidden"] = hidden_layers(args.n_layers, args.width)
    for k in ("learning_rate", "batch_size", "max_leaf_nodes", "max_iter", "l2_regularization"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    mean, std = blocked_cv_score(cfg, args.split, args.model, params, n_splits=args.n_splits or 3)
    print(f"candidate={json.dumps(params)}", flush=True)
    print(f"blocked_cv_r2={mean:.4f}", flush=True)     # <- captured by the tuner
    print(f"blocked_cv_r2_std={std:.4f}", flush=True)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.model_dir) / "result.json").write_text(
        json.dumps({"model": args.model, "split": args.split, "params": params,
                    "blocked_cv_r2": mean, "blocked_cv_r2_std": std}, indent=2))


if __name__ == "__main__":
    main()
