"""SageMaker script-mode entry point: run the results notebook in the container.

SageMaker copies the repository (``source_dir``) into the container, installs
``requirements.txt``, stages the S3 input channel under ``/opt/ml/input/data/data/``
and runs this file. The executed notebook and the figures it writes are copied to
``SM_MODEL_DIR``, which SageMaker uploads as ``model.tar.gz`` when the job finishes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--notebook", default="notebooks/02_results.ipynb")
    p.add_argument("--data-channel", default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args, _ = p.parse_known_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "nbconvert", "ipykernel"], check=True)

    (repo / "data").mkdir(exist_ok=True)
    for f in Path(args.data_channel).glob("*.nc"):
        shutil.copy(f, repo / "data" / f.name)
        print(f"staged {f.name}", flush=True)

    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", args.notebook], check=True)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.notebook, out / Path(args.notebook).name)
    if (repo / "docs" / "figures").exists():
        shutil.copytree(repo / "docs" / "figures", out / "figures", dirs_exist_ok=True)
    print(f"wrote {sorted(p.name for p in out.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
