"""Launch the comparison as a SageMaker training job.

Prerequisites (one-time):

    aws s3 cp data/satellite_and_WOA13_1_degree_Jan_v2.nc s3://<bucket>/modis-nutrients/data/
    pip install sagemaker

Run from the repository root:

    python aws/run_sagemaker.py --bucket <bucket> --role <SageMakerExecutionRoleArn>

The job executes ``notebooks/02_results.ipynb`` on an ``ml.m5.xlarge`` (CPU is
enough; about half an hour) and uploads the executed notebook and its figures to
``s3://<bucket>/modis-nutrients/output/<job>/output/model.tar.gz``.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    p.add_argument("--notebook", default="notebooks/02_results.ipynb")
    p.add_argument("--experiment", default=None, help="SageMaker Experiments name (optional)")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    import sagemaker
    from sagemaker.tensorflow import TensorFlow

    sess = sagemaker.Session()
    prefix = "modis-nutrients"
    job_name = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"

    estimator = TensorFlow(
        entry_point="aws/train_entry.py",
        source_dir=".",                      # the whole repo; requirements.txt is installed
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.16",
        py_version="py310",
        hyperparameters={"notebook": args.notebook},
        output_path=f"s3://{args.bucket}/{prefix}/output",
        sagemaker_session=sess,
        base_job_name=prefix,
    )
    inputs = {"data": f"s3://{args.bucket}/{prefix}/data/"}

    if args.experiment:
        from sagemaker.experiments.run import Run

        with Run(experiment_name=args.experiment, run_name=job_name, sagemaker_session=sess):
            estimator.fit(inputs, job_name=job_name, wait=args.wait)
    else:
        estimator.fit(inputs, job_name=job_name, wait=args.wait)

    print(f"submitted {job_name}")
    print(f"outputs -> s3://{args.bucket}/{prefix}/output/{job_name}/output/model.tar.gz")


if __name__ == "__main__":
    main()
