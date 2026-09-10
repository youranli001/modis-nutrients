"""Launch a SageMaker Automatic Model Tuning job.

    python aws/run_tuning.py --bucket <bucket> --role <SageMakerExecutionRoleArn> --model mlp

SageMaker runs ``aws/tune_entry.py`` once per trial with a candidate drawn from
the ranges below, reads ``blocked_cv_r2=...`` from the log, and uses Bayesian
optimisation to choose the next candidate. Results are viewable in the console
under *Training > Hyperparameter tuning jobs*, or with::

    tuner.analytics().dataframe()

Cost note: every trial is a separate training job with a few minutes of
instance start-up, which dominates for a model that fits in 30 s. Twelve trials
on ml.m5.xlarge is on the order of a dollar; the point is the workflow, not the
gain — the local grid already showed the MLP architecture is not the bottleneck.
"""

from __future__ import annotations

import argparse
import time

RANGES = {
    "mlp": lambda P: {
        "n_layers": P.Integer(1, 3),
        "width": P.Categorical(["32", "64", "128", "256"]),
        "learning_rate": P.Continuous(1e-4, 1e-2, scaling_type="Logarithmic"),
        "batch_size": P.Categorical(["128", "256", "512"]),
    },
    "gbm": lambda P: {
        "max_leaf_nodes": P.Integer(7, 127),
        "learning_rate": P.Continuous(0.02, 0.2, scaling_type="Logarithmic"),
        "l2_regularization": P.Continuous(0.0, 5.0),
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--model", choices=list(RANGES), default="mlp")
    p.add_argument("--split", default="block_buffer")
    p.add_argument("--config", default="default.yaml")
    p.add_argument("--max-jobs", type=int, default=12)
    p.add_argument("--max-parallel", type=int, default=3)
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    import sagemaker
    from sagemaker import tuner as T
    from sagemaker.tensorflow import TensorFlow

    class P:  # short names for the range types
        Categorical, Continuous, Integer = T.CategoricalParameter, T.ContinuousParameter, T.IntegerParameter

    sess = sagemaker.Session()
    prefix = "modis-nutrients"
    metric = [{"Name": "blocked_cv_r2", "Regex": r"blocked_cv_r2=([-+]?[0-9]*\.?[0-9]+)"}]

    estimator = TensorFlow(
        entry_point="aws/tune_entry.py",
        source_dir=".",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.16",
        py_version="py310",
        hyperparameters={"model": args.model, "split": args.split, "config": args.config},
        metric_definitions=metric,
        output_path=f"s3://{args.bucket}/{prefix}/tuning",
        sagemaker_session=sess,
    )
    tuner = T.HyperparameterTuner(
        estimator,
        objective_metric_name="blocked_cv_r2",
        objective_type="Maximize",
        hyperparameter_ranges=RANGES[args.model](P),
        metric_definitions=metric,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel,
        strategy="Bayesian",
        base_tuning_job_name=f"{prefix}-{args.model}",
    )
    job = f"{prefix}-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    tuner.fit({"data": f"s3://{args.bucket}/{prefix}/data/"}, job_name=job, wait=args.wait)
    print(f"submitted tuning job {job}")
    if args.wait:
        df = tuner.analytics().dataframe().sort_values("FinalObjectiveValue", ascending=False)
        print(df.head(10).to_string())


if __name__ == "__main__":
    main()
