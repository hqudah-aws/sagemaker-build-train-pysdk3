#!/usr/bin/env python

import argparse
import joblib
import os
import sys
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import sagemaker_mlflow  # noqa: F401
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    num_cpus = os.environ.get("SM_NUM_CPUS", 4)
    parser.add_argument("--n_jobs", type=int, default=num_cpus)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=120)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def start(args):
    """
    Train a Random Forest Regressor
    """

    # Load example dataset (diabetes regression)
    X, y = load_diabetes(return_X_y=True)

    # Alternatively read data from os.environ.get("SM_INPUT_DATA_DIR")

    # Split into train and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    hyperparameters = {
        "max_depth": args.max_depth,
        "verbose": 1,
        "n_jobs": args.n_jobs,
        "n_estimators": args.n_estimators,
    }

    model = RandomForestRegressor()
    model.set_params(**hyperparameters)
    model.fit(X_train, y_train)

    # Metrics we care about
    r_squared = model.score(X_validation, y_validation)
    print("r-squared: {}".format(r_squared))
    mse = mean_squared_error(y_validation, model.predict(X_validation))
    print("MSE: {}".format(mse))

    # Save the model
    joblib.dump(model, os.path.join(os.environ["SM_MODEL_DIR"], "model.joblib"))


if __name__ == "__main__":
    args, _ = parse_args()
    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.autolog()
    # Tag SageMaker AI Training job name so we can use it to pull
    # experiment data later if we register the model
    mlflow.set_tag("sm_training_job_name", os.environ.get("TRAINING_JOB_NAME"))

    start(args)
