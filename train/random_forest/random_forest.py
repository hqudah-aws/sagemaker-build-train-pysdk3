#!/usr/bin/env python

import argparse
import boto3
import joblib
import json
import os
import sys
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    # ARN of the Secrets Manager secret holding HF_TOKEN / WANDB_API_KEY.
    # Can also be provided via the SECRETS_ARN environment variable, which is how
    # the notebook passes it (ModelTrainer environment={"SECRETS_ARN": ...}).
    parser.add_argument(
        "--secrets_arn", type=str, default=os.environ.get("SECRETS_ARN")
    )

    return parser.parse_known_args()


def load_secrets_from_arn(secret_arn):
    """
    Fetch a JSON secret from AWS Secrets Manager and export each key as an
    environment variable, so downstream libraries (e.g. Hugging Face, Weights &
    Biases) pick them up automatically.

    The secret is expected to be a JSON object, e.g.:
        {"HF_TOKEN": "hf_...", "WANDB_API_KEY": "..."}

    Passing the secret ARN (instead of the raw values) keeps secrets out of the
    job definition, notebook, and CloudTrail. The training job's execution role
    just needs secretsmanager:GetSecretValue on this secret.
    """
    # The AWS region is embedded in the ARN, so we can build a client that
    # points at the right region even if it differs from the job's default.
    region = secret_arn.split(":")[3]
    client = boto3.client("secretsmanager", region_name=region)
    secret_string = client.get_secret_value(SecretId=secret_arn)["SecretString"]
    secrets = json.loads(secret_string)

    for key, value in secrets.items():
        os.environ[key] = value
        # NOTE: printing secret values is for VALIDATION ONLY. Remove this in
        # production — never log secret material to CloudWatch.
        print(f"[secrets] exported {key} from Secrets Manager (value: {value})")

    return secrets


def start(args):
    """
    Train a Random Forest Classifier
    """

    # Load example dataset (Pima Indians Diabetes, binary classification).
    # OpenML dataset id 37; features are 8 numeric clinical measurements and
    # the target is tested_positive / tested_negative for diabetes.
    X, y = fetch_openml(data_id=37, as_frame=False, return_X_y=True)

    # Encode the string target labels into 0/1 for the classifier
    y = LabelEncoder().fit_transform(y)

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

    model = RandomForestClassifier()
    model.set_params(**hyperparameters)
    model.fit(X_train, y_train)

    # Metrics we care about
    accuracy = model.score(X_validation, y_validation)
    print("accuracy: {}".format(accuracy))
    f1 = f1_score(y_validation, model.predict(X_validation))
    print("F1: {}".format(f1))

    # Save the model
    joblib.dump(model, os.path.join(os.environ["SM_MODEL_DIR"], "model.joblib"))


if __name__ == "__main__":
    args, _ = parse_args()

    # Load secrets from Secrets Manager (if an ARN was provided) and export them
    # as environment variables before anything else runs. In Example 1 this is
    # purely to validate the pattern; the same approach lets Example 2 fetch a
    # gated Hugging Face model and log to Weights & Biases.
    if args.secrets_arn:
        load_secrets_from_arn(args.secrets_arn)
    else:
        print("[secrets] no --secrets_arn / SECRETS_ARN provided; skipping")

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.autolog()
    # Tag SageMaker AI Training job name so we can use it to pull
    # experiment data later if we register the model
    mlflow.set_tag("sm_training_job_name", os.environ.get("TRAINING_JOB_NAME"))

    start(args)
