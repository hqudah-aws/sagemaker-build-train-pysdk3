"""
Prepare the google/dreambooth 'dog' dataset for Stable Diffusion LoRA fine-tuning.
Adds captions, saves in HF datasets format (as DatasetDict with 'train' split), and uploads to S3.

Usage:
    python prepare_sd_data.py
"""

import boto3
import os
from datasets import load_dataset, DatasetDict

# Config — matches the notebook
_session = boto3.session.Session()
AWS_ACCOUNT_ID = _session.client("sts").get_caller_identity()["Account"]
AWS_REGION = _session.region_name
S3_BUCKET = f"sagemaker-{AWS_REGION}-{AWS_ACCOUNT_ID}"
S3_PREFIX = "stable-diffusion/data"

LOCAL_DATA_DIR = "/tmp/sd-training-data"

print("Loading google/dreambooth 'dog' dataset...")
dataset = load_dataset("google/dreambooth", "dog", split="train")

# Add captions (standard DreamBooth-style token)
dataset = dataset.add_column("text", ["a photo of a sks dog"] * len(dataset))

# Wrap in DatasetDict so load_from_disk returns dict with "train" key
dataset_dict = DatasetDict({"train": dataset})

print(f"Dataset size: {len(dataset_dict['train'])} samples")
print(f"Columns: {dataset_dict['train'].column_names}")
print(f"Sample caption: {dataset_dict['train'][0]['text']}")

# Save in HF datasets format (DatasetDict so load_from_disk gets a "train" split)
dataset_dict.save_to_disk(LOCAL_DATA_DIR)
print(f"Saved dataset to {LOCAL_DATA_DIR}")

# Upload to S3
print(f"Uploading to s3://{S3_BUCKET}/{S3_PREFIX}/ ...")
s3 = _session.client("s3")
for root, dirs, files in os.walk(LOCAL_DATA_DIR):
    for file in files:
        local_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_path, LOCAL_DATA_DIR)
        s3_key = f"{S3_PREFIX}/{relative_path}"
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"  Uploaded: {s3_key}")

print(f"\n✓ Done! Data available at: s3://{S3_BUCKET}/{S3_PREFIX}/")
print("You can now run the training job.")
