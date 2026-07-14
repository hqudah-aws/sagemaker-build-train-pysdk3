# Bring Your Own Model with SageMaker AI: Script Mode in SDK v3

This project demonstrates **script mode** in the SageMaker Python SDK v3 through two end-to-end examples:

1. **Scikit-learn Random Forest** — Train a tabular ML model and deploy to a real-time endpoint
2. **Stable Diffusion 3.5 LoRA** — Fine-tune a generative AI model with multi-GPU distributed training

Both examples use `ModelTrainer` for training and `ModelBuilder` for deployment. The key pattern is `SourceCode` — your training and inference code is injected into minimal containers at runtime, so you never need to rebuild images when changing your algorithm code.

## Getting Started

### 1. Build and Push Docker Images

Before running the notebook, you need to build and push the training containers to Amazon ECR. From a terminal in the project root:

```bash
# Scikit-learn container
bash build.sh --env .env.docker.sklearn
bash push.sh --env .env.docker.sklearn

# Stable Diffusion container
bash build.sh --env .env.docker.stablediffusion
bash push.sh --env .env.docker.stablediffusion
```

These scripts handle ECR authentication, repository creation, and image pushing automatically.

### 2. Open the Notebook

Open `script_mode_sdkv3_blog (1).ipynb` and follow along. The notebook covers:

- Configuring training jobs with YAML recipe files
- Launching training with `ModelTrainer` and `SourceCode`
- Deploying models to real-time endpoints with `ModelBuilder`
- (Optional) MLflow experiment tracking

## Project Structure

```
├── docker/                  # Dockerfiles and requirements for training containers
│   ├── sklearn/
│   └── stable_diffusion/
├── train/                   # Training scripts and configs
│   ├── random_forest/       # Scikit-learn training script
│   ├── stable_diffusion/    # LoRA fine-tuning script + accelerate configs
│   ├── train.randomforest.yaml
│   └── train.stablediffusion.yaml
├── deploy/                  # Inference handlers
│   ├── random_forest/
│   └── stable_diffusion/
├── build.sh                 # Build Docker image locally
├── push.sh                  # Push Docker image to ECR
└── prepare_sd_data.py       # Prepare Stable Diffusion training data
```

## Prerequisites

- AWS account with SageMaker AI access
- IAM execution role with SageMaker and S3 permissions
- SageMaker Python SDK v3
- Docker (for building containers)
- S3 bucket for training data and model artifacts
- (For Stable Diffusion) A HuggingFace token with access to the model
