# SavVio Terraform Infrastructure — Setup Guide

## Prerequisites

Before you start, make sure these are installed on your machine:

| Tool | Version | Install |
|------|---------|---------|
| Terraform | >= 1.5 | `brew install terraform` (Mac) or [terraform.io/downloads](https://developer.hashicorp.com/terraform/downloads) |
| Google Cloud SDK (`gcloud`) | Latest | `brew install google-cloud-sdk` or [cloud.google.com/sdk/install](https://cloud.google.com/sdk/docs/install) |
| Git | Any | Already installed on most machines |

No Python venv needed — Terraform is a standalone binary. No `.env` files either — all config lives in `terraform.tfvars`.

---

## Authenticate with GCP

Run these once on your machine. This logs you in and sets the active project.

```bash
# Login to your Google account
gcloud auth login

# Set the project
gcloud config set project savvio-ai

# Create Application Default Credentials (Terraform uses these)
gcloud auth application-default login
```

---

## Step 1 — Clone SavVio repo and navigate to deployment

```bash
git clone https://github.com/nirajmehta960/SavVio.git
cd SavVio/deployment
```

---

## Step 2 — Terraform folder structure

```
deployment_pipeline/
└── terraform/
    ├── modules/
    │   ├── cloud_sql/
    │   │   ├── main.tf
    │   │   ├── variables.tf
    │   │   └── outputs.tf
    │   ├── cloud_run/
    │   │   ├── main.tf
    │   │   ├── variables.tf
    │   │   └── outputs.tf
    │   ├── storage/
    │   │   ├── main.tf
    │   │   ├── variables.tf
    │   │   └── outputs.tf
    │   ├── artifact_registry/
    │   │   ├── main.tf
    │   │   ├── variables.tf
    │   │   └── outputs.tf
    │   └── secrets/
    │       ├── main.tf
    │       ├── variables.tf
    │       └── outputs.tf
    └── environments/
        ├── dev/
        │   ├── backend.tf
        │   ├── main.tf
        │   ├── variables.tf
        │   ├── outputs.tf
        │   └── terraform.tfvars
        └── prod/
            ├── backend.tf
            ├── main.tf
            ├── variables.tf
            ├── outputs.tf
            └── terraform.tfvars
```

---
### File Walkthrough

| File | What it does | When you edit it |
|------|-------------|-----------------|
| `terraform.tfvars` | Environment-specific values (region, DB size, images) | When changing config for an environment |
| `variables.tf` | Declares what variables exist | When adding a new configurable value |
| `main.tf` (in env folder) | Wires modules together | When adding/removing infrastructure |
| `modules/*/main.tf` | Reusable resource definitions | When changing how a resource type works |
| `backend.tf` | Where Terraform state is stored | Rarely — only if changing state bucket |
---

## Step 3 — Create the Terraform state bucket (one-time only)

Terraform needs a place to store its state file. This bucket must exist before `terraform init`. Only needs to be run once — once ever.

```bash
gsutil mb -p savvio-ai -l us-east1 gs://savvio-ai-tf-state
```

If you get a "bucket already exists" error, verify if someone already created it.

---

## Step 4 — Deploy the dev environment

```bash
cd SavVio/deployment_pipeline/terraform/environments/dev

# Download provider plugins and configure backend
terraform init

# Preview what will be created (no changes made yet)
terraform plan

# Actually create the infrastructure
terraform apply
```

Terraform will show you a plan and ask `Do you want to perform these actions?`. Type `yes` and hit Enter.

**First run takes ~10-15 minutes** (Cloud SQL is slow to provision). Subsequent runs are fast.

---

## Step 5 — Verify the outputs

After `terraform apply` completes, it prints output values:

```
api_url               = "https://savvio-dev-api-xxxxx-ue.a.run.app"
frontend_url          = "https://savvio-dev-frontend-xxxxx-ue.a.run.app"
mlflow_url            = "https://savvio-dev-mlflow-xxxxx-ue.a.run.app"
db_connection_name    = "savvio-ai:us-east1:savvio-dev-db-instance"
dvc_bucket            = "savvio-dev-dvc-data"
mlflow_artifact_bucket = "savvio-dev-mlflow-artifacts"
docker_repo_url       = "us-east1-docker.pkg.dev/savvio-ai/savvio-dev-docker-repo"
```

You can re-print these anytime:

```bash
terraform output
```

## Step 6 — Deploying prod (when ready)

Same process, different folder:

```bash
cd SavVio/deployment_pipeline/terraform/environments/prod

terraform init
terraform plan
terraform apply
```

Prod creates the same resources but with stricter settings (IAM auth required, deletion protection on, always-on instances).

---

## How CI/CD Integrates

Your GitHub Actions workflow will:

1. Build Docker images
2. Push them to Artifact Registry
3. Run Terraform with the real image URIs

Example workflow step:

```yaml
- name: Deploy to dev
  run: |
    cd deployment_pipeline/terraform/environments/dev
    terraform init
    terraform apply -auto-approve \
      -var="api_image=us-east1-docker.pkg.dev/savvio-ai/savvio-dev-docker-repo/api:${{ github.sha }}" \
      -var="frontend_image=us-east1-docker.pkg.dev/savvio-ai/savvio-dev-docker-repo/frontend:${{ github.sha }}" \
      -var="mlflow_image=us-east1-docker.pkg.dev/savvio-ai/savvio-dev-docker-repo/mlflow:${{ github.sha }}"
```

---

## Tearing Down (dev only)

To destroy all dev infrastructure and stop billing:

```bash
cd SavVio/deployment_pipeline/terraform/environments/dev
terraform destroy
```

Type `yes` when prompted. This deletes everything: Cloud Run services, Cloud SQL, buckets, secrets, registry.

**Never run `terraform destroy` on prod** without team consensus.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `Error: bucket "savvio-ai-tf-state" does not exist` | Run Step 4 first |
| `Error: googleapi: Error 403: permission denied` | Run `gcloud auth application-default login` again |
| `Error: Cloud Run service placeholder image` | Expected — images deploy via CI/CD. Cloud Run will show "unhealthy" until real images are pushed |
| `Error: project "savvio-ai" not found` | Make sure the GCP project exists and you have Owner/Editor role |
| Terraform is stuck on Cloud SQL | Cloud SQL takes 10-15 min to provision. Be patient. |
| `Error: Secret already exists` | Another team member already applied. Run `terraform import` or coordinate who runs apply. |

---

