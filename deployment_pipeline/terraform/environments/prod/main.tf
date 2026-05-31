terraform {
  required_version = ">= 1.14.7"
  required_providers {
    google = { source = "hashicorp/google"
      version = "~> 7.25.0" }
    random = { source = "hashicorp/random"
      version = "~> 3.8.1" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "this" {}

locals {
  prefix = "savvio-${var.environment}"
  labels = {
    project     = "savvio"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---- APIs ----
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

# ---- Service Account ----
resource "google_service_account" "cloud_run" {
  account_id   = "${local.prefix}-run-sa"
  display_name = "SavVio ${var.environment} Cloud Run SA"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "run_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_storage_bucket_iam_member" "run_mlflow_storage" {
  bucket = module.mlflow_bucket.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_storage_bucket_iam_member" "run_dvc_storage" {
  bucket = module.dvc_bucket.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# ---- Cloud SQL ----
module "database" {
  source              = "../../modules/cloud_sql"
  instance_name       = "${local.prefix}-db-instance"
  region              = var.region
  tier                = var.db_tier
  database_name       = "${local.prefix}-db"
  user_name           = "${var.environment}-db-admin"
  deletion_protection = true   # PROD: protect
  labels              = local.labels
  depends_on          = [google_project_service.apis]
}

# ---- Secrets ----
module "db_password_secret" {
  source                         = "../../modules/secrets"
  secret_id                      = "${local.prefix}-db-password"
  secret_data                    = module.database.password
  accessor_service_account_email = google_service_account.cloud_run.email
  depends_on                     = [google_project_service.apis]
}

# ---- Storage ----
module "dvc_bucket" {
  source        = "../../modules/storage"
  bucket_name   = "${local.prefix}-dvc-data"
  location      = var.region
  force_destroy = false   # PROD: protect data
  labels        = local.labels
}

module "mlflow_bucket" {
  source        = "../../modules/storage"
  bucket_name   = "${local.prefix}-mlflow-artifacts"
  location      = var.region
  force_destroy = false
  labels        = local.labels
}

# ---- Artifact Registry ----
module "docker_repo" {
  source        = "../../modules/artifact_registry"
  repository_id = "${local.prefix}-docker-repo"
  location      = var.region
  labels        = local.labels
  depends_on    = [google_project_service.apis]
}

# ---- GCE VM (Airflow + ML Training) ----
resource "google_compute_instance" "pipeline_vm" {
  name         = "${local.prefix}-pipeline-vm"
  machine_type = "e2-standard-4"  # prod: more CPU for faster training
  zone         = var.zone
  labels       = local.labels
  tags         = ["ssh-access"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 100
    }
  }

  network_interface {
    network = "default"
    access_config {}  # ephemeral public IP for SSH
  }

  service_account {
    email  = google_service_account.cloud_run.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update && apt-get install -y docker.io docker-compose-plugin git
    systemctl enable docker
    systemctl start docker
    usermod -aG docker github-actions 2>/dev/null || true
  EOF

  depends_on = [google_project_service.apis]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${local.prefix}-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-access"]
}

# Airflow apiserver on pipeline VM (port 8080).
resource "google_compute_firewall" "allow_airflow_ui" {
  name    = "${local.prefix}-allow-airflow-ui"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-access"]
}

# ---- Cloud Run: API ----
module "api" {
  source                = "../../modules/cloud_run"
  service_name          = "${local.prefix}-api"
  region                = var.region
  image                 = var.api_image
  port                  = 8080
  service_account_email = google_service_account.cloud_run.email
  cloud_sql_connection  = module.database.connection_name
  public_access         = false   # PROD: require IAM auth
  min_instances         = 1       # PROD: no cold starts
  max_instances         = 5
  cpu                   = "2"
  memory                = "2Gi"
  labels                = local.labels

  env_vars = {
    ENVIRONMENT              = var.environment
    DB_USER                  = module.database.user_name
    DB_NAME                  = module.database.database_name
    INSTANCE_CONNECTION_NAME = module.database.connection_name
    MLFLOW_TRACKING_URI      = module.mlflow.service_url
  }
  secret_env_vars = {
    DB_PASS = { secret_id = module.db_password_secret.secret_id, version = "latest" }
  }

  depends_on = [google_project_service.apis, module.db_password_secret, module.mlflow]
}

# ---- Cloud Run: Frontend ----
module "frontend" {
  source                = "../../modules/cloud_run"
  service_name          = "${local.prefix}-frontend"
  region                = var.region
  image                 = var.frontend_image
  port                  = 8501
  service_account_email = google_service_account.cloud_run.email
  cloud_sql_connection  = ""
  public_access         = true    # PROD: user-facing
  min_instances         = 1
  max_instances         = 3
  cpu                   = "1"
  memory                = "512Mi"
  labels                = local.labels

  env_vars = {
    ENVIRONMENT = var.environment
    API_URL     = module.api.service_url
  }

  depends_on = [google_project_service.apis, module.api]
}

# ---- Cloud Run: MLflow ----
module "mlflow" {
  source                = "../../modules/cloud_run"
  service_name          = "${local.prefix}-mlflow"
  region                = var.region
  image                 = var.mlflow_image
  port                  = 5000
  service_account_email = google_service_account.cloud_run.email
  cloud_sql_connection  = module.database.connection_name
  public_access         = false   # PROD: internal only
  min_instances         = 1
  max_instances         = 2
  cpu                   = "1"
  memory                = "1Gi"
  labels                = local.labels

  env_vars = {
    ENVIRONMENT              = var.environment
    DB_USER                  = module.database.user_name
    DB_NAME                  = module.database.database_name
    INSTANCE_CONNECTION_NAME = module.database.connection_name
    MLFLOW_ARTIFACT_ROOT     = "gs://${module.mlflow_bucket.bucket_name}/artifacts"
  }
  secret_env_vars = {
    DB_PASS = { secret_id = module.db_password_secret.secret_id, version = "latest" }
  }

  depends_on = [google_project_service.apis, module.db_password_secret]
}

# ---- Service-to-service IAM ----
resource "google_cloud_run_v2_service_iam_member" "frontend_invokes_api" {
  name     = module.api.service_name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_cloud_run_v2_service_iam_member" "api_invokes_mlflow" {
  name     = module.mlflow.service_name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run.email}"
}
