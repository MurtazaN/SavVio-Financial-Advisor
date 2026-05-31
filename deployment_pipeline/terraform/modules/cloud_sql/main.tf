resource "random_password" "db_password" {
  length           = 24
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "google_sql_database_instance" "this" {
  name                = var.instance_name
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.tier
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"

    backup_configuration {
      enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
    }

    user_labels = var.labels
  }
}

resource "google_sql_database" "this" {
  name     = var.database_name
  instance = google_sql_database_instance.this.name
}

resource "google_sql_user" "this" {
  name     = var.user_name
  instance = google_sql_database_instance.this.name
  password = random_password.db_password.result
}

# pgvector is pre-installed on Cloud SQL PostgreSQL 15+.
# After provisioning, run: CREATE EXTENSION IF NOT EXISTS vector;
