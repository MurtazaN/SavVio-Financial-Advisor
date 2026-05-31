variable "service_name" {
  description = "Cloud Run service name"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "image" {
  description = "Container image URI"
  type        = string
}

variable "port" {
  description = "Container port"
  type        = number
  default     = 8080
}

variable "env_vars" {
  description = "Plain-text environment variables"
  type        = map(string)
  default     = {}
}

variable "secret_env_vars" {
  description = "Secret-backed environment variables"
  type = map(object({
    secret_id = string
    version   = string
  }))
  default = {}
}

variable "cloud_sql_connection" {
  description = "Cloud SQL connection name (empty string to skip)"
  type        = string
  default     = ""
}

variable "service_account_email" {
  description = "Service account for the Cloud Run service"
  type        = string
}

variable "public_access" {
  description = "Allow unauthenticated access"
  type        = bool
  default     = false
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 2
}

variable "cpu" {
  description = "CPU allocation"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory allocation"
  type        = string
  default     = "512Mi"
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------------------
# Monitoring (Prometheus / Grafana Cloud)
# ---------------------------------------------------------------------------

variable "metrics_enabled" {
  description = "Enable Prometheus metrics on the API container"
  type        = bool
  default     = true
}

variable "grafana_remote_write_url" {
  description = "Grafana Cloud Prometheus remote-write URL"
  type        = string
  default     = ""
}

variable "grafana_cloud_username" {
  description = "Grafana Cloud Prometheus instance username"
  type        = string
  default     = ""
}

variable "grafana_cloud_api_key_secret" {
  description = "Secret Manager secret ID for Grafana Cloud API key"
  type        = string
  default     = ""
}
