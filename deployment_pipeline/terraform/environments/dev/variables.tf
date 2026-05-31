variable "project_id"      { type = string }
variable "region"          { type = string }
variable "zone"            { type = string }
variable "environment"     { type = string }
variable "db_tier"         { type = string }
variable "api_image"       { type = string }
variable "frontend_image"  { type = string }
variable "mlflow_image"    { type = string }

# ---------------------------------------------------------------------------
# Grafana Cloud monitoring
# Pass via TF_VAR_* env vars or -var flags — never commit secrets to tfvars
# ---------------------------------------------------------------------------
variable "grafana_remote_write_url" {
  description = "Grafana Cloud Prometheus remote-write URL"
  type        = string
  default     = ""
}

variable "grafana_cloud_username" {
  description = "Grafana Cloud Prometheus instance numeric ID"
  type        = string
  default     = ""
}

variable "grafana_api_key" {
  description = "Grafana Cloud API key for remote-write authentication"
  type        = string
  sensitive   = true
  default     = ""
}
