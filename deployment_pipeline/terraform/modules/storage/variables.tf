variable "bucket_name" {
  description = "GCS bucket name (must be globally unique)"
  type        = string
}

variable "location" {
  description = "Bucket location"
  type        = string
}

variable "force_destroy" {
  description = "Allow bucket deletion even if non-empty"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}
