variable "repository_id" {
  description = "Artifact Registry repository ID"
  type        = string
}

variable "location" {
  description = "Repository location"
  type        = string
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}
