resource "google_storage_bucket" "this" {
  name                        = var.bucket_name
  location                    = var.location
  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy
  labels                      = var.labels

  versioning {
    enabled = true
  }
}
