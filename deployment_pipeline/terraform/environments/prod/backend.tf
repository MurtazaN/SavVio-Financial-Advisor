terraform {
  backend "gcs" {
    bucket = "savvio-ai-tf-state"
    prefix = "env/prod"
  }
}
