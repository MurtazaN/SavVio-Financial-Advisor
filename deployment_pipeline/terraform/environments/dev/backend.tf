terraform {
  backend "gcs" {
    bucket = "savvio-purchase-guardrail-tf-state"
    prefix = "env/dev"
  }
}
