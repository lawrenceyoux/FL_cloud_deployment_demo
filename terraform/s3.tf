locals {
  hospital_ids = ["hospital-1", "hospital-2", "hospital-3"]
}

# ── Per-hospital data buckets ─────────────────────────────────────────────────
resource "aws_s3_bucket" "hospital_data" {
  for_each = toset(local.hospital_ids)
  bucket   = "fl-demo-data-${each.key}"

  tags = { Project = "fl-demo", DataOwner = each.key }
}

resource "aws_s3_bucket_versioning" "hospital_data" {
  for_each = aws_s3_bucket.hospital_data
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}

# Block all public access on hospital data buckets
resource "aws_s3_bucket_public_access_block" "hospital_data" {
  for_each                = aws_s3_bucket.hospital_data
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Model artefacts bucket ────────────────────────────────────────────────────
resource "aws_s3_bucket" "models" {
  bucket = "fl-demo-models"
  tags   = { Project = "fl-demo" }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

# ── MLflow artefacts bucket ───────────────────────────────────────────────────
resource "aws_s3_bucket" "mlflow" {
  bucket = "fl-demo-mlflow"
  tags   = { Project = "fl-demo" }
}
