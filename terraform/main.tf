terraform {
  backend "s3" {
    bucket         = "fl-demo-terraform-state"
    key            = "eks/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fl-demo-terraform-locks"
    encrypt        = true
    # No profile — credentials come from GitHub Actions secrets
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars)
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
