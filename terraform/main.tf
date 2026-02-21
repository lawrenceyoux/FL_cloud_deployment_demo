terraform {
  backend "s3" {
    bucket         = "fl-demo-terraform-state"
    key            = "eks/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fl-demo-terraform-locks"
    encrypt        = true
    profile        = "fl-demo"
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
  region  = var.aws_region
  profile = var.aws_profile
}
