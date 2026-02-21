variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile"
  default     = "fl-demo"
}

variable "cluster_name" {
  description = "EKS cluster name"
  default     = "fl-demo-cluster"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  default     = "10.0.0.0/16"
}
