module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  # Allow the CI/CD runner and current caller to manage the cluster
  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    system = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 5
      desired_size   = 2

      labels = { role = "system" }
    }

    training = {
      instance_types = ["c5.2xlarge"]
      min_size       = 3
      max_size       = 10
      desired_size   = 3

      labels = { role = "training" }

      # Uncomment to use Spot instances and cut costs ~70%
      # capacity_type = "SPOT"
    }
  }

  # IRSA: enables pods to assume IAM roles via Kubernetes service accounts
  enable_irsa = true

  tags = {
    Project = "fl-demo"
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}
