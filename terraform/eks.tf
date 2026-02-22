module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.30"

  # Allow the CI/CD runner and current caller to manage the cluster
  cluster_endpoint_public_access = true

  # In terraform-aws-modules/eks v20+, the cluster creator is NOT automatically
  # granted K8s API access. This flag adds the Terraform caller (= the same IAM
  # identity used by GitHub Actions) as a cluster admin via the EKS Access API.
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  # Include both public and private subnets so LoadBalancers can be public
  # Nodes remain in private subnets, but AWS can create ELBs in public subnets
  subnet_ids = concat(module.vpc.private_subnets, module.vpc.public_subnets)

  eks_managed_node_groups = {
    # Pluralsight sandbox limits:
    #   - Allowed types: t2/t3/t3a/t4g in micro, small, or medium only
    #   - Max 9 concurrent instances total across all node groups
    #   - Spot instances not supported
    system = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3
      desired_size   = 1

      # Nodes must be in private subnets only; they reach the internet via the
      # NAT gateway so map_public_ip_on_launch is not required (and private
      # subnets don't have it set, which caused CREATE_FAILED when the cluster-
      # level subnet_ids included public subnets).
      subnet_ids = module.vpc.private_subnets

      labels = { role = "system" }
    }

    training = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3  # Increased to allow scaling for parallel FL training
      desired_size   = 2

      # Same reasoning as above — private subnets + NAT, no public IP needed.
      subnet_ids = module.vpc.private_subnets

      labels = { role = "training" }
    }
  }

  # IRSA: enables pods to assume IAM roles via Kubernetes service accounts
  enable_irsa = true

  tags = {
    Project = "fl-demo"
  }
}

# ── EBS CSI policy on node group roles ───────────────────────────────────────
# The EBS CSI driver runs on the nodes and needs EC2 permissions to create /
# attach / detach volumes. Attaching AmazonEBSCSIDriverPolicy to both node
# group roles is the simplest approach for a demo cluster (no IRSA needed).
resource "aws_iam_role_policy_attachment" "ebs_csi_system" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = module.eks.eks_managed_node_groups["system"].iam_role_name
}

resource "aws_iam_role_policy_attachment" "ebs_csi_training" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = module.eks.eks_managed_node_groups["training"].iam_role_name
}
# ── S3 model artifact access for training pods ────────────────────────────
# Grants the training node IAM role read/write access to fl-demo-models.
# Training pods (which run on training nodes) inherit this via the node
# instance profile — no IRSA / service account annotation needed for demo.
resource "aws_iam_role_policy" "training_s3_models" {
  name = "fl-training-s3-models"
  role = module.eks.eks_managed_node_groups["training"].iam_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "FlDemoModelsBucket"
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::fl-demo-models",
        "arn:aws:s3:::fl-demo-models/*"
      ]
    }]
  })
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
