module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  # Allow the CI/CD runner and current caller to manage the cluster
  cluster_endpoint_public_access = true

  # In terraform-aws-modules/eks v20+, the cluster creator is NOT automatically
  # granted K8s API access. This flag adds the Terraform caller (= the same IAM
  # identity used by GitHub Actions) as a cluster admin via the EKS Access API.
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    # Pluralsight sandbox limits:
    #   - Allowed types: t2/t3/t3a/t4g in micro, small, or medium only
    #   - Max 9 concurrent instances total across all node groups
    #   - Spot instances not supported
    system = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 2
      desired_size   = 2

      labels = { role = "system" }
    }

    training = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3  # Increased to allow scaling for parallel FL training
      desired_size   = 2

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

# ── Load Balancer permissions for Ingress Controller ──────────────────────────
# The NGINX Ingress Controller needs to create Network/Application Load Balancers
resource "aws_iam_policy" "node_elb" {
  name        = "${var.cluster_name}-node-elb-policy"
  description = "Allows EKS nodes to create and manage ELBs for ingress"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticloadbalancing:*",
          "ec2:DescribeAccountAttributes",
          "ec2:DescribeAddresses",
          "ec2:DescribeInternetGateways",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeSubnets",
          "ec2:DescribeVpcs",
          "ec2:DescribeInstances",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeTags",
          "ec2:CreateSecurityGroup",
          "ec2:CreateTags",
          "ec2:DeleteTags",
          "ec2:AuthorizeSecurityGroupIngress",
          "ec2:RevokeSecurityGroupIngress"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "node_elb_system" {
  policy_arn = aws_iam_policy.node_elb.arn
  role       = module.eks.eks_managed_node_groups["system"].iam_role_name
}

resource "aws_iam_role_policy_attachment" "node_elb_training" {
  policy_arn = aws_iam_policy.node_elb.arn
  role       = module.eks.eks_managed_node_groups["training"].iam_role_name
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
