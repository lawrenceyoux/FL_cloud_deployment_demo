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
      max_size       = 2
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

# ── IRSA: EBS CSI Driver ─────────────────────────────────────────────────────
# The AWS EBS CSI driver requires an IAM role bound to its Kubernetes
# service account (ebs-csi-controller-sa in kube-system) via IRSA.
# Without this the controller pod crashes with "AccessDenied" on EC2 calls.
data "aws_iam_policy_document" "ebs_csi_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.cluster_oidc_issuer_url}:sub"
      values   = ["system:serviceaccount:kube-system:ebs-csi-controller-sa"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.cluster_oidc_issuer_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi_driver" {
  name               = "fl-demo-ebs-csi-driver"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume.json
  tags = { Project = "fl-demo" }
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  role       = aws_iam_role.ebs_csi_driver.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
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

output "ebs_csi_driver_role_arn" {
  description = "IAM role ARN for the EBS CSI driver IRSA binding"
  value       = aws_iam_role.ebs_csi_driver.arn
}
