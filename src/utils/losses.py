"""
src/utils/losses.py
Loss functions for imbalanced stroke classification.

The stroke dataset is ~5% positive → naive BCE gives ~95% accuracy by
predicting "no stroke" for everything.  Two strategies provided:

  1. FocalLoss   — down-weights easy negatives, focuses on hard positives.
  2. WeightedBCE — the simpler approach: scale positive-class loss by pos_weight.

local_dev/simulate.py uses pos_weight=20 inline — this module packages
the same idea cleanly for use in train_baseline.py and FL clients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced datasets.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for the positive class (default: 0.25).
        gamma: Focusing parameter — higher values down-weight easy examples
               more aggressively (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt  = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce
        return focal.mean()


class WeightedBCE(nn.Module):
    """BCELoss that up-weights the positive (stroke=1) class.

    Equivalent to local_dev/simulate.py's manual pos_weight approach,
    but wrapped as a reusable Module.

    Args:
        pos_weight: Multiplier for positive-class loss.
                    Rule of thumb: n_negative / n_positive (≈ 20 for stroke data).
    """

    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.where(targets == 1,
                              torch.tensor(self.pos_weight, device=inputs.device),
                              torch.ones(1, device=inputs.device))
        return F.binary_cross_entropy(inputs, targets, weight=weights, reduction="mean")


def make_criterion(loss_type: str = "weighted_bce", pos_weight: float = 20.0) -> nn.Module:
    """Factory helper — matches the loss_type string to a Module.

    Used by train_baseline.py so the loss can be set via CLI arg.
    """
    if loss_type == "focal":
        return FocalLoss()
    if loss_type == "weighted_bce":
        return WeightedBCE(pos_weight=pos_weight)
    if loss_type == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unknown loss type: {loss_type}.  Choose: focal | weighted_bce | bce")
