"""
src/utils/metrics.py
Evaluation helpers for stroke prediction models.

evaluate_model() returns a flat dict of floats that can be logged
directly to MLflow with mlflow.log_metrics().
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Run inference on *dataloader* and return metrics + confusion matrix.

    Returns
    -------
    metrics : dict
        Keys: accuracy, precision, recall, f1, auc_roc, auc_pr, n_samples.
    conf_matrix : np.ndarray of shape (2, 2)
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)

            output = model(inputs)
            # Handle models that return (pred, embedding) tuples
            if isinstance(output, tuple):
                output = output[0]

            probs = output.squeeze().cpu()
            all_probs.extend(probs.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels).astype(int)
    preds_arr  = (probs_arr >= threshold).astype(int)

    # Guard against degenerate splits (no positives in val set)
    has_both_classes = labels_arr.sum() > 0 and (1 - labels_arr).sum() > 0

    metrics: dict[str, float] = {
        "accuracy":  float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall":    float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "f1":        float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels_arr, probs_arr)) if has_both_classes else 0.0,
        "auc_pr":    float(average_precision_score(labels_arr, probs_arr)) if has_both_classes else 0.0,
        "n_samples": float(len(labels_arr)),
    }

    conf = confusion_matrix(labels_arr, preds_arr)
    return metrics, conf


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Pretty-print metric dict to stdout (used in train_baseline.py)."""
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}"
        f"Acc={metrics['accuracy']:.3f}  "
        f"Prec={metrics['precision']:.3f}  "
        f"Rec={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"AUC-ROC={metrics['auc_roc']:.3f}  "
        f"AUC-PR={metrics['auc_pr']:.3f}"
    )
