"""
src/train_baseline.py
Phase 3: Baseline model training (centralized + per-hospital local).

Runs three experiments and prints a comparison table:

  1. Centralized  — all hospitals pooled (upper bound for FL)
  2. Hospital 1   — local only
  3. Hospital 2   — local only
  4. Hospital 3   — local only

Results are logged to MLflow so they can be compared against FL runs.

Usage
-----
# Quick local run (uses local_dev/data/processed/ by default):
  python -m src.train_baseline

# Override data dir:
  DATA_OUT_DIR=data/processed python -m src.train_baseline

# Change model / epochs:
  python -m src.train_baseline --model embeddings --epochs 30

# Point at remote MLflow:
  MLFLOW_TRACKING_URI=http://mlflow-server.mlops:5000 python -m src.train_baseline
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.models.stroke_classifier import (
    StrokeNet,
    StrokeNetEmbeddings,
    get_parameters,
    set_parameters,
)
from src.preprocessing.pipeline import load_hospital_tensors
from src.utils.losses import make_criterion
from src.utils.metrics import evaluate_model, print_metrics


# ---------------------------------------------------------------------------
# Optional S3 model artifact upload
# ---------------------------------------------------------------------------
def _maybe_upload_to_s3(model: nn.Module, run_name: str) -> None:
    """Upload model state-dict to S3 when USE_S3_ARTIFACTS=1.

    Requires:
      - USE_S3_ARTIFACTS=1  (env var)
      - S3_MODEL_BUCKET     (env var, e.g. "fl-demo-models")
      - AWS credentials available via IRSA, instance profile, or env vars.

    The S3 key pattern is:  baseline/<run_name>/<UTC-timestamp>/model.pt
    The S3 URI is logged to MLflow as the param ``s3_model_uri`` so it can
    be retrieved later for FL initialisation or deployment.
    """
    if os.environ.get("USE_S3_ARTIFACTS", "0") != "1":
        return
    bucket = os.environ.get("S3_MODEL_BUCKET", "")
    if not bucket:
        print("⚠️  USE_S3_ARTIFACTS=1 but S3_MODEL_BUCKET is not set — skipping S3 upload")
        return
    try:
        import boto3  # included transitively via mlflow; add explicitly if needed

        region = os.environ.get("AWS_REGION", "us-east-1")
        s3 = boto3.client("s3", region_name=region)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"baseline/{run_name}/{ts}/model.pt"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            s3.upload_file(tmp.name, bucket, key)

        s3_uri = f"s3://{bucket}/{key}"
        print(f"✅ Model uploaded → {s3_uri}")
        # Log alongside the MLflow run so the URI is discoverable in the UI
        mlflow.log_param("s3_model_uri", s3_uri)
    except Exception as exc:
        # Non-fatal: training already succeeded; S3 is supplementary storage
        print(f"⚠️  S3 upload failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Full training run with MLflow tracking
# ---------------------------------------------------------------------------
def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str,
    *,
    epochs: int = 30,
    lr: float = 0.001,
    loss_type: str = "weighted_bce",
    device: str = "cpu",
    extra_params: dict | None = None,
) -> dict:
    """Train *model*, log to MLflow, return final val metrics."""

    criterion = make_criterion(loss_type, pos_weight=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model     = model.to(device)

    # Single top-level run — no nesting so metrics are visible directly in UI
    with mlflow.start_run(run_name=run_name):
        params = {
            "model":       model.__class__.__name__,
            "epochs":      epochs,
            "lr":          lr,
            "loss":        loss_type,
            "train_size":  len(train_loader.dataset),
            "val_size":    len(val_loader.dataset),
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            val_metrics, _ = evaluate_model(model, val_loader, device=device)
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )

            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                      f"acc={val_metrics['accuracy']:.3f}  f1={val_metrics['f1']:.3f}")

        # Restore best checkpoint and log it
        if best_state:
            model.load_state_dict(best_state)

        final_metrics, conf = evaluate_model(model, val_loader, device=device)
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        mlflow.log_text(str(conf), "confusion_matrix.txt")
        # artifact_path= is the MLflow 2.x API; compatible with server v2.22.0
        mlflow.pytorch.log_model(model, artifact_path="model")
        # Optionally push the state-dict to S3 for downstream FL initialisation
        _maybe_upload_to_s3(model, run_name)

    return final_metrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_all_hospitals(batch_size: int, data_dir: str | None = None):
    """Load all 3 hospital datasets and return concatenated loaders + per-hospital loaders."""
    loaders = []
    input_dim = None
    for hid in range(1, 4):
        tr, va, dim = load_hospital_tensors(hid, data_dir, batch_size=batch_size)
        loaders.append((tr, va))
        input_dim = dim  # same input_dim for all hospitals

    # Centralized: concatenate all train sets, use hospital-1 val as proxy
    all_train = ConcatDataset([tr.dataset for tr, _ in loaders])
    all_val   = ConcatDataset([va.dataset for _, va in loaders])
    central_train = DataLoader(all_train, batch_size=batch_size, shuffle=True)
    central_val   = DataLoader(all_val,   batch_size=batch_size)

    return central_train, central_val, loaders, input_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3 baseline training")
    parser.add_argument("--model",   default="simple",
                        choices=["simple", "embeddings"],
                        help="Model architecture")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--lr",      type=float, default=0.001)
    parser.add_argument("--loss",    default="weighted_bce",
                        choices=["bce", "weighted_bce", "focal"])
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--data_dir", default=None,
                        help="Directory containing hospital_N.csv files")
    args = parser.parse_args()

    # MLflow setup
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("stroke-prediction-baseline")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  MLflow: {mlflow_uri}")

    # Load data
    central_train, central_val, hospital_loaders, input_dim = load_all_hospitals(
        args.batch, args.data_dir
    )
    print(f"Input features: {input_dim}")

    def make_model():
        if args.model == "embeddings":
            return StrokeNetEmbeddings(input_dim)
        return StrokeNet(input_dim)

    results: dict[str, dict] = {}

    # ── 1. Centralized baseline ─────────────────────────────────────────
    print("\n=== Centralized (all hospitals pooled) ===")
    m = run_training(
        make_model(), central_train, central_val,
        run_name="baseline_centralized",
        epochs=args.epochs, lr=args.lr, loss_type=args.loss, device=device,
        extra_params={"experiment_type": "centralized", "hospitals": "all"},
    )
    print_metrics(m, "Centralized")
    results["centralized"] = m

    # ── 2. Per-hospital local baselines ─────────────────────────────────
    for hid, (tr, va) in enumerate(hospital_loaders, start=1):
        label = f"local_hospital_{hid}"
        print(f"\n=== Hospital {hid} only ===")
        m = run_training(
            make_model(), tr, va,
            run_name=label,
            epochs=args.epochs, lr=args.lr, loss_type=args.loss, device=device,
            extra_params={"experiment_type": "local", "hospital_id": hid},
        )
        print_metrics(m, f"Hospital {hid}")
        results[label] = m

    # ── Comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Experiment':<30} {'Accuracy':>8} {'F1':>6} {'AUC-ROC':>8} {'Recall':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<30} {m['accuracy']:>8.3f} {m['f1']:>6.3f} "
              f"{m['auc_roc']:>8.3f} {m['recall']:>8.3f}")
    print("=" * 70)

    gap = results["centralized"]["accuracy"] - min(
        v["accuracy"] for k, v in results.items() if k != "centralized"
    )
    print(f"\nAccuracy gap (centralized vs worst local): {gap:.1%}")
    print("This gap motivates federated learning — FL should close most of it.")
    print(f"\nView in MLflow:  mlflow ui --backend-store-uri {mlflow_uri}")


if __name__ == "__main__":
    main()
