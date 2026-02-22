"""
src/federated/client.py
Flower FL client for cloud (EKS) deployment.

This is the cloud-packaged version of local_dev/client.py.
Algorithm is identical; configuration comes from env vars / CLI args so
Kubernetes ConfigMap values are injected automatically.

Environment variables (set via K8s ConfigMap fl-config):
  HOSPITAL_ID       — integer 1, 2, or 3  (required)
  SERVER_ADDRESS    — host:port of fl-server Service (default: fl-server:8080)
  DATA_OUT_DIR      — directory containing hospital_N.csv files
  LOCAL_EPOCHS      — local training epochs per round (default: 3)
  BATCH_SIZE        — (default: 32)
  LEARNING_RATE     — (default: 0.001)
  MLFLOW_TRACKING_URI — if set, per-client MLflow logging is enabled

Usage (Docker container / K8s pod):
  python -m src.federated.client

Usage (local test via src/ rather than local_dev/):
  HOSPITAL_ID=1 python -m src.federated.client
"""

from __future__ import annotations

import os

import flwr as fl
import mlflow
import torch
import torch.nn as nn

from src.models.stroke_classifier import StrokeNet, get_parameters, set_parameters
from src.preprocessing.pipeline import load_hospital_tensors
from src.utils.losses import make_criterion
from src.utils.metrics import evaluate_model


# ---------------------------------------------------------------------------
# Configuration from env (mirrors local_dev/client.py constants)
# ---------------------------------------------------------------------------
HOSPITAL_ID   = int(os.environ.get("HOSPITAL_ID", "1"))
SERVER_ADDRESS  = os.environ.get("SERVER_ADDRESS", "fl-server:8080")
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", "3"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "32"))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE", "0.001"))
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------
class HospitalClient(fl.client.NumPyClient):
    """Flower NumPyClient representing one hospital.

    Mirrors local_dev/client.py::HospitalClient exactly.  The only
    additions are:
      - Configurable via env vars instead of CLI flags
      - Optional per-round MLflow metric logging
    """

    def __init__(self, hospital_id: int):
        self.id = hospital_id
        self.train_loader, self.val_loader, input_dim = load_hospital_tensors(
            hospital_id, batch_size=BATCH_SIZE
        )
        self.model     = StrokeNet(input_dim)
        self.criterion = make_criterion("weighted_bce", pos_weight=20.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    def get_parameters(self, config: dict) -> list:
        return get_parameters(self.model)

    # ------------------------------------------------------------------
    def fit(self, parameters: list, config: dict) -> tuple:
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))

        self.model.train()
        for _ in range(local_epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X).squeeze(), y)
                loss.backward()
                self.optimizer.step()

        print(f"[Hospital #{self.id}] Local training complete ({local_epochs} epochs)")
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    # ------------------------------------------------------------------
    def evaluate(self, parameters: list, config: dict) -> tuple:
        set_parameters(self.model, parameters)
        self.model.eval()

        metrics, _ = evaluate_model(self.model, self.val_loader, device="cpu")
        loss_sum   = 0.0
        criterion  = nn.BCELoss()

        with torch.no_grad():
            for X, y in self.val_loader:
                out       = self.model(X).squeeze()
                loss_sum += criterion(out, y).item()
        avg_loss = loss_sum / len(self.val_loader)

        if MLFLOW_URI:
            mlflow.log_metrics(
                {f"hospital_{self.id}_{k}": v for k, v in metrics.items()},
            )

        print(
            f"[Hospital #{self.id}] "
            f"Acc={metrics['accuracy']:.1%}  "
            f"F1={metrics['f1']:.3f}  "
            f"AUC={metrics['auc_roc']:.3f}"
        )
        return avg_loss, int(metrics["n_samples"]), {
            "accuracy": metrics["accuracy"],
            "f1":       metrics["f1"],
            "auc_roc":  metrics["auc_roc"],
            "hospital_id": str(self.id),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("federated-stroke-prediction")

    client = HospitalClient(HOSPITAL_ID)
    print(f"[Hospital #{HOSPITAL_ID}] Connecting to FL server at {SERVER_ADDRESS}")
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)


if __name__ == "__main__":
    main()
