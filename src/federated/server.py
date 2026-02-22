"""
src/federated/server.py
Flower FL server for cloud (EKS) deployment.

Cloud-packaged version of local_dev/server.py.
Rounds, min-clients, and MLflow URI come from env vars (K8s ConfigMap fl-config).

Environment variables:
  NUM_ROUNDS             — FL rounds (default: 10)
  MIN_CLIENTS            — minimum available clients (default: 3)
  SERVER_HOST            — bind address (default: 0.0.0.0)
  SERVER_PORT            — bind port (default: 8080)
  MLFLOW_TRACKING_URI    — MLflow server URI (default: ./mlruns for local runs)
  STRATEGY               — fedavg | fedprox (default: fedavg)
  FEDPROX_MU             — proximal term for FedProx (default: 0.01)
  USE_S3_ARTIFACTS       — set "1" to upload final global model to S3 (default: 0)
  S3_MODEL_BUCKET        — S3 bucket name for global model (e.g. fl-demo-models)
  INPUT_DIM              — feature dimension for model reconstruction (default: 20)
  AWS_REGION             — AWS region for S3 upload (default: us-east-1)

Usage (Docker / K8s):
  python -m src.federated.server

Usage (local test via src/):
  NUM_ROUNDS=5 MIN_CLIENTS=3 python -m src.federated.server
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import flwr as fl
import mlflow
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ROUNDS    = int(os.environ.get("NUM_ROUNDS", "10"))
MIN_CLIENTS   = int(os.environ.get("MIN_CLIENTS", "3"))
SERVER_HOST   = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT   = os.environ.get("SERVER_PORT", "8080")
MLFLOW_URI    = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
STRATEGY_NAME = os.environ.get("STRATEGY", "fedavg")
FEDPROX_MU    = float(os.environ.get("FEDPROX_MU", "0.01"))
INPUT_DIM     = int(os.environ.get("INPUT_DIM", "20"))

SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Weighted average of client metrics by number of evaluation samples.

    Mirrors local_dev/simulate.py::fedavg logic.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    agg: Dict[str, float] = {}
    for n, m in metrics:
        w = n / total
        for k, v in m.items():
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0.0) + w * float(v)
    return agg


# ---------------------------------------------------------------------------
# Strategy with parameter capture
# Wraps FedAvg/FedProx to store the last aggregated numpy arrays so the
# server can save the final global model to S3 after training completes.
# ---------------------------------------------------------------------------
class _CaptureFinalParams:
    """Mixin: captures aggregated params from the last completed round."""

    final_params: Optional[List[np.ndarray]] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)  # type: ignore[misc]
        if aggregated[0] is not None:
            self.final_params = fl.common.parameters_to_ndarrays(aggregated[0])
            print(f"[server] Round {server_round}: aggregated {len(self.final_params)} param tensors")
        return aggregated


class FedAvgCapture(_CaptureFinalParams, fl.server.strategy.FedAvg):
    """FedAvg that retains the final global-model parameters."""
    pass


class FedProxCapture(_CaptureFinalParams, fl.server.strategy.FedProx):
    """FedProx that retains the final global-model parameters."""
    pass


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------
def make_strategy(name: str) -> _CaptureFinalParams:
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    if name == "fedprox":
        return FedProxCapture(proximal_mu=FEDPROX_MU, **common)
    return FedAvgCapture(**common)


# ---------------------------------------------------------------------------
# S3 global-model upload
# Called after start_server() returns.  Reconstructs the StrokeNet from the
# captured numpy parameter arrays and uploads the state_dict as model.pt.
# ---------------------------------------------------------------------------
def _upload_global_model_to_s3(
    params_ndarrays: List[np.ndarray],
    run_name: str,
    input_dim: int,
) -> None:
    """Upload the final global model weights to S3.

    Non-fatal: prints a clear warning if upload fails so the GHA step can
    detect it (the training result is already safe in MLflow).
    """
    if os.environ.get("USE_S3_ARTIFACTS", "0") != "1":
        print("[server] USE_S3_ARTIFACTS is not '1' — skipping S3 upload")
        return
    bucket = os.environ.get("S3_MODEL_BUCKET", "")
    if not bucket:
        print("⚠️  USE_S3_ARTIFACTS=1 but S3_MODEL_BUCKET not set — skipping S3 upload")
        return
    try:
        import boto3
        import torch
        from src.models.stroke_classifier import StrokeNet
        from src.models.stroke_classifier import set_parameters as sp

        model = StrokeNet(input_dim)
        sp(model, params_ndarrays)

        region = os.environ.get("AWS_REGION", "us-east-1")
        s3_client = boto3.client("s3", region_name=region)
        ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"global-model/{run_name}/{ts}/model.pt"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            s3_client.upload_file(tmp.name, bucket, key)

        s3_uri = f"s3://{bucket}/{key}"
        print(f"✅ Global FL model uploaded → {s3_uri}")
        # Surface the URI in MLflow so it is discoverable from the UI
        mlflow.log_param("s3_global_model_uri", s3_uri)

    except Exception as exc:
        # Print clearly so the GHA workflow step can grep for the failure
        print(f"❌ S3 global-model upload FAILED: {exc}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("federated-stroke-prediction")

    strategy  = make_strategy(STRATEGY_NAME)
    run_name  = f"fl_{STRATEGY_NAME}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    print(
        f"Starting FL server  "
        f"strategy={STRATEGY_NAME}  rounds={NUM_ROUNDS}  "
        f"min_clients={MIN_CLIENTS}  address={SERVER_ADDRESS}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "fl_framework":  "flower",
            "strategy":      STRATEGY_NAME,
            "num_rounds":    NUM_ROUNDS,
            "min_clients":   MIN_CLIENTS,
        })

        history = fl.server.start_server(
            server_address=SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

        # ── Log per-round metrics to MLflow ──────────────────────────
        for rnd, metrics_list in history.metrics_distributed.items():
            round_metrics = {}
            for key, value_list in metrics_list:
                round_metrics[f"distributed_{key}"] = value_list
            if round_metrics:
                mlflow.log_metrics(round_metrics, step=rnd)

        # ── Print final round summary ────────────────────────────────
        if history.metrics_distributed_fit:
            last_round   = max(history.metrics_distributed_fit.keys())
            final_metric = history.metrics_distributed_fit[last_round]
            print(f"Final round {last_round} metrics: {final_metric}")

        # ── Upload final global model to S3 ─────────────────────────
        if strategy.final_params is not None:
            _upload_global_model_to_s3(strategy.final_params, run_name, INPUT_DIM)
        else:
            print("⚠️  No aggregated parameters captured — S3 upload skipped")

    print("FL server finished all rounds.")


if __name__ == "__main__":
    main()
