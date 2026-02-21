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

Usage (Docker / K8s):
  python -m src.federated.server

Usage (local test via src/):
  NUM_ROUNDS=5 MIN_CLIENTS=3 python -m src.federated.server
"""

from __future__ import annotations

import os
from datetime import datetime
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
# Strategy factory
# ---------------------------------------------------------------------------
def make_strategy(name: str) -> fl.server.strategy.Strategy:
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
        return fl.server.strategy.FedProx(proximal_mu=FEDPROX_MU, **common)
    # default: FedAvg
    return fl.server.strategy.FedAvg(**common)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("federated-stroke-prediction")

    strategy = make_strategy(STRATEGY_NAME)
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

        # Log round-level metrics to MLflow
        for rnd, metrics_list in history.metrics_distributed.items():
            round_metrics = {}
            for key, value_list in metrics_list:
                round_metrics[f"distributed_{key}"] = value_list
            if round_metrics:
                mlflow.log_metrics(round_metrics, step=rnd)

        # Log final aggregated accuracy if available
        if history.metrics_distributed_fit:
            last_round   = max(history.metrics_distributed_fit.keys())
            final_metric = history.metrics_distributed_fit[last_round]
            print(f"Final round {last_round} metrics: {final_metric}")

    print("FL training complete.")


if __name__ == "__main__":
    main()
