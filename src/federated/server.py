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
    Also surfaces total sample count as ``n_samples`` in the returned dict.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    agg: Dict[str, float] = {"n_samples": float(total)}
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
# ---------------------------------------------------------------------------
# ASCII chart helpers
# ---------------------------------------------------------------------------
def _sparkbar(value: float, lo: float, hi: float, width: int = 30) -> str:
    """Return a filled bar proportional to value within [lo, hi]."""
    span  = max(hi - lo, 1e-9)
    filled = int(round((value - lo) / span * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _print_round_summary(
    rnd: int,
    num_rounds: int,
    eval_metrics: Dict[str, float],
    fit_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Print a compact per-round summary line after each evaluation round."""
    acc        = eval_metrics.get("accuracy",  float("nan"))
    auc_roc    = eval_metrics.get("auc_roc",   float("nan"))
    auc_pr     = eval_metrics.get("auc_pr",    float("nan"))
    f1         = eval_metrics.get("f1",        float("nan"))
    prec       = eval_metrics.get("precision", float("nan"))
    rec        = eval_metrics.get("recall",    float("nan"))
    n_samples  = eval_metrics.get("n_samples", float("nan"))
    train_loss = (fit_metrics or {}).get("train_loss", float("nan"))

    bar      = _sparkbar(auc_roc, 0.5, 1.0, width=20)
    tl_s     = f"{train_loss:.4f}" if train_loss == train_loss else "  n/a "
    pr_s     = f"{prec:.3f}"       if prec       == prec       else " n/a"
    rec_s    = f"{rec:.3f}"        if rec        == rec        else " n/a"
    aupr_s   = f"{auc_pr:.3f}"    if auc_pr     == auc_pr     else " n/a"
    ns_s     = f"{int(n_samples)}" if n_samples  == n_samples  else "n/a"
    print(
        f"  Round {rnd:>2}/{num_rounds}  "
        f"train_loss={tl_s}  "
        f"Acc={acc:.1%}  "
        f"Prec={pr_s}  Rec={rec_s}  "
        f"F1={f1:.3f}  "
        f"AUC-ROC={auc_roc:.3f} │{bar}│  "
        f"AUC-PR={aupr_s}  "
        f"n={ns_s}"
    )


def _print_final_chart(
    history: fl.server.history.History,
    num_rounds: int,
    strategy_name: str,
) -> None:
    """Print a full table + AUC-ROC sparkline chart after all rounds."""
    # Build lookup: metric_name -> {round: value}
    dist      = history.metrics_distributed
    fit_dist  = history.metrics_fit
    acc_map   = dict(dist.get("accuracy",  []))
    auc_map   = dict(dist.get("auc_roc",   []))
    aupr_map  = dict(dist.get("auc_pr",    []))
    f1_map    = dict(dist.get("f1",        []))
    prec_map  = dict(dist.get("precision", []))
    rec_map   = dict(dist.get("recall",    []))
    ns_map    = dict(dist.get("n_samples", []))
    tl_map    = dict(fit_dist.get("train_loss", []))
    loss_map  = {}
    for rnd, val in history.losses_distributed:
        loss_map[rnd] = val

    rounds = sorted(set(acc_map) | set(auc_map))
    if not rounds:
        return

    auc_values = [auc_map.get(r, float("nan")) for r in rounds]
    auc_lo = min((v for v in auc_values if not (v != v)), default=0.5)
    auc_hi = max((v for v in auc_values if not (v != v)), default=1.0)
    auc_lo = min(auc_lo, 0.5)  # always start scale at 0.5 for AUC

    W = 88
    print()
    print("═" * W)
    print(f"  FL Training Results  │  strategy={strategy_name}  rounds={num_rounds}")
    print("═" * W)
    print(
        f"  {'Rnd':>3}  {'TrnLoss':>8}  {'ValLoss':>8}  "
        f"{'Accuracy':>9}  {'Precision':>9}  {'Recall':>7}  "
        f"{'F1':>6}  {'AUC-ROC':>8}  {'AUC-PR':>7}  {'Samples':>8}"
    )
    print(
        f"  {'─'*3}  {'─'*8}  {'─'*8}  "
        f"{'─'*9}  {'─'*9}  {'─'*7}  "
        f"{'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}"
    )
    for r in rounds:
        acc   = acc_map.get(r,  float("nan"))
        auc   = auc_map.get(r,  float("nan"))
        aupr  = aupr_map.get(r, float("nan"))
        f1v   = f1_map.get(r,   float("nan"))
        prec  = prec_map.get(r, float("nan"))
        rec   = rec_map.get(r,  float("nan"))
        ns    = ns_map.get(r,   float("nan"))
        loss  = loss_map.get(r, float("nan"))
        tl    = tl_map.get(r,   float("nan"))
        acc_s  = f"{acc:.1%}"   if acc  == acc  else "   n/a  "
        auc_s  = f"{auc:.4f}"  if auc  == auc  else "  n/a  "
        aupr_s = f"{aupr:.4f}" if aupr == aupr else "  n/a "
        f1_s   = f"{f1v:.4f}"  if f1v  == f1v  else " n/a  "
        prec_s = f"{prec:.4f}" if prec == prec else "   n/a  "
        rec_s  = f"{rec:.4f}"  if rec  == rec  else "  n/a "
        ns_s   = f"{int(ns)}"  if ns   == ns   else "   n/a"
        loss_s = f"{loss:.4f}" if loss == loss else "  n/a  "
        tl_s   = f"{tl:.4f}"   if tl   == tl   else "  n/a  "
        print(
            f"  {r:>3}  {tl_s:>8}  {loss_s:>8}  "
            f"{acc_s:>9}  {prec_s:>9}  {rec_s:>7}  "
            f"{f1_s:>6}  {auc_s:>8}  {aupr_s:>7}  {ns_s:>8}"
        )
    print()
    # AUC-ROC bar chart
    bar_w = 28
    print(f"  AUC-ROC per round  (scale {auc_lo:.2f} → {auc_hi:.2f})")
    print(f"  {'─'*3}  {'─'*(bar_w+2)}")
    for r in rounds:
        auc = auc_map.get(r, float("nan"))
        if auc != auc:
            continue
        bar = _sparkbar(auc, auc_lo, auc_hi, width=bar_w)
        delta = ""
        prev  = auc_map.get(r - 1)
        if prev is not None:
            diff  = auc - prev
            delta = f"  +{diff:.3f}" if diff >= 0 else f"  {diff:.3f}"
        print(f"  {r:>3}  │{bar}│ {auc:.3f}{delta}")
    print("═" * W)
    print()


class _CaptureFinalParams:
    """Mixin: captures aggregated params and per-round eval metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-level attributes — avoid mutable class-level defaults
        self.final_params: Optional[List[np.ndarray]] = None
        self._round_eval_metrics: Dict[int, Dict[str, float]] = {}
        self._round_fit_metrics:  Dict[int, Dict[str, float]] = {}
        self._num_rounds: int = NUM_ROUNDS

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)  # type: ignore[misc]
        if aggregated[0] is not None:
            self.final_params = fl.common.parameters_to_ndarrays(aggregated[0])
        # Cache weighted-average fit metrics (e.g. train_loss) for this round
        agg_fit_metrics = aggregated[1] if aggregated and len(aggregated) > 1 else {}
        if agg_fit_metrics:
            self._round_fit_metrics[server_round] = dict(agg_fit_metrics)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """After each evaluate round, print a compact progress line."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)  # type: ignore[misc]
        # aggregated = (loss_or_None, {metric_name: value})
        agg_metrics = aggregated[1] if aggregated and len(aggregated) > 1 else {}
        if agg_metrics:
            self._round_eval_metrics[server_round] = dict(agg_metrics)
            fit_metrics = self._round_fit_metrics.get(server_round, {})
            _print_round_summary(server_round, self._num_rounds, agg_metrics, fit_metrics)
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
        s = FedProxCapture(proximal_mu=FEDPROX_MU, **common)
    else:
        s = FedAvgCapture(**common)
    return s


# ---------------------------------------------------------------------------
# S3 global-model upload
# Called after start_server() returns.  Reconstructs the StrokeNet from the
# captured numpy parameter arrays and uploads the state_dict as model.pt.
# ---------------------------------------------------------------------------
def _upload_global_model_to_s3(
    params_ndarrays: List[np.ndarray],
    run_name: str,
    input_dim: int,  # kept for API compat but derived from params when possible
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

        # Derive actual input_dim from the aggregated weights.
        # params_ndarrays[0] is the first linear layer weight: shape [hidden, input_dim]
        actual_input_dim = int(params_ndarrays[0].shape[1])
        if actual_input_dim != input_dim:
            print(
                f"[server] INPUT_DIM env var={input_dim} but actual feature dim "
                f"from params={actual_input_dim} — using {actual_input_dim}"
            )

        model = StrokeNet(actual_input_dim)
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
        # Flower returns metrics_distributed as:
        #   {metric_name: [(round_int, value), ...], ...}
        # Pivot to {round_int: {metric_name: value}} for MLflow step logging.
        metrics_by_round: Dict[int, Dict[str, float]] = {}
        for metric_name, round_values in history.metrics_distributed.items():
            for rnd, value in round_values:
                if rnd not in metrics_by_round:
                    metrics_by_round[rnd] = {}
                metrics_by_round[rnd][f"distributed_{metric_name}"] = float(value)

        for rnd in sorted(metrics_by_round):
            mlflow.log_metrics(metrics_by_round[rnd], step=rnd)

        # ── Print ASCII chart of all rounds ──────────────────────────
        _print_final_chart(history, NUM_ROUNDS, STRATEGY_NAME)

        # ── Upload final global model to S3 ─────────────────────────
        if strategy.final_params is not None:
            _upload_global_model_to_s3(strategy.final_params, run_name, INPUT_DIM)
        else:
            print("⚠️  No aggregated parameters captured — S3 upload skipped")

    print("FL server finished all rounds.")


if __name__ == "__main__":
    main()
