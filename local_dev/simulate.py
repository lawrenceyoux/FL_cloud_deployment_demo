"""
simulate.py - Run full FL training in a SINGLE process (no Docker/K8s/Ray needed).

Usage:
    python simulate.py

Pure Python FedAvg loop — no Flower simulation API, no Ray dependency.
MLflow logs to ./mlruns  (view with: mlflow ui)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from model import StrokeNet, get_parameters, set_parameters

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ROUNDS    = 10
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
DATA_DIR      = Path(__file__).parent / "data/processed"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_hospital(hospital_id: int):
    df = pd.read_csv(DATA_DIR / f"hospital_{hospital_id}.csv")
    X = df.drop(columns=["stroke"]).values.astype(np.float32)
    y = df["stroke"].values.astype(np.float32)
    split = int(0.8 * len(X))
    train = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
    val   = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val,   batch_size=BATCH_SIZE),
            X.shape[1], len(X[:split]))  # input_dim, n_train_samples


# ── Local training (one hospital, one round) ──────────────────────────────────
def local_train(model, train_loader, criterion, optimizer, pos_weight):
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X).squeeze()
            # Weight positive (stroke) samples by pos_weight
            weights = torch.where(y == 1, pos_weight, torch.tensor(1.0))
            loss = (weights * nn.functional.binary_cross_entropy(out, y, reduction='none')).mean()
            loss.backward()
            optimizer.step()
    return get_parameters(model)


def local_eval(model, val_loader, criterion):
    model.eval()
    correct = total = loss_sum = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            out       = model(X).squeeze()
            loss_sum += criterion(out, y).item()
            correct  += ((out > 0.5).float() == y).sum().item()
            total    += len(y)
            all_probs.extend(out.numpy())
            all_labels.extend(y.numpy())
    preds = (np.array(all_probs) > 0.5).astype(int)
    labels = np.array(all_labels).astype(int)
    return {
        "accuracy": correct / total,
        "recall":   recall_score(labels, preds, zero_division=0),
        "f1":       f1_score(labels, preds, zero_division=0),
        "auc":      roc_auc_score(labels, all_probs) if labels.sum() > 0 else 0.0,
    }


# ── FedAvg aggregation ────────────────────────────────────────────────────────
def fedavg(updates):
    """Weighted average of parameter lists by number of samples."""
    total = sum(n for n, _ in updates)
    avg = [
        sum(n * p[i] for n, p in updates) / total
        for i in range(len(updates[0][1]))
    ]
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mlflow.set_tracking_uri((Path(__file__).parent / "mlruns").as_uri())
    mlflow.set_experiment("fl-local-dev")

    # Initialise clients
    clients = []
    for hid in range(1, 4):
        train_loader, val_loader, input_dim, n_train = load_hospital(hid)
        model     = StrokeNet(input_dim)
        # pos_weight penalises missing a stroke ~20x more → forces model to learn minority class
        criterion = nn.BCELoss()
        pos_weight = torch.tensor(20.0)  # ~ratio of negatives to positives in stroke dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        clients.append({"id": hid, "model": model, "train_loader": train_loader,
                        "val_loader": val_loader, "criterion": criterion,
                        "optimizer": optimizer, "n_train": n_train,
                        "pos_weight": pos_weight})

    print(f"Input features: {input_dim}  |  Hospitals: {len(clients)}")

    # Global parameters start from hospital-1 model (random init)
    global_params = get_parameters(clients[0]["model"])

    with mlflow.start_run(run_name="simulation"):
        mlflow.log_params({"rounds": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
                           "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                           "hospitals": 3, "strategy": "FedAvg"})

        for rnd in range(1, NUM_ROUNDS + 1):
            updates = []

            # Each hospital trains on the current global model
            for c in clients:
                set_parameters(c["model"], global_params)
                local_params = local_train(c["model"], c["train_loader"],
                                           c["criterion"], c["optimizer"],
                                           c["pos_weight"])
                updates.append((c["n_train"], local_params))

            # Server aggregates
            global_params = fedavg(updates)

            # Evaluate global model on each hospital's validation set
            accs, f1s, recalls, aucs = [], [], [], []
            for c in clients:
                set_parameters(c["model"], global_params)
                m = local_eval(c["model"], c["val_loader"], c["criterion"])
                accs.append(m["accuracy"])
                f1s.append(m["f1"])
                recalls.append(m["recall"])
                aucs.append(m["auc"])
                mlflow.log_metrics({
                    f"hospital_{c['id']}_acc":    m["accuracy"],
                    f"hospital_{c['id']}_f1":     m["f1"],
                    f"hospital_{c['id']}_recall": m["recall"],
                    f"hospital_{c['id']}_auc":    m["auc"],
                }, step=rnd)

            mlflow.log_metrics({"avg_accuracy": np.mean(accs),
                                 "avg_f1":       np.mean(f1s),
                                 "avg_recall":   np.mean(recalls),
                                 "avg_auc":      np.mean(aucs)}, step=rnd)
            print(f"Round {rnd:2d} | "
                  f"Acc {np.mean(accs):.3f} | "
                  f"F1 {np.mean(f1s):.3f} | "
                  f"Recall {np.mean(recalls):.3f} | "
                  f"AUC {np.mean(aucs):.3f}")

    print("\nDone! View results:  mlflow ui  → http://localhost:5000")


if __name__ == "__main__":
    main()
