"""
client.py - FL client for multi-terminal mode (Option B).

Usage:
    python client.py --hospital 1
    python client.py --hospital 2
    python client.py --hospital 3

Each simulates one hospital. Start all 3 after server.py is running.
"""

import argparse
import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from model import StrokeNet, get_parameters, set_parameters

BATCH_SIZE    = 32
LOCAL_EPOCHS  = 3
LEARNING_RATE = 0.001
DATA_DIR      = Path(__file__).parent / "data/processed"


def load_hospital(hospital_id: int):
    df = pd.read_csv(DATA_DIR / f"hospital_{hospital_id}.csv")
    X = df.drop(columns=["stroke"]).values.astype(np.float32)
    y = df["stroke"].values.astype(np.float32)
    split = int(0.8 * len(X))
    train = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
    val   = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val,   batch_size=BATCH_SIZE),
            X.shape[1])


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int):
        self.id = hospital_id
        self.train_loader, self.val_loader, input_dim = load_hospital(hospital_id)
        self.model = StrokeNet(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X).squeeze(), y)
                loss.backward()
                self.optimizer.step()
        print(f"[Hospital {self.id}] Local training done")
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        correct = total = loss_sum = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                out  = self.model(X).squeeze()
                loss_sum += self.criterion(out, y).item()
                correct  += ((out > 0.5).float() == y).sum().item()
                total    += len(y)
        accuracy = correct / total
        print(f"[Hospital {self.id}] Accuracy: {accuracy:.3f}")
        return loss_sum / len(self.val_loader), total, {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    print(f"Hospital {args.hospital} connecting to server...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=HospitalClient(args.hospital),
    )
