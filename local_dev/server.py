"""
server.py - FL server for multi-terminal mode (Option B).

Usage (Terminal 1):
    python server.py

Then start 3 clients in separate terminals:
    python client.py --hospital 1
    python client.py --hospital 2
    python client.py --hospital 3
"""

import flwr as fl
import mlflow

NUM_ROUNDS = 10

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("fl-local-dev")

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
)

print("Starting FL server on localhost:8080 — waiting for 3 clients...")

with mlflow.start_run(run_name="multi-terminal"):
    mlflow.log_param("rounds", NUM_ROUNDS)

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
