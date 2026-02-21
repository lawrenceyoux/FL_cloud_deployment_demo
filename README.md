# Federated Learning Demo: Stroke Prediction

> **Privacy-preserving machine learning — 3 hospitals, one shared model, no shared data**

[![Flower](https://img.shields.io/badge/Flower-1.5%2B-blue)](https://flower.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Overview

This project demonstrates **federated learning (FL)** applied to stroke prediction, using a real-world tabular healthcare dataset split across three simulated hospitals. No raw patient data ever leaves a hospital — only model weight updates are exchanged and aggregated on the central server.

**What it shows:**

- Federated training with [Flower](https://flower.dev/) using the FedAvg strategy
- Non-IID data partitioning to simulate realistic hospital demographic differences
- Per-hospital and averaged metric tracking (accuracy, F1, recall, AUC-ROC) with MLflow
- Two runnable modes: a single-process simulation and a multi-terminal Flower server/client setup
- Cloud deployment scaffolding via Terraform (EKS) and Kubernetes manifests

### Use Case

**Problem**: Three hospitals want to collaboratively train a stroke-risk classifier but cannot share patient records due to privacy regulations.

**Solution**:
1. Each hospital trains a local copy of the model on its own data
2. Only model weights (not data) are sent to the central FL server
3. The server aggregates the weights using FedAvg and broadcasts the updated global model
4. This repeats for several rounds until the model converges

---

## Architecture

```
+------------------------------------------------------+
|                  FL Server (FedAvg)                  |
|         |              |              |              |
|    Hospital 1      Hospital 2     Hospital 3         |
|  (elderly/HTN)   (young/healthy)  (mixed/rural)      |
|   local data       local data      local data        |
+------------------------------------------------------+
                  MLflow Tracking
```

**Key components**:

| Component | Description |
|-----------|-------------|
| `StrokeNet` | Simple MLP (64 -> 32 -> 1) with Dropout and Sigmoid output |
| FedAvg | Weighted averaging of client parameters by number of training samples |
| Flower | Handles server/client communication in multi-terminal mode |
| MLflow | Logs per-hospital and averaged metrics every round |
| Terraform + Kubernetes | IaC scaffolding for EKS cloud deployment |

---

## Dataset

**Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
**Format**: Tabular CSV, ~5,000 rows, binary label (`stroke` = 0 or 1)

**Features used** (after preprocessing):
- Demographics: `age`, `gender`, `ever_married`, `Residence_type`
- Health: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
- Lifestyle: `work_type` (one-hot), `smoking_status` (one-hot)

**Non-IID hospital split**:

| Hospital | Patients | Profile | Approx. Stroke Rate |
|----------|----------|---------|---------------------|
| 1 | ~1 700 | Elderly / hypertension | ~12% |
| 2 | ~1 400 | Young / healthy | ~2% |
| 3 | ~2 000 | Mixed / rural | ~5% |

---

## Quick Start (local � no Docker, no cloud)

### Prerequisites

- Python 3.10+
- The Kaggle dataset CSV (see Step 1 below)

### Setup

```bash
cd local_dev
python -m venv fl_env
fl_env\Scripts\activate        # Windows
# source fl_env/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

`requirements.txt` installs: `torch`, `flwr[simulation]`, `pandas`, `scikit-learn`, `mlflow`.

### Step 1 - Get the data

Download the CSV from Kaggle and place it at:

```
local_dev/data/raw/healthcare-dataset-stroke-data.csv
```

### Step 2 - Preprocess and split into 3 hospitals

```bash
cd local_dev
python preprocess.py
```

Outputs `data/processed/hospital_1.csv`, `hospital_2.csv`, `hospital_3.csv`.

### Step 3 - Run federated learning

**Option A: Single-process simulation (easiest)**

```bash
python simulate.py
```

Runs a pure-Python FedAvg loop across all 3 hospitals in one process.  No Flower server/client networking needed.

**Option B: Multi-terminal (closer to real deployment)**

```bash
# Terminal 1 - server
python server.py

# Terminal 2
python client.py --hospital 1

# Terminal 3
python client.py --hospital 2

# Terminal 4
python client.py --hospital 3
```

### Step 4 - View results in MLflow

```bash
mlflow ui
# Open http://localhost:5000
```

Tracked per round: `avg_accuracy`, `avg_f1`, `avg_recall`, `avg_auc`, and per-hospital variants (`hospital_1_acc`, etc.).

---

## Training Configuration

| Parameter | Default |
|-----------|---------|
| Rounds | 10 |
| Local epochs per round | 3 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Weighted BCE (pos_weight = 20) |
| Aggregation | FedAvg (weighted by samples) |

The high `pos_weight` compensates for the strong class imbalance in the stroke dataset (~5% positive overall).

---

## Project Structure

```
FL_cloud_deployment_demo/
+-- local_dev/                     # All runnable local code
|   +-- preprocess.py              # Clean & split Kaggle CSV into 3 hospital files
|   +-- model.py                   # StrokeNet MLP definition + weight helpers
|   +-- simulate.py                # Single-process FedAvg simulation
|   +-- server.py                  # Flower FL server (multi-terminal mode)
|   +-- client.py                  # Flower FL client (multi-terminal mode)
|   +-- requirements.txt
|   +-- data/
|   |   +-- raw/                   # Place Kaggle CSV here
|   |   +-- processed/             # hospital_1/2/3.csv (generated)
|   +-- mlruns/                    # MLflow experiment logs
|
+-- kubernetes/                    # K8s manifests (cloud deployment)
|   +-- namespaces/
|   +-- configmaps/
|   +-- deployments/
|   +-- services/
|
+-- terraform/                     # EKS infrastructure as code
|   +-- main.tf
|   +-- eks.tf
|   +-- vpc.tf
|   +-- s3.tf
|   +-- variables.tf
|
+-- src/                           # Python package stubs (for future extension)
|   +-- federated/
|   +-- models/
|   +-- preprocessing/
|   +-- utils/
|
+-- tests/
    +-- unit/
    +-- integration/
```

---

## Cloud Deployment (EKS)

Terraform and Kubernetes manifests are provided under `terraform/` and `kubernetes/` to deploy the FL setup on AWS EKS.

```bash
cd terraform
terraform init
terraform apply -auto-approve
```

Kubernetes manifests in `kubernetes/` define the FL server and client deployments, services, namespace, and ConfigMaps.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| FL Framework | Flower 1.5+ |
| Deep Learning | PyTorch 2.0+ |
| Data | Pandas, scikit-learn |
| MLOps | MLflow |
| IaC | Terraform (EKS) |
| Orchestration | Kubernetes |

---

## Roadmap

- [ ] Add FedProx / FedOpt aggregation strategies
- [ ] Personalized FL (per-hospital fine-tuning)
- [ ] Docker images for server and clients
- [ ] Automated Kubernetes deployment pipeline
- [ ] Extend to MIMIC-IV or a richer dataset

---

## Contributing

Contributions welcome!  Please fork the repository, create a feature branch, and submit a pull request.

---

## License

MIT License - see [LICENSE](LICENSE)
