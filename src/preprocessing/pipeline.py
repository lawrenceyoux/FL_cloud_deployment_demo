"""
src/preprocessing/pipeline.py
Data preprocessing + non-IID hospital split for cloud deployment.

Identical logic to local_dev/preprocess.py — abstracted into importable
functions so they can be called from:
  - train_baseline.py  (local testing via src/)
  - GitHub Actions data-pipeline workflow
  - EKS init containers that load data from S3

Data paths are resolved via env vars so the same code works locally and
in Kubernetes without modification:

  DATA_RAW_PATH   (default: data/raw/healthcare-dataset-stroke-data.csv)
  DATA_OUT_DIR    (default: data/processed/)
  USE_S3          (default: 0)   — set to "1" in K8s to read/write S3
  S3_RAW_BUCKET   (default: fl-demo-data-hospital-1)
  S3_OUT_PREFIX   (default: processed/)
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import boto3
import pandas as pd


# ---------------------------------------------------------------------------
# Core transformation (identical to local_dev/preprocess.py::preprocess)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the raw Kaggle stroke CSV."""
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"].copy()

    # Fill missing BMI with median
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Binary encoding
    df["ever_married"]   = (df["ever_married"] == "Yes").astype(int)
    df["gender"]         = (df["gender"] == "Male").astype(int)
    df["Residence_type"] = (df["Residence_type"] == "Urban").astype(int)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=["work_type", "smoking_status"], dtype=int)

    return df


# ---------------------------------------------------------------------------
# Non-IID split (identical to local_dev/preprocess.py::non_iid_split)
# ---------------------------------------------------------------------------
def non_iid_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into 3 non-IID hospital datasets.

    Hospital 1 — elderly / hypertensive patients   → high stroke rate
    Hospital 2 — young patients                    → low stroke rate
    Hospital 3 — remainder (rural / mixed)         → medium stroke rate
    """
    pool1 = df[(df["age"] > 60) | (df["hypertension"] == 1)]
    h1    = pool1.sample(n=min(1700, len(pool1)), random_state=42)
    rest  = df.drop(h1.index)

    pool2 = rest[rest["age"] < 45]
    h2    = pool2.sample(n=min(1400, len(pool2)), random_state=42)
    rest  = rest.drop(h2.index)

    h3 = rest.sample(n=min(2000, len(rest)), random_state=42)
    return h1, h2, h3


# ---------------------------------------------------------------------------
# I/O helpers — local or S3 depending on USE_S3 env var
# ---------------------------------------------------------------------------
def _s3_client():
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV from a local path or S3."""
    use_s3 = os.environ.get("USE_S3", "0") == "1"

    if use_s3:
        bucket = os.environ["S3_RAW_BUCKET"]
        key    = os.environ.get("S3_RAW_KEY", "raw/healthcare-dataset-stroke-data.csv")
        s3     = _s3_client()
        obj    = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])

    src = path or os.environ.get(
        "DATA_RAW_PATH",
        str(Path(__file__).parents[2] / "data/raw/healthcare-dataset-stroke-data.csv"),
    )
    return pd.read_csv(src)


def save_hospital(df: pd.DataFrame, hospital_id: int, out_dir: str | None = None) -> str:
    """Save a hospital CSV to local disk or S3.  Returns the output path/URI."""
    use_s3 = os.environ.get("USE_S3", "0") == "1"
    filename = f"hospital_{hospital_id}.csv"

    if use_s3:
        bucket = f"fl-demo-data-hospital-{hospital_id}"
        prefix = os.environ.get("S3_OUT_PREFIX", "processed/")
        key    = f"{prefix.rstrip('/')}/{filename}"
        s3     = _s3_client()
        body   = df.to_csv(index=False).encode()
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        uri = f"s3://{bucket}/{key}"
        print(f"  Uploaded  → {uri}  ({len(df)} rows)")
        return uri

    dest = Path(out_dir or os.environ.get(
        "DATA_OUT_DIR",
        str(Path(__file__).parents[2] / "data/processed"),
    ))
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename
    df.to_csv(path, index=False)
    print(f"  Saved → {path}  ({len(df)} rows, stroke rate {df['stroke'].mean()*100:.1f}%)")
    return str(path)


# ---------------------------------------------------------------------------
# End-to-end pipeline — called by GitHub Actions and by train_baseline.py
# ---------------------------------------------------------------------------
def run_pipeline(
    raw_path: str | None = None,
    out_dir: str | None = None,
) -> tuple[str, str, str]:
    """Run the full pipeline: load → preprocess → split → save.

    Returns
    -------
    Tuple of three output paths/URIs (hospital_1, hospital_2, hospital_3).
    """
    print("Loading raw data…")
    df = load_raw(raw_path)
    print(f"  {len(df)} rows, {df['stroke'].mean()*100:.1f}% stroke rate")

    print("Preprocessing…")
    df = preprocess(df)

    print("Creating non-IID split…")
    h1, h2, h3 = non_iid_split(df)

    print("Saving hospital datasets…")
    paths = (
        save_hospital(h1, 1, out_dir),
        save_hospital(h2, 2, out_dir),
        save_hospital(h3, 3, out_dir),
    )
    print("Pipeline complete.")
    return paths


# ---------------------------------------------------------------------------
# PyTorch DataLoader helper (used by train_baseline.py and FL client)
# ---------------------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_hospital_tensors(
    hospital_id: int,
    data_dir: str | None = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, int]:
    """Load a processed CSV and return (train_loader, val_loader, input_dim).

    In K8s pods the CSV was downloaded from S3 into /tmp/data/ by an init
    container; data_dir should point there.  Locally it comes from
    local_dev/data/processed/ or data/processed/.
    """
    base = Path(data_dir or os.environ.get(
        "DATA_OUT_DIR",
        str(Path(__file__).parents[2] / "local_dev/data/processed"),
    ))
    df = pd.read_csv(base / f"hospital_{hospital_id}.csv")
    X  = df.drop(columns=["stroke"]).values.astype(np.float32)
    y  = df["stroke"].values.astype(np.float32)

    split = int((1 - val_split) * len(X))
    train_ds = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
    val_ds   = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, X.shape[1]
