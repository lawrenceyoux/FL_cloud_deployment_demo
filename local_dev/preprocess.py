"""
preprocess.py - Prepare Kaggle stroke dataset for local FL dev

Usage:
    python preprocess.py --input data/raw/healthcare-dataset-stroke-data.csv

Output:
    data/processed/hospital_1.csv  (elderly/hypertension-heavy → high stroke rate)
    data/processed/hospital_2.csv  (young/healthy             → low stroke rate)
    data/processed/hospital_3.csv  (mixed/rural               → medium stroke rate)
"""

import argparse
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent  # always points to local_dev/

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"].copy()                  # drop 1 row

    # Fill missing BMI with median
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Encode binary columns
    df["ever_married"]    = (df["ever_married"] == "Yes").astype(int)
    df["gender"]          = (df["gender"] == "Male").astype(int)
    df["Residence_type"]  = (df["Residence_type"] == "Urban").astype(int)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=["work_type", "smoking_status"], dtype=int)

    return df


def non_iid_split(df: pd.DataFrame):
    """Simulate 3 hospitals with different patient populations (non-IID)."""
    pool1 = df[(df["age"] > 60) | (df["hypertension"] == 1)]
    h1 = pool1.sample(n=min(1700, len(pool1)), random_state=42)
    rest = df.drop(h1.index)

    pool2 = rest[rest["age"] < 45]
    h2 = pool2.sample(n=min(1400, len(pool2)), random_state=42)
    rest = rest.drop(h2.index)

    h3 = rest.sample(n=min(2000, len(rest)), random_state=42)
    return h1, h2, h3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(HERE / "data/raw/healthcare-dataset-stroke-data.csv"))
    args = parser.parse_args()

    out_dir = HERE / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['stroke'].mean()*100:.1f}% stroke rate")

    df = preprocess(df)
    h1, h2, h3 = non_iid_split(df)

    for i, h in enumerate([h1, h2, h3], 1):
        path = out_dir / f"hospital_{i}.csv"
        h.to_csv(path, index=False)
        print(f"Hospital {i}: {len(h)} rows, stroke rate {h['stroke'].mean()*100:.1f}% → {path}")

    print("\nDone. Run simulate.py next.")


if __name__ == "__main__":
    main()
