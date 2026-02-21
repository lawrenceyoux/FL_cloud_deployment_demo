"""
validate.py  —  Data quality gates for the FL data pipeline

Two modes, selected with --stage:

  raw        Validate the Kaggle CSV before any transformation.
             Guards against: wrong file version, schema drift, truncation.

  processed  Validate hospital_1/2/3.csv after preprocess.py runs.
             Guards against: preprocessing bugs, nulls, wrong encoding.

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (GitHub Actions treats this as a job failure,
              blocking all downstream jobs in the pipeline).

Usage:
  python data/scripts/validate.py \
      --stage raw \
      --input local_dev/data/raw/healthcare-dataset-stroke-data.csv

  python data/scripts/validate.py \
      --stage processed \
      --input local_dev/data/processed/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

_PASS = "  ✓"
_FAIL = "  ✗ FAIL"


def _check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"{_PASS}  {msg_pass}")
    else:
        print(f"{_FAIL}  {msg_fail}", file=sys.stderr)
    return condition


# ── Raw validation ─────────────────────────────────────────────────────────────

RAW_REQUIRED_COLUMNS = {
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
}

ROW_COUNT_MIN = 4854   # 5110 × 0.95
ROW_COUNT_MAX = 5366   # 5110 × 1.05
STROKE_RATE_MIN = 0.04
STROKE_RATE_MAX = 0.06
BMI_NULL_MAX_RATIO = 0.05


def validate_raw(csv_path: Path) -> bool:
    print(f"\n── Raw validation: {csv_path} ──────────────────────")
    failures = 0

    # 1. File exists
    if not _check(csv_path.exists(),
                  f"File found: {csv_path}",
                  f"File NOT found: {csv_path}"):
        # Cannot continue without the file
        return False
    failures += 0

    df = pd.read_csv(csv_path)

    # 2. Row count
    n = len(df)
    ok = ROW_COUNT_MIN <= n <= ROW_COUNT_MAX
    failures += not _check(ok,
        f"Row count {n} within expected range [{ROW_COUNT_MIN}, {ROW_COUNT_MAX}]",
        f"Row count {n} outside expected range [{ROW_COUNT_MIN}, {ROW_COUNT_MAX}]")

    # 3. Required columns
    missing_cols = RAW_REQUIRED_COLUMNS - set(df.columns)
    failures += not _check(
        len(missing_cols) == 0,
        "All required columns present",
        f"Missing columns: {missing_cols}")

    # 4. Target column values
    if "stroke" in df.columns:
        invalid_targets = set(df["stroke"].dropna().unique()) - {0, 1}
        failures += not _check(
            len(invalid_targets) == 0,
            "stroke column values are all 0 or 1",
            f"stroke column contains unexpected values: {invalid_targets}")

    # 5. Overall stroke rate
    if "stroke" in df.columns:
        rate = df["stroke"].mean()
        failures += not _check(
            STROKE_RATE_MIN <= rate <= STROKE_RATE_MAX,
            f"Overall stroke rate {rate:.3f} within [{STROKE_RATE_MIN}, {STROKE_RATE_MAX}]",
            f"Overall stroke rate {rate:.3f} outside expected range [{STROKE_RATE_MIN}, {STROKE_RATE_MAX}]")

    # 6. BMI null rate
    if "bmi" in df.columns:
        null_ratio = df["bmi"].isna().mean()
        failures += not _check(
            null_ratio <= BMI_NULL_MAX_RATIO,
            f"BMI null rate {null_ratio:.3f} ≤ {BMI_NULL_MAX_RATIO}",
            f"BMI null rate {null_ratio:.3f} exceeds threshold {BMI_NULL_MAX_RATIO}")

    # 7. No fully-empty rows
    fully_empty = df.isna().all(axis=1).sum()
    failures += not _check(
        fully_empty == 0,
        "No fully-empty rows",
        f"{fully_empty} fully-empty row(s) found")

    print(f"\nRaw validation: {'PASSED' if failures == 0 else f'FAILED ({failures} check(s))'}\n")
    return failures == 0


# ── Processed validation ───────────────────────────────────────────────────────

# Columns expected after preprocess.py runs
PROCESSED_REQUIRED_COLUMNS = {
    "age", "gender", "hypertension", "heart_disease", "ever_married",
    "Residence_type", "avg_glucose_level", "bmi", "stroke",
    # one-hot: work_type
    "work_type_Govt_job", "work_type_Never_worked", "work_type_Private",
    "work_type_Self-employed", "work_type_children",
    # one-hot: smoking_status
    "smoking_status_Unknown", "smoking_status_formerly smoked",
    "smoking_status_never smoked", "smoking_status_smokes",
}

# Per-hospital expected stroke rate ranges
HOSPITAL_STROKE_RATES = {
    "hospital_1": (0.08, 0.18),
    "hospital_2": (0.01, 0.06),
    "hospital_3": (0.03, 0.09),
}

TOTAL_ROWS_MIN = int(5110 * 0.95)   # combined across all 3
TOTAL_ROWS_MAX = int(5110 * 1.05)


def validate_processed(processed_dir: Path) -> bool:
    print(f"\n── Processed validation: {processed_dir} ──────────────────────")
    failures = 0
    total_rows = 0

    for hospital_id, (rate_min, rate_max) in HOSPITAL_STROKE_RATES.items():
        csv_path = processed_dir / f"{hospital_id}.csv"
        print(f"\n  {hospital_id}.csv")

        # File exists
        if not _check(csv_path.exists(),
                      f"File found: {csv_path}",
                      f"File NOT found: {csv_path}"):
            failures += 1
            continue

        df = pd.read_csv(csv_path)
        total_rows += len(df)

        # No nulls
        null_count = df.isna().sum().sum()
        failures += not _check(
            null_count == 0,
            f"No nulls ({len(df)} rows)",
            f"{null_count} null value(s) found")

        # Required columns
        missing_cols = PROCESSED_REQUIRED_COLUMNS - set(df.columns)
        failures += not _check(
            len(missing_cols) == 0,
            "All expected columns present",
            f"Missing columns: {missing_cols}")

        # Stroke rate
        rate = df["stroke"].mean()
        failures += not _check(
            rate_min <= rate <= rate_max,
            f"Stroke rate {rate:.3f} within expected range [{rate_min}, {rate_max}]",
            f"Stroke rate {rate:.3f} outside expected range [{rate_min}, {rate_max}]")

        # All numeric dtypes (no object columns remaining)
        object_cols = list(df.select_dtypes(include="object").columns)
        failures += not _check(
            len(object_cols) == 0,
            "All columns are numeric (no object dtype remaining)",
            f"Object-dtype columns found: {object_cols}")

        # Row count sanity (each hospital should have at least 500 rows)
        failures += not _check(
            len(df) >= 500,
            f"Row count {len(df)} ≥ 500",
            f"Row count {len(df)} is suspiciously low (< 500)")

    # Total row count across all hospitals
    print(f"\n  Combined row count: {total_rows}")
    failures += not _check(
        TOTAL_ROWS_MIN <= total_rows <= TOTAL_ROWS_MAX,
        f"Combined row count {total_rows} within [{TOTAL_ROWS_MIN}, {TOTAL_ROWS_MAX}]",
        f"Combined row count {total_rows} outside expected range [{TOTAL_ROWS_MIN}, {TOTAL_ROWS_MAX}]")

    print(f"\nProcessed validation: {'PASSED' if failures == 0 else f'FAILED ({failures} check(s))'}\n")
    return failures == 0


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data quality gates for the FL data pipeline")
    parser.add_argument(
        "--stage", required=True, choices=["raw", "processed"],
        help="raw: validate Kaggle CSV | processed: validate hospital_N.csv files")
    parser.add_argument(
        "--input", required=True,
        help="Path to CSV (raw) or directory containing hospital_N.csv files (processed)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.stage == "raw":
        passed = validate_raw(input_path)
    else:
        passed = validate_processed(input_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
