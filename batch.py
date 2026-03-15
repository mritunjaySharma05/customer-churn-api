"""
batch.py — Overnight batch scoring pipeline.

Usage:
    python batch.py --input test_data/all_customers.csv

Outputs:
    scored_customers.csv   — original rows + churn_probability + churn_prediction
    logs/batch_log.txt     — summary stats (requests, failures, avg probability)
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
import requests

# ── Constants ──────────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000/predict"
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "batch_log.txt")
OUTPUT   = "scored_customers.csv"

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "tenure_years", "spend_per_month",
]

# ── Logging setup ──────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────
def row_to_payload(row: pd.Series) -> dict:
    """Convert a DataFrame row into the JSON payload expected by /predict."""
    customer = {col: (None if pd.isna(row[col]) else row[col])
                for col in FEATURE_COLS if col in row}
    return {"customer": customer}


def score_customer(row: pd.Series, session: requests.Session) -> tuple[float | None, str | None]:
    """POST a single customer to the API. Returns (probability, prediction)."""
    payload = row_to_payload(row)
    resp = session.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["churn_probability"], data["churn_prediction"]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch churn scoring pipeline")
    parser.add_argument("--input", required=True, help="Path to customer CSV file")
    args = parser.parse_args()

    # Load customers
    df = pd.read_csv(args.input, index_col=0)
    total      = len(df)
    failures   = 0
    probs      = []

    logger.info("=== Batch scoring started | customers=%d | input=%s ===", total, args.input)
    start_time = time.time()

    churn_probs        = []
    churn_predictions  = []

    with requests.Session() as session:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                prob, pred = score_customer(row, session)
                churn_probs.append(prob)
                churn_predictions.append(pred)
                probs.append(prob)
                if idx % 100 == 0:
                    logger.info("Progress: %d/%d processed", idx, total)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                churn_probs.append(None)
                churn_predictions.append(None)
                customer_id = row.get("customerID", idx)
                logger.warning("FAILED customer %s — %s", customer_id, exc)

    # Attach results to dataframe
    df["churn_probability"] = churn_probs
    df["churn_prediction"]  = churn_predictions
    df.to_csv(OUTPUT)

    elapsed      = time.time() - start_time
    avg_prob     = sum(probs) / len(probs) if probs else 0.0
    successful   = total - failures

    # Log summary
    logger.info("=== Batch scoring complete ===")
    logger.info("Total requests   : %d", total)
    logger.info("Successful       : %d", successful)
    logger.info("Failed           : %d", failures)
    logger.info("Avg churn prob   : %.4f", avg_prob)
    logger.info("Elapsed time     : %.1fs", elapsed)
    logger.info("Output saved to  : %s", OUTPUT)
    print(f"\nDone! Results → {OUTPUT} | Log → {LOG_FILE}")


if __name__ == "__main__":
    main()
