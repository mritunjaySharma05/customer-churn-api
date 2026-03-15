"""
app/utils.py — Helper functions for the churn prediction API.
"""
import os
import joblib
import pandas as pd

# ── Paths (relative to this file so they work regardless of cwd) ───────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(_HERE, "model.pkl")
TRANSFORMER_PATH = os.path.join(_HERE, "transformer.pkl")

# Columns expected by the model (same order used during training)
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "tenure_years", "spend_per_month",
]


def load_model():
    """Load the trained sklearn Pipeline from disk."""
    return joblib.load(MODEL_PATH)


def load_transformer():
    """Load the fitted ColumnTransformer from disk."""
    return joblib.load(TRANSFORMER_PATH)


def customer_json_to_df(customer_dict: dict) -> pd.DataFrame:
    """
    Convert a raw customer dict (from the /predict POST body) into a
    single-row DataFrame with the correct column order.

    Missing columns are filled with None so the preprocessing pipeline
    can apply its imputer strategy.
    """
    row = {col: customer_dict.get(col, None) for col in FEATURE_COLS}
    df = pd.DataFrame([row])
    # Ensure TotalCharges is numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df
