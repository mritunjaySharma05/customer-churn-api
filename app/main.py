"""
app/main.py — Flask real-time inference API for customer churn prediction.

Run from the project root:
    python -m app.main
"""
import logging
from flask import Flask, request, jsonify
from app.utils import load_model, customer_json_to_df

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App & model initialisation ─────────────────────────────────────────────
app   = Flask(__name__)
model = load_model()
logger.info("Model loaded successfully.")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with JSON body:
        { "customer": { "gender": "Female", "tenure": 12, ... } }

    Returns:
        { "churn_probability": 0.83, "churn_prediction": "Yes" }
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None or "customer" not in payload:
            return jsonify({"error": "Request body must contain a 'customer' key."}), 400

        customer_dict = payload["customer"]
        df = customer_json_to_df(customer_dict)

        # Full pipeline handles preprocessing + prediction
        prob        = float(model.predict_proba(df)[0][1])
        prediction  = "Yes" if prob >= 0.5 else "No"

        logger.info("Predicted churn=%s  prob=%.4f", prediction, prob)
        return jsonify({
            "churn_probability": round(prob, 4),
            "churn_prediction":  prediction,
        })

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
