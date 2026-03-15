# Customer Churn Prediction API

A production-ready Flask API that serves a trained churn prediction model in real time, paired with a batch scoring pipeline for overnight customer scoring.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API server
```bash
# From the project root
python -m app.main
```
The server listens on `http://localhost:8000`.

### 3. Test a single prediction
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @test_data/sample_input.json
```

Expected response:
```json
{
  "churn_probability": 0.62,
  "churn_prediction": "Yes"
}
```

### 4. Run batch scoring
```bash
python batch.py --input test_data/all_customers.csv
```
Outputs:
- `scored_customers.csv` — original rows + `churn_probability` + `churn_prediction`
- `logs/batch_log.txt` — total requests, failures, average probability

---

## Project Structure

```
customer-churn-api/
│
├── app/
│   ├── __init__.py        # Makes app a Python module
│   ├── main.py            # Flask app with /predict endpoint
│   ├── model.pkl          # Trained sklearn Pipeline
│   ├── transformer.pkl    # Fitted ColumnTransformer
│   └── utils.py           # Helper functions
│
├── test_data/
│   ├── sample_input.json  # Single customer JSON for /predict
│   └── all_customers.csv  # Batch input CSV
│
├── batch.py               # Batch scoring script
├── train.py               # Training script (run once to reproduce artefacts)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Details

| Attribute | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Preprocessing | SimpleImputer (mean) + OneHotEncoder + StandardScaler |
| Train / test split | 80 / 20 (stratified) |
| ROC-AUC (test) | 0.836 |
| Accuracy (test) | 80 % |
| Positive class | Churn = Yes |

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` — used as a liveness probe.

### `POST /predict`
**Request body:**
```json
{
  "customer": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 12,
    "Contract": "Month-to-month",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "tenure_years": 1.0,
    "spend_per_month": 65.5,
    ...
  }
}
```

**Response:**
```json
{
  "churn_probability": 0.7312,
  "churn_prediction": "Yes"
}
```

---

## Maintenance Plan

### 🧠 Retraining

Retrain the model **monthly** or whenever business conditions change significantly (e.g., new pricing plans, product launches). The trigger should be automatic: if the batch pipeline's average churn probability shifts by more than ±5 percentage points compared to the previous 4-week rolling baseline, flag for review and initiate retraining. To retrain, run `python train.py` with fresh data, validate that the new ROC-AUC meets or exceeds the existing benchmark (0.836), then replace `app/model.pkl` and `app/transformer.pkl` with the new artifacts and restart the server.

### 📉 Drift Detection

Monitor two types of drift via the batch pipeline logs:

- **Data drift**: Track the distribution of key inputs (`MonthlyCharges`, `tenure`, `Contract` type) each batch run. Flag if any feature mean shifts by more than 2 standard deviations from its training baseline. Use tools like Evidently AI or a lightweight rolling-stats script.
- **Model/prediction drift**: Log average churn probability from every batch run to `logs/batch_log.txt`. Plot a rolling 4-week trend; sustained upward or downward trends beyond the ±5 pp threshold signal concept drift and should trigger retraining evaluation.

### 🗃 Versioning

Follow a simple **date-stamped artefact versioning** strategy:

1. Name new artefacts with a timestamp: `model_20250801.pkl`, `transformer_20250801.pkl`.
2. Keep the last 3 versions archived in a `models/archive/` directory.
3. Maintain a `models/registry.json` that records each version's training date, dataset hash, ROC-AUC, and promotion date.
4. Use a `.gitignore`-excluded `models/current/` symlink pointing to the live artefacts, so `app/model.pkl` and `app/transformer.pkl` are always symlinked from the current version — making rollbacks a single `ln -sf` command.
5. Tag releases in Git: `git tag v1.0.0-model-20250801`.
