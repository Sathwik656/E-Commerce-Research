"""
Flask application - Customer Churn & Next-Order Spending prototype.

Batch workflow:
  1. Upload a churn feature CSV  (enriched behavioural features, optional customer_id)
  2. Upload a spending feature CSV (9 historical features, optional customer_id)
  3. The app predicts churn probability and next-order spending,
     then computes the prototype Customer Risk Value
     (churn probability x predicted next-order spending).

The risk value combines outputs from two models trained on two different
customer populations. It is a prototype indicator, not a validated financial
outcome. When both files contain a `customer_id` column the rows are joined on
it; otherwise rows are paired by position.

Run:  python app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

def load_models():
    churn_model = joblib.load(MODEL_DIR / "churn_model.pkl")
    spending_model = joblib.load(MODEL_DIR / "spending_model.pkl")
    with open(MODEL_DIR / "churn_features.json", encoding="utf-8") as fh:
        churn_features = json.load(fh)["features"]
    with open(MODEL_DIR / "spending_features.json", encoding="utf-8") as fh:
        spending_features = json.load(fh)["features"]
    return churn_model, spending_model, churn_features, spending_features


churn_model, spending_model, CHURN_FEATURES, SPENDING_FEATURES = load_models()

app = Flask(__name__)


@app.context_processor
def inject_features():
    return {"churn_features": CHURN_FEATURES, "spending_features": SPENDING_FEATURES}


def _read_upload(file_storage, feature_list, label, allow_negative=False):
    """Return (features_df, ids_or_none, error). Validates columns & values."""
    if not file_storage or not file_storage.filename:
        return None, None, f"No {label} CSV file uploaded."

    try:
        df = pd.read_csv(file_storage, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Could not read {label} CSV: {exc}"

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        return None, None, (
            f"{label} CSV is missing required columns: {', '.join(missing)}."
        )
    if df.empty:
        return None, None, f"{label} CSV contains no data rows."

    ids = df["customer_id"].astype(str) if "customer_id" in df.columns else None

    try:
        features = df[feature_list].apply(pd.to_numeric, errors="raise")
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{label} CSV has non-numeric feature values: {exc}"

    if features.isnull().any().any():
        return None, None, f"{label} CSV has missing feature values."
    if not allow_negative and (features < 0).any().any():
        return None, None, f"{label} CSV has negative feature values."

    return features.reset_index(drop=True), (
        ids.reset_index(drop=True) if ids is not None else None
    ), None


def _pair(churn_ids, spending_ids, n_churn, n_spend):
    """Return (churn_idx, spending_idx, pairing_note)."""
    if churn_ids is not None and spending_ids is not None:
        churn_frame = pd.DataFrame({"customer_id": churn_ids, "ci": range(n_churn)})
        spend_frame = pd.DataFrame({"customer_id": spending_ids, "si": range(n_spend)})
        merged = churn_frame.merge(spend_frame, on="customer_id", how="inner")
        if merged.empty:
            return None, None, (
                "No matching customer_id values between the two files."
            )
        note = f"Joined on customer_id ({len(merged)} matched rows)."
        return merged["ci"].tolist(), merged["si"].tolist(), note

    n = min(n_churn, n_spend)
    note = f"Paired by row order ({n} rows)."
    if n_churn != n_spend:
        note += (
            f" Files had different row counts ({n_churn} vs {n_spend});"
            " extra rows were ignored."
        )
    return list(range(n)), list(range(n)), note


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    churn_X, churn_ids, err_c = _read_upload(
        request.files.get("churn_file"), CHURN_FEATURES, "Churn", allow_negative=True
    )
    spending_X, spending_ids, err_s = _read_upload(
        request.files.get("spending_file"), SPENDING_FEATURES, "Spending"
    )
    if err_c or err_s:
        return render_template(
            "index.html", churn_error=err_c, spending_error=err_s
        ), 400

    churn_prob = churn_model.predict_proba(churn_X)[:, 1]
    churn_status = churn_model.predict(churn_X)
    spending_pred = spending_model.predict(spending_X)
    spending_pred = np.clip(spending_pred, 0, None)

    ci, si, note = _pair(
        churn_ids, spending_ids, len(churn_X), len(spending_X)
    )
    if ci is None:
        return render_template(
            "index.html", spending_error=note
        ), 400

    results = pd.DataFrame(
        {
            "customer_id": (
                churn_ids[ci].values
                if churn_ids is not None
                else [f"row_{i + 1}" for i in range(len(ci))]
            ),
            "churn_probability": np.round(churn_prob[ci], 4),
            "predicted_churn": np.where(churn_status[ci] == 1, "Churned", "Not Churned"),
            "predicted_next_order_spending_BRL": np.round(spending_pred[si], 2),
        }
    )
    results["customer_risk_value_BRL"] = np.round(
        results["churn_probability"] * results["predicted_next_order_spending_BRL"], 2
    )

    return render_template(
        "index.html",
        results=results.to_dict("records"),
        result_count=len(results),
        pairing_note=note,
        mean_risk=round(float(results["customer_risk_value_BRL"].mean()), 2),
        total_risk=round(float(results["customer_risk_value_BRL"].sum()), 2),
    )


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html", churn_error="Page not found."), 404


if __name__ == "__main__":
    app.run(debug=True)
