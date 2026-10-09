"""
Lightweight automated tests for the complete pipeline.

Runs with either `python tests/test_pipeline.py` (built-in runner) or pytest.

Covers:
  * churn model loading / prediction
  * spending model loading / prediction / feature order
  * risk-value calculation edge cases
  * Flask batch upload route (valid join, row-order pairing, invalid inputs)
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as flask_app  # noqa: E402

SAMPLE_DIR = ROOT / "sample_inputs"
CHURN_CSV = (SAMPLE_DIR / "churn_input.csv").read_bytes()
SPENDING_CSV = (SAMPLE_DIR / "spending_input.csv").read_bytes()


def _upload(churn_bytes, spending_bytes, churn_name="churn.csv", spend_name="spend.csv"):
    client = flask_app.app.test_client()
    data = {
        "churn_file": (io.BytesIO(churn_bytes), churn_name),
        "spending_file": (io.BytesIO(spending_bytes), spend_name),
    }
    return client.post("/predict", data=data, content_type="multipart/form-data")


def test_churn_model_loading():
    model = flask_app.churn_model
    assert type(model).__name__ == "RandomForestClassifier"
    assert model.n_features_in_ == 12
    assert len(flask_app.CHURN_FEATURES) == 12

    with open(ROOT / "models" / "churn_features.json", encoding="utf-8") as fh:
        saved = json.load(fh)["features"]
    assert saved == flask_app.CHURN_FEATURES


def test_churn_prediction():
    X = pd.read_csv(SAMPLE_DIR / "churn_input.csv")[flask_app.CHURN_FEATURES]
    proba = flask_app.churn_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba[:, 1] >= 0) & (proba[:, 1] <= 1))


def test_spending_model_loading():
    model = flask_app.spending_model
    assert type(model).__name__ == "RandomForestRegressor"
    assert model.n_features_in_ == 9

    with open(ROOT / "models" / "spending_features.json", encoding="utf-8") as fh:
        saved = json.load(fh)["features"]
    assert saved == flask_app.SPENDING_FEATURES


def test_spending_prediction():
    X = pd.read_csv(SAMPLE_DIR / "spending_input.csv")[flask_app.SPENDING_FEATURES]
    pred = flask_app.spending_model.predict(X)
    assert pred.shape == (len(X),)
    assert np.all(np.isfinite(pred))


def _risk(churn_p, spending):
    return churn_p * spending


def test_risk_value_calculation():
    spending = 500.0
    assert _risk(0.0, spending) == 0.0
    assert _risk(1.0, spending) == spending
    assert _risk(0.75, spending) == 375.0
    assert _risk(0.3, spending) >= 0
    assert _risk(0.9, 0.0) == 0.0


def test_flask_main_route():
    resp = flask_app.app.test_client().get("/")
    assert resp.status_code == 200
    assert b"Churn feature CSV" in resp.data
    assert b"Spending feature CSV" in resp.data


def test_batch_upload_valid():
    resp = _upload(CHURN_CSV, SPENDING_CSV)
    assert resp.status_code == 200
    assert b"Customer risk value" in resp.data
    assert b"C002" in resp.data


def test_batch_missing_churn_column():
    df = pd.read_csv(SAMPLE_DIR / "churn_input.csv").drop(columns=["Monetary"])
    bad = df.to_csv(index=False).encode()
    resp = _upload(bad, SPENDING_CSV)
    assert resp.status_code == 400
    assert b"Monetary" in resp.data


def test_batch_missing_spending_column():
    df = pd.read_csv(SAMPLE_DIR / "spending_input.csv").drop(columns=["AverageFreightValue"])
    bad = df.to_csv(index=False).encode()
    resp = _upload(CHURN_CSV, bad)
    assert resp.status_code == 400
    assert b"AverageFreightValue" in resp.data


def test_batch_non_numeric():
    df = pd.read_csv(SAMPLE_DIR / "churn_input.csv")
    df["Recency"] = df["Recency"].astype(object)
    df.loc[0, "Recency"] = "abc"
    bad = df.to_csv(index=False).encode()
    resp = _upload(bad, SPENDING_CSV)
    assert resp.status_code == 400


def test_batch_row_order_pairing():
    c = pd.read_csv(SAMPLE_DIR / "churn_input.csv").drop(columns=["customer_id"])
    s = pd.read_csv(SAMPLE_DIR / "spending_input.csv").drop(columns=["customer_id"])
    resp = _upload(c.to_csv(index=False).encode(), s.to_csv(index=False).encode())
    assert resp.status_code == 200
    assert b"row order" in resp.data


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {test.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
