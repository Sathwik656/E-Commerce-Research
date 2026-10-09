"""
Model 2 - Customer Next-Order Spending Prediction (unified Olist design).

Population  : repeat Olist customers (>= 2 delivered orders)
Features    : historical features computed from the customer's PRIOR orders
              only (no leakage from the target order)
Target      : NextOrderSpending = payment value of the customer's next order
              (conditional on that order existing - repeat customers only)
Evaluation  : out-of-time - trained on orders placed before 2018-01-01, tested
              on orders placed after.

Reads data/olist_spending_dataset_prepared.csv (built by prepare_data.py),
trains / evaluates a Random Forest Regressor, writes research outputs to
outputs/, and saves the model + ordered feature names to models/.

Run:  python prepare_data.py && python train_spending.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

from prepare_data import SPENDING_FEATURES, TRAIN_CUT

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for _d in (MODEL_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 300


def load_dataset():
    df = pd.read_csv(DATA_DIR / "olist_spending_dataset_prepared.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    for part in (train, test):
        if part[SPENDING_FEATURES].isnull().any().any():
            raise ValueError("Spending feature matrix contains missing values.")
    return df, train, test


def train_and_evaluate(train, test):
    X_train, y_train = train[SPENDING_FEATURES], train["NextOrderSpending"]
    X_test, y_test = test[SPENDING_FEATURES], test["NextOrderSpending"]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": r2_score(y_test, y_pred),
        "Median_AE": median_absolute_error(y_test, y_pred),
    }

    importance = (
        pd.DataFrame({"Feature": SPENDING_FEATURES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    train_customers = set(train["customer_unique_id"])
    test_customers = set(test["customer_unique_id"])
    split_info = {
        "train_observations": int(len(X_train)),
        "test_observations": int(len(X_test)),
        "train_customers": len(train_customers),
        "test_customers": len(test_customers),
        "customer_overlap": len(train_customers & test_customers),
        "split": f"temporal (order date < {TRAIN_CUT.date()})",
    }

    predictions = pd.DataFrame(
        {
            "observation_id": np.arange(1, len(y_test) + 1),
            "Actual_NextOrderSpending": y_test.values,
            "Predicted_NextOrderSpending": np.round(y_pred, 4),
            "Residual": np.round(residuals, 4),
        }
    )

    return {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "predictions": predictions,
        "split_info": split_info,
        "y_test": y_test,
        "y_pred": y_pred,
        "residuals": residuals,
        "train": train,
        "test": test,
    }


def save_and_verify(result):
    model_path = MODEL_DIR / "spending_model.pkl"
    joblib.dump(result["model"], model_path)

    with open(MODEL_DIR / "spending_features.json", "w", encoding="utf-8") as fh:
        json.dump({"features": SPENDING_FEATURES}, fh, indent=2)

    reloaded = joblib.load(model_path)
    sample = result["test"][SPENDING_FEATURES]
    verification = {
        "model_path": "models/spending_model.pkl",
        "features_path": "models/spending_features.json",
        "predictions_identical": bool(
            np.allclose(result["model"].predict(sample), reloaded.predict(sample))
        ),
        "model_type": type(reloaded).__name__,
        "n_features": int(reloaded.n_features_in_),
    }
    return model_path, verification


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_feature_importance(importance, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = importance.sort_values("Importance")
    ax.barh(data["Feature"], data["Importance"], color="#3b6ea5")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Feature Importance - Next-Order Spending (Olist)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_actual_vs_predicted(y_test, y_pred, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#3b6ea5", edgecolor="none")
    lims = [0, float(max(np.max(y_test), np.max(y_pred)))]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual next-order spending (BRL)")
    ax.set_ylabel("Predicted next-order spending (BRL)")
    ax.set_title("Actual vs Predicted Next-Order Spending")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_residuals(residuals, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40, color="#8a5cf6", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted, BRL)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution - Next-Order Spending")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_target_distribution(y, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y, bins=60, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("Next-order spending (BRL)")
    ax.set_ylabel("Frequency")
    ax.set_title("Next-Order Spending Target Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Risk value (Task 7): transparent, documented prototype formula
# --------------------------------------------------------------------------- #
def build_risk_examples(predictions):
    """Sensitivity of the risk formula to the churn probability."""
    sample = predictions.sort_values(
        "Predicted_NextOrderSpending", ascending=False
    ).head(5)
    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        predicted = float(row["Predicted_NextOrderSpending"])
        for churn_p in (0.0, 0.25, 0.50, 0.75, 1.0):
            rows.append(
                {
                    "example_id": i,
                    "churn_probability": churn_p,
                    "predicted_next_order_spending_BRL": round(predicted, 2),
                    "customer_risk_value_BRL": round(churn_p * predicted, 2),
                }
            )
    return pd.DataFrame(rows)


def main():
    print("Loading unified Olist next-order spending dataset ...")
    full, train, test = load_dataset()
    print(f"  train observations: {len(train)}  test observations: {len(test)}")

    result = train_and_evaluate(train, test)
    model_path, verification = save_and_verify(result)

    print("\n-- Out-of-time evaluation (orders after 2018-01-01) --")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print("Split info:", result["split_info"])

    metrics_rows = [
        {"Metric": "MAE", "Value": round(result["metrics"]["MAE"], 4)},
        {"Metric": "RMSE", "Value": round(result["metrics"]["RMSE"], 4)},
        {"Metric": "R2", "Value": round(result["metrics"]["R2"], 4)},
        {"Metric": "Median_AE", "Value": round(result["metrics"]["Median_AE"], 4)},
    ]
    pd.DataFrame(metrics_rows).to_csv(
        OUT_DIR / "spending_evaluation_metrics.csv", index=False
    )
    result["importance"].to_csv(OUT_DIR / "spending_feature_importance.csv", index=False)
    result["predictions"].to_csv(OUT_DIR / "spending_predictions.csv", index=False)
    build_risk_examples(result["predictions"]).to_csv(
        OUT_DIR / "customer_risk_value_examples.csv", index=False
    )

    # Data-driven risk-value thresholds (tertiles of predicted spending, since
    # the churn probability is ~constant on this minority repeatedly-buying
    # population). Documented as prototype cut-offs, not validated figures.
    p33, p66 = np.percentile(result["y_pred"], [33, 66])
    with open(MODEL_DIR / "risk_thresholds.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "basis": "churn_probability * predicted_next_order_spending (BRL)",
                "low_max": round(float(p33), 2),
                "medium_max": round(float(p66), 2),
            },
            fh,
            indent=2,
        )

    plot_feature_importance(result["importance"], OUT_DIR / "spending_feature_importance.png")
    plot_actual_vs_predicted(
        result["y_test"], result["y_pred"], OUT_DIR / "spending_actual_vs_predicted.png"
    )
    plot_residuals(result["residuals"], OUT_DIR / "spending_residuals.png")
    plot_target_distribution(full["NextOrderSpending"], OUT_DIR / "spending_target_distribution.png")

    summary = pd.DataFrame(
        [
            {"Item": "Source", "Value": "Olist Brazilian E-Commerce (delivered orders)"},
            {"Item": "Churn horizon (days)", "Value": 150},
            {"Item": "Repeat customers (>=2 orders)", "Value": int(full["customer_unique_id"].nunique())},
            {"Item": "Spending observations (train)", "Value": int(len(train))},
            {"Item": "Spending observations (test)", "Value": int(len(test))},
            {"Item": "NextOrderSpending mean (BRL)", "Value": round(float(full["NextOrderSpending"].mean()), 2)},
            {"Item": "NextOrderSpending median (BRL)", "Value": round(float(full["NextOrderSpending"].median()), 2)},
            {"Item": "NextOrderSpending max (BRL)", "Value": round(float(full["NextOrderSpending"].max()), 2)},
        ]
    )
    summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    config = {
        "model": "RandomForestRegressor",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "split": result["split_info"],
        "features": SPENDING_FEATURES,
        "target": "NextOrderSpending",
        "target_units": "BRL (Brazilian Real)",
        "metrics": {k: round(v, 6) for k, v in result["metrics"].items()},
        "verification": verification,
    }
    with open(OUT_DIR / "experiment_configuration.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\nModel saved to:", model_path)
    print("Reload verification:", verification)


if __name__ == "__main__":
    main()
