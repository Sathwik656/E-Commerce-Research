"""
Model 2 - Customer Next-Order Spending Prediction.

End-to-end pipeline (Tasks 4 - 6.7):
  * clean the Brazilian Olist order dataset
  * build customer-level historical features and the next-order spending target
  * customer-level train/test split
  * train / evaluate a Random Forest Regressor (200 trees)
  * feature importance, save + reload verification
  * write research tables/figures to outputs/ and the model to models/

Run:  python train_spending.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for _d in (MODEL_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.20

# Order matters: the Flask app must present features in exactly this order.
SPENDING_FEATURES = [
    "Recency",
    "PreviousOrderCount",
    "HistoricalSpending",
    "TotalItems",
    "TotalProducts",
    "TotalSellers",
    "AverageOrderValue",
    "AverageFreightValue",
    "AverageInstallments",
]

# Target order columns (current order) must never leak into the features.
REQUIRED_COLUMNS = [
    "customer_unique_id",
    "order_purchase_timestamp",
    "payment_value",
    "number_of_items",
    "number_of_products",
    "number_of_sellers",
    "total_item_price",
    "total_freight_value",
    "payment_installments",
]


# --------------------------------------------------------------------------- #
# Task 4 - inspect & clean
# --------------------------------------------------------------------------- #
def clean_olist(path=None):
    path = path or DATA_DIR / "olist_spending_dataset.csv"
    df = pd.read_csv(path)

    clean = df.copy()
    clean["order_purchase_timestamp"] = pd.to_datetime(
        clean["order_purchase_timestamp"], errors="coerce"
    )
    clean = clean.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])
    clean = clean[clean["payment_value"] > 0].copy()
    return clean


# --------------------------------------------------------------------------- #
# Task 5 - features + next-order target
# --------------------------------------------------------------------------- #
def build_spending_dataset(clean):
    delivered = clean[clean["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=REQUIRED_COLUMNS).copy()
    delivered = delivered.sort_values(
        by=["customer_unique_id", "order_purchase_timestamp"]
    ).reset_index(drop=True)

    g = delivered.groupby("customer_unique_id")

    delivered["PreviousOrderCount"] = g.cumcount()
    delivered["NextOrderSpending"] = delivered["payment_value"]

    def prior_cumsum(col):
        return g[col].cumsum() - delivered[col]

    delivered["HistoricalSpending"] = prior_cumsum("payment_value")
    delivered["TotalItems"] = prior_cumsum("number_of_items")
    delivered["TotalProducts"] = prior_cumsum("number_of_products")
    delivered["TotalSellers"] = prior_cumsum("number_of_sellers")

    with np.errstate(divide="ignore", invalid="ignore"):
        delivered["AverageOrderValue"] = (
            delivered["HistoricalSpending"] / delivered["PreviousOrderCount"]
        )
        delivered["AverageFreightValue"] = (
            prior_cumsum("total_freight_value") / delivered["PreviousOrderCount"]
        )
        delivered["AverageInstallments"] = (
            prior_cumsum("payment_installments") / delivered["PreviousOrderCount"]
        )

    delivered["PreviousPurchaseDate"] = g["order_purchase_timestamp"].shift(1)
    delivered["Recency"] = (
        delivered["order_purchase_timestamp"] - delivered["PreviousPurchaseDate"]
    ).dt.total_seconds() / 86400.0

    # First order per customer has no history -> drop it.
    obs = delivered[delivered["PreviousOrderCount"] > 0].copy()
    obs = obs[["customer_unique_id", *SPENDING_FEATURES, "NextOrderSpending"]]
    return delivered, obs


# --------------------------------------------------------------------------- #
# Task 6.3 - 6.7
# --------------------------------------------------------------------------- #
def train_and_evaluate(spending_features):
    X = spending_features[SPENDING_FEATURES].copy()
    y = spending_features["NextOrderSpending"].copy()
    customer_ids = spending_features["customer_unique_id"].copy()

    train_customers, test_customers = train_test_split(
        customer_ids.unique(), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_mask = customer_ids.isin(train_customers)
    test_mask = customer_ids.isin(test_customers)

    X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()

    overlap = len(set(train_customers).intersection(set(test_customers)))

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

    importance = pd.DataFrame(
        {"Feature": SPENDING_FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    split_info = {
        "train_observations": int(len(X_train)),
        "test_observations": int(len(X_test)),
        "train_customers": int(len(train_customers)),
        "test_customers": int(len(test_customers)),
        "customer_overlap": int(overlap),
    }

    predictions = pd.DataFrame(
        {
            "observation_id": np.arange(1, len(y_test) + 1),
            "Actual_NextOrderSpending": y_test.values,
            "Predicted_NextOrderSpending": y_pred,
            "Residual": residuals,
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
        "X": X,
        "y": y,
    }


def save_and_verify(result):
    model_path = MODEL_DIR / "spending_model.pkl"
    joblib.dump(result["model"], model_path)

    features_path = MODEL_DIR / "spending_features.json"
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump({"features": SPENDING_FEATURES}, fh, indent=2)

    reloaded = joblib.load(model_path)
    sample = result["X"]
    original = result["model"].predict(sample)
    loaded = reloaded.predict(sample)

    verification = {
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "features_path": str(features_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_identical": bool(np.allclose(original, loaded)),
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
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance - Next-Order Spending Prediction")
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
# Risk value demonstration (Task 7)
# --------------------------------------------------------------------------- #
def build_risk_examples(predictions):
    sample = predictions.head(5).copy()
    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        predicted = float(row["Predicted_NextOrderSpending"])
        for churn_p in (0.0, 0.25, 0.50, 0.75, 1.0):
            rows.append(
                {
                    "example_id": i,
                    "churn_probability": churn_p,
                    "predicted_next_order_spending_BRL": round(predicted, 4),
                    "customer_risk_value_BRL": round(churn_p * predicted, 4),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    print("Loading and cleaning Olist data ...")
    clean = clean_olist()
    delivered, spending_features = build_spending_dataset(clean)

    print(f"Cleaned rows: {len(clean)}")
    print(f"Delivered orders: {len(delivered)}")
    print(
        "Customers in delivered data:",
        delivered["customer_unique_id"].nunique(),
    )
    order_counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
    print("Customers with 2+ orders:", int((order_counts >= 2).sum()))
    print("Regression observations:", len(spending_features))
    print("Missing values in features:", int(spending_features.isnull().sum().sum()))

    result = train_and_evaluate(spending_features)
    model_path, verification = save_and_verify(result)

    print("\n-- Evaluation (held-out customers) --")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print("Split info:", result["split_info"])

    # ---- persist tables ----
    metrics_rows = [
        {"Metric": "MAE", "Value": round(result["metrics"]["MAE"], 4)},
        {"Metric": "RMSE", "Value": round(result["metrics"]["RMSE"], 4)},
        {"Metric": "R2", "Value": round(result["metrics"]["R2"], 4)},
        {"Metric": "Median_AE", "Value": round(result["metrics"]["Median_AE"], 4)},
    ]
    pd.DataFrame(metrics_rows).to_csv(
        OUT_DIR / "spending_evaluation_metrics.csv", index=False
    )
    result["importance"].to_csv(
        OUT_DIR / "spending_feature_importance.csv", index=False
    )
    result["predictions"].to_csv(OUT_DIR / "spending_predictions.csv", index=False)
    build_risk_examples(result["predictions"]).to_csv(
        OUT_DIR / "customer_risk_value_examples.csv", index=False
    )

    # ---- figures ----
    plot_feature_importance(
        result["importance"], OUT_DIR / "spending_feature_importance.png"
    )
    plot_actual_vs_predicted(
        result["y_test"], result["y_pred"], OUT_DIR / "spending_actual_vs_predicted.png"
    )
    plot_residuals(result["residuals"], OUT_DIR / "spending_residuals.png")
    plot_target_distribution(result["y"], OUT_DIR / "spending_target_distribution.png")

    # ---- dataset summary + config ----
    summary = pd.DataFrame(
        [
            {"Item": "Raw Olist rows", "Value": 99441},
            {"Item": "Cleaned rows", "Value": len(clean)},
            {"Item": "Delivered orders", "Value": len(delivered)},
            {
                "Item": "Customers (delivered)",
                "Value": int(delivered["customer_unique_id"].nunique()),
            },
            {"Item": "Customers with 2+ orders", "Value": int((order_counts >= 2).sum())},
            {"Item": "Regression observations", "Value": len(spending_features)},
            {
                "Item": "NextOrderSpending mean (BRL)",
                "Value": round(float(spending_features["NextOrderSpending"].mean()), 4),
            },
            {
                "Item": "NextOrderSpending median (BRL)",
                "Value": round(float(spending_features["NextOrderSpending"].median()), 4),
            },
            {
                "Item": "NextOrderSpending max (BRL)",
                "Value": round(float(spending_features["NextOrderSpending"].max()), 4),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    config = {
        "model": "RandomForestRegressor",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split": "customer-level (no overlap)",
        "features": SPENDING_FEATURES,
        "target": "NextOrderSpending",
        "target_units": "BRL (Brazilian Real)",
        "metrics": {k: round(v, 6) for k, v in result["metrics"].items()},
        "split_info": result["split_info"],
        "verification": verification,
    }
    with open(OUT_DIR / "experiment_configuration.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\nModel saved to:", model_path)
    print("Reload verification:", verification)
    print("Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
