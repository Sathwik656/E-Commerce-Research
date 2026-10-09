"""
Model 1 - Customer Churn Prediction (improved).

Builds an enriched customer-level feature set from the raw Online Retail II
workbook, trains a tuned Random Forest Classifier, writes research outputs to
outputs/, and saves the model + ordered feature names to models/.

Improvement over the original 8-feature RF (accuracy 0.6291):
  * 22 behavioural features (tenure, spend trend, return/cancellation ratio,
    order-value dispersion, activity ratios, ...)
  * tuned Random Forest (500 trees, max_depth=6, min_samples_leaf=2,
    max_features=0.8)

The enriched customer dataset is written to
data/customer_churn_dataset_enriched.csv; the original
data/customer_churn_dataset.csv is left untouched.

Run:  python train_churn.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for _d in (MODEL_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 500
FUTURE_WINDOW_DAYS = 90

# Order matters: the Flask app must present features in exactly this order.
CHURN_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AverageOrderValue",
    "AverageQuantityPerOrder",
    "UniqueProducts",
    "ActiveDays",
    "Tenure",
    "FirstPurchaseDaysAgo",
    "OrderValueStd",
    "OrderValueMax",
    "OrderValueMin",
    "OrderValueMedian",
    "OrdersPerActiveDay",
    "ProductsPerOrder",
    "SpendPerDay",
    "CountryCount",
    "Cancellations",
    "CancelledValue",
    "ReturnRatio",
    "SpendTrend",
]


def build_dataset():
    """Reproduce the churn target and build the enriched feature matrix."""
    raw = pd.read_excel(DATA_DIR / "online_retail_II.xlsx", engine="openpyxl")
    raw = raw.dropna(subset=["Customer ID"]).copy()
    raw["Customer ID"] = raw["Customer ID"].astype(int)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["TotalAmount"] = raw["Quantity"] * raw["Price"]
    raw["IsCancel"] = raw["Invoice"].str.startswith("C")

    max_date = raw["InvoiceDate"].max()
    obs_end = max_date - pd.DateOffset(days=FUTURE_WINDOW_DAYS)

    sale = raw[(~raw["IsCancel"]) & (raw["Quantity"] > 0) & (raw["Price"] > 0)]
    hist = sale[sale["InvoiceDate"] <= obs_end].copy()
    future = sale[sale["InvoiceDate"] > obs_end].copy()
    future_customers = set(future["Customer ID"].unique())

    g = hist.groupby("Customer ID")
    per_order = (
        hist.groupby(["Customer ID", "Invoice"])["TotalAmount"].sum()
        .groupby("Customer ID")
    )
    first, last = g["InvoiceDate"].min(), g["InvoiceDate"].max()

    feat = pd.DataFrame(index=g.size().index)
    feat["Recency"] = (obs_end - last).dt.days
    feat["Frequency"] = g["Invoice"].nunique()
    feat["Monetary"] = g["TotalAmount"].sum()
    feat["TotalQuantity"] = g["Quantity"].sum()
    feat["AverageOrderValue"] = feat["Monetary"] / feat["Frequency"]
    feat["AverageQuantityPerOrder"] = feat["TotalQuantity"] / feat["Frequency"]
    feat["UniqueProducts"] = g["StockCode"].nunique()
    feat["ActiveDays"] = g["InvoiceDate"].apply(lambda x: x.dt.date.nunique())
    feat["Tenure"] = (last - first).dt.days
    feat["FirstPurchaseDaysAgo"] = (obs_end - first).dt.days
    feat["OrderValueStd"] = per_order.std().fillna(0)
    feat["OrderValueMax"] = per_order.max()
    feat["OrderValueMin"] = per_order.min()
    feat["OrderValueMedian"] = per_order.median()
    feat["OrdersPerActiveDay"] = feat["Frequency"] / feat["ActiveDays"]
    feat["ProductsPerOrder"] = feat["UniqueProducts"] / feat["Frequency"]
    feat["SpendPerDay"] = feat["Monetary"] / (feat["Tenure"] + 1)
    feat["CountryCount"] = g["Country"].nunique()

    cancel = raw[(raw["IsCancel"]) | (raw["Quantity"] < 0)]
    cg = cancel.groupby("Customer ID")
    feat["Cancellations"] = cg.size().reindex(feat.index).fillna(0)
    feat["CancelledValue"] = cg["TotalAmount"].sum().abs().reindex(feat.index).fillna(0)
    feat["ReturnRatio"] = feat["Cancellations"] / (feat["Frequency"] + feat["Cancellations"])

    mid = first + (last - first) / 2
    merged = hist.merge(mid.rename("mid"), left_on="Customer ID", right_index=True)
    first_half = merged[merged["InvoiceDate"] <= merged["mid"]].groupby("Customer ID")["TotalAmount"].sum()
    second_half = merged[merged["InvoiceDate"] > merged["mid"]].groupby("Customer ID")["TotalAmount"].sum()
    feat["SpendTrend"] = (
        second_half.reindex(feat.index).fillna(0) - first_half.reindex(feat.index).fillna(0)
    ) / (feat["Monetary"] + 1)

    feat = feat.fillna(0)
    feat["Churn"] = (~feat.index.isin(future_customers)).astype(int)
    feat.index.name = "Customer ID"
    return feat.reset_index()


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Customer Churn Prediction")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Not Churned", "Churned"])
    ax.set_yticks([0, 1], ["Not Churned", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, label="Number of Customers")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_importance(importance, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    data = importance.sort_values("Importance")
    ax.barh(data["Feature"], data["Importance"], color="#c0653b")
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance - Customer Churn Prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_test, proba, auc, path):
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#c0653b", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Customer Churn Prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_metrics_comparison(churn_metrics, path):
    spending_path = OUT_DIR / "spending_evaluation_metrics.csv"
    if not spending_path.exists():
        return False
    spending = pd.read_csv(spending_path).set_index("Metric")["Value"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    c_names = list(churn_metrics.keys())
    axes[0].bar(c_names, [churn_metrics[k] for k in c_names], color="#c0653b")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Churn (classification)\nmetrics on held-out customers")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=30)

    s_names = ["MAE", "RMSE", "R2", "Median_AE"]
    axes[1].bar(s_names, [spending.get(k, 0) for k in s_names], color="#3b6ea5")
    axes[1].set_title("Spending (regression) metrics\non held-out customers (BRL)")
    axes[1].set_ylabel("Error / R2 (different scale)")

    fig.suptitle(
        "Model Metrics by Task - NOT directly comparable (different tasks/units)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return True


def main():
    print("Building enriched churn dataset from raw data ...")
    df = build_dataset()
    df.to_csv(DATA_DIR / "customer_churn_dataset_enriched.csv", index=False)
    print("Customers:", len(df), "features:", len(CHURN_FEATURES),
          "churn rate:", round(float(df["Churn"].mean()) * 100, 2), "%")

    X = df[CHURN_FEATURES].copy()
    y = df["Churn"].copy()
    if X.isnull().any().any():
        raise ValueError("Enriched feature matrix contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    churn_probability = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    model_path = MODEL_DIR / "churn_model.pkl"
    joblib.dump(model, model_path)
    with open(MODEL_DIR / "churn_features.json", "w", encoding="utf-8") as fh:
        import json

        json.dump({"features": CHURN_FEATURES}, fh, indent=2)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, churn_probability)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }
    print("\n-- Held-out evaluation (improved model) --")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"5-fold CV accuracy: {cv_acc:.4f}  CV ROC-AUC: {cv_auc:.4f}")
    print("Confusion matrix:")
    print(cm)

    pd.DataFrame(
        [{"Metric": k, "Value": round(v, 4)} for k, v in metrics.items()]
    ).to_csv(OUT_DIR / "churn_evaluation_metrics.csv", index=False)

    report = classification_report(
        y_test, y_pred, target_names=["Not Churned", "Churned"], output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / "churn_classification_report.csv")

    pd.DataFrame(
        cm,
        index=["Actual_NotChurned", "Actual_Churned"],
        columns=["Pred_NotChurned", "Pred_Churned"],
    ).to_csv(OUT_DIR / "churn_confusion_matrix.csv")

    importance = pd.DataFrame(
        {"Feature": CHURN_FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance.to_csv(OUT_DIR / "churn_feature_importance.csv", index=False)

    plot_confusion_matrix(cm, OUT_DIR / "churn_confusion_matrix.png")
    plot_feature_importance(importance, OUT_DIR / "churn_feature_importance.png")
    plot_roc_curve(y_test, churn_probability, roc_auc, OUT_DIR / "churn_roc_curve.png")
    made = plot_metrics_comparison(metrics, OUT_DIR / "model_metrics_comparison.png")
    print("Comparison figure written:", made)
    print("Model saved to:", model_path)
    print("Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
