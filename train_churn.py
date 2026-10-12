"""
Model 1 - Customer Churn Prediction (unified Olist design).

Population  : repeat Olist customers (>= 2 delivered orders) as of a snapshot
Features    : RFM + basket / freight / installments / recency / tenure history
              (computed strictly before the snapshot date - no leakage)
Target      : Churn = 1 if the customer places NO delivered order in the
              150 days after the snapshot date
Imbalance   : the repeat-cohort snapshot is ~96% "not churned" (Olist is a
              mostly one-shot marketplace), so the minority class is
              randomly oversampled to 50/50 for training and the held-out
              snapshot is class-balanced for the headline evaluation.
Evaluation  : out-of-time - trained on the 2018-01-01 snapshot, tested on the
              2018-04-01 snapshot (the two snapshots never overlap in time).

Reads data/olist_churn_dataset.csv (built by prepare_data.py), trains a
regularised Random Forest Classifier on a balanced training set, writes research
outputs to outputs/, and saves the model + ordered feature names to models/.

Run:  python prepare_data.py && python train_churn.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from prepare_data import CHURN_FEATURES, HORIZON_DAYS, TRAIN_CUT, TEST_CUT

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for _d in (MODEL_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 500
# Fixed by 5-fold cross-validation on the training snapshot (balanced accuracy).
MODEL_PARAMS = dict(
    n_estimators=N_ESTIMATORS,
    max_depth=4,
    min_samples_leaf=100,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def make_model():
    return RandomForestClassifier(**MODEL_PARAMS)


def load_dataset():
    df = pd.read_csv(DATA_DIR / "olist_churn_dataset.csv")
    train = df[df["snapshot"] == "train"].copy()
    test = df[df["snapshot"] == "test"].copy()
    for part in (train, test):
        if part[CHURN_FEATURES].isnull().any().any():
            raise ValueError("Churn feature matrix contains missing values.")
    return train, test


def oversample_minority(X, y, random_state=RANDOM_STATE):
    """Randomly oversample the minority class up to the majority count (50/50)."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    counts = y.value_counts()
    minority = counts.idxmin()
    n_majority = int(counts.max())
    X_min = X[y == minority]
    y_min = y[y == minority]
    X_over = X_min.sample(n_majority, replace=True, random_state=random_state)
    y_over = y_min.sample(n_majority, replace=True, random_state=random_state)
    X_bal = pd.concat([X, X_over], ignore_index=True)
    y_bal = pd.concat([y, y_over], ignore_index=True)
    return X_bal, y_bal


def balanced_subset(df, random_state=RANDOM_STATE):
    """Return a class-balanced copy of df (undersample the majority class)."""
    pos = df[df["Churn"] == 1]
    neg = df[df["Churn"] == 0]
    n = min(len(pos), len(neg))
    bal = pd.concat(
        [pos.sample(n, random_state=random_state), neg.sample(n, random_state=random_state)]
    )
    return bal.sample(frac=1, random_state=random_state).reset_index(drop=True)


def cross_val_auc(X, y, cv=5):
    """Mean out-of-fold ROC-AUC, oversampling the minority inside each fold."""
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        Xb, yb = oversample_minority(X.iloc[tr_idx], y.iloc[tr_idx])
        m = make_model().fit(Xb, yb)
        proba = m.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[va_idx], proba))
    return float(np.mean(scores))


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Customer Churn (Olist)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Not Churned", "Churned"])
    ax.set_yticks([0, 1], ["Not Churned", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, label="Number of customers")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_importance(importance, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    data = importance.sort_values("Importance")
    ax.barh(data["Feature"], data["Importance"], color="#c0653b")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Feature Importance - Customer Churn (Olist)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_test, proba, auc, path):
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#c0653b", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve - Customer Churn (Olist)")
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
    axes[0].set_title("Churn (classification)\nmetrics on held-out snapshot")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=30)

    s_names = ["MAE", "RMSE", "R2", "Median_AE"]
    axes[1].bar(s_names, [spending.get(k, 0) for k in s_names], color="#3b6ea5")
    axes[1].set_title("Spending (regression)\nmetrics on held-out orders (BRL)")
    axes[1].set_ylabel("Error / R2 (different scale)")

    fig.suptitle(
        "Model metrics by task - NOT directly comparable (different tasks/units)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return True


def main():
    print("Loading unified Olist churn dataset ...")
    train, test = load_dataset()
    print(
        f"Train snapshot {TRAIN_CUT.date()}: {len(train)} customers, "
        f"churn {train.Churn.mean() * 100:.1f}%"
    )
    print(
        f"Test snapshot  {TEST_CUT.date()}: {len(test)} customers, "
        f"churn {test.Churn.mean() * 100:.1f}%"
    )

    X_train, y_train = train[CHURN_FEATURES], train["Churn"]
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train)
    print(
        f"Balanced training set: {len(X_train_bal)} rows "
        f"(churn {y_train_bal.mean() * 100:.1f}%)"
    )

    test_bal = balanced_subset(test)
    X_test, y_test = test_bal[CHURN_FEATURES], test_bal["Churn"]
    print(
        f"Balanced test set: {len(test_bal)} customers "
        f"(churn {y_test.mean() * 100:.1f}%)"
    )

    model = make_model()
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    cv_auc = cross_val_auc(X_train, y_train)

    joblib.dump(model, MODEL_DIR / "churn_model.pkl")
    with open(MODEL_DIR / "churn_features.json", "w", encoding="utf-8") as fh:
        json.dump({"features": CHURN_FEATURES}, fh, indent=2)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
    }
    cm = confusion_matrix(y_test, y_pred)

    full_proba = model.predict_proba(test[CHURN_FEATURES])[:, 1]
    full_auc = roc_auc_score(test["Churn"], full_proba)

    print("\n-- Out-of-time evaluation (2018-04-01 snapshot, class-balanced) --")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"5-fold CV ROC-AUC (train snapshot): {cv_auc:.4f}")
    print(f"Balanced test base rate: {y_test.mean():.4f}")
    print(f"Full (imbalanced) test ROC-AUC / base rate: {full_auc:.4f} / "
          f"{test['Churn'].mean():.4f}")
    print("Confusion matrix (balanced test):\n", cm)

    pd.DataFrame(
        [{"Metric": k, "Value": round(v, 4)} for k, v in metrics.items()]
        + [
            {"Metric": "CV_ROC_AUC", "Value": round(cv_auc, 4)},
            {"Metric": "BalancedTestBaseRate", "Value": round(float(y_test.mean()), 4)},
            {"Metric": "FullTest_ROC_AUC", "Value": round(full_auc, 4)},
            {"Metric": "FullTestBaseRate", "Value": round(float(test["Churn"].mean()), 4)},
        ]
    ).to_csv(OUT_DIR / "churn_evaluation_metrics.csv", index=False)

    report = classification_report(
        y_test, y_pred, target_names=["Not Churned", "Churned"],
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / "churn_classification_report.csv")

    pd.DataFrame(
        cm,
        index=["Actual_NotChurned", "Actual_Churned"],
        columns=["Pred_NotChurned", "Pred_Churned"],
    ).to_csv(OUT_DIR / "churn_confusion_matrix.csv")

    importance = (
        pd.DataFrame({"Feature": CHURN_FEATURES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUT_DIR / "churn_feature_importance.csv", index=False)

    plot_confusion_matrix(cm, OUT_DIR / "churn_confusion_matrix.png")
    plot_feature_importance(importance, OUT_DIR / "churn_feature_importance.png")
    plot_roc_curve(y_test, proba, metrics["ROC-AUC"], OUT_DIR / "churn_roc_curve.png")
    made = plot_metrics_comparison(metrics, OUT_DIR / "model_metrics_comparison.png")

    print("Comparison figure written:", made)
    print("Model saved to:", MODEL_DIR / "churn_model.pkl")


if __name__ == "__main__":
    main()
