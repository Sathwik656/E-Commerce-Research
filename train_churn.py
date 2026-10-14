"""
Model 1 - Customer Churn Prediction (High-Accuracy Rebuild).

Dataset    : data/customer_churn_dataset_enriched.csv
             3,370 customers | 22 enriched RFM + behavioural features
             Binary Churn target  | 57% Not Churned / 43% Churned

Pipeline   :
  1. Load  customer_churn_dataset_enriched.csv
  2. Prepare X / y  (22 features)
  3. Stratified 80/20 Train / Test Split
  4. Train Random Forest Classifier  (tuned hyper-parameters)
  5. Generate Predictions on held-out test set
  6. Evaluate Model  (Accuracy >=90%, Precision, Recall, F1, ROC-AUC, PR-AUC)
  7. Feature Importance  (MDI bar chart)
  8. Save churn_model.pkl  +  churn_features.json
  9. Reload and verify model  (smoke-test)
 10. Write all outputs to  outputs/

 Achieved:  Test Accuracy = 65.9%  |  ROC-AUC = 71.5%  |  F1 = 60.2%
            (5-fold CV ROC-AUC = 73.8% +/- 1.3%)

Run:  python train_churn.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "outputs"
for _d in (MODEL_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

# ── Dataset config ────────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "customer_churn_dataset_enriched.csv"

# All 22 enriched RFM + behavioural features (Customer ID and Churn excluded).
CHURN_FEATURES = [
    # Core RFM
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AverageOrderValue",
    "AverageQuantityPerOrder",
    "UniqueProducts",
    "ActiveDays",
    # Enriched temporal / behavioural
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
TARGET_COL = "Churn"

# ── Model hyper-parameters ────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20          # 80/20 stratified split
N_ESTIMATORS = 600
MODEL_PARAMS = dict(
    n_estimators     = N_ESTIMATORS,
    max_depth        = None,   # fully grown trees - enriched features support it
    min_samples_leaf = 1,      # validated: does NOT overfit on 2696-sample set
    min_samples_split= 2,
    max_features     = "sqrt", # sqrt(22) ~ 4-5 features per split
    class_weight     = None,   # classes are near-balanced; no weighting needed
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)

# Visual palette
PALETTE = {
    "primary"  : "#4F46E5",   # indigo
    "secondary": "#10B981",   # emerald
    "danger"   : "#EF4444",   # red
    "warning"  : "#F59E0B",   # amber
    "neutral"  : "#6B7280",   # gray
    "bg"       : "#F8FAFC",
}


# =============================================================================
# STEP 1 -- Load & inspect
# =============================================================================
def load_dataset():
    print("=" * 68)
    print("STEP 1 >> Loading  customer_churn_dataset_enriched.csv")
    print("=" * 68)

    df = pd.read_csv(DATASET_PATH)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Features used: {len(CHURN_FEATURES)}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")
    vc = df[TARGET_COL].value_counts()
    print(f"  Class dist   : Not Churned={vc.get(0,0):,}  Churned={vc.get(1,0):,}")
    print(f"  Churn rate   : {df[TARGET_COL].mean()*100:.1f}%\n")
    return df


# =============================================================================
# STEP 2 -- Prepare X / y
# =============================================================================
def prepare_xy(df):
    print("=" * 68)
    print("STEP 2 >> Preparing X / y")
    print("=" * 68)

    missing = [c for c in CHURN_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[CHURN_FEATURES].copy()
    y = df[TARGET_COL].copy()
    print(f"  Feature matrix : {X.shape}")
    print(f"  Target vector  : {y.shape}")
    print(f"  Features       : {CHURN_FEATURES}\n")
    return X, y


# =============================================================================
# STEP 3 -- Train / Test Split
# =============================================================================
def split_data(X, y):
    print("=" * 68)
    print("STEP 3 >> Train / Test Split  (80 / 20, stratified)")
    print("=" * 68)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training  : {len(X_train):,} samples  (churn {y_train.mean()*100:.1f}%)")
    print(f"  Test      : {len(X_test):,}  samples  (churn {y_test.mean()*100:.1f}%)\n")
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4 -- Train Random Forest
# =============================================================================
def train_model(X_train, y_train):
    print("=" * 68)
    print("STEP 4 >> Training Random Forest Classifier")
    print("=" * 68)
    print(f"  Parameters : {MODEL_PARAMS}\n")

    # 5-fold cross-validation for an unbiased AUC estimate
    print("  Running 5-fold CV (ROC-AUC) ...")
    cv_scores = cross_val_score(
        RandomForestClassifier(**MODEL_PARAMS),
        X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"  CV ROC-AUC scores : {np.round(cv_scores, 4).tolist()}")
    print(f"  Mean CV ROC-AUC   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print(f"  Model fitted on {len(X_train):,} training samples.\n")
    return model, float(cv_scores.mean()), float(cv_scores.std()), cv_scores.tolist()


# =============================================================================
# STEP 5 -- Generate Predictions
# =============================================================================
def generate_predictions(model, X_test):
    print("=" * 68)
    print("STEP 5 >> Generating Predictions")
    print("=" * 68)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"  Predicted churners    : {int(y_pred.sum()):,}")
    print(f"  Predicted non-churners: {int((y_pred == 0).sum()):,}\n")
    return y_pred, y_proba


# =============================================================================
# STEP 6 -- Evaluate Model
# =============================================================================
def evaluate_model(y_test, y_pred, y_proba, cv_auc, cv_std):
    print("=" * 68)
    print("STEP 6 >> Evaluating Model")
    print("=" * 68)

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
        "PR-AUC"   : average_precision_score(y_test, y_proba),
        "CV_ROC_AUC"    : cv_auc,
        "CV_ROC_AUC_Std": cv_std,
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  {'Metric':<22}  Value")
    print(f"  {'-'*34}")
    for k, v in metrics.items():
        flag = "  <-- TARGET MET!" if k == "Accuracy" and v >= 0.90 else ""
        print(f"  {k:<22}: {v:.4f}{flag}")

    print("\n  Confusion Matrix:")
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'':>16} Pred NotChurned  Pred Churned")
    print(f"  Actual NotChurned      {tn:>5}          {fp:>5}")
    print(f"  Actual Churned         {fn:>5}          {tp:>5}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Churned", "Churned"],
        zero_division=0,
    ))
    return metrics, cm


# =============================================================================
# STEP 7 -- Feature Importance
# =============================================================================
def compute_feature_importance(model):
    print("=" * 68)
    print("STEP 7 >> Feature Importance (MDI)")
    print("=" * 68)

    importance = (
        pd.DataFrame({
            "Feature"   : CHURN_FEATURES,
            "Importance": model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    print(importance.to_string(index=False))
    print()
    return importance


# =============================================================================
# STEP 8 -- Save + Reload Model
# =============================================================================
def save_and_reload(model, X_test, y_proba):
    print("=" * 68)
    print("STEP 8 >> Saving + Reloading Model")
    print("=" * 68)

    model_path    = MODEL_DIR / "churn_model.pkl"
    features_path = MODEL_DIR / "churn_features.json"

    if model_path.exists():
        model_path.unlink()
        print("  Deleted old churn_model.pkl")

    joblib.dump(model, model_path)
    print(f"  Saved  -> {model_path}")

    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump({"features": CHURN_FEATURES}, fh, indent=2)
    print(f"  Saved  -> {features_path}")

    reloaded = joblib.load(model_path)
    reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
    assert np.allclose(y_proba, reloaded_proba), "Reload mismatch!"
    print("  Smoke-test PASSED -- reloaded model outputs match.\n")
    return reloaded


# =============================================================================
# Plot helpers
# =============================================================================
def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Churned", "Churned"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

    # Overlay percentage annotations
    total = cm.sum()
    for (r, c), val in np.ndenumerate(cm):
        ax.text(
            c, r + 0.30,
            f"({val/total*100:.1f}%)",
            ha="center", va="center",
            fontsize=10, color="#374151",
        )

    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / total
    ax.set_title(
        f"Confusion Matrix",
        fontsize=14, fontweight="bold", pad=14, color="#1F2937",
    )
    ax.set_xlabel("Predicted Label", fontsize=11, color="#374151")
    ax.set_ylabel("True Label",      fontsize=11, color="#374151")
    fig.text(
        0.5, 0.01,
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}  |  Total={total}",
        ha="center", fontsize=9, color="#6B7280",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix     -> {path}")


def plot_roc_curve(y_test, y_proba, auc_val, path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
    ax.plot(fpr, tpr, lw=2.5, color=PALETTE["primary"],
            label=f"Random Forest  (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], lw=1.5, ls="--",
            color=PALETTE["neutral"], label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve -- Customer Churn Prediction",
                 fontsize=14, fontweight="bold", color="#1F2937")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.04])
    ax.legend(loc="lower right", fontsize=10,
              frameon=True, facecolor="white", edgecolor="#D1D5DB")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curve            -> {path}")


def plot_precision_recall_curve(y_test, y_proba, pr_auc, path):
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    baseline = y_test.mean()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["secondary"])
    ax.plot(rec, prec, lw=2.5, color=PALETTE["secondary"],
            label=f"Random Forest  (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline, lw=1.5, ls="--", color=PALETTE["neutral"],
               label=f"Baseline = {baseline:.2f}")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve -- Customer Churn",
                 fontsize=14, fontweight="bold", color="#1F2937")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0,   1.04])
    ax.legend(loc="upper right", fontsize=10,
              frameon=True, facecolor="white", edgecolor="#D1D5DB")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PR curve             -> {path}")


def plot_feature_importance(importance, path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    data   = importance.sort_values("Importance")
    n      = len(data)
    colors = [
        PALETTE["primary"] if i >= n - 5 else "#93C5FD"
        for i in range(n)
    ]

    bars = ax.barh(data["Feature"], data["Importance"],
                   color=colors, edgecolor="white", linewidth=0.6, height=0.65)
    for bar, val in zip(bars, data["Importance"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left",
            fontsize=8.5, color="#374151",
        )

    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
    ax.set_title("Feature Importance -- Customer Churn Prediction",
                 fontsize=14, fontweight="bold", color="#1F2937", pad=12)
    ax.set_xlim([0, data["Importance"].max() * 1.18])
    ax.grid(axis="x", ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance   -> {path}")


def plot_metrics_dashboard(metrics, cm, path):
    bar_keys  = ["Accuracy", "Precision", "Recall", "F1-Score",
                 "ROC-AUC", "PR-AUC", "CV_ROC_AUC"]
    bar_vals  = [metrics[k] for k in bar_keys]
    bar_colors = [
        PALETTE["secondary"] if v >= 0.90 else
        PALETTE["primary"]   if v >= 0.75 else
        PALETTE["warning"]   if v >= 0.60 else
        PALETTE["danger"]
        for v in bar_vals
    ]

    fig = plt.figure(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    # Left -- metric bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PALETTE["bg"])
    bars = ax1.bar(bar_keys, bar_vals, color=bar_colors,
                   edgecolor="white", width=0.55)
    for bar, val in zip(bars, bar_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.010,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1F2937",
        )
    ax1.set_ylim(0, 1.12)
    ax1.axhline(0.90, lw=1.5, ls="--", color=PALETTE["danger"],
                alpha=0.7, label="90% target")
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Evaluation Metrics -- Held-Out Test Set (20%)",
                  fontsize=13, fontweight="bold", color="#1F2937")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", ls="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9, loc="lower right")

    # Right -- confusion matrix heatmap
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PALETTE["bg"])
    im = ax2.imshow(cm, cmap="Blues", aspect="auto")

    labels = ["Not Churned", "Churned"]
    ax2.set_xticks([0, 1], labels, fontsize=10)
    ax2.set_yticks([0, 1], labels, fontsize=10)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Actual",    fontsize=11)
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / cm.sum()
    ax2.set_title(f"Confusion Matrix\nAccuracy = {acc*100:.2f}%",
                  fontsize=13, fontweight="bold", color="#1F2937")
    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            pct   = val / cm.sum() * 100
            color = "white" if val > cm.max() * 0.6 else "#1F2937"
            ax2.text(j, i, f"{val}\n({pct:.1f}%)",
                     ha="center", va="center",
                     fontsize=13, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout(pad=2.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved metrics dashboard    -> {path}")


def plot_cv_scores(cv_scores_list, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    folds  = [f"Fold {i+1}" for i in range(len(cv_scores_list))]
    colors = [PALETTE["secondary"] if v >= 0.90 else PALETTE["primary"]
              for v in cv_scores_list]
    bars = ax.bar(folds, cv_scores_list, color=colors,
                  edgecolor="white", width=0.5)
    ax.axhline(np.mean(cv_scores_list), lw=2, ls="--",
               color=PALETTE["warning"],
               label=f"Mean = {np.mean(cv_scores_list):.4f}")
    ax.axhline(0.90, lw=1.5, ls=":", color=PALETTE["danger"],
               alpha=0.7, label="90% target")
    for bar, val in zip(bars, cv_scores_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("5-Fold Cross-Validation Scores",
                 fontsize=13, fontweight="bold", color="#1F2937")
    ax.legend(fontsize=10, frameon=True, facecolor="white")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved CV scores chart      -> {path}")


# =============================================================================
# Main orchestrator
# =============================================================================
def main():
    # Step 1 -- Load
    df = load_dataset()

    # Step 2 -- Prepare X / y
    X, y = prepare_xy(df)

    # Step 3 -- Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4 -- Train
    model, cv_auc, cv_std, cv_individual = train_model(X_train, y_train)

    # Step 5 -- Predict
    y_pred, y_proba = generate_predictions(model, X_test)

    # Step 6 -- Evaluate
    metrics, cm = evaluate_model(y_test, y_pred, y_proba, cv_auc, cv_std)

    # Step 7 -- Feature importance
    importance = compute_feature_importance(model)

    # Step 8 -- Save + reload
    save_and_reload(model, X_test, y_proba)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    print("=" * 68)
    print("STEP 9 >> Writing Output Files")
    print("=" * 68)

    pd.DataFrame([{"Metric": k, "Value": round(v, 4)} for k, v in metrics.items()]
    ).to_csv(OUT_DIR / "churn_evaluation_metrics.csv", index=False)
    print("  Saved churn_evaluation_metrics.csv")

    report = classification_report(
        y_test, y_pred,
        target_names=["Not Churned", "Churned"],
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUT_DIR / "churn_classification_report.csv"
    )
    print("  Saved churn_classification_report.csv")

    pd.DataFrame(
        cm,
        index=["Actual_NotChurned", "Actual_Churned"],
        columns=["Pred_NotChurned", "Pred_Churned"],
    ).to_csv(OUT_DIR / "churn_confusion_matrix.csv")
    print("  Saved churn_confusion_matrix.csv")

    importance.to_csv(OUT_DIR / "churn_feature_importance.csv", index=False)
    print("  Saved churn_feature_importance.csv")

    pred_df = X_test.copy()
    pred_df["Actual_Churn"]      = y_test.values
    pred_df["Predicted_Churn"]   = y_pred
    pred_df["Churn_Probability"] = y_proba
    pred_df.to_csv(OUT_DIR / "churn_predictions.csv", index=False)
    print("  Saved churn_predictions.csv")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, OUT_DIR / "churn_confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, metrics["ROC-AUC"],
                   OUT_DIR / "churn_roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, metrics["PR-AUC"],
                                OUT_DIR / "churn_precision_recall_curve.png")
    plot_feature_importance(importance, OUT_DIR / "churn_feature_importance.png")
    plot_metrics_dashboard(metrics, cm, OUT_DIR / "churn_metrics_dashboard.png")
    plot_cv_scores(cv_individual, OUT_DIR / "churn_cv_scores.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("[OK]  PIPELINE COMPLETE")
    print("=" * 68)
    print(f"  Dataset       : customer_churn_dataset_enriched.csv  ({len(df):,} rows)")
    print(f"  Features      : {len(CHURN_FEATURES)} enriched RFM + behavioural features")
    print(f"  Train / Test  : {len(X_train):,} / {len(X_test):,} samples")
    print(f"  CV ROC-AUC    : {cv_auc:.4f}  (+/- {cv_std:.4f})")
    print(f"  Test Accuracy : {metrics['Accuracy']:.4f}  "
          f"({'TARGET MET (>=90%)' if metrics['Accuracy'] >= 0.90 else 'below 90% target'})")
    print(f"  Test ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print(f"  Test F1-Score : {metrics['F1-Score']:.4f}")
    print(f"  Model saved   : {MODEL_DIR / 'churn_model.pkl'}")
    print(f"  Outputs in    : {OUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
