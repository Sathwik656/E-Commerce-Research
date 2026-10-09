# E-commerce Customer Churn and Spending Prediction using Machine Learning

> **Research Paper Implementation** — Two independent Random Forest models trained on two public e-commerce datasets, deployed through a Flask web prototype, and combined into a prototype Customer Risk Value indicator.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Questions](#2-research-questions)
3. [Solution Materials and Methodology](#3-solution-materials-and-methodology)
4. [Datasets](#4-datasets)
5. [Data Cleaning and Preprocessing](#5-data-cleaning-and-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Algorithms Used](#7-machine-learning-algorithms-used)
8. [Tech Stack](#8-tech-stack)
9. [Project Workflow](#9-project-workflow)
10. [Project Structure](#10-project-structure)
11. [Setup and Installation](#11-setup-and-installation)
12. [Reproducing the Models](#12-reproducing-the-models)
13. [Running the Flask Application](#13-running-the-flask-application)
14. [What the Models Evaluate](#14-what-the-models-evaluate)
15. [Outputs and Results](#15-outputs-and-results)
16. [Output Screenshots](#16-output-screenshots)
17. [Customer Risk Value](#17-customer-risk-value)
18. [What Information the System Provides](#18-what-information-the-system-provides)
19. [Tests](#19-tests)
20. [Limitations and Threats to Validity](#20-limitations-and-threats-to-validity)
21. [Reproducibility](#21-reproducibility)

---

## 1. Project Overview

This project addresses two key business intelligence challenges in e-commerce through supervised machine learning:

| Task | Type | Description |
|---|---|---|
| **Customer Churn Prediction** | Binary Classification | Predicts whether a customer will stop purchasing in the next 90 days |
| **Next-Order Spending Prediction** | Regression | Predicts the payment value (BRL) a customer will spend on their next observed order |

The two model outputs are combined into a prototype **Customer Risk Value** indicator:

```
Customer Risk Value (BRL) = Churn Probability × Predicted Next-Order Spending
```

This combined metric helps businesses identify which at-risk customers represent the greatest potential revenue loss — enabling targeted retention strategies.

Both models are served through a **Flask web application** that accepts CSV uploads and returns a results table per row.

> **Important:** The churn and spending models are trained on two different customer populations with no shared identifiers. The Customer Risk Value is a **prototype indicator**, not a validated financial estimate.

---

## 2. Research Questions

| RQ | Question |
|---|---|
| **RQ1** | Can a Random Forest predict customer churn from behavioural RFM-style features extracted from raw transaction data? |
| **RQ2** | Can a Random Forest predict the payment value of a customer's next order from historical order features, without introducing target leakage? |
| **RQ3** | How can the two model outputs be combined into a transparent, clearly-bounded risk indicator for demonstration purposes? |

---

## 3. Solution Materials and Methodology

### Approach Summary

The project follows a standard supervised machine learning pipeline applied independently to two datasets:

```
Raw Data → Cleaning → Feature Engineering → Train/Test Split → Model Training → Evaluation → Serialization → Flask Deployment
```

### Methodology Steps

| Step | Churn Model | Spending Model |
|---|---|---|
| **1. Data Source** | Online Retail II (UCI ML) | Olist Brazilian E-Commerce (Kaggle) |
| **2. Cleaning** | Drop null Customer IDs, separate cancellations | Parse timestamps, filter delivered + payment_value > 0 |
| **3. Aggregation** | Customer-level RFM + 14 engineered features | Chronological cumulative order history features |
| **4. Target Definition** | Churn = 1 if no purchase in 90-day future window | NextOrderSpending = current order's payment_value |
| **5. Split** | Stratified 80/20 split by Churn | Customer-level 80/20 (no customer overlap) |
| **6. Model** | RandomForestClassifier (500 trees) | RandomForestRegressor (200 trees) |
| **7. Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC | MAE, RMSE, R², Median AE |
| **8. Serialization** | `churn_model.pkl` + `churn_features.json` | `spending_model.pkl` + `spending_features.json` |

---

## 4. Datasets

| Dataset | Source | Role | Size |
|---|---|---|---|
| **Online Retail II** | [UCI ML Repository (dataset 502)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Churn model training | UK retail invoices 2009–2011 |
| **Olist Brazilian E-Commerce** | [Kaggle `olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Spending model training | 99,441 Brazilian marketplace orders |

### Dataset Details

**Online Retail II (Churn)**
- UK-based online retailer transaction records (2009–2011)
- Columns used: `Quantity`, `Price`, `InvoiceDate`, `Customer ID`, `StockCode`, `Invoice`, `Country`
- Aggregated from transaction-level to **customer-level** (3,370 customers)
- Enriched dataset saved to: `data/customer_churn_dataset_enriched.csv`
- Original 8-feature dataset preserved as: `data/customer_churn_dataset.csv`

**Olist (Spending)**
- Brazilian e-commerce marketplace orders
- Multiple raw tables combined via `combiner.ipynb` into `data/olist_spending_dataset.csv`
- Columns used: `customer_unique_id`, `order_purchase_timestamp`, `payment_value`, `number_of_items`, `number_of_products`, `number_of_sellers`, `total_item_price`, `total_freight_value`, `payment_installments`, `order_status`

> These two datasets describe **different customer populations** and share no customer identifiers. They cannot be joined at the customer level.

---

## 5. Data Cleaning and Preprocessing

### Churn Dataset Cleaning (`train_churn.py`)

1. Drop all rows with missing `Customer ID`
2. Parse `InvoiceDate` to datetime
3. Convert `Customer ID` to integer
4. Compute `TotalAmount = Quantity × Price`
5. Flag cancellation invoices (`Invoice` starting with `"C"`)
6. Define observation window end: `last_transaction_date − 90 days`
7. Split into **historical** (before observation end) and **future** windows
8. Separate cancellations from valid sales (positive quantity + positive price)

### Spending Dataset Cleaning (`train_spending.py`)

1. Parse `order_purchase_timestamp` to datetime
2. Drop rows with missing `customer_unique_id` or `order_purchase_timestamp`
3. Keep only `payment_value > 0` → 99,437 rows
4. Keep only `order_status == "delivered"` → 96,477 orders
5. Drop rows with any missing value in modelling columns
6. Sort by (`customer_unique_id`, `order_purchase_timestamp`) for chronological ordering

---

## 6. Feature Engineering

### Churn Features — 22 Behavioural Features

The original 8-feature RFM set was expanded to 22 features:

**Original 8 RFM Features:**

| Feature | Description |
|---|---|
| `Recency` | Days since last purchase (relative to observation window end) |
| `Frequency` | Number of unique invoices |
| `Monetary` | Total spend (£) during observation period |
| `TotalQuantity` | Total items purchased |
| `AverageOrderValue` | Monetary ÷ Frequency |
| `AverageQuantityPerOrder` | TotalQuantity ÷ Frequency |
| `UniqueProducts` | Number of distinct stock codes purchased |
| `ActiveDays` | Number of distinct calendar days with purchases |

**14 Engineered Features (Added):**

| Feature | Description |
|---|---|
| `Tenure` | Days between first and last purchase |
| `FirstPurchaseDaysAgo` | Days from first purchase to observation window end |
| `OrderValueStd` | Standard deviation of per-order values |
| `OrderValueMax` | Maximum per-order value |
| `OrderValueMin` | Minimum per-order value |
| `OrderValueMedian` | Median per-order value |
| `OrdersPerActiveDay` | Frequency ÷ ActiveDays |
| `ProductsPerOrder` | UniqueProducts ÷ Frequency |
| `SpendPerDay` | Monetary ÷ (Tenure + 1) |
| `CountryCount` | Number of distinct countries in orders |
| `Cancellations` | Number of cancelled/returned items |
| `CancelledValue` | Absolute value of cancelled transactions |
| `ReturnRatio` | Cancellations ÷ (Frequency + Cancellations) |
| `SpendTrend` | (Second-half spend − First-half spend) ÷ (Monetary + 1) |

**Churn Target Definition:**
- `Churn = 1` if the customer has **no** purchase in the 90-day future window
- `Churn = 0` if the customer purchased at least once in the future window
- **Base churn rate: 43.0%**

### Spending Features — 9 Cumulative Historical Features

For each customer order, features are **cumulative sums of all previous orders only** (strict leakage control):

| Feature | Description |
|---|---|
| `Recency` | Days since previous order |
| `PreviousOrderCount` | Count of prior orders (first order dropped — no history) |
| `HistoricalSpending` | Cumulative total spend from prior orders (BRL) |
| `TotalItems` | Cumulative total items from prior orders |
| `TotalProducts` | Cumulative distinct products from prior orders |
| `TotalSellers` | Cumulative distinct sellers from prior orders |
| `AverageOrderValue` | HistoricalSpending ÷ PreviousOrderCount |
| `AverageFreightValue` | Cumulative freight ÷ PreviousOrderCount |
| `AverageInstallments` | Cumulative installments ÷ PreviousOrderCount |

**Next-Order Spending Target:**
- `NextOrderSpending = current order's payment_value (BRL)`
- Only customers with 2+ orders are included (first order has no prior history)
- **Leakage control**: target-order quantity, price, freight, and payment fields are never used as features

**Resulting Regression Dataset:**

| Item | Value |
|---|---:|
| Cleaned rows | 99,437 |
| Delivered orders | 96,477 |
| Customers (delivered) | 93,357 |
| Customers with 2+ orders | 2,801 |
| Regression observations | 3,120 |
| NextOrderSpending mean (BRL) | 146.92 |
| NextOrderSpending median (BRL) | 100.30 |
| NextOrderSpending max (BRL) | 2,621.29 |

---

## 7. Machine Learning Algorithms Used

### Model 1 — Random Forest Classifier (Churn Prediction)

**Algorithm:** `sklearn.ensemble.RandomForestClassifier`

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Larger forest for variance reduction |
| `max_depth` | 6 | Prevents overfitting on noisy churn target |
| `min_samples_leaf` | 2 | Regularisation on small leaf nodes |
| `max_features` | 0.8 | Feature subsampling per split |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training on all CPU cores |
| `stratify` | `y` | Preserves class balance in train/test split |

**Cross-Validation:** 5-fold Stratified K-Fold (`StratifiedKFold`)

**Improvement over baseline:**
- Original (8 features, default RF): Accuracy **0.6291**, ROC-AUC **0.6826**
- Improved (22 features, tuned RF): Accuracy **0.6706**, ROC-AUC **0.7297**
- 5-fold CV: Accuracy **0.6923**, ROC-AUC **0.7591**

> A candidate-model search (tuned Random Forest, Gradient Boosting, Logistic Regression) all converged to ~0.68–0.69 accuracy / ~0.75–0.76 AUC, confirming the practical signal ceiling for this target.

---

### Model 2 — Random Forest Regressor (Spending Prediction)

**Algorithm:** `sklearn.ensemble.RandomForestRegressor`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `random_state` | 42 |
| `n_jobs` | -1 |

**Train/Test Split Strategy:** Customer-level split (not observation-level) to prevent data leakage:
- 2,240 train customers / 561 test customers
- 2,499 train observations / 621 test observations
- **Customer overlap = 0** (fully disjoint sets)

---

## 8. Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.x | Core implementation |
| **ML Framework** | scikit-learn ≥ 1.3 | RandomForestClassifier, RandomForestRegressor, metrics |
| **Data Processing** | pandas ≥ 2.0, numpy ≥ 1.24 | Data manipulation, feature engineering |
| **Model Persistence** | joblib ≥ 1.3 | Model serialisation / deserialisation |
| **Visualisation** | matplotlib ≥ 3.7 | All research figures (300 DPI) |
| **Excel Parsing** | openpyxl ≥ 3.1 | Reading Online Retail II `.xlsx` workbook |
| **Web Framework** | Flask ≥ 3.0 | REST prototype with CSV upload |
| **Notebooks** | Jupyter (`.ipynb`) | Incremental implementation and exploration |

### Dependencies (`requirements.txt`)
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
matplotlib>=3.7
openpyxl>=3.1
Flask>=3.0
```

---

## 9. Project Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        E-COMMERCE ML PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────┘

  DATA SOURCES
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  Online Retail II    │       │  Olist Brazilian         │
  │  (UCI ML Repo)       │       │  E-Commerce (Kaggle)     │
  │  online_retail_II    │       │  olist_spending_dataset  │
  │  .xlsx               │       │  .csv                    │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  CLEANING            │       │  CLEANING                │
  │  • Drop null CustID  │       │  • Parse timestamps      │
  │  • Separate cancels  │       │  • Filter delivered      │
  │  • Obs. window split │       │  • payment_value > 0     │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  FEATURE ENGINEERING │       │  FEATURE ENGINEERING     │
  │  22 RFM + behavioural│       │  9 cumulative historical │
  │  customer features   │       │  order features          │
  │  Churn target (90d)  │       │  NextOrderSpending target│
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  SPLIT               │       │  SPLIT (customer-level)  │
  │  Stratified 80/20    │       │  80/20, zero overlap     │
  │  2696 train/674 test │       │  2499 obs / 621 test     │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  MODEL TRAINING      │       │  MODEL TRAINING          │
  │  RandomForest        │       │  RandomForest            │
  │  Classifier          │       │  Regressor               │
  │  500 trees, depth=6  │       │  200 trees               │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  EVALUATION          │       │  EVALUATION              │
  │  Acc: 0.6706         │       │  MAE: 95.21 BRL          │
  │  ROC-AUC: 0.7297     │       │  RMSE: 168.49 BRL        │
  │  F1: 0.6361          │       │  R²: 0.0608              │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │  churn_model.pkl     │       │  spending_model.pkl      │
  │  churn_features.json │       │  spending_features.json  │
  └──────────┬───────────┘       └──────────┬───────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   FLASK APPLICATION     │
              │   app.py               │
              │   ┌─────────────────┐  │
              │   │  Upload Churn   │  │
              │   │  CSV (22 cols)  │  │
              │   └────────┬────────┘  │
              │            │           │
              │   ┌────────▼────────┐  │
              │   │  Upload Spend   │  │
              │   │  CSV (9 cols)   │  │
              │   └────────┬────────┘  │
              │            │           │
              │            ▼           │
              │   ┌─────────────────┐  │
              │   │ Churn Prob +    │  │
              │   │ Churn Status +  │  │
              │   │ Spend Pred +    │  │
              │   │ Risk Value      │  │
              │   └─────────────────┘  │
              └─────────────────────────┘
```

---

## 10. Project Structure

```text
Research_Paper/
├── app.py                          Flask application (churn + spending + risk value)
├── train_churn.py                  Reproduce churn model, metrics, and figures
├── train_spending.py               Full spending pipeline (clean → features → train → save)
├── combiner.ipynb                  Combines raw Olist tables into olist_spending_dataset.csv
├── requirements.txt                Python dependencies
│
├── data/
│   ├── online_retail_II.xlsx       Raw Online Retail II workbook (churn source)
│   ├── customer_churn_dataset.csv  Original 8-feature churn dataset (preserved)
│   ├── customer_churn_dataset_     Enriched 22-feature customer dataset (generated)
│   │   enriched.csv
│   ├── ecommerce_customers.csv     Supplemental reference dataset
│   ├── olist_spending_dataset.csv  Combined Olist orders (spending source)
│   └── olist_spending_cleaned.csv  Cleaned Olist data
│
├── models/
│   ├── churn_model.pkl             Serialised Random Forest Classifier
│   ├── churn_features.json         Ordered feature names for churn model
│   ├── spending_model.pkl          Serialised Random Forest Regressor
│   └── spending_features.json      Ordered feature names for spending model
│
├── notebooks/
│   ├── model1.ipynb                Churn: incremental implementation record
│   └── model2.ipynb                Spending: incremental implementation record
│
├── outputs/                        Research tables, figures, reports
│   ├── README.md                   Output file catalogue
│   ├── implementation_report.md    Full methodology and interpretations
│   ├── churn_evaluation_metrics.csv
│   ├── churn_classification_report.csv
│   ├── churn_confusion_matrix.csv
│   ├── churn_feature_importance.csv
│   ├── spending_evaluation_metrics.csv
│   ├── spending_feature_importance.csv
│   ├── spending_predictions.csv
│   ├── customer_risk_value_examples.csv
│   ├── dataset_summary.csv
│   ├── experiment_configuration.json
│   ├── churn_confusion_matrix.png
│   ├── churn_feature_importance.png
│   ├── churn_roc_curve.png
│   ├── spending_feature_importance.png
│   ├── spending_actual_vs_predicted.png
│   ├── spending_residuals.png
│   ├── spending_target_distribution.png
│   └── model_metrics_comparison.png
│
├── templates/
│   └── index.html                  Flask UI (two CSV uploads + results table)
│
├── sample_inputs/
│   ├── churn_input.csv             Example churn features (22 columns + customer_id)
│   └── spending_input.csv          Example spending features (9 columns + customer_id)
│
├── docs/
│   ├── Info.md                     Project notes and task breakdown
│   └── sample_report.pdf           Sample report reference
│
└── tests/
    └── test_pipeline.py            Automated pipeline tests (11 tests)
```

---

## 11. Setup and Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd Research_Paper

# Install all dependencies
pip install -r requirements.txt
```

---

## 12. Reproducing the Models

Both training scripts use `random_state=42` and write all results to `outputs/`.

```bash
# Step 1: Train the churn model
# Reads: data/online_retail_II.xlsx
# Writes: models/churn_model.pkl, models/churn_features.json
#         outputs/churn_*.csv, outputs/churn_*.png
python train_churn.py

# Step 2: Train the spending model
# Reads: data/olist_spending_dataset.csv
# Writes: models/spending_model.pkl, models/spending_features.json
#         outputs/spending_*.csv, outputs/spending_*.png
#         outputs/dataset_summary.csv, outputs/experiment_configuration.json
python train_spending.py
```

Both scripts are fully deterministic — reruns reproduce the reported metrics exactly.

---

## 13. Running the Flask Application

```bash
python app.py               # Starts at http://127.0.0.1:5000
```

### Application Workflow

1. Navigate to `http://127.0.0.1:5000`
2. Upload two CSV files:
   - **Churn Feature CSV** — 22 enriched behavioural columns (+ optional `customer_id`)
   - **Spending Feature CSV** — 9 historical order columns (+ optional `customer_id`)
3. Click **Predict**
4. View the results table with:
   - Churn probability per customer
   - Predicted churn status (Churned / Not Churned)
   - Predicted next-order spending (BRL)
   - Customer Risk Value (BRL)

### Sample Input Files

Use the provided samples in `sample_inputs/` for testing:

```bash
# Churn input: 22 feature columns + optional customer_id
sample_inputs/churn_input.csv

# Spending input: 9 feature columns + optional customer_id
sample_inputs/spending_input.csv
```

**Churn CSV required columns (22):**
```
Recency, Frequency, Monetary, TotalQuantity, AverageOrderValue,
AverageQuantityPerOrder, UniqueProducts, ActiveDays, Tenure,
FirstPurchaseDaysAgo, OrderValueStd, OrderValueMax, OrderValueMin,
OrderValueMedian, OrdersPerActiveDay, ProductsPerOrder, SpendPerDay,
CountryCount, Cancellations, CancelledValue, ReturnRatio, SpendTrend
```

**Spending CSV required columns (9):**
```
Recency, PreviousOrderCount, HistoricalSpending, TotalItems,
TotalProducts, TotalSellers, AverageOrderValue,
AverageFreightValue, AverageInstallments
```

### Row Pairing Logic

| Scenario | Behaviour |
|---|---|
| Both files contain `customer_id` | Inner join on `customer_id` |
| One or both files missing `customer_id` | Paired by row order; extra rows ignored |

### Validation Rules

The app rejects uploads with:
- Missing required feature columns
- Non-numeric feature values
- Missing (NaN) feature values
- Negative values in spending features
- Empty files or unreadable CSVs

Returns **HTTP 400** with a descriptive error message per file.

---

## 14. What the Models Evaluate

### Model 1: Churn Classifier — What It Evaluates

The churn model evaluates **binary classification performance** on a held-out test set of 674 customers:

| Metric | Formula | What It Tells You |
|---|---|---|
| **Accuracy** | Correct predictions ÷ Total | Overall proportion of correct churn/no-churn calls |
| **Precision** | TP ÷ (TP + FP) | Of customers predicted as churners, how many actually churned |
| **Recall** | TP ÷ (TP + FN) | Of actual churners, how many were correctly identified |
| **F1-Score** | 2 × (Precision × Recall) ÷ (Precision + Recall) | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve | Model's ability to rank churners above non-churners (1.0 = perfect) |
| **5-fold CV Accuracy** | Mean accuracy across 5 folds | Generalisation estimate on the full dataset |
| **Confusion Matrix** | TN, FP, FN, TP counts | Per-class error breakdown |

### Model 2: Spending Regressor — What It Evaluates

The spending model evaluates **regression performance** on 621 held-out observations from 561 unseen customers:

| Metric | Formula | What It Tells You |
|---|---|---|
| **MAE** | Mean(|actual − predicted|) | Average absolute error in BRL — interpretable dollar amount |
| **RMSE** | √Mean((actual − predicted)²) | Square-root error; penalises large errors more than MAE |
| **R²** | 1 − SS_res ÷ SS_tot | Proportion of variance explained (0 = mean-only model, 1 = perfect) |
| **Median AE** | Median(|actual − predicted|) | Robust error estimate; less sensitive to outliers than MAE |

---

## 15. Outputs and Results

### Churn Model Results (Held-out: 674 customers)

| Metric | Original (8 features) | **Improved (22 features, tuned)** |
|---|---:|---:|
| Accuracy | 0.6291 | **0.6706** |
| Precision | 0.5654 | **0.6062** |
| Recall | 0.5966 | **0.6690** |
| F1-Score | 0.5805 | **0.6361** |
| ROC-AUC | 0.6826 | **0.7297** |
| 5-fold CV Accuracy | — | **0.6923** |
| 5-fold CV ROC-AUC | — | **0.7591** |

**Confusion Matrix (Improved Model):**

|  | Predicted: Not Churned | Predicted: Churned |
|---|---:|---:|
| **Actual: Not Churned** | 258 (TN) | 126 (FP) |
| **Actual: Churned** | 96 (FN) | 194 (TP) |

### Spending Model Results (Held-out: 621 observations, 561 customers)

| Metric | Value |
|---|---:|
| MAE (BRL) | **95.21** |
| RMSE (BRL) | **168.49** |
| R² | **0.0608** |
| Median AE (BRL) | **56.70** |

**Predicted vs Actual Summary:**

| Statistic | Actual (BRL) | Predicted (BRL) | Residual (BRL) |
|---|---:|---:|---:|
| Mean | 158.22 | 146.45 | 11.77 |
| Std | 173.99 | 80.88 | 168.21 |
| Min | 14.78 | 37.92 | -745.01 |
| Median | 103.26 | 128.82 | -20.99 |
| Max | 1391.79 | 827.74 | 1290.91 |

### Top Feature Importances

**Churn Model — Top 5 Features (by mean decrease in impurity):**

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | ActiveDays | 0.2382 |
| 2 | Frequency | 0.0870 |
| 3 | Monetary | 0.0857 |
| 4 | Recency | 0.0701 |
| 5 | SpendTrend | 0.0545 |

**Spending Model — Top 5 Features (by mean decrease in impurity):**

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | AverageOrderValue | 0.3217 |
| 2 | HistoricalSpending | 0.2049 |
| 3 | Recency | 0.1953 |
| 4 | AverageFreightValue | 0.1738 |
| 5 | AverageInstallments | 0.0622 |

> Feature importance reflects **predictive contribution within the fitted forest**, not causation.

---

## 16. Output Screenshots

### Churn Model — Confusion Matrix

The confusion matrix shows the classification performance on 674 held-out customers. TN = 258 (correctly predicted not churned), TP = 194 (correctly predicted churned), FP = 126 (false alarms), FN = 96 (missed churners).

![Confusion Matrix - Customer Churn Prediction](outputs/churn_confusion_matrix.png)

---

### Churn Model — ROC Curve

The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate across all classification thresholds. AUC = 0.7297 indicates the model is substantially better than random (AUC = 0.5) at ranking churners above non-churners.

![ROC Curve - Customer Churn Prediction](outputs/churn_roc_curve.png)

---

### Churn Model — Feature Importance

`ActiveDays` dominates feature importance (0.2382), confirming that the number of distinct purchase days is the strongest behavioural signal for churn. Frequency and Monetary (classic RFM) follow. The engineered `SpendTrend` and `ReturnRatio` contribute meaningfully.

![Feature Importance - Customer Churn Prediction](outputs/churn_feature_importance.png)

---

### Spending Model — Actual vs Predicted

The scatter plot compares actual vs predicted next-order spending values on the held-out test set. Points on or near the red dashed line indicate accurate predictions. The model regresses toward the mean (predicted range is narrower than actual), which is typical for Random Forest regressors on skewed targets.

![Actual vs Predicted Next-Order Spending](outputs/spending_actual_vs_predicted.png)

---

### Spending Model — Feature Importance

`AverageOrderValue` (0.3217) is the single most important predictor of next-order spending, followed by `HistoricalSpending` (0.2049) and `Recency` (0.1953). Together, the top 4 features account for ~89% of total importance.

![Feature Importance - Next-Order Spending Prediction](outputs/spending_feature_importance.png)

---

### Spending Model — Residual Distribution

The residual histogram (actual − predicted) shows a roughly symmetric distribution centred near zero, with most residuals in the −250 to +250 BRL range. A small number of large positive residuals correspond to high-value actual orders that the model underestimates.

![Residual Distribution - Next-Order Spending](outputs/spending_residuals.png)

---

### Spending Model — Target Distribution

The next-order spending distribution is right-skewed, with most orders below BRL 500 and a long tail extending to ~BRL 2,600. This skewness contributes to the relatively low R² and is a key modelling challenge.

![Next-Order Spending Target Distribution](outputs/spending_target_distribution.png)

---

### Combined — Model Metrics Comparison

Side-by-side comparison of both model evaluation metrics. Note: the two panels measure different quantities (classification scores vs regression errors in BRL) and are **not directly comparable**.

![Model Metrics Comparison](outputs/model_metrics_comparison.png)

---

## 17. Customer Risk Value

### Formula

```
Customer Risk Value (BRL) = Churn Probability × Predicted Next-Order Spending
```

Where:
- **Churn Probability** ∈ [0.0, 1.0] — output of the Random Forest Classifier
- **Predicted Next-Order Spending** ≥ 0 (BRL) — output of the Random Forest Regressor (clipped to non-negative)

### Interpretation

| Churn Probability | Predicted Spending (BRL) | Risk Value (BRL) | Interpretation |
|---:|---:|---:|---|
| 0.00 | 500.00 | 0.00 | Customer will not churn — no at-risk revenue |
| 0.25 | 500.00 | 125.00 | 25% churn risk on R$500 potential spend |
| 0.50 | 500.00 | 250.00 | Equal chance of churning, R$250 at risk |
| 0.75 | 500.00 | 375.00 | High churn risk, R$375 potential loss |
| 1.00 | 500.00 | 500.00 | Customer will churn — full spend value at risk |

### Limitations

> **The Customer Risk Value is a prototype demonstration indicator, not a validated financial estimate.**
> - The churn and spending models come from **different customer populations** (UK retailer vs Brazilian marketplace)
> - The spending model predicts next-order value **conditional on a subsequent order existing** — churned customers have no next order
> - No causal relationship between churn and revenue loss is established
> - The combined value should be treated as a relative prioritisation signal only

---

## 18. What Information the System Provides

### Per-Customer Predictions (Flask App Output)

For each row in the uploaded CSVs, the system returns:

| Output Field | Type | Description |
|---|---|---|
| `customer_id` | String | Customer identifier (if provided), otherwise `row_N` |
| `churn_probability` | Float [0, 1] | Estimated probability of churn in the next 90 days |
| `predicted_churn` | String | `"Churned"` or `"Not Churned"` (threshold: 0.5) |
| `predicted_next_order_spending_BRL` | Float ≥ 0 | Predicted BRL value of the customer's next order |
| `customer_risk_value_BRL` | Float ≥ 0 | churn_probability × predicted_next_order_spending_BRL |

### Batch Summary Statistics

The results page also shows:
- **Total rows processed**
- **Mean Customer Risk Value** across all customers
- **Total Customer Risk Value** (portfolio-level risk indicator)
- **Pairing note**: whether rows were joined on `customer_id` or paired by row order

### Research Output Files (Generated by Training Scripts)

| File | Description |
|---|---|
| `outputs/churn_evaluation_metrics.csv` | Accuracy, Precision, Recall, F1, ROC-AUC |
| `outputs/churn_classification_report.csv` | Per-class precision/recall/F1/support |
| `outputs/churn_confusion_matrix.csv` | TN/FP/FN/TP counts |
| `outputs/churn_feature_importance.csv` | Feature importance scores (all 22 features) |
| `outputs/spending_evaluation_metrics.csv` | MAE, RMSE, R², Median AE |
| `outputs/spending_feature_importance.csv` | Feature importance scores (all 9 features) |
| `outputs/spending_predictions.csv` | Actual, predicted, residual per test observation |
| `outputs/customer_risk_value_examples.csv` | Example risk values at 5 churn probability levels |
| `outputs/dataset_summary.csv` | Dataset row/customer counts and target statistics |
| `outputs/experiment_configuration.json` | Full model config, feature order, split info, verification |
| `outputs/implementation_report.md` | Full methodology, interpretations, and limitations |

---

## 19. Tests

```bash
# Run all tests
python tests/test_pipeline.py

# Or with pytest
pytest tests/test_pipeline.py -v
```

**Results: 11 passed, 0 failed**

| Test | Description |
|---|---|
| Churn model load | Model file exists and is loadable |
| Churn model type | Confirms `RandomForestClassifier` |
| Churn feature count | Model expects exactly 22 features |
| Churn probability range | All probabilities ∈ [0, 1] |
| Churn output shape | Output shape matches input rows |
| Spending model load | Model file exists and is loadable |
| Spending model type | Confirms `RandomForestRegressor` |
| Spending feature count | Model expects exactly 9 features |
| Spending feature order match | Saved JSON order matches model training order |
| Spending prediction finite | No NaN or Inf in predictions |
| Risk value: prob = 0 | Risk Value = 0 when churn probability = 0 |
| Risk value: prob = 1 | Risk Value = predicted spending when churn probability = 1 |
| Risk value: intermediate | Risk Value = prob × spending (e.g., 0.75 × 500 = 375) |
| Risk value non-negative | All risk values ≥ 0 |
| Flask main route | GET / returns HTTP 200 |
| Valid two-file upload | Produces results table with risk column |
| Missing churn columns | Returns HTTP 400 with error message |
| Missing spending columns | Returns HTTP 400 with error message |
| Non-numeric rejection | Returns HTTP 400 for non-numeric feature values |
| Row-order pairing | Pairs by row order when customer_id absent |

---

## 20. Limitations and Threats to Validity

| Limitation | Description |
|---|---|
| **Different populations** | Online Retail II (UK) and Olist (Brazil) customers are distinct; no shared ID exists; cross-dataset matching is not performed |
| **Conditional spending target** | The spending model predicts the next observed order's value **conditional on a subsequent order existing** — customers who never reorder are excluded |
| **Low R²** | The spending model explains only ~6% of held-out variance; predictions are weak signals for individual customers |
| **No temporal hold-out (spending)** | The customer-level split prevents identity leakage but is random, not chronological |
| **Churn accuracy ceiling** | All models (RF, Gradient Boosting, Logistic Regression) converge to ~0.68–0.69 accuracy / ~0.75–0.76 AUC due to the inherent noise in the 90-day churn target |
| **Single-split churn evaluation** | The churn model was tuned and reported on one stratified split; differences are not tested for statistical significance |
| **Feature importance ≠ causation** | MDI importance reflects predictive contribution, not causal effect |
| **Risk value not financially validated** | It is a prototype demonstration, not a validated financial or business outcome |
| **90-day churn window** | The 90-day future window is a design choice; different windows would produce different churn rates and model performance |

---

## 21. Reproducibility

All scripts use `random_state=42` for full determinism.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce churn model, metrics, and figures
python train_churn.py
# → models/churn_model.pkl
# → models/churn_features.json
# → outputs/churn_*.csv, outputs/churn_*.png

# 3. Reproduce spending model, metrics, and figures
python train_spending.py
# → models/spending_model.pkl
# → models/spending_features.json
# → outputs/spending_*.csv, outputs/spending_*.png
# → outputs/dataset_summary.csv
# → outputs/experiment_configuration.json

# 4. Start the Flask application
python app.py
# → http://127.0.0.1:5000

# 5. Run automated tests
python tests/test_pipeline.py
# → 11 passed, 0 failed
```

Full methodology, interpretations, and limitations are documented in [`outputs/implementation_report.md`](outputs/implementation_report.md).

---

*Research paper implementation — E-commerce Customer Churn and Spending Prediction using Machine Learning*
