# Implementation Report

**Project:** E-Commerce Customer Churn and Spending Prediction Using Machine Learning
**Models:** Random Forest Classifier (churn) and Random Forest Regressor (next-order spending)
**Models served via:** Flask prototype (`app.py`)

All figures and tables referenced below are in `outputs/`. All numbers reported
are the actual values produced by the scripts in this repository; illustrative
values are explicitly labelled as such.

---

## 1. Project overview

The project builds two independent supervised-learning models on two public
e-commerce datasets:

1. **Churn prediction** — predict whether a customer will stop purchasing,
   using transaction-level retail data aggregated to customer level.
2. **Next-order spending prediction** — predict the amount a customer will
   spend on their next observed order, using only their prior order history.

The two outputs are combined into a prototype **Customer Risk Value**
indicator and exposed through a small Flask web application.

## 2. Research objective and questions

- **RQ1.** Can a Random Forest predict customer churn from behavioural
  recency/frequency/monetary (RFM-style) features?
- **RQ2.** Can a Random Forest predict the payment value of a customer's next
  order from historical order features without target leakage?
- **RQ3.** How can the two outputs be combined into a transparent,
  clearly-bounded risk indicator for demonstration purposes?

## 3. Dataset descriptions and provenance

| Dataset | Source | Role |
|---|---|---|
| Online Retail II | UCI ML Repository (dataset 502) | Churn model |
| Olist Brazilian E-Commerce | Kaggle `olistbr/brazilian-ecommerce` | Spending model |

- **Online Retail II** — UK online-retail invoices (2009–2011) with
  `Quantity`, `Price`, `InvoiceDate`, `Customer ID`. Aggregated to customer
  level; the enriched modelling dataset (`data/customer_churn_dataset_
  enriched.csv`) has 3,370 customers, 22 features, and the `Churn` target. The
  original 8-feature dataset `data/customer_churn_dataset.csv` is preserved.
- **Olist** — Brazilian marketplace orders. Raw order/item/payment/customer
  tables were combined (in `combiner.ipynb`) into
  `data/olist_spending_dataset.csv` (99,441 orders). The spending pipeline
  consumes this file directly.

> These two datasets describe **different customer populations** and share no
> customer identifiers. They cannot be joined at the customer level.

## 4. Data cleaning and preprocessing

**Churn (reproduced in `train_churn.py` from the raw workbook):**
rows with a missing `Customer ID` were dropped; the feature set is built from
non-cancellation, positive-quantity, positive-price sales, while cancellations
are retained separately to derive return-behaviour features. The model input is
the 22-feature enriched customer-level dataset.

**Spending (`train_spending.py`):**
1. Parse `order_purchase_timestamp`.
2. Drop rows missing `customer_unique_id` or timestamp.
3. Keep `payment_value > 0` (leaves 99,437 rows).
4. Keep `order_status == "delivered"` (96,477 orders).
5. Drop rows missing any modelling column (`REQUIRED_COLUMNS`).

## 5. Churn feature construction and target definition

- Observation window ends 90 days before the last transaction date.
- Target `Churn = 1` if the customer has **no** purchase in the 90-day future
  window, else `0`. Base churn rate in the dataset is **43.0%**.
- Features (historical window only, **22 total**): the original 8 RFM features
  (`Recency`, `Frequency`, `Monetary`, `TotalQuantity`, `AverageOrderValue`,
  `AverageQuantityPerOrder`, `UniqueProducts`, `ActiveDays`) plus 14 engineered
  features: `Tenure`, `FirstPurchaseDaysAgo`, `OrderValueStd`, `OrderValueMax`,
  `OrderValueMin`, `OrderValueMedian`, `OrdersPerActiveDay`, `ProductsPerOrder`,
  `SpendPerDay`, `CountryCount`, `Cancellations`, `CancelledValue`,
  `ReturnRatio`, and `SpendTrend` (first-half vs second-half spend).

**Model improvement.** The original 8-feature, default Random Forest reached
accuracy 0.6291 / ROC-AUC 0.6826. Adding the 14 historical-order features and
tuning the forest (500 trees, `max_depth=6`, `min_samples_leaf=2`,
`max_features=0.8`) raised held-out accuracy to **0.6706** and ROC-AUC to
**0.7297**; 5-fold cross-validated accuracy is **0.6923** (CV ROC-AUC 0.7591).
The enriched dataset is saved to `data/customer_churn_dataset_enriched.csv`;
the original `data/customer_churn_dataset.csv` is preserved unchanged.

## 6. Spending feature construction and next-order target definition

An earlier future-window target produced ~99% zero values (documented in
`notebooks/model2.ipynb` cells 66–75) and was abandoned. The current method
uses chronological order histories:

- Orders are sorted by (`customer_unique_id`, `order_purchase_timestamp`).
- For each order, all features are **cumulative sums of previous orders only**:
  `HistoricalSpending`, `TotalItems`, `TotalProducts`, `TotalSellers`,
  `AverageOrderValue`, `AverageFreightValue`, `AverageInstallments`,
  `PreviousOrderCount`, and `Recency` (days since the previous order).
- `NextOrderSpending` = the **current** order's `payment_value`.
- The customer's first order (`PreviousOrderCount == 0`) is dropped because it
  has no history.

**Leakage control.** Target-order quantity, price, freight, and payment fields
are never used as features. Only data from prior orders is used. Verified
result: zero missing values in the final feature matrix.

Resulting dataset (`outputs/dataset_summary.csv`):

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

## 7. Train/test split methodology

- **Churn:** stratified 80/20 split, `random_state=42` → 2,696 train / 674 test
  customers.
- **Spending:** split by **customer**, not by observation, so no customer
  appears in both sets. `train_test_split(unique_customers, test_size=0.20,
  random_state=42)` → 2,240 train / 561 test customers; 2,499 train / 621 test
  observations; **customer overlap = 0**.

## 8. Random Forest configurations

| Model | Estimator | n_estimators | random_state | Other |
|---|---|---|---|---|
| Churn | `RandomForestClassifier` | 500 | 42 | `max_depth=6`, `min_samples_leaf=2`, `max_features=0.8`, `n_jobs=-1`, `stratify=y` |
| Spending | `RandomForestRegressor` | 200 | 42 | `n_jobs=-1` |

The churn forest was tuned after the original default configuration plateaued;
the spending regressor uses defaults.

## 9. Evaluation metrics and actual results

### Churn (held-out 674 customers) — improved model

| Metric | Original (8 features) | Improved (22 features, tuned) |
|---|---:|---:|
| Accuracy | 0.6291 | **0.6706** |
| Precision | 0.5654 | **0.6062** |
| Recall | 0.5966 | **0.6690** |
| F1-score | 0.5805 | **0.6361** |
| ROC-AUC | 0.6826 | **0.7297** |

5-fold cross-validation of the improved model: accuracy **0.6923**, ROC-AUC
**0.7591**. Confusion matrix: `[[258, 126], [96, 194]]` (TN, FP, FN, TP).
Per-class detail is in `outputs/churn_classification_report.csv`.

A candidate-model search (tuned Random Forest, Gradient Boosting, and scaled
Logistic Regression on the enriched features) all converged to ~0.68–0.69
accuracy / ~0.75–0.76 AUC, indicating the practical signal ceiling for this
target. The improved Random Forest was retained to keep the research design's
required estimator.

### Spending (held-out customers, 621 observations) — new actual results

| Metric | Value |
|---|---:|
| MAE (BRL) | 95.2083 |
| RMSE (BRL) | 168.4860 |
| R² | 0.0608 |
| Median AE (BRL) | 56.6951 |

Interpretation:

- **MAE** — predictions are, on average, R$95.21 away from the actual
  next-order payment value.
- **RMSE** — R$168.49; larger than MAE, indicating a few large errors driven by
  high-value orders.
- **R² = 0.0608** — the model explains ~6% of the variance in held-out
  next-order spending. This is a **low explanatory power**, not
  "6% accuracy". Next-order spend is highly variable and largely driven by
  factors not present in prior-order features.
- **Median AE** — half of the test predictions are within R$56.70.

Predicted vs actual summary (`outputs/spending_predictions.csv`):

| Statistic | Actual (BRL) | Predicted (BRL) | Residual (BRL) |
|---|---:|---:|---:|
| Mean | 158.22 | 146.45 | 11.77 |
| Std | 173.99 | 80.88 | 168.21 |
| Min | 14.78 | 37.92 | -745.01 |
| Median | 103.26 | 128.82 | -20.99 |
| Max | 1391.79 | 827.74 | 1290.91 |

The narrow predicted range relative to actual spread shows the regressor
regresses toward the mean, typical of tree ensembles on skewed targets.

## 10. Feature-importance findings

**Spending** (`spending_feature_importance.csv`):

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | AverageOrderValue | 0.3217 |
| 2 | HistoricalSpending | 0.2049 |
| 3 | Recency | 0.1953 |
| 4 | AverageFreightValue | 0.1738 |
| 5 | AverageInstallments | 0.0622 |
| 6 | TotalItems | 0.0203 |
| 7 | TotalProducts | 0.0108 |
| 8 | TotalSellers | 0.0066 |
| 9 | PreviousOrderCount | 0.0044 |

**Churn** (`churn_feature_importance.csv`): ActiveDays (0.2382), Frequency
(0.0870), Monetary (0.0857), Recency (0.0701), SpendTrend (0.0545),
TotalQuantity (0.0527), UniqueProducts (0.0457), ReturnRatio (0.0452),
OrderValueMin (0.0388), CancelledValue (0.0360), Tenure (0.0337), and the
remaining engineered features.

Importance reflects **predictive contribution within the fitted forest**, not
causation.

## 11. Risk-value formula and interpretation

\[
R_i = P(\text{Churn}_i=1)\times \widehat{S}_i
\]

where \(P\) is the churn probability and \(\widehat{S}\) the predicted
next-order spending (BRL). Validated edge cases (also in the test suite):
prob 0 → R$0; prob 1 → R$ = predicted spending; prob 0.75 × R$500 = R$375.
Illustrative examples are in `outputs/customer_risk_value_examples.csv`.

**Interpretation limitation.** This is a **risk-weighted spending indicator**,
not validated recoverable revenue, profit, or the causal financial impact of
churn. Because the churn and spending models come from different datasets, the
combined value is a prototype only.

## 12. Flask application architecture

`app.py` loads `models/churn_model.pkl` and `models/spending_model.pkl` once at
startup (no retraining per request) and reads `models/spending_features.json`
to guarantee the training feature order. It exposes a **batch CSV upload**
workflow:

- `GET /` — landing page with two file inputs (`templates/index.html`).
- `POST /predict` — accepts a **churn feature CSV** and a **spending feature
  CSV**, validates them, predicts churn probability + status and next-order
  spending (BRL) for every row, and computes the prototype Customer Risk Value.

Validation rejects missing files, unreadable CSVs, missing required columns,
non-numeric or negative feature values, and empty files, returning HTTP 400
with a readable message per file. Rows are joined on `customer_id` when both
files contain it (inner join); otherwise they are paired by row order. Sample
inputs are provided in `sample_inputs/churn_input.csv` and
`sample_inputs/spending_input.csv`. A persistent disclaimer states the two
datasets are independent and the risk value is a prototype. No external
services are used.

## 13. Test results

`tests/test_pipeline.py` (run via `python tests/test_pipeline.py` or pytest):
**11 passed, 0 failed.**

Coverage: churn model load + type + feature count; churn probability in [0,1]
and correct output shape; spending model load + type + feature count + saved
feature-order match; finite spending prediction; risk-value edge cases (0, 1,
intermediate, non-negative); Flask main route; valid two-file upload and risk
table; missing-column detection for each file; non-numeric rejection; and
row-order pairing when `customer_id` is absent.

## 14. Limitations and threats to validity

- **Different populations.** Online Retail II and Olist customers are distinct;
  no shared ID exists, and cross-dataset matching is not performed.
- **Conditional target.** The spending model predicts the next observed order's
  value **conditional on a subsequent order existing** in the data; customers
  who never reorder are not represented in the regression set.
- **Low R².** The spending model explains only ~6% of held-out variance;
  predictions should be treated as weak signals.
- **No temporal hold-out for spending.** The customer-level split prevents
  identity leakage but is random, not chronological, so some future orders can
  inform training on other customers.
- **Churn accuracy ceiling.** Enriched features and tuned models (RF, Gradient
  Boosting, Logistic Regression) all converge to ~0.68–0.69 held-out accuracy
  and ~0.75–0.76 AUC. The 90-day churn target is inherently noisy (many
  "churners" are one-time buyers), and no historical feature exceeds |r|≈0.37
  with the target. Accuracy near 0.90 would require target leakage (using
  future-window information as a feature) or a different, easier target
  definition; neither is used here.
- **Single-split churn estimate / no significance testing.** The churn model was
  tuned and reported on one stratified split (5-fold CV is also given);
  differences are not tested for statistical significance.
- **Feature importance ≠ causation.**
- **Risk value not financially validated.** It is a demonstration, not a
  validated financial outcome.
- **Churn target window** (90 days) is a design choice, not validated.

## 15. Reproducibility instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce churn outputs + model
python train_churn.py

# 3. Reproduce spending outputs + model
python train_spending.py

# 4. Start the Flask app
python app.py            # http://127.0.0.1:5000

# 5. Run tests
python tests/test_pipeline.py
```

Scripts use paths relative to the project root and written with
`random_state=42`; reruns reproduce the reported churn metrics exactly and the
spending metrics deterministically. `outputs/README.md` documents every output
file. The notebook `notebooks/model2.ipynb` is retained as the incremental
record; `train_spending.py` reimplements cells up to Task 6.2 and completes
Tasks 6.3–6.7.
