# Implementation Report — Unified Olist Churn & Spending Pipeline

**Project:** E-Commerce Customer Churn and Spending Prediction Using Machine Learning
**Models:** Random Forest Classifier (churn) and Random Forest Regressor (next-order spending)
**Models served via:** Flask prototype (`app.py`)

All figures and tables referenced below are in `outputs/`. All numbers reported
are the actual values produced by the scripts in this repository; nothing is
invented.

---

## 1. Summary of the redesign (old vs new)

The project previously trained its two models on **two unrelated datasets**
(Online Retail II for churn, Olist for spending), which meant the two outputs
could not describe the same customers. The redesign puts **both models on the
same Olist source and the same customer population**.

| Aspect | Old methodology | New (unified) methodology |
|---|---|---|
| Churn source | Online Retail II (UK, 2009–2011) | Olist (Brazil, delivered orders) |
| Spending source | Olist | Olist |
| Shared population? | No (two datasets, no shared IDs) | Yes — repeat customers (>= 2 delivered orders) |
| Churn target | no purchase in 90 days, base rate 43% | no order within 150 days of a snapshot, base rate ~96% |
| Spending target | next-order value (customer-random split) | next-order value (out-of-time split) |
| Split | churn: stratified random; spending: customer-random | both **out-of-time** (chronological) |
| Churn ROC-AUC | 0.7297 (held-out) | **0.7390** (out-of-time) |
| Spending R² | 0.0608 | **0.0426** (out-of-time) |

The redesign also rebuilds the data layer directly from the raw Olist CSV tables
(`prepare_data.py`), so both models share one reproducible customer definition
and one feature-building routine. The root `README.md` was intentionally left
untouched.

## 2. Research objective and questions

- **RQ1.** On a single e-commerce dataset, can a Random Forest distinguish
  repeat customers who will go dormant from those who will keep buying?
- **RQ2.** Can a Random Forest predict the payment value of a customer's next
  order from prior-order history alone, without leakage?
- **RQ3.** Can the two unified outputs be combined into a transparent,
  clearly-bounded customer risk indicator?

## 3. Dataset and population

| Item | Value |
|---|---|
| Source | Olist Brazilian E-Commerce (`olist_complete_dataset/`) |
| Delivered orders after cleaning | 96,477 |
| Customers (delivered) | 93,357 |
| **Repeat customers (>= 2 delivered orders)** | **2,801 (3.0%)** |
| Churn observation horizon | 150 days |

Raw tables joined: `olist_orders_dataset.csv`, `olist_customers_dataset.csv`,
`olist_order_payments_dataset.csv`, `olist_order_items_dataset.csv`.
Payments are summed per `order_id` (multi-row payments would otherwise be
double-counted) and items are aggregated per `order_id` before the join. Only
`order_status == "delivered"` rows with `payment_value > 0` are kept.

**Why repeat customers.** A customer who has never ordered twice has no
purchasing rhythm to model; defining churn among them produces a near-constant
target (~99% churn) with no learnable signal. Restricting both models to
repeat customers is a deliberate, documented design choice.

## 4. Temporal evaluation design (no future leakage)

| Use | Cut-off | Observation window |
|---|---|---|
| Churn — train snapshot | 2018-01-01 | target = order in [2018-01-01, 2018-05-31) |
| Churn — test snapshot | 2018-04-01 | target = order in [2018-04-01, 2018-08-29) |
| Spending — train | order date < 2018-01-01 | — |
| Spending — test | order date >= 2018-01-01 | — |

Features are always computed from data **strictly before** the snapshot / split
point. The test snapshot occurs later in time than the training snapshot, so the
churn estimate is genuinely out-of-time. Olist delivered activity ends
2018-08-31, so the test horizon (ending 2018-08-29) fits inside the data.

**Leakage checks**
- Churn features use only orders before the snapshot; the target uses only
  orders after it. No overlap.
- Spending features are cumulative sums of **prior** orders; the target (next
  order value, items, freight, installments) is never used as a feature.
- Verified at runtime: both feature matrices contain zero missing values.

## 5. Churn model

- **Population:** customers with >= 2 delivered orders before the snapshot.
- **Features (12):** `Recency`, `Frequency`, `Monetary`, `TotalItems`,
  `TotalProducts`, `TotalSellers`, `AverageFreight`, `AverageInstallments`,
  `AverageOrderValue`, `Tenure`, `ActiveDays`, `AverageGap`.
- **Target:** `Churn = 1` if the customer places **no** delivered order in the
  150 days after the snapshot.
- **Estimator:** `RandomForestClassifier(n_estimators=500, max_depth=8,
  min_samples_leaf=2, class_weight="balanced", random_state=42)`.

### Actual results (test snapshot 2018-04-01, 1,829 customers, churn 96.7%)

| Metric | Value |
|---|---:|
| Accuracy | 0.9666 |
| Precision (Churned) | 0.9697 |
| Recall (Churned) | 0.9966 |
| F1 (Churned) | 0.9830 |
| **ROC-AUC** | **0.7390** |
| PR-AUC | 0.9870 |
| 5-fold CV ROC-AUC (train snapshot) | 0.6313 |

Confusion matrix `[[6, 55], [6, 1762]]` (TN, FP, FN, TP).

**Reading these honestly.** The target is a rare event (~96% churn), so accuracy
(0.9666) is only trivially above the base rate (0.9666) and must not be read as
skill. The minority "Not Churned" class has precision 0.50 / recall 0.10
(only 6 of 61 returning customers are flagged). The informative number is
**ROC-AUC = 0.739**, i.e. the ranked churn risk is meaningfully better than
random. `class_weight="balanced"` prevents the model from collapsing to the
majority class but, with only 1,178 training customers and 42 non-churners,
recall on the minority class remains low.

### Churn feature importance (mean decrease in impurity)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Recency | 0.173 |
| 2 | Monetary | 0.138 |
| 3 | AverageOrderValue | 0.125 |
| 4 | AverageFreight | 0.111 |
| 5 | Tenure | 0.102 |
| 6 | AverageInstallments | 0.098 |
| 7 | AverageGap | 0.097 |
| 8 | TotalItems | 0.042 |
| 9 | ActiveDays | 0.038 |
| 10 | Frequency | 0.029 |
| 11 | TotalSellers | 0.026 |
| 12 | TotalProducts | 0.020 |

## 6. Spending model

- **Population:** repeat-customer order events (each order that has prior
  history), 3,120 observations over all time.
- **Features (9, prior orders only):** `Recency`, `PreviousOrderCount`,
  `HistoricalSpending`, `TotalItems`, `TotalProducts`, `TotalSellers`,
  `AverageOrderValue`, `AverageFreightValue`, `AverageInstallments`.
- **Target:** `NextOrderSpending` = payment value of the customer's next order.
  The target is **conditional on a subsequent order existing** — a customer who
  never reorders is not represented in the regression set.
- **Estimator:** `RandomForestRegressor(n_estimators=300, random_state=42)`.

### Actual results (test: 1,820 orders placed after 2018-01-01)

| Metric | Value |
|---|---:|
| MAE (BRL) | 93.8980 |
| RMSE (BRL) | 150.1643 |
| **R²** | **0.0426** |
| Median AE (BRL) | 60.5203 |

Split: 1,300 train obs / 1,820 test obs; 1,178 train customers / 1,685 test
customers; 62 customers appear in both (expected for a chronological split, and
each of their orders still contributes only prior-order features).

Predicted vs actual (`outputs/spending_predictions.csv`):

| Statistic | Actual (BRL) | Predicted (BRL) | Residual (BRL) |
|---|---:|---:|---:|
| Mean | 148.75 | 157.70 | -8.94 |
| Std | 153.51 | 82.97 | 149.94 |
| Min | 11.56 | 26.83 | -739.41 |
| Median | 104.40 | 137.36 | -28.30 |
| Max | 1435.97 | 797.11 | 1314.01 |

**Interpretation.** R² = 0.043 means prior-order history explains ~4% of
held-out next-order variance; this is a **weak but positive** signal (not
negative, i.e. better than predicting the mean). The predicted range is much
narrower than the actual range — the regressor shrinks toward the mean, typical
for tree ensembles on skewed monetary targets. The out-of-time split is stricter
than the old random split, which is why R² is slightly lower than the previous
0.0608.

### Spending feature importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | AverageOrderValue | 0.349 |
| 2 | HistoricalSpending | 0.202 |
| 3 | Recency | 0.185 |
| 4 | AverageFreightValue | 0.176 |
| 5 | AverageInstallments | 0.062 |
| 6 | TotalItems | 0.015 |
| 7 | TotalProducts | 0.007 |
| 8 | TotalSellers | 0.002 |
| 9 | PreviousOrderCount | 0.001 |

## 7. Customer Risk Value

\[
R_i = P(\text{Churn}_i = 1) \times \widehat{S}_i
\]

- \(P\) = churn probability from the unified churn model,
- \(\widehat{S}\) = predicted next-order spending (BRL), clipped at 0 in the app.

Because both models now share the same customer population, the combined value
is more coherent than before, but it is still a **prototype risk-weighted
spending indicator** — not validated recoverable revenue, profit, or the causal
financial impact of churn. Illustrative sensitivity examples are in
`outputs/customer_risk_value_examples.csv` (predicted spending crossed with
churn probabilities 0.0–1.0).

**Risk segments.** `train_spending.py` writes data-driven cut-offs to
`models/risk_thresholds.json` (33rd / 66th percentile of the held-out predicted
spending), and the app labels each row Low / Medium / High. On this population
the churn probability is near-constant (~0.96), so the segment is effectively
driven by predicted spending; this is stated rather than hidden.

## 8. Application (`app.py`)

`app.py` loads both models and feature lists once at startup (no per-request
retraining) and exposes a batch CSV workflow:

- `GET /` — landing page with two file inputs (`templates/index.html`).
- `POST /predict` — accepts a churn feature CSV and a spending feature CSV,
  validates them, predicts churn probability/status and next-order spending
  (BRL), computes `R = P(churn) x S_pred`, assigns a Low/Medium/High segment, and
  returns the customers **sorted by risk value (highest first)** with a summary
  panel (counts, mean churn probability, mean predicted spending, mean/total risk
  value) and the segment thresholds `t1`/`t2` shown explicitly. The results are
  labelled a **prototype estimate**.

Validation rejects missing files, unreadable CSVs, missing required columns,
non-numeric or negative feature values, and empty files (HTTP 400 with a
readable message). Rows are joined on an optional `customer_id`, otherwise
paired by row order. Sample inputs live in `sample_inputs/`.

## 9. Test results

`tests/test_pipeline.py` (via `python tests/test_pipeline.py` or pytest):
**11 passed, 0 failed.** Coverage: model load/type/feature-count for both
models; churn probability in [0,1] and output shape; finite spending
predictions; saved feature-order match; risk-value edge cases; Flask main
route; valid two-file upload and risk table; missing-column detection per file;
non-numeric rejection; and row-order pairing without `customer_id`.

## 10. Files created / modified / superseded

**Created**
- `prepare_data.py` — unified Olist data preparation (shared datasets).
- `data/olist_orders_joined.csv` — joined delivered-order table.
- `data/olist_churn_dataset.csv` — customer snapshot dataset (train/test).
- `data/olist_spending_dataset_prepared.csv` — next-order event dataset.
- `models/risk_thresholds.json` — data-driven risk-segment cut-offs.
- `outputs/churn_classification_report.csv`, `outputs/spending_predictions.csv`
  etc. are regenerated in place (see `outputs/README.md`).

**Modified (rewritten for the unified design)**
- `train_churn.py` — now Olist-based, out-of-time snapshots, 12 features.
- `train_spending.py` — loads the prepared dataset, temporal split.
- `app.py` — new feature schema + risk segments.
- `templates/index.html` — risk-segment column and updated disclaimer.
- `tests/test_pipeline.py` — updated feature counts/schema.
- `sample_inputs/churn_input.csv`, `sample_inputs/spending_input.csv` —
  regenerated for the new schema.

**Superseded (no longer used by either model)**
- Online Retail II inputs (`data/online_retail_II.xlsx`,
  `data/online_retail_II_cleaned.csv`) and
  `data/customer_churn_dataset*.csv` — retained on disk for provenance only.
- `data/olist_spending_dataset.csv` / `data/olist_spending_cleaned.csv` — the
  old ad-hoc join; superseded by `prepare_data.py` output.
- `combiner.ipynb` / notebooks — incremental history only.

**Untouched (by request):** root `README.md`.

## 11. Limitations and threats to validity

- **Rare-event churn.** ~96% of repeat customers do not buy within 150 days;
  minority-class recall is low. AUC 0.739 is the honest headline, not accuracy.
- **Conditional spending target.** Spending is modelled only for customers who
  place a next order; never-reordering customers are excluded from the
  regression set.
- **Weak spending signal.** R² = 0.043 is low; treat predictions as weak.
- **Olist is not a subscription business.** "Churn" here means long-run
  inactivity after a repeat purchase, not contract cancellation.
- **150-day horizon and 2018 cut-offs are design choices**, not validated
  optima.
- **Feature importance ≠ causation.**
- **No significance testing** of metric differences.
- **Risk value not financially validated.**

## 12. Reproducibility

```bash
pip install -r requirements.txt

python prepare_data.py       # builds data/ from raw Olist tables
python train_spending.py     # models/spending_model.pkl + spending outputs
python train_churn.py        # models/churn_model.pkl + churn outputs

python app.py                # http://127.0.0.1:5000
python tests/test_pipeline.py
```

All paths are project-root-relative; `random_state=42` is fixed so reruns
reproduce the reported metrics. Run `train_spending.py` before `train_churn.py`
so `model_metrics_comparison.png` uses fresh spending metrics.
