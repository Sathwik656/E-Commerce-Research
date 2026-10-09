"""
Unified Olist data preparation for the churn + next-order-spending pipeline.

Both models now use the SAME source (the raw Olist CSV tables in
olist_complete_dataset/) and the SAME customer population: repeat customers,
i.e. customers with at least two delivered orders. This replaces the old
two-dataset setup (Online Retail II for churn, Olist for spending).

Writes:
  data/olist_orders_joined.csv            one row per delivered order, after
                                          joining customer / payment / item
                                          aggregates (no double counting)
  data/olist_churn_dataset.csv            customer snapshot dataset (train and
                                          test cutoffs, RFM + history features,
                                          Churn target)
  data/olist_spending_dataset_prepared.csv next-order event dataset (prior
                                          features only, NextOrderSpending
                                          target, train/test split)

Run:  python prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "olist_complete_dataset"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# --- temporal design (both models are evaluated out-of-time) --------------- #
TRAIN_CUT = pd.Timestamp("2018-01-01")   # train snapshot / spending split cut
TEST_CUT = pd.Timestamp("2018-04-01")    # test snapshot cut
HORIZON_DAYS = 150                        # churn observation horizon (days)

# Order matters: the Flask app must present features in exactly this order.
CHURN_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalItems",
    "TotalProducts",
    "TotalSellers",
    "AverageFreight",
    "AverageInstallments",
    "AverageOrderValue",
    "Tenure",
    "ActiveDays",
    "AverageGap",
]

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


def load_joined_orders():
    """Join the raw Olist tables into one clean, delivered-order table."""
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    customers = pd.read_csv(
        RAW / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id", "customer_state"],
    )
    payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")

    # Aggregate payments per order (multi-row payments would double-count).
    pay = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    )
    # Aggregate items per order.
    it = items.groupby("order_id").agg(
        number_of_items=("order_item_id", "count"),
        number_of_products=("product_id", "nunique"),
        number_of_sellers=("seller_id", "nunique"),
        total_item_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    )

    df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(pay, on="order_id", how="left")
        .merge(it, on="order_id", how="left")
    )
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(
        subset=["customer_unique_id", "order_purchase_timestamp", "payment_value"]
    )
    df = df[df["payment_value"] > 0].copy()

    for col in (
        "number_of_items",
        "number_of_products",
        "number_of_sellers",
        "total_item_price",
        "total_freight_value",
    ):
        df[col] = df[col].fillna(0)
    df["payment_installments"] = df["payment_installments"].fillna(1)

    df = df.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    ).reset_index(drop=True)
    return df


def _churn_snapshot(o, cutoff):
    """Build one customer-level snapshot of features + the churn target."""
    before = o[o["order_purchase_timestamp"] < cutoff]
    window = o[
        (o["order_purchase_timestamp"] >= cutoff)
        & (o["order_purchase_timestamp"] < cutoff + pd.Timedelta(HORIZON_DAYS, unit="D"))
    ]
    active = set(window["customer_unique_id"])

    g = before.groupby("customer_unique_id")
    f = g.agg(
        Frequency=("order_id", "size"),
        Monetary=("payment_value", "sum"),
        TotalItems=("number_of_items", "sum"),
        TotalProducts=("number_of_products", "sum"),
        TotalSellers=("number_of_sellers", "sum"),
        AverageFreight=("total_freight_value", "mean"),
        AverageInstallments=("payment_installments", "mean"),
        first=("order_purchase_timestamp", "min"),
        last=("order_purchase_timestamp", "max"),
    )
    f["Recency"] = (cutoff - f["last"]).dt.days
    f["AverageOrderValue"] = f["Monetary"] / f["Frequency"]
    f["Tenure"] = (f["last"] - f["first"]).dt.days
    f["ActiveDays"] = (
        before.assign(_d=before["order_purchase_timestamp"].dt.date)
        .groupby("customer_unique_id")["_d"]
        .nunique()
    )
    f["AverageGap"] = np.where(f["Frequency"] > 1, f["Tenure"] / (f["Frequency"] - 1), 0.0)

    # Repeat cohort only: churn is only meaningful for demonstrated repeaters.
    f = f[f["Frequency"] >= 2].copy()
    f["Churn"] = (~f.index.isin(active)).astype(int)
    return f[CHURN_FEATURES + ["Churn"]]


def build_churn_dataset(o):
    train = _churn_snapshot(o, TRAIN_CUT).assign(snapshot="train")
    test = _churn_snapshot(o, TEST_CUT).assign(snapshot="test")
    train.index.name = test.index.name = "customer_unique_id"
    return pd.concat([train, test]).reset_index()


def build_spending_dataset(o):
    d = o.copy()
    g = d.groupby("customer_unique_id")

    d["PreviousOrderCount"] = g.cumcount()

    def prior_cumsum(col):
        return g[col].cumsum() - d[col]

    d["HistoricalSpending"] = prior_cumsum("payment_value")
    d["TotalItems"] = prior_cumsum("number_of_items")
    d["TotalProducts"] = prior_cumsum("number_of_products")
    d["TotalSellers"] = prior_cumsum("number_of_sellers")

    with np.errstate(divide="ignore", invalid="ignore"):
        d["AverageOrderValue"] = d["HistoricalSpending"] / d["PreviousOrderCount"]
        d["AverageFreightValue"] = (
            prior_cumsum("total_freight_value") / d["PreviousOrderCount"]
        )
        d["AverageInstallments"] = (
            prior_cumsum("payment_installments") / d["PreviousOrderCount"]
        )

    d["PreviousPurchaseDate"] = g["order_purchase_timestamp"].shift(1)
    d["Recency"] = (
        d["order_purchase_timestamp"] - d["PreviousPurchaseDate"]
    ).dt.total_seconds() / 86400.0

    # First order per customer has no history -> drop it. This makes the target
    # conditional on the customer actually placing a subsequent order.
    obs = d[d["PreviousOrderCount"] > 0].copy()
    obs = obs[
        ["customer_unique_id", "order_purchase_timestamp", *SPENDING_FEATURES, "payment_value"]
    ].rename(columns={"payment_value": "NextOrderSpending"})
    obs["split"] = np.where(
        obs["order_purchase_timestamp"] < TRAIN_CUT, "train", "test"
    )
    return obs


def main():
    print("Joining raw Olist tables ...")
    orders = load_joined_orders()
    orders.to_csv(DATA / "olist_orders_joined.csv", index=False)
    print(f"  delivered orders: {len(orders)}")
    print(f"  customers: {orders['customer_unique_id'].nunique()}")

    churn = build_churn_dataset(orders)
    churn.to_csv(DATA / "olist_churn_dataset.csv", index=False)
    tr = churn[churn.snapshot == "train"]
    te = churn[churn.snapshot == "test"]
    print("Churn dataset written:", DATA / "olist_churn_dataset.csv")
    print(
        f"  train customers: {len(tr)} (churn {tr.Churn.mean():.3f})  "
        f"test customers: {len(te)} (churn {te.Churn.mean():.3f})"
    )

    spending = build_spending_dataset(orders)
    spending.to_csv(DATA / "olist_spending_dataset_prepared.csv", index=False)
    print("Spending dataset written:", DATA / "olist_spending_dataset_prepared.csv")
    print(
        f"  train observations: {(spending.split == 'train').sum()}  "
        f"test observations: {(spending.split == 'test').sum()}  "
        f"target mean {spending.NextOrderSpending.mean():.2f} BRL"
    )


if __name__ == "__main__":
    main()
