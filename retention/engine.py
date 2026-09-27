"""Retention analysis engine: the notebook pipeline packaged for any retailer's upload.

Steps: detect columns -> check the data is customer transactions -> clean ->
build customer snapshots -> train and validate a churn model -> segment ->
value at risk -> budget optimisation.
"""
import re

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, identity, kron
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------- column detection

STANDARD_FIELDS = {
    "customer_id": "Customer",
    "order_id": "Order / invoice",
    "order_date": "Order date",
    "quantity": "Quantity",
    "unit_price": "Unit price",
    "amount": "Line total (if no quantity and price)",
    "product_id": "Product",
    "country": "Country",
}

SYNONYMS = {
    "customer_id": ["customerid", "customer", "customerno", "clientid", "client", "memberid",
                    "userid", "accountid", "shopperid", "buyerid"],
    "order_id": ["invoice", "invoiceno", "invoiceid", "orderid", "order", "orderno", "ordernumber",
                 "transactionid", "transactionno", "transaction", "receiptid", "receipt", "basketid"],
    "order_date": ["invoicedate", "orderdate", "date", "transactiondate", "purchasedate",
                   "datetime", "timestamp", "saledate", "orderdatetime"],
    "quantity": ["quantity", "qty", "units", "unitssold", "itemcount", "items"],
    "unit_price": ["price", "unitprice", "itemprice", "priceeach", "sellingprice", "unitcost"],
    "amount": ["amount", "total", "linetotal", "sales", "revenue", "value", "spend", "salesamount",
               "totalprice", "ordervalue", "netamount"],
    "product_id": ["stockcode", "productid", "product", "sku", "itemcode", "itemid", "productcode"],
    "country": ["country", "region", "market"],
}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def guess_mapping(df):
    """Guess which uploaded column matches each standard field (None if not found)."""
    normalised = {_norm(c): c for c in df.columns}
    mapping, used = {}, set()
    for field, names in SYNONYMS.items():
        mapping[field] = None
        for n in names:
            if n in normalised and normalised[n] not in used:
                mapping[field] = normalised[n]
                used.add(normalised[n])
                break
    return mapping


def check_data(df, mapping):
    """Decide whether the upload looks like retail customer transactions.

    Returns (ok, problems, warnings) with plain-English messages.
    """
    problems, warnings = [], []
    m = mapping
    if not m.get("customer_id"):
        problems.append("No customer column found. Churn is about customers, so each purchase "
                        "must say which customer made it.")
    if not m.get("order_date"):
        problems.append("No date column found. We need to know when each purchase happened.")
    if not (m.get("amount") or (m.get("quantity") and m.get("unit_price"))):
        problems.append("No spend information found. We need either a line total, "
                        "or both quantity and unit price.")
    if problems:
        return False, problems, warnings

    dates = pd.to_datetime(df[m["order_date"]], errors="coerce", format="mixed")
    if dates.notna().mean() < 0.9:
        problems.append(f"The column '{m['order_date']}' does not contain valid dates.")
    else:
        span = (dates.max() - dates.min()).days
        if span < 90:
            problems.append(f"The data covers only {span} days. At least 3 months of purchases "
                            "are needed to see who stops buying.")
    for field in ["quantity", "unit_price", "amount"]:
        col = m.get(field)
        if col and pd.to_numeric(df[col], errors="coerce").notna().mean() < 0.9:
            problems.append(f"The column '{col}' (used as {STANDARD_FIELDS[field].lower()}) is not numeric.")

    ids = df[m["customer_id"]].dropna()
    if len(ids) and ids.nunique() / len(ids) > 0.95:
        warnings.append("Almost every row has a different customer, so few customers bought more "
                        "than once. Churn predictions will be less reliable.")
    if not m.get("order_id"):
        warnings.append("No order column found: purchases by the same customer on the same day "
                        "will be treated as one order.")
    return not problems, problems, warnings


# ---------------------------------------------------------------- cleaning

def clean(df, mapping, cancel_prefix="C"):
    """Convert the upload into standard transactions, with a plain-English cleaning report."""
    m = {k: v for k, v in mapping.items() if v}
    t = pd.DataFrame({field: df[col] for field, col in m.items()})
    report = []

    def note(label, count, why):
        if count:
            report.append({"step": label, "rows": int(count), "why": why})

    total = len(t)
    t["order_date"] = pd.to_datetime(t["order_date"], errors="coerce", format="mixed")
    for field in ["quantity", "unit_price", "amount"]:
        if field in t:
            t[field] = pd.to_numeric(t[field], errors="coerce")
    if "amount" not in t:
        t["amount"] = t["quantity"] * t["unit_price"]
    t["customer_id"] = t["customer_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    if "order_id" not in t:
        t["order_id"] = t["customer_id"] + "_" + t["order_date"].dt.strftime("%Y-%m-%d")
    t["order_id"] = t["order_id"].astype("string")

    before = len(t)
    t = t.drop_duplicates()
    note("Duplicate rows removed", before - len(t), "The same line appeared more than once and would double-count purchases.")

    bad = t["order_date"].isna() | t["amount"].isna()
    note("Rows with an invalid date or amount removed", bad.sum(), "They cannot be placed in time or valued.")
    t = t[~bad]

    missing = t["customer_id"].isna()
    note("Rows without a customer removed", missing.sum(),
         "They cannot be linked to a customer (often guest checkouts), so they cannot inform churn.")
    t = t[~missing]

    is_cancel = t["amount"] < 0
    if cancel_prefix:
        is_cancel |= t["order_id"].str.startswith(cancel_prefix).fillna(False)
    cancellations = t[is_cancel].copy()
    cancellations["amount"] = -cancellations["amount"].abs()
    note("Cancellations and returns set aside", is_cancel.sum(),
         "They are not purchases. They are subtracted from each customer's spend instead.")
    t = t[~is_cancel]

    zero = t["amount"] <= 0
    note("Zero-value lines removed", zero.sum(), "Free items or adjustments are not real spending.")
    t = t[~zero]

    report.insert(0, {"step": "Rows uploaded", "rows": total, "why": ""})
    report.append({"step": "Purchase lines used for analysis", "rows": len(t), "why": ""})
    return t.reset_index(drop=True), cancellations.reset_index(drop=True), report


# ---------------------------------------------------------------- customer features

FEATURES = ["recency_days", "tenure_days", "frequency", "monetary", "avg_order_value",
            "orders_last_90d", "spend_last_90d", "days_per_order", "cancelled_orders"]

# (wording when the value is low, wording when the value is high)
FEATURE_WORDS = {
    "recency_days": ("a recent last order", "a long time since their last order"),
    "tenure_days": ("a short history with you", "a long history with you"),
    "frequency": ("few orders", "many orders"),
    "monetary": ("low total spend", "high total spend"),
    "avg_order_value": ("small orders", "large orders"),
    "orders_last_90d": ("few orders in the last 90 days", "many orders in the last 90 days"),
    "spend_last_90d": ("low spend in the last 90 days", "high spend in the last 90 days"),
    "days_per_order": ("short gaps between orders", "long gaps between orders"),
    "cancelled_orders": ("few cancellations or returns", "many cancellations or returns"),
}


def snapshot(tx, cancels, cutoff, window, with_label=True):
    hist = tx[tx["order_date"] < cutoff]
    orders = (hist.groupby(["customer_id", "order_id"])
                  .agg(order_date=("order_date", "min"), value=("amount", "sum")).reset_index())
    g = orders.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (cutoff - g["order_date"].max()).dt.days,
        "tenure_days": (cutoff - g["order_date"].min()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["value"].sum(),
    })
    recent = orders[orders["order_date"] >= cutoff - pd.Timedelta(days=90)].groupby("customer_id")
    f["orders_last_90d"] = recent["order_id"].nunique()
    f["spend_last_90d"] = recent["value"].sum()
    past_c = cancels[cancels["order_date"] < cutoff].groupby("customer_id")
    f["cancelled_orders"] = past_c["order_id"].nunique()
    f["cancelled_value"] = past_c["amount"].sum()
    f = f.fillna(0)
    f["monetary"] = (f["monetary"] + f["cancelled_value"]).clip(lower=0)
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    f["days_per_order"] = f["tenure_days"] / f["frequency"]
    if with_label:
        fut = tx[(tx["order_date"] >= cutoff) & (tx["order_date"] < cutoff + window)]
        f["churned"] = (~f.index.isin(fut["customer_id"].unique())).astype(int)
    return f.drop(columns="cancelled_value").reset_index()


def _model():
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=1000))


# ---------------------------------------------------------------- analysis

def analyse(tx, cancels):
    """Validate a churn model on past data, then score today's customers."""
    start = tx["order_date"].min().normalize()
    end = tx["order_date"].max().normalize() + pd.Timedelta(days=1)
    span = (end - start).days
    window_days = int(min(180, span // 3))
    window = pd.Timedelta(days=window_days)
    test_cut, train_cut = end - window, end - 2 * window

    train = snapshot(tx, cancels, train_cut, window)
    test = snapshot(tx, cancels, test_cut, window)
    current = snapshot(tx, cancels, end, window, with_label=False)
    for name, s in [("earlier", train), ("recent", test)]:
        if len(s) < 50 or s["churned"].nunique() < 2:
            raise ValueError(f"Not enough customer history in the {name} period to train a reliable model.")

    model = _model().fit(train[FEATURES], train["churned"])
    auc = roc_auc_score(test["churned"], model.predict_proba(test[FEATURES])[:, 1])
    baseline_auc = roc_auc_score(test["churned"], test["recency_days"])

    final = _model().fit(pd.concat([train, test])[FEATURES], pd.concat([train, test])["churned"])
    current["churn_prob"] = final.predict_proba(current[FEATURES])[:, 1]
    current["value_if_retained"] = current["monetary"] / current["tenure_days"].clip(lower=90) * window_days
    current["value_at_risk"] = current["churn_prob"] * current["value_if_retained"]

    coefs = pd.Series(final[-1].coef_[0], index=FEATURES)
    drivers = []
    for feat, c in coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(4).items():
        low, high = FEATURE_WORDS[feat]
        drivers.append(f"Customers with {high if c > 0 else low} are more likely to stop buying.")

    current = add_segments(current)
    return {
        "customers": current,
        "window_days": window_days,
        "auc": auc,
        "baseline_auc": baseline_auc,
        "drivers": drivers,
        "period": (start.date(), (end - pd.Timedelta(days=1)).date()),
        "test_churn_rate": test["churned"].mean(),
    }


SEGMENT_TEXT = {
    "Loyal high-value": "Buy often, spend the most and bought recently. Protect this group.",
    "Active developing": "Bought recently but not often yet. Encourage the next purchase.",
    "At-risk regulars": "Used to buy regularly but have gone quiet. The best group to win back.",
    "Lapsed one-off buyers": "Bought once, a long time ago. Usually not worth an expensive offer.",
}


def add_segments(current, k=4):
    rfm = np.log1p(current[["recency_days", "frequency", "monetary"]])
    scaled = StandardScaler().fit_transform(rfm)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
    centres = pd.DataFrame(km.cluster_centers_, columns=["recency", "frequency", "monetary"])
    # Rank clusters from best (recent, frequent, high spend) to worst, then name them in that order.
    quality = (centres["frequency"] + centres["monetary"] - centres["recency"]).sort_values(ascending=False)
    names = list(SEGMENT_TEXT)
    # The two middle clusters: the more recent one is "developing", the other "at risk".
    best, mid1, mid2, worst = quality.index
    if centres.loc[mid1, "recency"] > centres.loc[mid2, "recency"]:
        mid1, mid2 = mid2, mid1
    order = {best: names[0], mid1: names[1], mid2: names[2], worst: names[3]}
    current["segment"] = pd.Series(km.labels_).map(order).values
    return current


def optimise(current, budget, call_capacity, offers):
    """Choose at most one offer per customer to maximise expected revenue saved within budget."""
    names = list(offers)
    n = len(current)
    cost = np.array([offers[o]["cost"] for o in names])
    save = np.array([offers[o]["save_rate"] for o in names])
    gain = current["value_at_risk"].to_numpy()[:, None] * save[None, :]
    net = (gain - cost[None, :]).ravel()
    is_call = np.array([1 if offers[o].get("uses_call") else 0 for o in names])
    constraints = [
        LinearConstraint(kron(identity(n), csr_matrix(np.ones((1, len(names))))), 0, 1),
        LinearConstraint(np.tile(cost, n)[None, :], 0, budget),
        LinearConstraint(np.tile(is_call, n)[None, :], 0, call_capacity),
    ]
    sol = milp(c=-net, constraints=constraints, integrality=np.ones(n * len(names)), bounds=Bounds(0, 1))
    x = sol.x.reshape(n, len(names)).round().astype(int)
    plan = current.copy()
    plan["recommended_action"] = "No action"
    for j, o in enumerate(names):
        plan.loc[x[:, j] == 1, "recommended_action"] = offers[o]["label"]
    plan["expected_revenue_saved"] = (x * gain).sum(axis=1)
    plan["offer_cost"] = (x * cost).sum(axis=1)

    # Comparison: the common rule of thumb of contacting the highest churn risk first.
    order = current["churn_prob"].sort_values(ascending=False).index.to_numpy()
    naive = np.zeros_like(x)
    j_call = int(np.argmax(cost))
    calls = int(min(call_capacity, budget // cost[j_call]))
    naive[order[:calls], j_call] = 1
    j_cheap = int(np.argmin(cost))
    extra = int((budget - calls * cost[j_call]) // cost[j_cheap])
    naive[order[calls:calls + extra], j_cheap] = 1
    rule = {"saved": float((naive * gain).sum()), "spent": float((naive * cost).sum())}
    return plan, rule
