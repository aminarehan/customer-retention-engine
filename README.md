# Customer Retention Decision Engine

Predicts which customers of a UK online retailer are likely to stop buying, estimates how much revenue is at risk, and decides **which customers to target with which retention offer within a fixed budget**.

**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 transactions from a UK online retailer, December 2009 to December 2011. Dataset by Daqing Chen, UCI Machine Learning Repository, licensed under CC BY 4.0. A Parquet copy is included in `data/raw/` for the app's demo.

## Web app

**Live app:** https://amina-retention-planner.streamlit.app/

`app.py` is a Streamlit app that runs the same pipeline on **any retailer's customer purchase data**:

1. Upload a CSV, Excel or Parquet file, or try the demo data.
2. The app detects the customer, order, date, quantity and price columns, and lets the user correct them. Files that aren't customer purchase data (for example, missing a customer, date or spend column) are flagged with a plain-English explanation.
3. It explains what it cleaned and why.
4. It reports, in plain language, how many customers are likely to stop buying, the revenue at risk, the main reasons customers leave, the customer groups and how reliable the predictions are.
5. The user sets their own budget, call capacity, offer costs and success rates. The app returns an optimised contact list to download.

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The reusable pipeline lives in `retention/engine.py`, and the app is a thin layer on top of it.

## Pipeline

| Notebook | What it does |
|---|---|
| `01_load_and_explore` | Loads both Excel sheets, saves a Parquet copy, and investigates data quality issues |
| `02_clean_and_features` | Applies the retailer's mapping file, runs quality checks, builds customer features and the churn label |
| `03_models_and_optimisation` | Segmentation, churn models, SHAP explanations, value at risk and budget optimisation |

The retailer-specific cleaning rules live in `mappings/online_retail_ii.json`, not in code. A different retailer would need a new mapping file, not new cleaning code.

## Key data decisions

- **Duplicates:** 34,335 exact duplicate rows removed (mainly an overlap between the two source sheets).
- **Cancellations:** 8,292 cancelled invoices excluded as purchases but kept per customer as a feature. Cancellations are also **netted against spend**: some customers placed very large orders and cancelled them minutes later (one customer's £77,600 order was cancelled 16 minutes after it was placed), which would otherwise hugely overstate their value.
- **Missing customer IDs:** 22.8% of rows have no customer ID (likely guest checkouts). They cannot be linked to a customer, so they are excluded from customer-level modelling. This is a limitation.
- **Non-product codes:** a simple "stock code starts with a digit" rule would have removed genuine products (DCGS…, PADS, SP1002). Non-product codes are therefore a **reviewed list**: service charges, accounting entries, test products and gift vouchers.
- Result: 776,579 clean transactions, 5,852 customers, £17.1m revenue. All automated quality checks pass.

## Churn definition and validation

**Churn = no purchase in the next 180 days.** Models are trained on an earlier snapshot (history before 15 Dec 2010) and tested on a later one (history before 13 Jun 2011), so the model is always evaluated on a future period it has never seen.

| Model | AUC (test period) | Precision | Recall |
|---|---|---|---|
| Recency rule (no ML) | 0.770 | – | – |
| **Logistic regression** | **0.792** | 0.649 | 0.856 |
| Gradient boosting | 0.780 | 0.632 | 0.882 |

The simpler, more interpretable logistic regression outperformed gradient boosting, so it was chosen. The ML models improve on a simple recency rule, though modestly.

**Main churn drivers (SHAP):** order frequency, days between orders, customer tenure, cancelled orders and recent orders. Customers who order often and regularly are much less likely to churn. Customers with more cancellations were *less* likely to churn, so in this data cancellations reflect active, engaged buyers rather than dissatisfaction.

## Customer segments (k-means on RFM)

| Segment | Customers | Median days since last order | Median orders | Median spend |
|---|---|---|---|---|
| Lapsed one-time buyers | 1,982 | 400 | 1 | £261 |
| Loyal high-value | 1,147 | 16 | 13 | £4,889 |
| Active developing | 1,236 | 23 | 3 | £706 |
| At-risk regulars | 1,487 | 176 | 5 | £1,430 |

k=2 had the highest silhouette score but only separated active from inactive customers. k=4 (second best) was chosen because it gives more actionable groups.

## Value at risk and retention optimisation

- **Value if retained:** each customer's historical 6-month spend run-rate. Checked against what customers who stayed actually spent in the test period (rank correlation 0.67; £3.9m estimated vs £4.2m actual).
- **Value at risk** = churn probability × value if retained: **£1.23m** across current customers.
- **Optimisation:** a mixed-integer linear program (SciPy `milp`, HiGHS solver) assigns each customer at most one offer (email voucher or call plus voucher) to maximise expected revenue saved, within the budget and the team's call capacity.

| Same £5,000 budget | Customers targeted | Expected revenue saved | Net benefit |
|---|---|---|---|
| Rule of thumb: call highest churn risk first | 200 | £1,069 | −£3,931 |
| **Optimised (value at risk + budget)** | **1,120** | **£64,994** | **£59,994** |

The customers most likely to churn are mostly low-value, long-lapsed one-time buyers. Targeting by value at risk, rather than churn probability alone, spends the same budget far more effectively.

**Important:** offer costs, success rates, budget and call capacity are **illustrative assumptions**, not measured values. In practice they would be measured with controlled experiments (A/B tests) before relying on these figures. Revenue is used rather than profit because margin data is not available.

## Limitations

- One retailer, one channel (online), and data from 2009 to 2011.
- 22.8% of transactions cannot be linked to a customer.
- Retention offer effects are assumed, not measured.
- The value estimate is a simple run-rate; a probabilistic model (such as BG/NBD with Gamma-Gamma) would be a natural next step.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyterlab ipykernel
```
Download `online_retail_II.xlsx` from UCI into `data/raw/`, then run the notebooks in order.

Outputs are saved in `reports/`: model comparison (`roc_curves.png`), churn drivers (`shap_summary.png`), strategy comparison (`strategy_comparison.png`), the recommended target list and `results.json`.
