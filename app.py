"""Customer Retention Planner - a web app for any retailer's customer purchase data.

Run with:  streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from retention import engine

st.set_page_config(page_title="Customer Retention Planner", page_icon="📉", layout="wide")

DEMO_FILE = Path(__file__).parent / "data" / "raw" / "online_retail_II.parquet"


@st.cache_data(show_spinner="Reading your file...")
def read_file(data: bytes, name: str) -> pd.DataFrame:
    suffix = name.lower().rsplit(".", 1)[-1]
    buf = io.BytesIO(data)
    if suffix == "csv":
        return pd.read_csv(buf, encoding_errors="replace", low_memory=False)
    if suffix in ("xlsx", "xls"):
        sheets = pd.read_excel(buf, sheet_name=None, engine="calamine")
        return pd.concat(sheets.values(), ignore_index=True)
    if suffix == "parquet":
        return pd.read_parquet(buf)
    raise ValueError("Please upload a CSV, Excel or Parquet file.")


def money(x):
    return f"{st.session_state.get('currency', '£')}{x:,.0f}"


# ---------------------------------------------------------------- sidebar: business settings
with st.sidebar:
    st.header("Your retention settings")
    st.caption("These are business assumptions. Adjust them to match your offers and team.")
    st.selectbox("Currency", ["£", "$", "€"], key="currency")
    budget = st.number_input("Retention budget", min_value=100, value=5000, step=500)
    call_capacity = st.number_input("Customers your team can call", min_value=0, value=300, step=50)
    st.markdown("**Email voucher**")
    email_cost = st.number_input("Cost per customer", min_value=0.1, value=2.0, step=0.5, key="ec")
    email_rate = st.slider("Share of leaving customers it keeps", 0.0, 0.5, 0.05, 0.01, key="er")
    st.markdown("**Phone call + voucher**")
    call_cost = st.number_input("Cost per customer", min_value=0.1, value=25.0, step=1.0, key="cc")
    call_rate = st.slider("Share of leaving customers it keeps", 0.0, 0.8, 0.25, 0.01, key="cr")

offers = {
    "email": {"label": "Send email voucher", "cost": email_cost, "save_rate": email_rate},
    "call": {"label": "Call and offer voucher", "cost": call_cost, "save_rate": call_rate, "uses_call": True},
}

# ---------------------------------------------------------------- header
st.title("Customer Retention Planner")
st.write("Upload your customer purchase data to find out **who is likely to stop buying**, "
         "**how much revenue is at risk**, and **who to contact first** within your budget.")

# ---------------------------------------------------------------- step 1: upload
st.subheader("1. Upload your data")
st.caption("One row per purchase (or per item in a purchase), with a customer, a date and the amount spent. "
           "CSV, Excel or Parquet.")
col_a, col_b = st.columns([3, 1])
with col_a:
    upload = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "parquet"], label_visibility="collapsed")
with col_b:
    use_demo = DEMO_FILE.exists() and st.button("Try the demo data", help="A real UK online retailer, 2009 to 2011")

if use_demo:
    st.session_state["source"] = ("demo", DEMO_FILE.read_bytes(), DEMO_FILE.name)
elif upload is not None:
    st.session_state["source"] = ("upload", upload.getvalue(), upload.name)

if "source" not in st.session_state:
    st.info("Upload a file, or try the demo data, to begin.")
    st.stop()

kind, data, name = st.session_state["source"]
try:
    df = read_file(data, name)
except Exception as exc:  # unreadable file
    st.error(f"We couldn't read this file: {exc}")
    st.stop()
st.success(f"Loaded **{name}**: {len(df):,} rows and {len(df.columns)} columns.")
with st.expander("Preview the first rows"):
    st.dataframe(df.head(10), width="stretch")

# ---------------------------------------------------------------- step 2: check columns
st.subheader("2. Check we've understood your columns")
st.caption("We matched your columns automatically. Correct any that are wrong.")
guess = engine.guess_mapping(df)
options = ["— not in my data —"] + list(df.columns)
mapping = {}
cols = st.columns(4)
for i, (field, label) in enumerate(engine.STANDARD_FIELDS.items()):
    default = options.index(guess[field]) if guess[field] in options else 0
    choice = cols[i % 4].selectbox(label, options, index=default, key=f"map_{field}_{name}")
    mapping[field] = None if choice == options[0] else choice

ok, problems, warnings = engine.check_data(df, mapping)
if not ok:
    st.error("**This doesn't look like customer purchase data, so we can't analyse it yet.**")
    for p in problems:
        st.markdown(f"- {p}")
    st.caption("This tool needs one row per purchase with a customer, a date and the amount spent. "
               "If your data has these, pick the right columns above.")
    st.stop()
for w in warnings:
    st.warning(w)

run_key = (name, len(df), tuple(sorted((k, v or "") for k, v in mapping.items())))
if st.button("Analyse my customers", type="primary"):
    with st.spinner("Cleaning the data and training the model..."):
        try:
            tx, cancels, report = engine.clean(df, mapping)
            st.session_state["result"] = (run_key, report, engine.analyse(tx, cancels))
        except ValueError as exc:
            st.error(f"We couldn't complete the analysis: {exc}")
            st.stop()

if "result" not in st.session_state or st.session_state["result"][0] != run_key:
    st.stop()
_, report, result = st.session_state["result"]
customers = result["customers"]
days = result["window_days"]

# ---------------------------------------------------------------- step 3: data quality
st.subheader("3. What we did to your data")
start, end = result["period"]
st.write(f"Your data covers **{start:%d %b %Y} to {end:%d %b %Y}**. Before analysing it, we:")
for r in report[1:-1]:
    st.markdown(f"- **{r['step']}:** {r['rows']:,} rows. {r['why']}")
st.markdown(f"This left **{report[-1]['rows']:,} purchase lines** from **{len(customers):,} customers**.")

# ---------------------------------------------------------------- step 4: results
st.subheader("4. Your customers at a glance")
likely = customers["churn_prob"] >= 0.5
m1, m2, m3 = st.columns(3)
m1.metric("Customers analysed", f"{len(customers):,}")
m2.metric(f"Likely to stop buying in the next {days} days", f"{likely.sum():,}",
          f"{likely.mean():.0%} of customers", delta_color="off")
m3.metric(f"Revenue at risk over the next {days} days", money(customers["value_at_risk"].sum()))

st.markdown("**Why customers stop buying**")
for d in result["drivers"]:
    st.markdown(f"- {d}")

st.markdown("**Your customer groups**")
seg = (customers.groupby("segment")
       .agg(customers=("customer_id", "count"),
            days_since_last_order=("recency_days", "median"),
            orders=("frequency", "median"),
            typical_spend=("monetary", "median"),
            revenue_at_risk=("value_at_risk", "sum"))
       .reindex(engine.SEGMENT_TEXT.keys()).dropna(how="all"))
seg_view = pd.DataFrame({
    "Customers": seg["customers"].astype(int).map("{:,}".format),
    "Typical days since last order": seg["days_since_last_order"].round().astype(int),
    "Typical number of orders": seg["orders"].round().astype(int),
    "Typical spend": seg["typical_spend"].map(money),
    "Revenue at risk": seg["revenue_at_risk"].map(money),
    "What it means": [engine.SEGMENT_TEXT[s] for s in seg.index],
})
st.dataframe(seg_view, width="stretch")

with st.expander("How reliable are these predictions?"):
    st.write(f"We tested the model on the most recent {days} days, which it had not seen. "
             f"When comparing a customer who left with one who stayed, it correctly identified the "
             f"one more likely to leave **{result['auc']:.0%} of the time**. A simple "
             f"'days since last order' rule managed {result['baseline_auc']:.0%}.")
    st.caption("A score of 50% would be guessing; 100% would be perfect. Predictions describe likelihood, not certainty.")

# ---------------------------------------------------------------- step 5: retention plan
st.subheader("5. Your retention plan")
st.caption("Adjust the budget and offers in the left-hand panel. The plan updates automatically.")
with st.spinner("Finding the best use of your budget..."):
    plan, rule = engine.optimise(customers, budget, call_capacity, offers)
contacted = plan[plan["recommended_action"] != "No action"]
saved, spent = contacted["expected_revenue_saved"].sum(), contacted["offer_cost"].sum()

p1, p2, p3 = st.columns(3)
p1.metric("Customers to contact", f"{len(contacted):,}",
          f"{(contacted['recommended_action'] == offers['call']['label']).sum():,} calls, "
          f"{(contacted['recommended_action'] == offers['email']['label']).sum():,} emails", delta_color="off")
p2.metric("Budget used", money(spent), f"of {money(budget)}", delta_color="off")
p3.metric("Expected revenue kept", money(saved))

gap = saved - rule["saved"]
if gap > 0:
    st.success(f"Contacting simply the customers **most likely to leave** would keep about "
               f"**{money(rule['saved'])}** with the same budget. This plan keeps **{money(gap)} more**, "
               f"because it focuses on customers who are both at risk **and** valuable.")

show = contacted.sort_values("expected_revenue_saved", ascending=False)
table = pd.DataFrame({
    "Customer": show["customer_id"],
    "Group": show["segment"],
    "Chance of leaving": show["churn_prob"].map("{:.0%}".format),
    "Value if they stay": show["value_if_retained"].map(money),
    "Recommended action": show["recommended_action"],
})
st.dataframe(table.head(50), width="stretch", hide_index=True)
st.download_button("Download the full contact list (CSV)",
                   table.to_csv(index=False).encode("utf-8"),
                   file_name="retention_plan.csv", mime="text/csv")

st.info("**Please note:** offer costs and success rates are assumptions you set on the left. "
        "Test them on a small group of customers first, then update the settings with what you learn.")
