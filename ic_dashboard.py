#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="Gas Peakers – IC Sensitivity Dashboard",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------
# Helper Functions
# -------------------------------
def fmt_currency(x):
    return f"£{x:,.0f}"

def fmt_pct(x, dp=2):
    return f"{x*100:.{dp}f}%" if x is not None else "—"

def irr_safe(cashflows):
    try:
        return float(npf.irr(cashflows))
    except Exception:
        return None

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("⚙️ Assumptions")

initial_rent = st.sidebar.number_input("Year 1 NOI (£)", min_value=0.0, value=186231.0, step=1000.0)
growth = st.sidebar.slider("Rent Growth (p.a.)", 0.0, 0.05, 0.015, 0.0025)
entry_cap = st.sidebar.slider("Entry Cap Rate", 0.04, 0.08, 0.0625, 0.0005)
exit_cap = st.sidebar.slider("Exit Cap Rate", 0.05, 0.09, 0.0675, 0.0005)
hold = st.sidebar.slider("Hold Period (years)", 5, 20, 10, 1)
ltv = st.sidebar.slider("Loan-to-Value (LTV)", 0.0, 0.7, 0.50, 0.01)
rate = st.sidebar.slider("Interest Rate (fixed)", 0.00, 0.10, 0.05, 0.0025)
acq_cost = st.sidebar.slider("Acquisition Costs (% of price)", 0.00, 0.05, 0.02, 0.0025)
sell_cost = st.sidebar.slider("Selling Costs (% of exit value)", 0.00, 0.04, 0.015, 0.0025)

# -------------------------------
# Core Calculations
# -------------------------------
purchase_price = initial_rent / entry_cap
acq_cost_amt = purchase_price * acq_cost
purchase_price_allin = purchase_price + acq_cost_amt

loan_amt = purchase_price * ltv
equity = purchase_price_allin - loan_amt
annual_interest = loan_amt * rate

years = list(range(1, hold + 1))
noi = [initial_rent * ((1 + growth) ** (y-1)) for y in years]

exit_noi = noi[-1]
exit_value = exit_noi / exit_cap
exit_value_net = exit_value * (1 - sell_cost)
net_sale_proceeds = exit_value_net - loan_amt

equity_cf = [-equity] + [n - annual_interest for n in noi]
equity_cf[-1] += net_sale_proceeds

levered_irr = irr_safe(equity_cf)
unlevered_cf = [-purchase_price_allin] + noi[:-1] + [noi[-1] + exit_value_net]
unlevered_irr = irr_safe(unlevered_cf)

avg_cash_yield = np.mean([(n - annual_interest) / equity for n in noi]) if equity > 0 else np.nan

# -------------------------------
# Dashboard Layout
# -------------------------------
st.title("⚡ Gas Peakers – Investment Committee Dashboard")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Purchase Price", fmt_currency(purchase_price))
col2.metric("Equity Required", fmt_currency(equity))
col3.metric("Levered IRR", fmt_pct(levered_irr))
col4.metric("Unlevered IRR", fmt_pct(unlevered_irr))

# Cash Flow Table
df = pd.DataFrame({
    "Year": [0] + years,
    "NOI (£)": [0] + [round(x,0) for x in noi],
    "Interest (£)": [0] + [round(annual_interest,0)]*hold,
    "Equity CF (£)": [round(x,0) for x in equity_cf]
})
st.subheader("💰 Cash Flow Projection")
st.dataframe(df, use_container_width=True)

# Cash Flow Chart
fig = go.Figure()
fig.add_trace(go.Bar(x=df["Year"], y=df["Equity CF (£)"], name="Equity Cash Flow"))
fig.update_layout(title="Equity Cash Flows Over Time", xaxis_title="Year", yaxis_title="£")
st.plotly_chart(fig, use_container_width=True)

# Sensitivity Heatmap
st.subheader("📊 Sensitivity Analysis")
cap_rates_exit = [0.0625, 0.0675, 0.0725]
rent_growth_rates = [0.0, 0.015, 0.03]
records = []
for g in rent_growth_rates:
    for ce in cap_rates_exit:
        noi_test = [initial_rent * ((1 + g) ** (y-1)) for y in years]
        exit_noi_test = noi_test[-1]
        exit_val_test = exit_noi_test / ce
        exit_val_net = exit_val_test * (1 - sell_cost)
        net_sale_test = exit_val_net - loan_amt
        eq_cf_test = [-equity] + [n - annual_interest for n in noi_test]
        eq_cf_test[-1] += net_sale_test
        irr = irr_safe(eq_cf_test)
        records.append({
            "Rent Growth": f"{g*100:.1f}%",
            "Exit Cap Rate": f"{ce*100:.2f}%",
            "Levered IRR": irr*100 if irr else None
        })
df_sens = pd.DataFrame(records)
st.dataframe(df_sens.pivot(index="Rent Growth", columns="Exit Cap Rate", values="Levered IRR").round(2))
