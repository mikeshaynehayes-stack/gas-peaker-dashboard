#!/usr/bin/env python3
"""Gas Peakers – Investment Committee sensitivity dashboard (Streamlit)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import Assumptions, run_model

st.set_page_config(page_title="Gas Peakers – IC Sensitivity Dashboard", page_icon="⚡", layout="wide")


# -------------------------------
# Formatting helpers
# -------------------------------
def fmt_gbp(x):
    return "—" if x is None else f"£{x:,.0f}"


def fmt_pct(x, dp=2):
    return "—" if x is None else f"{x * 100:.{dp}f}%"


def fmt_x(x, dp=2):
    return "—" if x is None else f"{x:.{dp}f}x"


def pct_range(centre, step, n, lo=0.0):
    """n values centred on `centre`, spaced by `step`, floored at `lo`."""
    half = n // 2
    vals = [round(centre + (i - half) * step, 6) for i in range(n)]
    return [v for v in vals if v >= lo]


# -------------------------------
# Sidebar – assumptions
# -------------------------------
sb = st.sidebar

def pct(label, lo, hi, default, step, **kw):
    """Slider shown in % but returning a decimal."""
    v = sb.slider(label, round(lo * 100, 4), round(hi * 100, 4), round(default * 100, 4),
                  round(step * 100, 4), format="%.2f%%", **kw)
    return round(v / 100, 6)

sb.title("⚙️ Assumptions")

sb.subheader("Income")
year1_rent = sb.number_input("Year 1 rent / NOI (£)", min_value=0.0, value=186_231.0, step=1_000.0)
rent_mode = sb.radio("Rent growth basis", ["Annual growth", "Periodic review"], horizontal=True,
                     help="Periodic review: rent steps up every N years by indexation (e.g. CPI), "
                          "subject to a collar and cap, upward-only.")
if rent_mode == "Periodic review":
    growth = pct("Assumed CPI (p.a.)", 0.0, 0.06, 0.025, 0.0025)
    review_every = sb.slider("Review every (years)", 1, 10, 5)
    collar = pct("Collar (p.a.)", 0.0, 0.03, 0.01, 0.0025)
    cap = pct("Cap (p.a.)", 0.0, 0.06, 0.04, 0.0025)
else:
    growth = pct("Rent growth (p.a.)", 0.0, 0.05, 0.015, 0.0025)
    review_every, collar, cap = 5, 0.0, 1.0

sb.subheader("Pricing")
entry_yield = pct("Entry yield", 0.04, 0.10, 0.0625, 0.0005)
exit_yield = pct("Exit yield", 0.04, 0.12, 0.0675, 0.0005)
forward_exit = sb.checkbox("Value exit on forward (next-year) income", value=True,
                           help="Market convention: exit value = year N+1 income ÷ exit yield.")
acq_cost = pct("Purchaser's costs (% of price)", 0.0, 0.08, 0.02, 0.0025,
                     help="UK investment convention is c.6.8% incl. SDLT.")
sell_cost = pct("Selling costs (% of exit value)", 0.0, 0.04, 0.015, 0.0025)
hold = sb.slider("Hold period (years)", 3, 25, 10)

sb.subheader("Debt")
ltv = pct("Loan-to-value", 0.0, 0.75, 0.50, 0.01)
rate = pct("Interest rate (fixed)", 0.0, 0.10, 0.05, 0.0025)
amortising = sb.checkbox("Amortising loan", value=False)
amort_years = sb.slider("Amortisation period (years)", 5, 30, 20) if amortising else 20

sb.subheader("Returns")
discount_rate = pct("Target return / discount rate (for NPV)", 0.0, 0.20, 0.08, 0.0025)

a = Assumptions(
    year1_rent=year1_rent, rent_mode=rent_mode, growth=growth, review_every=review_every,
    collar=collar, cap=cap, entry_yield=entry_yield, exit_yield=exit_yield,
    forward_exit=forward_exit, hold=hold, ltv=ltv, rate=rate, amortising=amortising,
    amort_years=amort_years, acq_cost=acq_cost, sell_cost=sell_cost, discount_rate=discount_rate,
)
r = run_model(a)

# -------------------------------
# Headline metrics
# -------------------------------
st.title("⚡ Gas Peakers – Investment Committee Dashboard")

c = st.columns(4)
c[0].metric("Purchase price (net)", fmt_gbp(r["price"]))
c[1].metric("Equity required", fmt_gbp(r["equity"]))
c[2].metric("Levered IRR", fmt_pct(r["levered_irr"]))
c[3].metric("Unlevered IRR", fmt_pct(r["unlevered_irr"]))

c = st.columns(4)
c[0].metric("Equity multiple", fmt_x(r["equity_multiple"]))
c[1].metric("Avg cash-on-cash", fmt_pct(r["avg_cash_yield"]))
c[2].metric("Min DSCR", fmt_x(r["min_dscr"]))
c[3].metric(f"NPV @ {discount_rate * 100:.2f}%", fmt_gbp(r["npv"]))

if r["min_dscr"] is not None and r["min_dscr"] < 1.0:
    st.error("Rent does not cover debt service in at least one year (DSCR < 1.0x).")
elif r["min_dscr"] is not None and r["min_dscr"] < 1.25:
    st.warning("Minimum DSCR is below 1.25x – likely outside typical lender covenants.")

with st.expander("Sources & uses / exit"):
    s1, s2 = st.columns(2)
    s1.markdown(
        f"| Sources & uses | £ |\n|---|---:|\n"
        f"| Net purchase price | {r['price']:,.0f} |\n"
        f"| Purchaser's costs | {r['acq_cost']:,.0f} |\n"
        f"| **Total cost** | **{r['all_in']:,.0f}** |\n"
        f"| Debt | {r['loan']:,.0f} |\n"
        f"| Equity | {r['equity']:,.0f} |"
    )
    s2.markdown(
        f"| Exit (end of year {hold}) | £ |\n|---|---:|\n"
        f"| Income capitalised | {r['exit_noi']:,.0f} |\n"
        f"| Gross exit value | {r['exit_gross']:,.0f} |\n"
        f"| Net of selling costs | {r['exit_net']:,.0f} |\n"
        f"| Loan repaid | {r['loan_repaid']:,.0f} |\n"
        f"| Payback year | {r['payback'] or 'Not within hold'} |"
    )

# -------------------------------
# Cash flows
# -------------------------------
st.subheader("💰 Cash flow projection")
years = list(range(0, hold + 1))
df = pd.DataFrame({
    "Year": years,
    "Rent / NOI (£)": [0.0] + r["noi"],
    "Interest (£)": [0.0] + [-x if x else 0.0 for x in r["interest"]],
    "Loan repayment (£)": [0.0] + [-x if x else 0.0 for x in r["principal"]],
    "Exit proceeds net of debt (£)": [0.0] * hold + [r["exit_net"] - r["loan_repaid"]],
    "Equity CF (£)": r["equity_cf"],
})
df["Cumulative equity CF (£)"] = df["Equity CF (£)"].cumsum()
st.dataframe(df.style.format({col: "{:,.0f}" for col in df.columns if col != "Year"}),
             use_container_width=True, hide_index=True)
st.download_button("Download cash flows (CSV)", df.to_csv(index=False).encode(), "cash_flows.csv", "text/csv")

fig = go.Figure()
fig.add_bar(x=df["Year"], y=df["Equity CF (£)"], name="Equity cash flow")
fig.add_scatter(x=df["Year"], y=df["Cumulative equity CF (£)"], name="Cumulative", mode="lines+markers")
fig.update_layout(title="Equity cash flows", xaxis_title="Year", yaxis_title="£", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Sensitivities
# -------------------------------
st.subheader("📊 Sensitivity analysis – levered IRR")


def heatmap(x_name, x_vals, y_name, y_vals, field_x, field_y, x_fmt, y_fmt):
    z = []
    for yv in y_vals:
        row = []
        for xv in x_vals:
            irr = run_model(a.with_(**{field_x: xv, field_y: yv}))["levered_irr"]
            row.append(None if irr is None else irr * 100)
        z.append(row)
    xl = [x_fmt(v) for v in x_vals]
    yl = [y_fmt(v) for v in y_vals]
    text = [["—" if v is None else f"{v:.2f}%" for v in row] for row in z]
    f = go.Figure(go.Heatmap(z=z, x=xl, y=yl, text=text, texttemplate="%{text}",
                             colorscale="RdYlGn", colorbar=dict(title="IRR %"),
                             hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<br>IRR: %{{text}}<extra></extra>"))
    f.update_layout(xaxis_title=x_name, yaxis_title=y_name, height=380, margin=dict(t=30),
                    xaxis_type="category", yaxis_type="category")
    return f


pct1 = lambda v: f"{v * 100:.1f}%"
pct2 = lambda v: f"{v * 100:.2f}%"
growth_label = "CPI assumption" if rent_mode == "Periodic review" else "Rent growth"

h1, h2 = st.columns(2)
with h1:
    st.caption(f"Exit yield vs {growth_label.lower()}")
    st.plotly_chart(heatmap("Exit yield", pct_range(exit_yield, 0.0025, 5, 0.01),
                            growth_label, pct_range(growth, 0.005, 5),
                            "exit_yield", "growth", pct2, pct1), use_container_width=True)
with h2:
    st.caption("Entry yield vs exit yield")
    st.plotly_chart(heatmap("Exit yield", pct_range(exit_yield, 0.0025, 5, 0.01),
                            "Entry yield", pct_range(entry_yield, 0.0025, 5, 0.01),
                            "exit_yield", "entry_yield", pct2, pct2), use_container_width=True)

h3, h4 = st.columns(2)
with h3:
    st.caption("Interest rate vs LTV")
    st.plotly_chart(heatmap("LTV", [v for v in pct_range(ltv, 0.10, 5) if v <= 0.8],
                            "Interest rate", pct_range(rate, 0.005, 5),
                            "ltv", "rate", lambda v: f"{v * 100:.0f}%", pct2), use_container_width=True)
with h4:
    st.caption("Hold period vs exit yield")
    holds = [h for h in (hold - 4, hold - 2, hold, hold + 2, hold + 4) if 1 <= h <= 40]
    st.plotly_chart(heatmap("Exit yield", pct_range(exit_yield, 0.0025, 5, 0.01),
                            "Hold (years)", holds, "exit_yield", "hold", pct2, str),
                    use_container_width=True)

st.caption("Model treats the site as an income investment (landlord's interest under a ground lease). "
           "It does not model plant operating economics (capacity market, spark spreads).")
