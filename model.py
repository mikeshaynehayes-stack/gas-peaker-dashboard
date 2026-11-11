"""Cash flow model for a gas peaker site held as a rental investment.

The model is kept separate from the Streamlit UI so it can be tested
and reused for the sensitivity tables.
"""
from dataclasses import dataclass, replace

import numpy as np
import numpy_financial as npf


@dataclass(frozen=True)
class Assumptions:
    year1_rent: float = 186_231.0          # Year 1 net income (£)
    rent_mode: str = "Annual growth"      # or "Periodic review"
    growth: float = 0.015                  # annual growth / assumed CPI
    review_every: int = 5                  # years between reviews (periodic mode)
    collar: float = 0.0                    # min p.a. uplift at review
    cap: float = 0.04                      # max p.a. uplift at review
    entry_yield: float = 0.0625
    exit_yield: float = 0.0675
    forward_exit: bool = True              # value exit on next year's income
    hold: int = 10
    ltv: float = 0.50
    rate: float = 0.05
    amortising: bool = False
    amort_years: int = 20
    acq_cost: float = 0.02                 # % of net price
    sell_cost: float = 0.015               # % of gross exit value
    discount_rate: float = 0.08            # for NPV

    def with_(self, **kw):
        return replace(self, **kw)


def irr_safe(cashflows):
    try:
        v = float(npf.irr(cashflows))
        return None if np.isnan(v) else v
    except Exception:
        return None


def rent_schedule(a: Assumptions, n_years: int):
    """Rent for years 1..n_years."""
    if a.rent_mode == "Periodic review":
        eff = min(max(a.growth, a.collar), a.cap)
        out = []
        for y in range(1, n_years + 1):
            reviews_done = (y - 1) // a.review_every
            out.append(a.year1_rent * (1 + eff) ** (reviews_done * a.review_every))
        return out
    return [a.year1_rent * (1 + a.growth) ** (y - 1) for y in range(1, n_years + 1)]


def debt_schedule(a: Assumptions, loan: float):
    """Return (interest, principal, closing balance) lists for years 1..hold."""
    interest, principal, balance = [], [], []
    bal = loan
    if a.amortising and loan > 0:
        pmt = float(-npf.pmt(a.rate, a.amort_years, loan)) if a.rate > 0 else loan / a.amort_years
    for y in range(1, a.hold + 1):
        i = bal * a.rate
        if a.amortising and loan > 0 and y <= a.amort_years:
            p = min(pmt - i, bal)
        else:
            p = 0.0
        bal -= p
        interest.append(i)
        principal.append(p)
        balance.append(bal)
    return interest, principal, balance


def run_model(a: Assumptions) -> dict:
    rents = rent_schedule(a, a.hold + 1)
    noi = rents[: a.hold]

    price = a.year1_rent / a.entry_yield
    acq = price * a.acq_cost
    all_in = price + acq
    loan = price * a.ltv
    equity = all_in - loan

    interest, principal, balance = debt_schedule(a, loan)
    debt_service = [i + p for i, p in zip(interest, principal)]

    exit_noi = rents[a.hold] if a.forward_exit else noi[-1]
    exit_gross = exit_noi / a.exit_yield
    exit_net = exit_gross * (1 - a.sell_cost)
    loan_repaid = balance[-1]

    cfads = [n - ds for n, ds in zip(noi, debt_service)]
    equity_cf = [-equity] + cfads
    equity_cf[-1] += exit_net - loan_repaid

    unlev_cf = [-all_in] + list(noi)
    unlev_cf[-1] += exit_net

    distributions = sum(cf for cf in equity_cf[1:])
    dscr = [n / ds if ds > 0 else None for n, ds in zip(noi, debt_service)]
    dscr_valid = [d for d in dscr if d is not None]

    cum, payback = -equity, None
    for y, cf in enumerate(equity_cf[1:], start=1):
        cum += cf
        if payback is None and cum >= 0:
            payback = y

    return {
        "price": price,
        "acq_cost": acq,
        "all_in": all_in,
        "loan": loan,
        "equity": equity,
        "noi": noi,
        "interest": interest,
        "principal": principal,
        "balance": balance,
        "exit_noi": exit_noi,
        "exit_gross": exit_gross,
        "exit_net": exit_net,
        "loan_repaid": loan_repaid,
        "equity_cf": equity_cf,
        "unlev_cf": unlev_cf,
        "levered_irr": irr_safe(equity_cf),
        "unlevered_irr": irr_safe(unlev_cf),
        "equity_multiple": distributions / equity if equity > 0 else None,
        "avg_cash_yield": float(np.mean(cfads)) / equity if equity > 0 else None,
        "min_dscr": min(dscr_valid) if dscr_valid else None,
        "npv": float(npf.npv(a.discount_rate, equity_cf)),
        "payback": payback,
    }
