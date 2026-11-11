"""Basic checks for model.py. Run with: python -m pytest -q"""
import numpy_financial as npf
import pytest

from model import Assumptions, rent_schedule, run_model


def original_irr(rent=186231.0, g=0.015, entry=0.0625, exit_=0.0675, hold=10,
                 ltv=0.5, rate=0.05, acq=0.02, sell=0.015):
    """Logic from the first version of ic_dashboard.py (trailing-income exit, interest-only)."""
    price = rent / entry
    equity = price * (1 + acq) - price * ltv
    interest = price * ltv * rate
    noi = [rent * (1 + g) ** (y - 1) for y in range(1, hold + 1)]
    cf = [-equity] + [n - interest for n in noi]
    cf[-1] += noi[-1] / exit_ * (1 - sell) - price * ltv
    return npf.irr(cf)


def test_matches_original_when_using_trailing_exit():
    r = run_model(Assumptions(forward_exit=False))
    assert r["levered_irr"] == pytest.approx(original_irr(), abs=1e-9)


def test_forward_exit_increases_value_when_growing():
    assert run_model(Assumptions(forward_exit=True))["exit_gross"] > \
        run_model(Assumptions(forward_exit=False))["exit_gross"]


def test_periodic_review_steps():
    a = Assumptions(rent_mode="Periodic review", growth=0.03, review_every=5, collar=0.01, cap=0.04)
    s = rent_schedule(a, 11)
    assert s[0] == s[4] == a.year1_rent
    assert s[5] == pytest.approx(a.year1_rent * 1.03 ** 5)
    assert s[10] == pytest.approx(a.year1_rent * 1.03 ** 10)


def test_collar_and_cap_apply():
    lo = Assumptions(rent_mode="Periodic review", growth=0.0, collar=0.01, cap=0.04)
    hi = Assumptions(rent_mode="Periodic review", growth=0.08, collar=0.01, cap=0.04)
    assert rent_schedule(lo, 6)[5] == pytest.approx(lo.year1_rent * 1.01 ** 5)
    assert rent_schedule(hi, 6)[5] == pytest.approx(hi.year1_rent * 1.04 ** 5)


def test_amortising_loan_reduces_balance():
    r = run_model(Assumptions(amortising=True, amort_years=20))
    assert 0 < r["loan_repaid"] < r["loan"]
    assert sum(r["principal"]) == pytest.approx(r["loan"] - r["loan_repaid"])


def test_fully_amortised_within_hold():
    r = run_model(Assumptions(amortising=True, amort_years=5, hold=10))
    assert r["loan_repaid"] == pytest.approx(0, abs=1e-6)


def test_zero_irr_is_reported_not_blank():
    # Build a case where equity just gets its money back.
    r = run_model(Assumptions(ltv=0, acq_cost=0, sell_cost=0, growth=0, entry_yield=0.05,
                              exit_yield=0.05, forward_exit=False))
    assert r["levered_irr"] == pytest.approx(0.05, abs=1e-9)
    assert r["levered_irr"] is not None


def test_no_debt_dscr_none():
    assert run_model(Assumptions(ltv=0))["min_dscr"] is None
