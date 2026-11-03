from datetime import date

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from bubble.data import build_dataset, repair_split_misalignment, SeriesData
from bubble.garch import fit_garch
from bubble.gsadf import _bsadf_batch, default_min_window, run_gsadf, simulate_illustrative_bubble
from bubble.periods import detect_periods
from bubble.stats import describe_returns, max_drawdown


def _prices(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_full_sample_adf_matches_statsmodels(lag):
    y = np.cumsum(np.random.default_rng(0).standard_normal(250)) + 30
    _, fwd = _bsadf_batch(y[None, :], lag, default_min_window(250))
    assert fwd[0, -1] == pytest.approx(adfuller(y, maxlag=lag, autolag=None, regression="c")[0], rel=1e-8)


def test_gsadf_detects_simulated_bubble():
    res = run_gsadf(simulate_illustrative_bubble(), reps=99)
    assert res.reject_95
    assert len(res.episodes) >= 1


def test_gsadf_random_walk_rarely_rejects():
    rejections = 0
    for seed in range(8):
        p = _prices(np.exp(np.cumsum(0.01 * np.random.default_rng(100 + seed).standard_normal(300)) + 4))
        rejections += run_gsadf(p, reps=99, seed=1).reject_95
    assert rejections <= 3


def test_max_drawdown():
    p = _prices([100, 120, 90, 60, 130])
    mdd = max_drawdown(p)
    assert mdd["max_drawdown"] == pytest.approx(-0.5)
    assert mdd["peak_price"] == 120 and mdd["trough_price"] == 60
    assert mdd["recovery_date"] == p.index[4]


def test_describe_returns_annualisation():
    p = _prices(np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0.001, 0.02, 500)])))
    d = describe_returns(p, 252)
    assert d["Annualised volatility"] == pytest.approx(d["Daily std. deviation"] * np.sqrt(252))
    assert d["Annualised mean return"] == pytest.approx(d["Mean daily log return"] * 252)


def test_garch_recovers_persistence():
    rng = np.random.default_rng(3)
    n, omega, alpha, beta = 3000, 0.05, 0.10, 0.85
    r, s2 = np.zeros(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    prices = _prices(100 * np.exp(np.cumsum(np.r_[0, r / 100])))
    g = fit_garch(prices, dist="normal")
    assert g.persistence == pytest.approx(alpha + beta, abs=0.04)


def test_split_repair_only_with_matching_record():
    close = np.r_[np.full(10, 140.0), np.full(10, 100.0)]
    df = pd.DataFrame({c: close for c in ["Open", "High", "Low", "Close", "AdjClose"]} | {"Volume": 1e6},
                      index=pd.bdate_range("2024-01-01", periods=20))
    fixed, repairs = repair_split_misalignment(df, pd.Series([1.4], index=[df.index[12]]), 0.20)
    assert len(repairs) == 1 and fixed["Close"].pct_change().abs().max() < 1e-9
    untouched, none = repair_split_misalignment(df, pd.Series(dtype=float), 0.20)
    assert none == [] and untouched["Close"].equals(df["Close"])


def test_missing_policy_drop_does_not_invent_values():
    idx = pd.bdate_range("2024-01-01", periods=80)
    full = pd.DataFrame({c: np.linspace(10, 20, 80) for c in ["Open", "High", "Low", "Close", "AdjClose"]} | {"Volume": 1.0}, index=idx)
    gappy = full.copy()
    gappy.loc[idx[30:33], "Volume"] = 0
    a = SeriesData("A", "A", "equity", full, "test", "", True)
    b = SeriesData("B", "B", "equity", gappy, "test", "", True)
    ds = build_dataset([a, b], date(2024, 1, 1), date(2024, 12, 31), "drop")
    assert ds["B"].missing["missing"] == 3 and ds["B"].missing["used"] == 77
    ds = build_dataset([a, b], date(2024, 1, 1), date(2024, 12, 31), "ffill")
    assert ds["B"].missing["filled"] == 3


def test_detect_periods():
    p = _prices(np.r_[np.linspace(100, 80, 50), np.linspace(80, 200, 100), np.linspace(200, 120, 50)])
    per = detect_periods(p, date(2024, 1, 1))
    assert p.loc[per.peak] == 200 and p.loc[per.rally_start] == 80 and p.loc[per.trough] == 120
