"""Descriptive return statistics and drawdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def drawdown(prices: pd.Series) -> pd.Series:
    """D_t = P_t / max_{s<=t} P_s - 1."""
    return prices / prices.cummax() - 1.0


def max_drawdown(prices: pd.Series) -> dict:
    dd = drawdown(prices)
    trough = dd.idxmin()
    peak = prices.loc[:trough].idxmax()
    after = prices.loc[trough:]
    recovered = after[after >= prices.loc[peak]]
    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovered.index[0] if len(recovered) else None,
        "peak_price": float(prices.loc[peak]),
        "trough_price": float(prices.loc[trough]),
    }


def describe_returns(prices: pd.Series, periods_per_year: int = 252) -> dict:
    r = np.log(prices).diff().dropna()
    n = len(r)
    mu, var = r.mean(), r.var(ddof=1)
    sd = np.sqrt(var)
    jb = sps.jarque_bera(r)
    mdd = max_drawdown(prices)
    return {
        "Observations": n,
        "Start": prices.index[0].date(),
        "End": prices.index[-1].date(),
        "Total return": prices.iloc[-1] / prices.iloc[0] - 1,
        "Mean daily log return": mu,
        "Annualised mean return": mu * periods_per_year,
        "Daily variance": var,
        "Daily std. deviation": sd,
        "Annualised volatility": sd * np.sqrt(periods_per_year),
        "Skewness": float(sps.skew(r, bias=False)),
        "Excess kurtosis": float(sps.kurtosis(r, bias=False)),
        "Min daily return": r.min(),
        "Max daily return": r.max(),
        "Max drawdown": mdd["max_drawdown"],
        "MDD peak": mdd["peak_date"].date(),
        "MDD trough": mdd["trough_date"].date(),
        "MDD recovered": mdd["recovery_date"].date() if mdd["recovery_date"] is not None else "Not recovered",
        "Jarque-Bera p-value": float(jb.pvalue),
    }


PERCENT_FIELDS = {
    "Total return", "Mean daily log return", "Annualised mean return", "Daily std. deviation",
    "Annualised volatility", "Min daily return", "Max daily return", "Max drawdown",
}


def rolling_volatility(prices: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    r = np.log(prices).diff()
    return r.rolling(window).std() * np.sqrt(periods_per_year)
