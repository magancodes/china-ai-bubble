"""Pre-rally / rally / correction period analysis.

Period boundaries are either derived from the reference series' own price path or set
manually by the user; they are never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import max_drawdown


@dataclass
class Periods:
    pre_start: pd.Timestamp
    rally_start: pd.Timestamp
    peak: pd.Timestamp
    trough: pd.Timestamp | None
    method: str

    def as_dict(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
        return {
            "Pre-rally": (self.pre_start, self.rally_start),
            "Rally": (self.rally_start, self.peak),
            "Correction": (self.peak, self.trough) if self.trough is not None and self.trough > self.peak else None,
        }


def detect_periods(prices: pd.Series, peak_search_from: date, rally_lookback_days: int = 365) -> Periods:
    """Peak = highest close on/after `peak_search_from`; rally start = lowest close within the
    look-back window before the peak; correction trough = lowest close after the peak."""
    prices = prices.dropna()
    window = prices.loc[pd.Timestamp(peak_search_from) :]
    if window.empty:
        raise ValueError("No data on or after the peak search date.")
    peak = window.idxmax()
    before = prices.loc[peak - pd.Timedelta(days=rally_lookback_days) : peak]
    rally_start = before.idxmin()
    after = prices.loc[peak:]
    trough = after.idxmin() if len(after) > 1 and after.idxmin() != peak else None
    return Periods(prices.index[0], rally_start, peak, trough, "Derived from price path")


def period_metrics(prices: pd.Series, volume: pd.Series, periods: Periods, periods_per_year: int = 252) -> pd.DataFrame:
    rows = []
    pre_vol = None
    for name, span in periods.as_dict().items():
        if span is None:
            continue
        s, e = span
        p = prices.loc[s:e].dropna()
        if len(p) < 3:
            continue
        r = np.log(p).diff().dropna()
        v = volume.loc[s:e].dropna()
        avg_vol = float(v.mean()) if len(v) else np.nan
        if name == "Pre-rally":
            pre_vol = avg_vol
        rows.append(
            {
                "Period": name,
                "Start": p.index[0].date(),
                "End": p.index[-1].date(),
                "Trading days": len(p),
                "Start price": float(p.iloc[0]),
                "End price": float(p.iloc[-1]),
                "Cumulative return": float(p.iloc[-1] / p.iloc[0] - 1),
                "Mean daily log return": float(r.mean()),
                "Annualised volatility": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
                "Max drawdown": max_drawdown(p)["max_drawdown"],
                "Avg daily volume": avg_vol,
                "Volume vs pre-rally": avg_vol / pre_vol if pre_vol else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_periods(prices: pd.Series, a: tuple, b: tuple) -> dict:
    """Welch t-test on mean daily log returns and Brown-Forsythe test on variances."""
    ra = np.log(prices.loc[a[0] : a[1]]).diff().dropna()
    rb = np.log(prices.loc[b[0] : b[1]]).diff().dropna()
    if len(ra) < 5 or len(rb) < 5:
        return {}
    t = sps.ttest_ind(ra, rb, equal_var=False)
    bf = sps.levene(ra, rb, center="median")
    return {
        "n_a": len(ra),
        "n_b": len(rb),
        "mean_a": float(ra.mean()),
        "mean_b": float(rb.mean()),
        "welch_t": float(t.statistic),
        "welch_p": float(t.pvalue),
        "var_ratio": float(rb.var(ddof=1) / ra.var(ddof=1)),
        "bf_stat": float(bf.statistic),
        "bf_p": float(bf.pvalue),
    }
