"""Rule-based risk indicators.

Thresholds are user-adjustable heuristics, not statistical findings. Each indicator is
computed only from the fetched price/volume history, except valuation ratios which are a
current snapshot from Yahoo Finance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    abnormal_z: float = 3.0
    vol_ratio: float = 1.5
    pe: float = 100.0
    pb: float = 10.0
    volume_surge: float = 1.5
    ma_premium: float = 0.30
    momentum_60d: float = 0.50


def abnormal_returns(prices: pd.Series, window: int = 60) -> pd.Series:
    """z_t = (r_t - mean_{t-window..t-1}) / std_{t-window..t-1}, using only prior data."""
    r = np.log(prices).diff()
    mu = r.rolling(window).mean().shift(1)
    sd = r.rolling(window).std().shift(1)
    return ((r - mu) / sd).dropna()


def indicators(prices: pd.Series, volume: pd.Series, valuation: dict | None, th: Thresholds) -> list[dict]:
    prices = prices.dropna()
    r = np.log(prices).diff()
    z = abnormal_returns(prices)
    recent_z = z.iloc[-20:]
    n_abn = int((recent_z.abs() > th.abnormal_z).sum())
    vol20 = r.iloc[-20:].std()
    vol120 = r.iloc[-120:].std()
    vratio = float(vol20 / vol120) if vol120 > 0 else np.nan
    v = volume.dropna()
    vsurge = float(v.iloc[-20:].mean() / v.iloc[-120:].mean()) if len(v) >= 120 else np.nan
    ma200 = prices.rolling(200).mean().iloc[-1]
    prem = float(prices.iloc[-1] / ma200 - 1) if not np.isnan(ma200) else np.nan
    mom = float(prices.iloc[-1] / prices.iloc[-61] - 1) if len(prices) > 61 else np.nan

    out = [
        _row("Abnormal returns", "Days in last 20 with |z| > %.1f" % th.abnormal_z, n_abn, n_abn >= 1,
             f"Latest z = {z.iloc[-1]:+.2f}", "Return z-score vs. prior 60-day mean and std."),
        _row("Rising volatility", "20d / 120d realised volatility", vratio, vratio > th.vol_ratio,
             f"threshold {th.vol_ratio:.2f}", "Short-term volatility relative to its medium-term level."),
        _row("Volume surge", "20d / 120d average volume", vsurge, vsurge > th.volume_surge,
             f"threshold {th.volume_surge:.2f}", "Turnover spikes are a classic sign of speculative participation."),
        _row("Price vs. 200-day MA", "Premium over 200-day moving average", prem, prem > th.ma_premium,
             f"threshold {th.ma_premium:+.0%}", "Large deviations from trend indicate overextension."),
        _row("60-day momentum", "Return over last 60 trading days", mom, mom > th.momentum_60d,
             f"threshold {th.momentum_60d:+.0%}", "Very rapid price appreciation."),
    ]
    pe = (valuation or {}).get("trailingPE")
    pb = (valuation or {}).get("priceToBook")
    if pe is not None or pb is not None:
        flag = (pe is not None and (pe > th.pe or pe < 0)) or (pb is not None and pb > th.pb)
        pe_txt = f"P/E {pe:.1f}" if pe is not None else "P/E n/a"
        pb_txt = f"P/B {pb:.1f}" if pb is not None else "P/B n/a"
        out.append(_row("High valuation", "Trailing P/E and P/B (current snapshot)", pe if pe is not None else pb, flag,
                        f"{pe_txt}, {pb_txt}; thresholds P/E {th.pe:.0f}, P/B {th.pb:.0f}",
                        "Current ratios from Yahoo Finance; historical valuations are not available from this source."))
    else:
        out.append(_row("High valuation", "Trailing P/E and P/B", np.nan, False, "Not available for this series",
                        "Valuation ratios are not published for indices / ETFs on Yahoo Finance."))
    return out


def _row(name, metric, value, flag, detail, explanation):
    return {"Indicator": name, "Metric": metric, "Value": value, "Warning": bool(flag) if value == value else False,
            "Detail": detail, "Explanation": explanation}


def overall_level(rows: list[dict]) -> str:
    n = sum(r["Warning"] for r in rows)
    if n >= 3:
        return "High"
    if n >= 1:
        return "Elevated"
    return "Normal"


def warning_history(prices: pd.Series, volume: pd.Series, th: Thresholds) -> pd.DataFrame:
    """Daily boolean history of the price/volume-based indicators (no valuation history exists)."""
    prices = prices.dropna()
    r = np.log(prices).diff()
    z = abnormal_returns(prices).reindex(prices.index)
    vr = r.rolling(20).std() / r.rolling(120).std()
    v = volume.reindex(prices.index)
    vs = v.rolling(20).mean() / v.rolling(120).mean()
    prem = prices / prices.rolling(200).mean() - 1
    mom = prices / prices.shift(60) - 1
    return pd.DataFrame(
        {
            "Abnormal return": z.abs() > th.abnormal_z,
            "Rising volatility": vr > th.vol_ratio,
            "Volume surge": vs > th.volume_surge,
            "Above 200-day MA": prem > th.ma_premium,
            "60-day momentum": mom > th.momentum_60d,
        }
    )
