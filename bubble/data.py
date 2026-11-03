"""Data collection, validation and cleaning.

Sources
-------
* Yahoo Finance (via ``yfinance``) for individual equities. Prices are provided both
  unadjusted (``Close``) and dividend/split adjusted (``AdjClose``).
* Sina Finance daily K-line API for the STAR 50 and CSI 300 indices, because Yahoo
  Finance only serves the latest quote for these index symbols.
* If an index cannot be fetched from Sina, a tracking ETF on Yahoo Finance is used
  and the series is explicitly flagged as a *proxy*.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

from .universe import BY_SYMBOL, Security, daily_price_limit

SYMBOL_PATTERN = re.compile(r"^[0-9A-Z^][0-9A-Z.\-^=]{0,14}$")
A_SHARE_PATTERN = re.compile(r"^\d{6}\.(SS|SZ|BJ)$")
MIN_OBSERVATIONS = 60
COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]

SINA_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var=/CN_MarketDataService.getKLineData"
YAHOO_URL = "https://finance.yahoo.com/quote/{symbol}/history"


class DataError(RuntimeError):
    pass


@dataclass
class SeriesData:
    symbol: str
    name: str
    kind: str
    raw: pd.DataFrame
    source: str
    source_url: str
    adjusted: bool
    is_proxy: bool = False
    proxy_symbol: str | None = None
    fetched_at: str = ""
    notes: list[str] = field(default_factory=list)
    validation: list[dict] = field(default_factory=list)
    clean: pd.DataFrame | None = None
    missing: dict = field(default_factory=dict)
    original: pd.DataFrame | None = None
    repairs: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.name} (proxy: {self.proxy_symbol})" if self.is_proxy else self.name

    @property
    def prices(self) -> pd.Series:
        """Price series used for all return-based calculations."""
        assert self.clean is not None
        return self.clean["AdjClose"].dropna()


def normalise_symbol(text: str) -> str:
    return text.strip().upper().replace(".SH", ".SS")


def check_symbol_format(symbol: str) -> tuple[bool, str]:
    if not symbol:
        return False, "Empty symbol."
    if not SYMBOL_PATTERN.match(symbol):
        return False, "Symbol contains invalid characters."
    if A_SHARE_PATTERN.match(symbol):
        return True, "Valid A-share symbol format (6 digits + .SS/.SZ/.BJ)."
    return True, "Non A-share Yahoo Finance symbol format; will be validated by fetching data."


def _to_sina_symbol(symbol: str) -> str | None:
    code, _, exch = symbol.partition(".")
    prefix = {"SS": "sh", "SZ": "sz", "BJ": "bj"}.get(exch)
    if prefix is None or not code.isdigit() or len(code) != 6:
        return None
    return prefix + code


def fetch_yahoo(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = yf.Ticker(symbol).history(
            start=pd.Timestamp(start), end=pd.Timestamp(end) + pd.Timedelta(days=1), auto_adjust=False, actions=False
        )
    if hist is None or hist.empty:
        return pd.DataFrame(columns=COLUMNS)
    hist = hist.rename(columns={"Adj Close": "AdjClose"})
    if "AdjClose" not in hist:
        hist["AdjClose"] = hist["Close"]
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    hist.index = pd.DatetimeIndex(idx).normalize()
    hist.index.name = "Date"
    return hist[COLUMNS].astype(float)


def fetch_yahoo_splits(symbol: str) -> pd.Series:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splits = yf.Ticker(symbol).splits
    if splits is None or splits.empty:
        return pd.Series(dtype=float)
    idx = splits.index.tz_localize(None) if getattr(splits.index, "tz", None) is not None else splits.index
    return pd.Series(splits.values, index=pd.DatetimeIndex(idx).normalize())


def repair_split_misalignment(df: pd.DataFrame, splits: pd.Series, limit: float, window_days: int = 60) -> tuple[pd.DataFrame, list[dict]]:
    """Repair price discontinuities caused by a vendor applying a recorded split on the wrong date.

    A daily move beyond the exchange price limit is only repaired when the vendor's own split
    record contains a ratio k within `window_days` such that the remaining move (1 + r) * k - 1
    lies inside the limit. Prices before the discontinuity are then divided by k.
    """
    df = df.copy()
    repairs: list[dict] = []
    if splits.empty:
        return df, repairs
    for _ in range(5):
        r = df["Close"].pct_change()
        breaches = r[r.abs() > limit + 0.005]
        if breaches.empty:
            break
        day, move = breaches.index[0], float(breaches.iloc[0])
        near = splits[(splits.index >= day - pd.Timedelta(days=window_days)) & (splits.index <= day + pd.Timedelta(days=window_days))]
        near = near[near > 0]
        if near.empty:
            break
        resid = ((1 + move) * near - 1).abs()
        k = float(near.loc[resid.idxmin()])
        if resid.min() > limit:
            break
        before = df.index < day
        for c in ("Open", "High", "Low", "Close", "AdjClose"):
            df.loc[before, c] = df.loc[before, c] / k
        df.loc[before, "Volume"] = df.loc[before, "Volume"] * k
        repairs.append({"date": day.date(), "move": move, "ratio": k, "split_record": resid.idxmin().date(),
                        "residual": float((1 + move) * k - 1)})
    return df, repairs


def fetch_sina(symbol: str, start: date, end: date) -> pd.DataFrame:
    sina = _to_sina_symbol(symbol)
    if sina is None:
        return pd.DataFrame(columns=COLUMNS)
    days_needed = (date.today() - start).days
    datalen = int(min(max(days_needed * 0.75 + 30, 100), 6000))
    resp = requests.get(
        SINA_URL,
        params={"symbol": sina, "scale": 240, "ma": "no", "datalen": datalen},
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
        timeout=20,
    )
    resp.raise_for_status()
    match = re.search(r"\((.*)\)", resp.text, re.S)
    if not match or match.group(1).strip() in {"", "null"}:
        return pd.DataFrame(columns=COLUMNS)
    rows = json.loads(match.group(1))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["day"])
    df = df.set_index("Date").rename(columns=str.capitalize)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["AdjClose"] = df["Close"]
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df[COLUMNS]


def load_security(sec: Security, start: date, end: date) -> SeriesData:
    """Fetch one security from the most appropriate real data source."""
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    errors: list[str] = []

    def attempt(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # network / parsing errors are reported, never hidden
            errors.append(f"{fn.__name__}: {exc}")
            return pd.DataFrame(columns=COLUMNS)

    if sec.kind == "index":
        df = attempt(fetch_sina, sec.symbol, start, end)
        if len(df) >= MIN_OBSERVATIONS:
            return SeriesData(
                sec.symbol, sec.name, sec.kind, df, "Sina Finance (daily index K-line)",
                f"https://finance.sina.com.cn/realstock/company/{_to_sina_symbol(sec.symbol)}/nc.shtml",
                adjusted=False, fetched_at=fetched_at,
                notes=["Index levels are price-return levels; no dividend adjustment applies."] + errors,
            )
        df = attempt(fetch_yahoo, sec.symbol, start, end)
        if len(df) >= MIN_OBSERVATIONS:
            return SeriesData(
                sec.symbol, sec.name, sec.kind, df, "Yahoo Finance", YAHOO_URL.format(symbol=sec.symbol),
                adjusted=False, fetched_at=fetched_at, notes=errors,
            )
        if sec.proxy_etf:
            df = attempt(fetch_yahoo, sec.proxy_etf, start, end)
            if len(df) >= MIN_OBSERVATIONS:
                return SeriesData(
                    sec.symbol, sec.name, sec.kind, df, "Yahoo Finance (tracking ETF proxy)",
                    YAHOO_URL.format(symbol=sec.proxy_etf), adjusted=True, is_proxy=True,
                    proxy_symbol=sec.proxy_etf, fetched_at=fetched_at,
                    notes=[f"Index history unavailable; {sec.proxy_etf} ETF used as a proxy."] + errors,
                )
    else:
        df = attempt(fetch_yahoo, sec.symbol, start, end)
        if len(df) >= MIN_OBSERVATIONS:
            original = df
            limit = daily_price_limit(sec.symbol)
            repairs: list[dict] = []
            if limit:
                splits = attempt(fetch_yahoo_splits, sec.symbol)
                if isinstance(splits, pd.Series):
                    df, repairs = repair_split_misalignment(df, splits, limit)
            notes = [
                f"Split repair: {r['move']:+.1%} jump on {r['date']} matches Yahoo's recorded {r['ratio']:g}-for-1 split "
                f"({r['split_record']}); earlier prices divided by {r['ratio']:g}. Residual move {r['residual']:+.1%}."
                for r in repairs
            ]
            sd = SeriesData(
                sec.symbol, sec.name, sec.kind, df, "Yahoo Finance", YAHOO_URL.format(symbol=sec.symbol),
                adjusted=True, fetched_at=fetched_at, notes=notes + errors,
            )
            sd.original = original
            sd.repairs = repairs
            return sd
        df = attempt(fetch_sina, sec.symbol, start, end)
        if len(df) >= MIN_OBSERVATIONS:
            return SeriesData(
                sec.symbol, sec.name, sec.kind, df, "Sina Finance (unadjusted)",
                f"https://finance.sina.com.cn/realstock/company/{_to_sina_symbol(sec.symbol)}/nc.shtml",
                adjusted=False, fetched_at=fetched_at,
                notes=["Yahoo Finance unavailable; unadjusted Sina prices used."] + errors,
            )
    detail = "; ".join(errors) if errors else "no source returned enough observations"
    raise DataError(f"{sec.symbol}: fewer than {MIN_OBSERVATIONS} observations available ({detail}).")


def custom_security(symbol: str) -> Security:
    if symbol in BY_SYMBOL:
        return BY_SYMBOL[symbol]
    kind = "index" if symbol.startswith("^") else "equity"
    return Security(symbol, symbol, kind, "User-added", "User-supplied ticker validated against live data.")


def lookup_name(symbol: str) -> str | None:
    import yfinance as yf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(symbol).info
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def validate(sd: SeriesData, start: date, end: date) -> list[dict]:
    """Run data-quality checks. Each check is reported with its outcome; none modify data."""
    df = sd.raw
    checks: list[dict] = []

    def add(name, passed, detail, severity="error"):
        checks.append({"Check": name, "Result": "Pass" if passed else ("Warning" if severity == "warn" else "Fail"), "Detail": detail})

    ok, msg = check_symbol_format(sd.symbol)
    add("Symbol format", ok, msg)
    add("Observations", len(df) >= MIN_OBSERVATIONS, f"{len(df)} daily rows returned (minimum {MIN_OBSERVATIONS}).")
    if df.empty:
        return checks
    first, last = df.index.min().date(), df.index.max().date()
    gap_start = (first - start).days
    add("Start coverage", gap_start <= 10, f"First observation {first} (requested {start}).", "warn")
    gap_end = (min(end, date.today()) - last).days
    add("End coverage", gap_end <= 7, f"Last observation {last} (requested {end}).", "warn")
    dup = int(df.index.duplicated().sum())
    add("Duplicate dates", dup == 0, f"{dup} duplicated dates (later rows kept).", "warn")
    nonpos = int((df["Close"] <= 0).sum())
    add("Positive prices", nonpos == 0, f"{nonpos} non-positive closing prices (set to missing).", "warn")
    bad_hl = int(((df["High"] < df["Low"]) | (df["Close"] > df["High"] * 1.001) | (df["Close"] < df["Low"] * 0.999)).sum())
    add("OHLC consistency", bad_hl == 0, f"{bad_hl} rows where Close lies outside [Low, High].", "warn")
    nan_close = int(df["Close"].isna().sum())
    add("Missing closes", nan_close == 0, f"{nan_close} rows with missing close.", "warn")
    zero_vol = int((df["Volume"] == 0).sum())
    add("Zero-volume days", zero_vol == 0, f"{zero_vol} zero-volume rows treated as trading suspensions (set to missing).", "warn")
    limit = daily_price_limit(sd.symbol) if sd.kind == "equity" and not sd.is_proxy else None
    if limit:
        r = df["Close"].pct_change().dropna()
        breaches = r[r.abs() > limit + 0.005]
        detail = f"{len(breaches)} daily moves beyond the ±{limit:.0%} exchange limit"
        if len(breaches):
            detail += " (" + ", ".join(f"{d.date()}: {v:+.1%}" for d, v in breaches.head(5).items()) + ")"
        add("Exchange price-limit consistency", len(breaches) == 0, detail + " (after split repairs).", "warn")
    if sd.repairs:
        detail = "; ".join(f"{r['date']}: {r['move']:+.1%} jump realigned with recorded {r['ratio']:g}-for-1 split "
                           f"({r['split_record']})" for r in sd.repairs)
        add("Split-adjustment repair", False, detail + ".", "warn")
    return checks


def clean_series(sd: SeriesData, start: date, end: date) -> pd.DataFrame:
    df = sd.raw.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    invalid = (df["Close"] <= 0) | df["Close"].isna() | (df["Volume"] == 0)
    df.loc[invalid, ["Open", "High", "Low", "Close", "AdjClose"]] = np.nan
    df.loc[invalid, "Volume"] = np.nan
    return df


def build_dataset(series: list[SeriesData], start: date, end: date, policy: str = "drop") -> dict[str, SeriesData]:
    """Validate, clean and align series on the union trading calendar.

    policy="drop": missing observations are excluded (returns are only computed between
    consecutive valid prices). policy="ffill": prices are carried forward for up to five
    trading days, which creates zero returns on filled days.
    """
    for sd in series:
        sd.validation = validate(sd, start, end)
        sd.clean = clean_series(sd, start, end)
    calendar = pd.DatetimeIndex(sorted(set().union(*[sd.clean.index for sd in series]))) if series else pd.DatetimeIndex([])
    out = {}
    for sd in series:
        aligned = sd.clean.reindex(calendar)
        first_valid = aligned["Close"].first_valid_index()
        window = aligned.loc[first_valid:] if first_valid is not None else aligned.iloc[0:0]
        missing_before = int(window["Close"].isna().sum())
        if policy == "ffill":
            price_cols = ["Open", "High", "Low", "Close", "AdjClose"]
            window = window.copy()
            window[price_cols] = window[price_cols].ffill(limit=5)
        filled = missing_before - int(window["Close"].isna().sum())
        sd.clean = window.dropna(subset=["Close"])
        sd.missing = {
            "calendar_days": len(window),
            "missing": missing_before,
            "missing_pct": 100 * missing_before / max(len(window), 1),
            "filled": filled,
            "used": len(sd.clean),
        }
        out[sd.symbol] = sd
    return out


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def price_panel(ds: dict[str, SeriesData], column: str = "AdjClose") -> pd.DataFrame:
    return pd.DataFrame({sym: sd.clean[column] for sym, sd in ds.items()})


def valuation_snapshot(symbol: str) -> dict:
    """Current valuation ratios from Yahoo Finance. These are a present-day snapshot, not history."""
    import yfinance as yf

    keys = ["trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months", "marketCap", "currency"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(symbol).info
    except Exception as exc:
        return {"error": str(exc)}
    out = {k: info.get(k) for k in keys}
    out["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out
