"""Shared state, cached computations and UI helpers for the dashboard."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from bubble import data as bd
from bubble.garch import fit_garch
from bubble.gsadf import default_min_window, run_gsadf, simulate_critical_values
from bubble.universe import BY_SYMBOL, UNIVERSE, Security

PROJECT_TITLE = "A Case Study of the 2026 Chinese AI Stock Market Bubble"
DEFAULT_START = date(2024, 1, 1)


# ---- settings --------------------------------------------------------------------

def settings() -> dict:
    ss = st.session_state
    return {
        "start": ss.get("start", DEFAULT_START),
        "end": ss.get("end", date.today()),
        "symbols": ss.get("symbols", [s.symbol for s in UNIVERSE]),
        "custom": ss.get("custom", []),
        "policy": ss.get("policy", "drop"),
        "ann": ss.get("ann", 252),
    }


def sidebar() -> None:
    ss = st.session_state
    ss.setdefault("start", DEFAULT_START)
    ss.setdefault("end", date.today())
    ss.setdefault("symbols", [s.symbol for s in UNIVERSE])
    ss.setdefault("custom", [])
    ss.setdefault("policy", "drop")
    ss.setdefault("ann", 252)
    if pending := ss.pop("pending_symbol", None):
        if pending not in ss["custom"]:
            ss["custom"] = ss["custom"] + [pending]
        if pending not in ss["symbols"]:
            ss["symbols"] = ss["symbols"] + [pending]
    with st.sidebar:
        st.markdown("### Data settings")
        c1, c2 = st.columns(2)
        c1.date_input("Start", key="start", min_value=date(2015, 1, 1), max_value=date.today())
        c2.date_input("End", key="end", min_value=date(2015, 1, 1), max_value=date.today())
        options = [s.symbol for s in UNIVERSE] + ss["custom"]
        ss["symbols"] = [s for s in ss["symbols"] if s in options]
        st.multiselect("Securities", options, key="symbols", format_func=label_for)
        st.radio(
            "Missing values",
            ["drop", "ffill"],
            key="policy",
            format_func=lambda p: "Exclude missing days" if p == "drop" else "Forward-fill (max 5 days)",
            help="Excluding is the default: returns are only computed between two observed prices. "
            "Forward-filling creates artificial zero returns on filled days.",
        )
        st.selectbox("Trading days per year (annualisation)", [252, 242], key="ann",
                     help="252 is the textbook convention; Chinese exchanges trade roughly 242 days a year.")
        if st.button("Refresh data from sources", width="stretch", icon=":material/refresh:"):
            fetch_security.clear()
            valuation.clear()
            st.rerun()
        st.caption("Equities: Yahoo Finance (yfinance). Indices: Sina Finance daily K-line. "
                   "All figures are computed live from these sources.")


def label_for(symbol: str) -> str:
    sec = BY_SYMBOL.get(symbol)
    if sec:
        return f"{sec.name} ({symbol})"
    name = st.session_state.get("custom_names", {}).get(symbol)
    return f"{name} ({symbol})" if name else symbol


def short_label(ds: dict, symbol: str) -> str:
    sd = ds.get(symbol)
    return sd.label if sd else label_for(symbol)


# ---- data ------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_security(symbol: str, name: str, kind: str, proxy: str | None, start: date, end: date) -> bd.SeriesData:
    sec = Security(symbol, name, kind, "", "", proxy)
    return bd.load_security(sec, start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def valuation(symbol: str) -> dict:
    return bd.valuation_snapshot(symbol)


def _security(symbol: str) -> Security:
    return BY_SYMBOL.get(symbol) or bd.custom_security(symbol)


def dataset() -> tuple[dict[str, bd.SeriesData], list[str]]:
    """Load, validate and clean every selected security. Returns (dataset, errors)."""
    cfg = settings()
    if cfg["start"] >= cfg["end"]:
        st.error("Start date must be before end date.")
        st.stop()
    loaded, errors = [], []
    pending = [s for s in cfg["symbols"]]
    if not pending:
        st.info("Select at least one security in the sidebar.")
        st.stop()
    with st.spinner("Fetching historical prices..."):
        for sym in pending:
            sec = _security(sym)
            name = st.session_state.get("custom_names", {}).get(sym, sec.name)
            try:
                loaded.append(fetch_security(sec.symbol, name, sec.kind, sec.proxy_etf, cfg["start"], cfg["end"]))
            except Exception as exc:
                errors.append(f"{label_for(sym)}: {exc}")
    if not loaded:
        st.error("No data could be loaded.\n\n" + "\n\n".join(errors))
        st.stop()
    ds = bd.build_dataset(loaded, cfg["start"], cfg["end"], cfg["policy"])
    return ds, errors


def require_data() -> dict[str, bd.SeriesData]:
    ds, errors = dataset()
    for e in errors:
        st.warning(e, icon=":material/error:")
    return ds


# ---- cached models ---------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=64)
def garch(prices: pd.Series, dist: str, ann: int):
    return fit_garch(prices, dist=dist, periods_per_year=ann)


@st.cache_data(show_spinner=False, max_entries=16)
def critical_values(n: int, lag: int, w0: int, reps: int, seed: int) -> dict:
    return simulate_critical_values(n, lag, w0, reps, seed)


@st.cache_data(show_spinner=False, max_entries=64)
def gsadf(prices: pd.Series, lag: int, reps: int, seed: int, min_duration: int | None):
    n = len(prices.dropna())
    w0 = max(default_min_window(n), lag + 6)
    cv = critical_values(n, lag, w0, reps, seed)
    return run_gsadf(prices, lag=lag, min_window=w0, reps=reps, seed=seed, min_duration=min_duration, cv=cv)


# ---- UI helpers ------------------------------------------------------------------

def actual_badge(text: str = "Actual result: computed from real market data") -> None:
    st.badge(text, icon=":material/verified:", color="green")


def illustrative_badge(text: str = "Illustrative example: simulated data, not market observations") -> None:
    st.badge(text, icon=":material/science:", color="orange")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def table(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 4, height: int | None = None, **kw) -> None:
    cfg = {}
    for c in df.columns:
        if c in pct:
            cfg[c] = st.column_config.NumberColumn(c, format="percent")
        elif pd.api.types.is_float_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(c, format=f"%.{digits}f")
    st.dataframe(df, column_config=cfg, width="stretch", height=height or "auto", **kw)


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    csv = df.to_csv(index=not isinstance(df.index, pd.RangeIndex)).encode()
    st.download_button(label, csv, file_name=filename, mime="text/csv",
                       icon=":material/download:")


def series_picker(ds: dict, label: str, key: str, default: str | None = None) -> str:
    opts = list(ds)
    idx = opts.index(default) if default in opts else 0
    return st.selectbox(label, opts, index=idx, key=key, format_func=lambda s: short_label(ds, s))
