import pandas as pd
import streamlit as st

from bubble import data as bd

from .common import actual_badge, download_csv, page_header, require_data, series_picker, settings


def _add_custom_ticker():
    st.subheader("Add and validate a ticker")
    st.caption("Use Yahoo Finance symbols, e.g. `688111.SS` (Shanghai), `002415.SZ` (Shenzhen), `0981.HK`, `NVDA`.")
    with st.form("custom_ticker", clear_on_submit=False, border=True):
        c1, c2 = st.columns([3, 1])
        raw = c1.text_input("Ticker symbol", placeholder="e.g. 688111.SS", label_visibility="collapsed")
        submitted = c2.form_submit_button("Validate & add", width="stretch", icon=":material/add:")
    if not submitted:
        return
    sym = bd.normalise_symbol(raw)
    ok, msg = bd.check_symbol_format(sym)
    if not ok:
        st.error(f"`{raw}`: {msg}")
        return
    cfg = settings()
    with st.spinner(f"Fetching {sym}..."):
        try:
            sd = bd.load_security(bd.custom_security(sym), cfg["start"], cfg["end"])
        except Exception as exc:
            st.error(f"`{sym}` could not be validated: {exc}")
            return
    name = bd.lookup_name(sym) or sym
    st.session_state.setdefault("custom_names", {})[sym] = name
    st.session_state["pending_symbol"] = sym
    st.session_state["flash"] = f"{name} (`{sym}`) validated: {len(sd.raw)} daily observations from {sd.source}. Added to the selection."
    st.rerun()


def render():
    page_header("Data Collection", "Sources, validation checks and missing-value handling for every series.")
    if msg := st.session_state.pop("flash", None):
        st.success(msg, icon=":material/check_circle:")
    ds = require_data()
    cfg = settings()

    actual_badge("Live data: fetched from public market data sources")
    rows = []
    for sym, sd in ds.items():
        fails = sum(c["Result"] == "Fail" for c in sd.validation)
        warns = sum(c["Result"] == "Warning" for c in sd.validation)
        rows.append({
            "Symbol": sym,
            "Series": sd.label,
            "Source": sd.source,
            "Source link": sd.source_url,
            "Adjusted for dividends": "Yes" if sd.adjusted else "No",
            "First": sd.clean.index.min().date(),
            "Last": sd.clean.index.max().date(),
            "Raw rows": len(sd.raw),
            "Missing days": sd.missing["missing"],
            "Missing %": sd.missing["missing_pct"] / 100,
            "Filled": sd.missing["filled"],
            "Observations used": sd.missing["used"],
            "Checks": "Pass" if fails + warns == 0 else f"{fails} fail · {warns} warning",
            "Fetched": sd.fetched_at,
        })
    summary = pd.DataFrame(rows)
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Source link": st.column_config.LinkColumn("Source link", display_text="Open"),
            "Missing %": st.column_config.NumberColumn(format="percent"),
        },
    )
    download_csv(summary, "data_sources.csv", "Download source summary")

    proxies = [sd.label for sd in ds.values() if sd.is_proxy]
    if proxies:
        st.warning("Proxy series in use (index history unavailable): " + ", ".join(proxies), icon=":material/swap_horiz:")

    with st.expander("Missing-value methodology"):
        st.markdown(
            f"- The trading calendar is the union of all dates observed across loaded series "
            f"({cfg['start']} to {cfg['end']}).\n"
            "- A day is *missing* for a series if it is on the calendar but the series has no valid close. "
            "Zero-volume rows (trading suspensions) and non-positive prices are also treated as missing.\n"
            f"- Current policy: **{'exclude missing days' if cfg['policy'] == 'drop' else 'forward-fill up to 5 days'}**. "
            "No values are interpolated or invented.\n"
            "- Returns use dividend-adjusted closes for equities (Yahoo Finance) and index levels for indices (Sina Finance)."
        )

    st.subheader("Validation report")
    sym = series_picker(ds, "Series", "data_series")
    sd = ds[sym]
    checks = pd.DataFrame(sd.validation)
    st.dataframe(
        checks.style.map(lambda v: {"Pass": "color:#239b56", "Warning": "color:#d68910", "Fail": "color:#c0392b"}.get(v, ""),
                         subset=["Result"]),
        hide_index=True, width="stretch",
    )
    for note in sd.notes:
        st.caption(f"Note: {note}")

    st.subheader("Data preview")
    c1, c2 = st.columns([1, 1])
    view = c1.radio("Show", ["Cleaned data used in analysis", "Raw data as downloaded"], horizontal=True, key="data_view")
    frame = sd.clean if view.startswith("Cleaned") else (sd.original if sd.original is not None else sd.raw)
    c2.metric("Rows", len(frame))
    st.dataframe(frame.sort_index(ascending=False), width="stretch", height=320)
    download_csv(frame, f"{sym}_{'clean' if view.startswith('Cleaned') else 'raw'}.csv")

    st.divider()
    _add_custom_ticker()
