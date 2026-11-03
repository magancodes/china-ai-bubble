from datetime import date

import pandas as pd
import streamlit as st

from bubble import charts
from bubble.periods import Periods, compare_periods, detect_periods, period_metrics

from .common import actual_badge, download_csv, page_header, require_data, series_picker, settings, short_label, table

PCT = {"Cumulative return", "Mean daily log return", "Annualised volatility", "Max drawdown"}


def saved_periods(ds) -> tuple[str, Periods]:
    """Periods last chosen on the Crash Analysis page, or data-derived defaults for the STAR 50."""
    saved = st.session_state.get("crash_saved")
    if saved and saved[0] in ds:
        return saved
    ref = "000688.SS" if "000688.SS" in ds else list(ds)[0]
    p = ds[ref].prices
    return ref, detect_periods(p, default_peak_from(p))


def current_periods(ds, ref: str) -> Periods:
    ss = st.session_state
    if ss.get("crash_mode") == "Manual" and all(k in ss for k in ("m_pre", "m_rally", "m_peak", "m_trough")):
        return Periods(pd.Timestamp(ss["m_pre"]), pd.Timestamp(ss["m_rally"]), pd.Timestamp(ss["m_peak"]),
                       pd.Timestamp(ss["m_trough"]), "Manual")
    p = ds[ref].prices
    peak_from = ss.get("crash_peak_from", default_peak_from(p))
    return detect_periods(p, peak_from, ss.get("crash_lookback", 365))


def default_peak_from(prices: pd.Series) -> date:
    first_2026 = date(2026, 1, 1)
    return first_2026 if prices.index.max() >= pd.Timestamp(first_2026) else prices.index[0].date()


def all_series_metrics(ds, periods: Periods, ann: int) -> pd.DataFrame:
    frames = []
    for s, sd in ds.items():
        m = period_metrics(sd.prices, sd.clean["Volume"], periods, ann)
        m.insert(0, "Series", short_label(ds, s))
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def tests_table(ds, periods: Periods) -> pd.DataFrame:
    spans = periods.as_dict()
    rows = []
    for s, sd in ds.items():
        for a, b in (("Pre-rally", "Rally"), ("Rally", "Correction")):
            if spans.get(a) is None or spans.get(b) is None:
                continue
            t = compare_periods(sd.prices, spans[a], spans[b])
            if not t:
                continue
            rows.append({
                "Series": short_label(ds, s), "Comparison": f"{a} vs {b}",
                "Mean r (first phase)": t["mean_a"], "Mean r (second phase)": t["mean_b"],
                "Welch t": t["welch_t"], "Welch p": t["welch_p"],
                "Variance ratio (2nd / 1st)": t["var_ratio"],
                "Brown-Forsythe F": t["bf_stat"], "Brown-Forsythe p": t["bf_p"],
            })
    return pd.DataFrame(rows)


def render():
    page_header("Crash Analysis", "Pre-rally, rally and correction phases compared with actual historical data.")
    ds = require_data()
    cfg = settings()
    ref_default = "000688.SS" if "000688.SS" in ds else list(ds)[0]

    c1, c2 = st.columns([2, 1])
    with c1:
        ref = series_picker(ds, "Reference series for dating the phases", "crash_ref", ref_default)
    mode = c2.radio("Period definition", ["Data-derived", "Manual"], horizontal=True, key="crash_mode")
    prices = ds[ref].prices

    if mode == "Data-derived":
        c1, c2 = st.columns(2)
        c1.date_input("Search for the peak on or after", key="crash_peak_from", value=default_peak_from(prices),
                      min_value=prices.index[0].date(), max_value=prices.index[-1].date())
        c2.slider("Rally start: look-back window before the peak (calendar days)", 60, 900, 365, 15, key="crash_lookback")
        periods = current_periods(ds, ref)
        st.caption("Peak = highest close of the reference series after the search date. Rally start = lowest close within "
                   "the look-back window before the peak. Correction trough = lowest close after the peak. "
                   "Pre-rally = start of the sample to the rally start.")
    else:
        auto = detect_periods(prices, default_peak_from(prices))
        c = st.columns(4)
        c[0].date_input("Pre-rally start", key="m_pre", value=auto.pre_start.date())
        c[1].date_input("Rally start", key="m_rally", value=auto.rally_start.date())
        c[2].date_input("Peak", key="m_peak", value=auto.peak.date())
        c[3].date_input("Correction end", key="m_trough", value=(auto.trough or prices.index[-1]).date())
        periods = current_periods(ds, ref)
        if not (periods.pre_start < periods.rally_start < periods.peak):
            st.error("Dates must satisfy: pre-rally start < rally start < peak.")
            return

    st.session_state["crash_saved"] = (ref, periods)
    spans = periods.as_dict()
    actual_badge()
    m = st.columns(4)
    m[0].metric("Rally start", str(periods.rally_start.date()), help=f"Close {prices.loc[:periods.rally_start].iloc[-1]:,.2f}")
    m[1].metric("Peak", str(periods.peak.date()), help=f"Close {prices.loc[:periods.peak].iloc[-1]:,.2f}")
    rally_ret = prices.loc[:periods.peak].iloc[-1] / prices.loc[:periods.rally_start].iloc[-1] - 1
    m[2].metric("Rally return", f"{rally_ret:+.1%}")
    if spans["Correction"]:
        corr = prices.loc[:periods.trough].iloc[-1] / prices.loc[:periods.peak].iloc[-1] - 1
        m[3].metric("Correction (peak to trough)", f"{corr:+.1%}", help=f"Trough {periods.trough.date()}")
    else:
        m[3].metric("Correction", "None yet", help="The highest close is the latest observation; no decline after the peak.")
        st.info("The reference series' highest close is its most recent observation, so no correction phase exists in this sample.")

    st.plotly_chart(charts.periods_chart(prices, spans, short_label(ds, ref)), width="stretch")

    st.subheader("Phase statistics for all series (same dates)")
    metrics = all_series_metrics(ds, periods, cfg["ann"])
    metric = st.segmented_control("Compare", ["Cumulative return", "Annualised volatility", "Max drawdown"],
                                  default="Cumulative return", key="crash_metric") or "Cumulative return"
    st.plotly_chart(charts.period_bars(metrics, metric), width="stretch")
    table(metrics, pct=PCT, digits=4, hide_index=True)
    download_csv(metrics, "crash_phase_statistics.csv")

    st.subheader("Are the phases statistically different?")
    st.markdown("Welch's t-test (H₀: equal mean daily log return) and the Brown–Forsythe test "
                "(H₀: equal variance) compare consecutive phases.")
    tests = tests_table(ds, periods)
    if tests.empty:
        st.info("Not enough observations in consecutive phases to run the tests.")
    else:
        table(tests, digits=4, hide_index=True)
        download_csv(tests, "crash_phase_tests.csv")
        rc = tests[tests["Comparison"] == "Rally vs Correction"]
        if len(rc):
            sig_var = int((rc["Brown-Forsythe p"] < 0.05).sum())
            higher = int((rc["Variance ratio (2nd / 1st)"] > 1).sum())
            st.markdown(f"**From the data:** return variance in the correction was higher than in the rally for "
                        f"{higher} of {len(rc)} series, and the difference in variance is significant at 5% for {sig_var}.")
