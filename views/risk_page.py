import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bubble import charts
from bubble.risk import Thresholds, abnormal_returns, indicators, overall_level, warning_history

from .common import actual_badge, download_csv, page_header, require_data, series_picker, short_label, valuation

LEVEL_COLOUR = {"Normal": "green", "Elevated": "orange", "High": "red"}


def thresholds_from_state() -> Thresholds:
    saved = st.session_state.get("risk_thresholds")
    return saved if saved else Thresholds()


def risk_overview(ds, th: Thresholds) -> tuple[pd.DataFrame, dict]:
    rows, details = [], {}
    for s, sd in ds.items():
        val = valuation(s) if sd.kind == "equity" and not sd.is_proxy else None
        ind = indicators(sd.prices, sd.clean["Volume"], val, th)
        details[s] = (ind, val)
        row = {"Series": short_label(ds, s), "As of": sd.prices.index[-1].date(), "Risk level": overall_level(ind),
               "Warnings": sum(i["Warning"] for i in ind)}
        for i in ind:
            row[i["Indicator"]] = "⚠" if i["Warning"] else "·"
        rows.append(row)
    return pd.DataFrame(rows), details


def render():
    page_header("Risk Monitoring", "Rule-based warning indicators for abnormal returns, valuation, volatility and speculation.")
    ds = require_data()

    with st.expander("Warning thresholds (heuristics, adjustable)", icon=":material/tune:"):
        st.caption("These thresholds are rules of thumb chosen by the analyst, not statistically estimated quantities.")
        d = thresholds_from_state()
        c = st.columns(4)
        th = Thresholds(
            abnormal_z=c[0].number_input("Abnormal return |z| >", 1.5, 6.0, d.abnormal_z, 0.5),
            vol_ratio=c[1].number_input("20d/120d volatility >", 1.0, 3.0, d.vol_ratio, 0.1),
            volume_surge=c[2].number_input("20d/120d volume >", 1.0, 4.0, d.volume_surge, 0.1),
            momentum_60d=c[3].number_input("60-day return >", 0.1, 3.0, d.momentum_60d, 0.05),
        )
        c = st.columns(4)
        th.ma_premium = c[0].number_input("Premium over 200d MA >", 0.05, 2.0, d.ma_premium, 0.05)
        th.pe = c[1].number_input("Trailing P/E >", 10.0, 500.0, d.pe, 10.0)
        th.pb = c[2].number_input("Price/Book >", 1.0, 50.0, d.pb, 1.0)
        st.session_state["risk_thresholds"] = th

    with st.spinner("Computing indicators and fetching valuation snapshots..."):
        overview, details = risk_overview(ds, th)

    actual_badge("Indicators computed from real price and volume data; valuations are a current Yahoo Finance snapshot")
    st.subheader("Current status")
    st.dataframe(
        overview.style.map(lambda v: {"High": "background-color:#fadbd8", "Elevated": "background-color:#fdebd0",
                                      "Normal": "background-color:#d5f5e3"}.get(v, ""), subset=["Risk level"]),
        hide_index=True, width="stretch",
    )
    download_csv(overview, "risk_overview.csv")
    st.caption("Risk level: Normal = no warnings, Elevated = 1–2 warnings, High = 3 or more.")

    st.subheader("Indicator detail")
    sym = series_picker(ds, "Series", "risk_series")
    ind, val = details[sym]
    level = overall_level(ind)
    st.badge(f"{level} risk · {sum(i['Warning'] for i in ind)} active warnings", color=LEVEL_COLOUR[level])
    cols = st.columns(3)
    for k, i in enumerate(ind):
        with cols[k % 3].container(border=True):
            v = i["Value"]
            if i["Indicator"] in {"Price vs. 200-day MA", "60-day momentum"}:
                shown = f"{v:+.1%}" if v == v else "n/a"
            elif i["Indicator"] == "Abnormal returns":
                shown = f"{int(v)} day(s)"
            else:
                shown = f"{v:.2f}" if v == v else "n/a"
            st.markdown(f"{'🔴' if i['Warning'] else '🟢'} **{i['Indicator']}**")
            st.markdown(f"### {shown}")
            st.caption(f"{i['Metric']} · {i['Detail']}")
            st.caption(i["Explanation"])
    if val and "as_of" in val:
        st.caption(f"Valuation snapshot fetched {val['as_of']} from Yahoo Finance; it reflects current ratios, "
                   "not the ratios at the time of the peak.")

    st.subheader("Warning history")
    sd = ds[sym]
    hist = warning_history(sd.prices, sd.clean["Volume"], th)
    st.plotly_chart(charts.warning_timeline(hist), width="stretch")
    st.caption("Days on which each price/volume rule was triggered (red). Valuation history is not available from the source.")

    z = abnormal_returns(sd.prices)
    fig = go.Figure(go.Scatter(x=z.index, y=z, mode="lines", name="z-score", line=dict(color="#1f4e79", width=1)))
    out = z[z.abs() > th.abnormal_z]
    fig.add_trace(go.Scatter(x=out.index, y=out, mode="markers", name=f"|z| > {th.abnormal_z}", marker=dict(color="#c0392b", size=7)))
    for y in (th.abnormal_z, -th.abnormal_z):
        fig.add_hline(y=y, line_dash="dash", line_color="#c0392b")
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Abnormal return z-score")
    st.plotly_chart(fig, width="stretch")
    counts = hist.sum().rename("Days triggered").to_frame()
    counts["Share of sample"] = counts["Days triggered"] / len(hist)
    st.dataframe(counts, width="stretch", column_config={"Share of sample": st.column_config.NumberColumn(format="percent")})
    download_csv(hist.astype(int), f"{sym}_warning_history.csv", "Download warning history")
