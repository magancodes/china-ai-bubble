import pandas as pd
import streamlit as st

from bubble import charts
from bubble.data import price_panel
from bubble.stats import PERCENT_FIELDS, describe_returns

from .common import actual_badge, download_csv, page_header, require_data, settings, short_label, table


def stats_table(ds, ann) -> pd.DataFrame:
    rows = {short_label(ds, s): describe_returns(sd.prices, ann) for s, sd in ds.items()}
    return pd.DataFrame(rows).T


def render():
    page_header("Statistical Analysis", "Moments of daily log returns, annualised volatility and maximum drawdown.")
    ds = require_data()
    cfg = settings()
    ann = cfg["ann"]
    labels = {s: short_label(ds, s) for s in ds}

    with st.expander("Formulas used", icon=":material/function:"):
        st.latex(r"r_t=\ln(P_t/P_{t-1}),\quad \bar r=\frac1n\sum_{t=1}^n r_t,\quad s^2=\frac{1}{n-1}\sum_{t=1}^n (r_t-\bar r)^2")
        st.latex(r"\mu_{\text{ann}}=%d\,\bar r,\qquad \sigma_{\text{ann}}=\sqrt{%d}\,s" % (ann, ann))
        st.latex(r"\text{MDD}=\min_t \left(\frac{P_t}{\max_{s\le t} P_s}-1\right)")
        st.markdown("Skewness and excess kurtosis are the bias-corrected sample estimators; the Jarque–Bera test "
                    "has H₀: returns are normally distributed.")

    actual_badge()
    st.subheader("Descriptive statistics of daily log returns")
    df = stats_table(ds, ann)
    numeric = df.copy()
    for c in numeric.columns:
        if c not in {"Start", "End", "MDD peak", "MDD trough", "MDD recovered"}:
            numeric[c] = pd.to_numeric(numeric[c])
    table(numeric, pct=PERCENT_FIELDS, digits=6)
    download_csv(numeric, "descriptive_statistics.csv")

    hi_vol = numeric["Annualised volatility"].idxmax()
    lo_vol = numeric["Annualised volatility"].idxmin()
    worst = numeric["Max drawdown"].idxmin()
    non_normal = int((numeric["Jarque-Bera p-value"] < 0.05).sum())
    st.markdown(
        f"**Findings from the loaded data.** The highest annualised volatility is **{hi_vol}** "
        f"({numeric.loc[hi_vol, 'Annualised volatility']:.1%}); the lowest is **{lo_vol}** "
        f"({numeric.loc[lo_vol, 'Annualised volatility']:.1%}). The deepest peak-to-trough fall is **{worst}** "
        f"({numeric.loc[worst, 'Max drawdown']:.1%}, peak {numeric.loc[worst, 'MDD peak']}, trough "
        f"{numeric.loc[worst, 'MDD trough']}). Normality is rejected at the 5% level for {non_normal} of "
        f"{len(numeric)} series (Jarque–Bera)."
    )

    st.subheader("Drawdowns")
    st.plotly_chart(charts.drawdowns(price_panel(ds), labels), width="stretch")

    st.subheader("Rolling volatility")
    window = st.select_slider("Rolling window (trading days)", [10, 20, 60, 120], value=20)
    st.plotly_chart(charts.rolling_vol(price_panel(ds), labels, window, ann), width="stretch")

    st.subheader("Risk–return comparison")
    import plotly.express as px

    rr = numeric.reset_index(names="Series")
    fig = px.scatter(rr, x="Annualised volatility", y="Annualised mean return", text="Series",
                     size=rr["Max drawdown"].abs(), size_max=28, template="plotly_white")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=460, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bubble size is proportional to the magnitude of the maximum drawdown.")
