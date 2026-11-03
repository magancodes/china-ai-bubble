import pandas as pd
import streamlit as st

from bubble.stats import describe_returns
from bubble.universe import UNIVERSE

from .common import PROJECT_TITLE, actual_badge, require_data, settings


def render():
    st.title(PROJECT_TITLE)
    st.caption("B.Sc. Mathematics project · Interactive research dashboard")
    st.markdown(
        "This dashboard studies the speculative rally and subsequent correction in Chinese AI and semiconductor "
        "equities. It applies descriptive statistics, **GARCH(1,1)** volatility modelling and the "
        "**Phillips–Shi–Yu (2015) GSADF** explosive-root test to daily prices of the **STAR 50** and **CSI 300** "
        "indices and eight AI/semiconductor companies. Every number on the analytical pages is computed live from "
        "fetched market data; simulated examples are confined to clearly labelled boxes."
    )

    ds = require_data()
    cfg = settings()

    st.subheader("Headline figures")
    actual_badge()
    cols = st.columns(4)
    for col, sym in zip(cols, ["000688.SS", "000300.SS", "688256.SS", "688981.SS"]):
        if sym not in ds:
            continue
        d = describe_returns(ds[sym].prices, cfg["ann"])
        col.metric(ds[sym].label, f"{d['Total return']:+.1%}", help=f"Total price change {d['Start']} to {d['End']}")
        col.caption(f"Max drawdown {d['Max drawdown']:.1%} · Ann. vol {d['Annualised volatility']:.1%}")

    st.subheader("Securities studied")
    rows = []
    for s in UNIVERSE:
        sd = ds.get(s.symbol)
        rows.append({
            "Symbol": s.symbol, "Name": s.name, "Type": s.kind.title(), "Listing": s.segment, "Profile": s.description,
            "Loaded": "Yes" if sd else "Not selected / unavailable",
            "Observations": sd.missing["used"] if sd else None,
            "Source": sd.source if sd else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Mathematical toolkit")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Log returns and volatility**")
        st.latex(r"r_t=\ln\frac{P_t}{P_{t-1}},\qquad \hat\sigma_{\text{ann}}=\sqrt{252}\,\hat\sigma_{\text{daily}}")
        st.markdown("**Maximum drawdown**")
        st.latex(r"\text{MDD}=\min_t\left(\frac{P_t}{\max_{s\le t}P_s}-1\right)")
    with c2:
        st.markdown("**GARCH(1,1)**")
        st.latex(r"\sigma_t^2=\omega+\alpha\,\varepsilon_{t-1}^2+\beta\,\sigma_{t-1}^2,\qquad \text{persistence}=\alpha+\beta")
        st.markdown("**GSADF bubble test**")
        st.latex(r"\text{GSADF}(r_0)=\sup_{r_2\in[r_0,1]}\ \sup_{r_1\in[0,r_2-r_0]}\ \text{ADF}_{r_1}^{r_2}")

    st.subheader("How to use this dashboard")
    st.markdown(
        "1. **Data Collection** – check sources, validation results and missing values; add your own tickers.\n"
        "2. **Market Visualisation** – prices, returns, volume and cross-asset comparisons.\n"
        "3. **Statistical Analysis** – moments of returns, volatility and drawdowns.\n"
        "4. **GARCH(1,1)** – conditional volatility and persistence of shocks.\n"
        "5. **Bubble Detection** – GSADF test with Monte Carlo critical values and date-stamped explosive periods.\n"
        "6. **Crash Analysis** – pre-rally vs rally vs correction, with formal tests of differences.\n"
        "7. **Risk Monitoring** – rule-based warning indicators.\n"
        "8. **Reports** – export everything to CSV and PDF."
    )
    st.info(
        "Interpretation note: a statistically significant GSADF result indicates *explosive price dynamics* "
        "(a mathematical property consistent with a rational bubble). It does not by itself prove irrational "
        "speculation, and it does not predict future prices.",
        icon=":material/info:",
    )
