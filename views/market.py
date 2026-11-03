import numpy as np
import pandas as pd
import streamlit as st

from bubble import charts
from bubble.data import price_panel

from .common import actual_badge, page_header, require_data, series_picker, settings, short_label


def render():
    page_header("Market Visualisation", "Prices, daily returns, trading volume and cross-asset comparisons.")
    ds = require_data()
    cfg = settings()
    labels = {s: short_label(ds, s) for s in ds}
    actual_badge()

    tab_single, tab_compare, tab_returns, tab_volume, tab_corr = st.tabs(
        ["Price & volume", "Index vs company comparison", "Daily returns", "Trading volume", "Correlation"]
    )

    with tab_single:
        c1, c2 = st.columns([3, 1])
        with c1:
            sym = series_picker(ds, "Series", "mkt_series")
        candles = c2.toggle("Candlesticks", value=True)
        df = ds[sym].clean
        st.plotly_chart(charts.price_volume(df, labels[sym], candles), width="stretch")
        st.caption("Red = up day, green = down day (Chinese market convention). Moving averages use closing prices. "
                   "Candles show unadjusted prices as traded.")

    with tab_compare:
        default = [s for s in ["000688.SS", "000300.SS", "688256.SS", "688981.SS"] if s in ds]
        chosen = st.multiselect("Series to compare", list(ds), default=default or list(ds)[:3],
                                format_func=lambda s: labels[s], key="mkt_compare")
        log_scale = st.toggle("Log scale", value=True, key="mkt_log")
        if chosen:
            panel = price_panel({s: ds[s] for s in chosen})
            st.plotly_chart(charts.normalised(panel, labels, log_scale), width="stretch")
            last = panel.ffill().iloc[-1] / panel.bfill().iloc[0] - 1
            perf = pd.DataFrame({"Series": [labels[s] for s in chosen], "Total return": last[chosen].values})
            perf["Relative to CSI 300"] = (
                (1 + perf["Total return"]) / (1 + last["000300.SS"]) - 1 if "000300.SS" in last else np.nan
            )
            st.dataframe(perf.sort_values("Total return", ascending=False), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent")
                                        for c in ["Total return", "Relative to CSI 300"]})
            st.caption("Each series is rebased to 100 on its first valid observation in the selected window.")

    with tab_returns:
        sym_r = series_picker(ds, "Series", "mkt_ret_series")
        r = np.log(ds[sym_r].prices).diff().dropna()
        st.plotly_chart(charts.returns_series(r, labels[sym_r]), width="stretch")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(charts.returns_hist(r, labels[sym_r]), width="stretch")
        with c2:
            st.markdown("**Largest moves**")
            big = r.reindex(r.abs().sort_values(ascending=False).index).head(10)
            st.dataframe(pd.DataFrame({"Date": big.index.date, "Log return": big.values}), hide_index=True,
                         column_config={"Log return": st.column_config.NumberColumn(format="percent")}, width="stretch")
        st.caption("Fat tails relative to the fitted normal curve motivate volatility models such as GARCH.")

    with tab_volume:
        vpanel = pd.DataFrame({s: ds[s].clean["Volume"] for s in ds})
        st.plotly_chart(charts.volume_compare(vpanel, labels), width="stretch")
        st.caption("Values above 1 mean recent trading activity exceeds its medium-term norm; "
                   "volume surges often accompany speculative phases. Volumes are in shares (indices: index-constituent shares).")

    with tab_corr:
        rets = np.log(price_panel(ds)).diff()
        st.plotly_chart(charts.correlation(rets.dropna(how="all"), labels), width="stretch")
        st.caption(f"Pearson correlation of daily log returns over {cfg['start']} to {cfg['end']} (pairwise complete observations).")
