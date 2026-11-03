import pandas as pd
import streamlit as st

from bubble import charts
from bubble.garch import persistence_label

from .common import actual_badge, download_csv, garch, page_header, require_data, settings, short_label, table


def garch_summary(ds, symbols, dist, ann) -> tuple[pd.DataFrame, dict]:
    rows, results = [], {}
    for s in symbols:
        try:
            g = garch(ds[s].prices, dist, ann)
        except Exception as exc:
            st.warning(f"GARCH failed for {short_label(ds, s)}: {exc}")
            continue
        results[s] = g
        p = g.params["p-value"]
        rows.append({
            "Series": short_label(ds, s),
            "omega": g.omega,
            "alpha": g.alpha,
            "beta": g.beta,
            "alpha p-value": float(p["alpha[1]"]),
            "beta p-value": float(p["beta[1]"]),
            "Persistence (alpha+beta)": g.persistence,
            "Half-life (days)": g.half_life,
            "Long-run ann. vol": g.unconditional_vol_ann,
            "Latest cond. vol (ann.)": float(g.cond_vol_ann.iloc[-1]),
            "ARCH-LM p (raw returns)": g.arch_lm_pvalue,
            "Ljung-Box p (std. resid²)": g.ljung_box_sq_pvalue,
            "Log-likelihood": g.loglik,
            "Converged": g.converged,
        })
    return pd.DataFrame(rows), results


def render():
    page_header("GARCH(1,1) Volatility Model", "Conditional volatility and the persistence of volatility shocks.")
    ds = require_data()
    cfg = settings()
    ann = cfg["ann"]

    with st.expander("Model specification", icon=":material/function:", expanded=False):
        st.latex(r"r_t = \mu + \varepsilon_t,\qquad \varepsilon_t=\sigma_t z_t,\qquad z_t \overset{iid}{\sim} D(0,1)")
        st.latex(r"\sigma_t^2=\omega+\alpha\,\varepsilon_{t-1}^2+\beta\,\sigma_{t-1}^2,\qquad \omega>0,\ \alpha,\beta\ge0")
        st.latex(r"\text{persistence}=\alpha+\beta,\qquad \text{half-life}=\frac{\ln 0.5}{\ln(\alpha+\beta)},\qquad "
                 r"\bar\sigma^2=\frac{\omega}{1-\alpha-\beta}")
        st.markdown("Parameters are estimated by maximum likelihood (`arch` package) on daily log returns in percent. "
                    "The ARCH-LM test (Engle, 10 lags) checks for volatility clustering before fitting; the Ljung–Box test "
                    "on squared standardised residuals checks whether the model has captured it.")

    c1, c2 = st.columns([3, 1])
    default = [s for s in ["000688.SS", "000300.SS", "688256.SS"] if s in ds]
    symbols = c1.multiselect("Series", list(ds), default=default or list(ds)[:2], key="garch_syms",
                             format_func=lambda s: short_label(ds, s))
    dist = c2.selectbox("Innovation distribution", ["t", "normal"], format_func=lambda d: "Student's t" if d == "t" else "Normal")
    if not symbols:
        st.info("Select at least one series.")
        return

    with st.spinner("Estimating GARCH(1,1) by maximum likelihood..."):
        summary, results = garch_summary(ds, symbols, dist, ann)
    if summary.empty:
        return

    actual_badge()
    st.subheader("Parameter estimates")
    table(summary, pct={"Long-run ann. vol", "Latest cond. vol (ann.)"}, digits=4, hide_index=True)
    download_csv(summary, "garch_summary.csv")

    for _, r in summary.iterrows():
        hl = f"{r['Half-life (days)']:.1f} trading days" if pd.notna(r["Half-life (days)"]) else "undefined (α+β ≥ 1)"
        st.markdown(f"- **{r['Series']}**: α+β = {r['Persistence (alpha+beta)']:.4f} → "
                    f"{persistence_label(r['Persistence (alpha+beta)']).lower()}; half-life of a volatility shock {hl}.")

    st.subheader("Conditional volatility")
    focus = st.selectbox("Series", list(results), format_func=lambda s: short_label(ds, s), key="garch_focus")
    g = results[focus]
    st.plotly_chart(charts.garch_vol(ds[focus].prices, g.cond_vol_ann, g.forecast_vol_ann, ann), width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persistence α+β", f"{g.persistence:.4f}")
    c2.metric("Half-life", f"{g.half_life:.1f} days" if g.half_life else "∞")
    c3.metric("Peak conditional vol", f"{g.cond_vol_ann.max():.1%}", help=str(g.cond_vol_ann.idxmax().date()))
    c4.metric("20-day-ahead forecast", f"{g.forecast_vol_ann.iloc[-1]:.1%}")
    with st.expander("Full estimation output"):
        table(g.params, digits=5)
        st.caption(f"{g.nobs} observations · distribution: {g.dist} · AIC {g.aic:.1f} · BIC {g.bic:.1f} · "
                   f"optimiser converged: {g.converged}")
    download_csv(pd.DataFrame({"conditional_vol_annualised": g.cond_vol_ann}), f"{focus}_garch_conditional_vol.csv",
                 "Download conditional volatility series")
