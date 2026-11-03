import pandas as pd
import streamlit as st

from bubble import charts
from bubble.gsadf import illustrative_regimes, simulate_illustrative_bubble

from .common import (actual_badge, download_csv, gsadf, illustrative_badge, page_header, require_data,
                     short_label, table)


def gsadf_summary(ds, symbols, lag, reps, seed, min_dur, progress=None) -> tuple[pd.DataFrame, dict]:
    rows, results = [], {}
    for i, s in enumerate(symbols):
        if progress:
            progress.progress(i / len(symbols), text=f"Testing {short_label(ds, s)} ({i + 1}/{len(symbols)})...")
        res = gsadf(ds[s].prices, lag, reps, seed, min_dur)
        results[s] = res
        rows.append({
            "Series": short_label(ds, s),
            "Obs.": res.n,
            "SADF": res.sadf,
            "SADF 95% cv": res.sadf_cv[0.95],
            "SADF p-value": res.sadf_pvalue,
            "GSADF": res.gsadf,
            "GSADF 90% cv": res.gsadf_cv[0.90],
            "GSADF 95% cv": res.gsadf_cv[0.95],
            "GSADF 99% cv": res.gsadf_cv[0.99],
            "GSADF p-value": res.gsadf_pvalue,
            "Explosive at 5%?": "Yes" if res.reject_95 else "No",
            "Episodes": len(res.episodes),
        })
    if progress:
        progress.empty()
    return pd.DataFrame(rows), results


def episodes_table(ds, results) -> pd.DataFrame:
    frames = []
    for s, r in results.items():
        if len(r.episodes):
            e = r.episodes.copy()
            e.insert(0, "Series", short_label(ds, s))
            frames.append(e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _illustration(lag, reps, seed):
    with st.expander("Illustrative example: how the test behaves on a simulated bubble", icon=":material/science:"):
        illustrative_badge()
        st.markdown(
            "The series below is **simulated**, not market data. It is a random walk with one explosive regime "
            "$P_t = 1.01P_{t-1}+\\varepsilon_t$ followed by a collapse. The true bubble window is known, so you can "
            "see whether the BSADF sequence crosses its critical value inside it. None of these numbers enter the "
            "case-study results or the exported reports."
        )
        if not st.toggle("Run the simulated example", key="gsadf_illustration"):
            return
        sim = simulate_illustrative_bubble()
        res = gsadf(sim, lag, reps, seed, None)
        reg = illustrative_regimes()
        fig = charts.bsadf_chart(sim, res, "Simulated series (illustrative)")
        fig.add_vrect(x0=reg["bubble_start"], x1=reg["peak"], line=dict(color="#239b56", dash="dash"), row=1, col=1,
                      annotation_text="true explosive regime", annotation_position="top left")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Simulated GSADF = {res.gsadf:.2f} vs 95% critical value {res.gsadf_cv[0.95]:.2f}.")


def render():
    page_header("Bubble Detection: GSADF Test", "Phillips, Shi & Yu (2015) right-tailed unit-root test with date-stamping.")
    ds = require_data()

    with st.expander("Method", icon=":material/function:"):
        st.markdown("For a window of the log price $y_t=\\ln P_t$ starting at fraction $r_1$ and ending at $r_2$ of the sample, estimate")
        st.latex(r"\Delta y_t=a+b\,y_{t-1}+\sum_{i=1}^{k}\phi_i\,\Delta y_{t-i}+e_t,\qquad \text{ADF}_{r_1}^{r_2}=\hat b/\text{se}(\hat b)")
        st.latex(r"\text{BSADF}_{r_2}=\sup_{r_1\in[0,\,r_2-r_0]}\text{ADF}_{r_1}^{r_2},\qquad "
                 r"\text{GSADF}=\sup_{r_2\in[r_0,1]}\text{BSADF}_{r_2}")
        st.markdown(
            "- $H_0$: unit root ($b=0$, prices follow a martingale). $H_1$: mildly explosive root ($b>0$).\n"
            "- Minimum window $r_0 = 0.01 + 1.8/\\sqrt{T}$ (PSY recommendation).\n"
            "- Critical values: Monte Carlo under $y_t = T^{-1} + y_{t-1} + \\varepsilon_t$, $\\varepsilon_t\\sim N(0,1)$, "
            "with the same $T$, $r_0$ and lag $k$ as the data.\n"
            "- Date-stamping: an explosive episode begins when BSADF first exceeds its 95% critical value sequence and "
            "must last at least $\\lceil \\ln T \\rceil$ observations."
        )

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    default = [s for s in ["000688.SS", "000300.SS", "688256.SS"] if s in ds]
    symbols = c1.multiselect("Series", list(ds), default=default or list(ds)[:2], key="gsadf_syms",
                             format_func=lambda s: short_label(ds, s))
    lag = c2.number_input("ADF lag k", 0, 6, 0, help="PSY (2015) use k = 0 in their S&P 500 application.")
    reps = c3.selectbox("Monte Carlo reps", [199, 499, 999, 1999], index=1)
    seed = c4.number_input("Random seed", 0, 10**9, 20260101)
    min_dur_auto = st.toggle("Minimum episode duration = ⌈ln T⌉", value=True)
    min_dur = None if min_dur_auto else st.slider("Minimum episode duration (trading days)", 1, 60, 10)

    if not symbols:
        st.info("Select at least one series.")
        _illustration(lag, reps, seed)
        return

    progress = st.progress(0.0, text="Simulating critical values...")
    summary, results = gsadf_summary(ds, symbols, lag, reps, seed, min_dur, progress)

    actual_badge()
    st.subheader("Test statistics")
    table(summary, digits=3, hide_index=True)
    download_csv(summary, "gsadf_tests.csv")
    for _, r in summary.iterrows():
        verdict = ("rejects the unit-root null in favour of explosive behaviour" if r["Explosive at 5%?"] == "Yes"
                   else "does not reject the unit-root null")
        st.markdown(f"- **{r['Series']}**: GSADF = {r['GSADF']:.3f} vs 95% critical value {r['GSADF 95% cv']:.3f} "
                    f"(Monte Carlo p = {r['GSADF p-value']:.3f}); the test {verdict} at the 5% level.")

    st.subheader("Date-stamped explosive episodes")
    eps = episodes_table(ds, results)
    if eps.empty:
        st.info("No BSADF exceedances of the 95% critical value lasting the minimum duration were found.")
    else:
        table(eps, pct={"Return over episode"}, digits=3, hide_index=True)
        download_csv(eps, "gsadf_episodes.csv")

    focus = st.selectbox("Chart", list(results), format_func=lambda s: short_label(ds, s), key="gsadf_focus")
    st.plotly_chart(charts.bsadf_chart(ds[focus].prices, results[focus], short_label(ds, focus)), width="stretch")
    r = results[focus]
    st.caption(f"T = {r.n}, minimum window = {r.min_window} observations, lag k = {r.lag}, "
               f"{r.reps} Monte Carlo replications (seed {r.seed}), minimum episode length {r.min_duration} days. "
               "Note that BSADF can also exceed its critical value during a sharp collapse, since a crash is also a "
               "departure from martingale behaviour; read episodes together with the price path.")
    bs = pd.DataFrame({"BSADF": r.bsadf, "cv90": r.cv90, "cv95": r.cv95, "cv99": r.cv99})
    download_csv(bs, f"{focus}_bsadf_sequence.csv", "Download BSADF sequence")

    _illustration(lag, reps, seed)
