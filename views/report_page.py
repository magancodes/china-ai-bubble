from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from bubble import report as rp
from bubble.data import price_panel
from bubble.stats import PERCENT_FIELDS

from .common import PROJECT_TITLE, page_header, require_data, settings, short_label
from .crash import PCT as PERIOD_PCT
from .crash import all_series_metrics, saved_periods, tests_table
from .garch_page import garch_summary
from .gsadf_page import episodes_table, gsadf_summary
from .risk_page import risk_overview, thresholds_from_state
from .statistics import stats_table

LIMITATIONS = [
    "Data are taken from public sources (Yahoo Finance for equities, Sina Finance for indices) and may contain "
    "vendor errors; the validation checks reported in Section 1 are designed to surface them, not to correct them.",
    "Returns are daily log returns of dividend-adjusted closing prices for equities and price-index levels for indices.",
    "GARCH(1,1) assumes a constant unconditional variance; structural breaks during a bubble can inflate the "
    "estimated persistence.",
    "A significant GSADF statistic indicates mildly explosive price dynamics. It is consistent with, but does not "
    "prove, a speculative bubble, and BSADF can also exceed its critical value during rapid collapses.",
    "Risk-monitoring thresholds are heuristics chosen by the analyst. Valuation ratios are a present-day snapshot.",
    "Nothing in this report is investment advice or a forecast of future prices.",
]


def _sources_table(ds) -> pd.DataFrame:
    return pd.DataFrame([{
        "Series": sd.label, "Symbol": sym, "Source": sd.source,
        "First": sd.clean.index.min().date(), "Last": sd.clean.index.max().date(),
        "Obs. used": sd.missing["used"], "Missing days": sd.missing["missing"],
        "Failed checks": sum(c["Result"] == "Fail" for c in sd.validation),
        "Warnings": sum(c["Result"] == "Warning" for c in sd.validation),
        "Fetched": sd.fetched_at,
    } for sym, sd in ds.items()])


def build(ds, cfg, include, garch_syms, gsadf_syms, reps, progress):
    labels = {s: short_label(ds, s) for s in ds}
    sections, csvs = [], {}
    panel = price_panel(ds)

    progress.progress(0.05, text="Data sources...")
    src = _sources_table(ds)
    csvs["01_data_sources"] = src
    validation = pd.concat([pd.DataFrame(sd.validation).assign(Series=sd.label) for sd in ds.values()])
    csvs["01_validation_checks"] = validation
    csvs["01_adjusted_close_prices"] = panel
    sections.append(rp.Section(
        "1. Data collection and validation",
        [f"Sample: {cfg['start']} to {cfg['end']}. Missing-value policy: "
         f"{'missing days excluded' if cfg['policy'] == 'drop' else 'forward-filled for at most 5 days'}. "
         "Every statistic in this report is computed from the series listed below."],
        [("Data sources", rp.format_table(src))],
    ))

    if "Statistics" in include:
        progress.progress(0.15, text="Descriptive statistics...")
        st_df = stats_table(ds, cfg["ann"])
        csvs["02_descriptive_statistics"] = st_df
        moments = st_df[["Observations", "Mean daily log return", "Annualised mean return", "Daily variance",
                         "Daily std. deviation", "Annualised volatility", "Skewness", "Excess kurtosis",
                         "Jarque-Bera p-value"]]
        dd = st_df[["Total return", "Min daily return", "Max daily return", "Max drawdown", "MDD peak", "MDD trough",
                    "MDD recovered"]]
        sections.append(rp.Section(
            "2. Descriptive statistics",
            [f"Daily log returns r_t = ln(P_t / P_(t-1)); annualisation uses {cfg['ann']} trading days per year."],
            [("Moments of daily log returns", rp.format_table(moments.rename_axis("Series"), PERCENT_FIELDS, 6)),
             ("Returns and maximum drawdown", rp.format_table(dd.rename_axis("Series"), PERCENT_FIELDS))],
            [("Rebased prices (first valid day = 100, log scale)", rp.fig_normalised(panel, labels)),
             ("Drawdown from running peak", rp.fig_drawdowns(panel, labels))],
        ))

    if "GARCH" in include and garch_syms:
        progress.progress(0.3, text="GARCH(1,1)...")
        gs, results = garch_summary(ds, garch_syms, "t", cfg["ann"])
        csvs["03_garch_summary"] = gs
        csvs["03_garch_conditional_vol"] = pd.DataFrame({labels[s]: r.cond_vol_ann for s, r in results.items()})
        sections.append(rp.Section(
            "3. GARCH(1,1) conditional volatility",
            ["sigma_t^2 = omega + alpha e_(t-1)^2 + beta sigma_(t-1)^2 estimated by maximum likelihood with "
             "Student-t innovations. Persistence = alpha + beta; half-life = ln(0.5)/ln(alpha + beta)."],
            [("GARCH(1,1) estimates", rp.format_table(
                gs[["Series", "omega", "alpha", "beta", "alpha p-value", "beta p-value", "Persistence (alpha+beta)",
                    "Half-life (days)", "Long-run ann. vol", "Latest cond. vol (ann.)", "ARCH-LM p (raw returns)",
                    "Ljung-Box p (std. resid²)"]], {"Long-run ann. vol", "Latest cond. vol (ann.)"}))],
            [("Annualised conditional volatility", rp.fig_garch({labels[s]: r.cond_vol_ann for s, r in results.items()}))],
        ))

    if "GSADF" in include and gsadf_syms:
        progress.progress(0.45, text="GSADF bubble tests (Monte Carlo critical values)...")
        summ, results = gsadf_summary(ds, gsadf_syms, 0, reps, 20260101, None)
        eps = episodes_table(ds, results)
        csvs["04_gsadf_tests"] = summ
        csvs["04_gsadf_episodes"] = eps
        for s, r in results.items():
            csvs[f"04_bsadf_{s}"] = pd.DataFrame({"BSADF": r.bsadf, "cv95": r.cv95})
        tables = [("SADF and GSADF test statistics (lag 0, Monte Carlo critical values)", rp.format_table(summ, digits=3))]
        if not eps.empty:
            tables.append(("Date-stamped explosive episodes (BSADF > 95% critical value)",
                           rp.format_table(eps, {"Return over episode"}, 3)))
        sections.append(rp.Section(
            "4. Bubble detection: GSADF test (Phillips, Shi & Yu, 2015)",
            [f"Right-tailed ADF tests on log prices over all windows with minimum size r0 = 0.01 + 1.8/sqrt(T). "
             f"Critical values from {reps} Monte Carlo replications of a random walk with drift T^(-1). "
             "H0: unit root; H1: mildly explosive root."],
            tables,
            [(f"BSADF date-stamping: {labels[s]}", rp.fig_bsadf(ds[s].prices, r, labels[s])) for s, r in results.items()],
        ))

    if "Crash" in include:
        progress.progress(0.75, text="Crash analysis...")
        ref, periods = saved_periods(ds)
        metrics = all_series_metrics(ds, periods, cfg["ann"])
        tests = tests_table(ds, periods)
        csvs["05_phase_statistics"] = metrics
        csvs["05_phase_tests"] = tests
        spans = periods.as_dict()
        desc = (f"Phases dated on {labels[ref]} ({periods.method.lower()}): pre-rally {periods.pre_start.date()} to "
                f"{periods.rally_start.date()}, rally to the peak on {periods.peak.date()}"
                + (f", correction to the trough on {periods.trough.date()}." if spans["Correction"] else "; no correction observed."))
        tables = [("Phase statistics", rp.format_table(metrics.drop(columns=["Start price", "End price"]), PERIOD_PCT))]
        if not tests.empty:
            tables.append(("Welch t-test (means) and Brown-Forsythe test (variances)", rp.format_table(tests)))
        sections.append(rp.Section("5. Crash analysis", [desc], tables,
                                   [("Market phases", rp.fig_periods(ds[ref].prices, spans, labels[ref]))]))

    if "Risk" in include:
        progress.progress(0.88, text="Risk indicators...")
        th = thresholds_from_state()
        ov, _ = risk_overview(ds, th)
        csvs["06_risk_overview"] = ov
        ov_pdf = ov.replace({"⚠": "WARN", "·": "-"})
        sections.append(rp.Section(
            "6. Risk monitoring",
            [f"Thresholds: |z| > {th.abnormal_z}, 20d/120d volatility > {th.vol_ratio}, 20d/120d volume > {th.volume_surge}, "
             f"premium over 200-day MA > {th.ma_premium:.0%}, 60-day return > {th.momentum_60d:.0%}, "
             f"P/E > {th.pe:.0f} or P/B > {th.pb:.0f}. These are heuristic rules; valuations are a current snapshot."],
            [("Current warning indicators", rp.format_table(ov_pdf))],
        ))

    sections.append(rp.Section("7. Methodology notes and limitations", [f"- {t}" for t in LIMITATIONS]))
    progress.progress(0.95, text="Rendering PDF...")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = [("Generated", generated), ("Sample period", f"{cfg['start']} to {cfg['end']}"),
            ("Securities", ", ".join(labels.values())),
            ("Data sources", "Yahoo Finance (equities, adjusted closes); Sina Finance (index levels)"),
            ("Status of results", "All tables and charts are computed from real market data. No simulated data are included.")]
    pdf = rp.build_pdf(PROJECT_TITLE, "B.Sc. Mathematics project: statistical analysis report", meta, sections)
    progress.empty()
    return pdf, rp.csv_zip(csvs), generated


def render():
    page_header("Reports", "Export statistical tables, charts and test results as PDF and CSV.")
    ds = require_data()
    cfg = settings()

    include = st.pills("Sections", ["Statistics", "GARCH", "GSADF", "Crash", "Risk"],
                       default=["Statistics", "GARCH", "GSADF", "Crash", "Risk"], selection_mode="multi")
    c1, c2 = st.columns(2)
    garch_syms = c1.multiselect("GARCH series", list(ds), default=list(ds), format_func=lambda s: short_label(ds, s))
    default_g = [s for s in ["000688.SS", "000300.SS", "688256.SS"] if s in ds]
    gsadf_syms = c2.multiselect("GSADF series", list(ds), default=default_g, format_func=lambda s: short_label(ds, s))
    reps = st.select_slider("GSADF Monte Carlo replications", [199, 499, 999], value=499)
    st.caption("The crash-analysis section uses the phase dates last set on the Crash Analysis page "
               "(data-derived STAR 50 phases by default) and the risk thresholds last set on the Risk Monitoring page.")

    if st.button("Generate report", type="primary", icon=":material/description:"):
        progress = st.progress(0.0, text="Starting...")
        try:
            pdf, zipped, generated = build(ds, cfg, include or [], garch_syms, gsadf_syms, reps, progress)
            st.session_state["report"] = (pdf, zipped, generated)
        except Exception as exc:
            progress.empty()
            st.error(f"Report generation failed: {exc}")

    if rep := st.session_state.get("report"):
        pdf, zipped, generated = rep
        st.success(f"Report generated {generated}.", icon=":material/check_circle:")
        c1, c2 = st.columns(2)
        c1.download_button("Download PDF report", pdf, "chinese_ai_bubble_report.pdf", "application/pdf",
                           icon=":material/picture_as_pdf:", width="stretch", type="primary")
        c2.download_button("Download all tables (CSV, zip)", zipped, "chinese_ai_bubble_tables.zip", "application/zip",
                           icon=":material/folder_zip:", width="stretch")
