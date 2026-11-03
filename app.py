import streamlit as st

from views import crash, data_page, garch_page, gsadf_page, market, overview, report_page, risk_page, statistics
from views.common import sidebar

st.set_page_config(page_title="Chinese AI Bubble Case Study", page_icon=":material/query_stats:", layout="wide")

pages = {
    "Project": [st.Page(overview.render, title="Overview", icon=":material/home:", url_path="overview", default=True)],
    "Analysis": [
        st.Page(data_page.render, title="Data Collection", icon=":material/database:", url_path="data"),
        st.Page(market.render, title="Market Visualisation", icon=":material/candlestick_chart:", url_path="market"),
        st.Page(statistics.render, title="Statistical Analysis", icon=":material/functions:", url_path="statistics"),
        st.Page(garch_page.render, title="GARCH(1,1) Model", icon=":material/ssid_chart:", url_path="garch"),
        st.Page(gsadf_page.render, title="Bubble Detection (GSADF)", icon=":material/bubble_chart:", url_path="gsadf"),
        st.Page(crash.render, title="Crash Analysis", icon=":material/trending_down:", url_path="crash"),
        st.Page(risk_page.render, title="Risk Monitoring", icon=":material/warning:", url_path="risk"),
    ],
    "Output": [st.Page(report_page.render, title="Reports", icon=":material/description:", url_path="reports")],
}

nav = st.navigation(pages)
sidebar()
nav.run()
