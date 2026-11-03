# A Case Study of the 2026 Chinese AI Stock Market Bubble

An interactive Streamlit dashboard for a B.Sc. Mathematics project. It collects daily market data for the
**STAR 50** and **CSI 300** indices and eight Chinese AI / semiconductor companies, measures the rally and the
subsequent correction, tests for explosive (bubble) behaviour, and models volatility and risk.

| Symbol | Security | Segment |
| --- | --- | --- |
| 000688.SS | STAR 50 Index | SSE STAR Market |
| 000300.SS | CSI 300 Index | Shanghai + Shenzhen |
| 688256.SS | Cambricon Technologies | STAR Market |
| 688981.SS | SMIC | STAR Market |
| 688041.SS | Hygon Information Technology | STAR Market |
| 002230.SZ | iFLYTEK | SZSE Main Board |
| 688008.SS | Montage Technology | STAR Market |
| 002371.SZ | NAURA Technology | SZSE Main Board |
| 000977.SZ | Inspur Electronic Information | SZSE Main Board |
| 300308.SZ | Zhongji Innolight | ChiNext |

More Yahoo Finance tickers can be validated and added from the **Data Collection** page.

## Features

1. **Data Collection**: live download, ticker validation, data-quality checks, missing-value handling, and source links.
2. **Market Visualisation**: candlestick and volume charts, rebased comparisons, daily returns, volume surges and correlations.
3. **Statistical Analysis**: mean, variance, standard deviation, annualised volatility, skewness, kurtosis, Jarque–Bera and maximum drawdown.
4. **GARCH(1,1)**: maximum-likelihood estimates, persistence α+β, shock half-life, conditional volatility and forecasts, and ARCH-LM / Ljung–Box diagnostics.
5. **Bubble Detection**: SADF and GSADF tests (Phillips, Shi & Yu 2015) with Monte Carlo critical values and BSADF date-stamping of explosive episodes.
6. **Crash Analysis**: pre-rally, rally and correction phases, either derived from the price path or set manually. Phases are compared with Welch t-tests and Brown–Forsythe variance tests.
7. **Risk Monitoring**: warnings for abnormal returns, rising volatility, volume surges, overextension and high valuation, with adjustable thresholds.
8. **Reports**: a PDF report (tables and charts) and a zip of CSV tables. Most tables can also be downloaded as CSV from their own page.

## Data sources and integrity

- **Equities** come from Yahoo Finance via `yfinance`. Returns use dividend-adjusted closes.
- **Indices** come from the Sina Finance daily K-line API, because Yahoo Finance only serves the latest quote for
  `000688.SS` and `000300.SS`. If Sina is unreachable, the app falls back to a tracking ETF (588000.SS / 510300.SS)
  and labels the series as a proxy everywhere it appears.
- **Missing data**: zero-volume days (trading suspensions) and invalid prices count as missing. By default they are
  excluded. Forward-filling for up to five days is optional. Values are never interpolated or invented.
- **Split repair**: if a daily move breaks the exchange price limit (±10% main board, ±20% STAR/ChiNext) and exactly
  matches a split recorded by Yahoo Finance nearby, earlier prices are rescaled by that ratio. Each repair is listed on
  the Data Collection page. For example, Innolight's 1.4-for-1 bonus-share issue in 2024 is misdated in Yahoo's data.
- **Illustrative vs actual**: actual results carry a green "Actual result" badge. The only simulated data is the
  optional GSADF demonstration, which has an orange "Illustrative example" badge. It is never part of the results or the reports.
- **Valuation ratios** (P/E, P/B) are a current Yahoo Finance snapshot, not historical values.

## Running locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8517> (the port is set in `.streamlit/config.toml`). It needs internet access to
reach Yahoo Finance and Sina Finance. Downloads are cached for six hours. Use **Refresh data from sources** in the
sidebar to download again.

The GSADF page simulates its critical values when it first runs. With 499 replications this takes about 15–30
seconds per sample size, and the result is cached after that.

Run the tests:

```bash
pytest -q
```

## Deploying

- **Streamlit Community Cloud** (free): push this repository to GitHub, go to <https://share.streamlit.io>, choose
  **Create app**, pick the repository, branch `main`, main file `app.py`, and Python 3.12 under *Advanced settings*.
- **Any Docker host** (Render, Railway, Fly.io, Hugging Face Spaces, ...): the included `Dockerfile` serves the app on
  `$PORT` (default 8501). Test it locally with `docker build -t china-ai-bubble . && docker run -p 8501:8501 china-ai-bubble`.

No secrets are needed, but the host must allow outbound requests to Yahoo Finance and Sina Finance.

## Project layout

```
app.py              Streamlit entry point and navigation
bubble/             Analysis library (no Streamlit dependency)
  data.py           Fetching, validation, cleaning, split repair
  stats.py          Descriptive statistics and drawdowns
  garch.py          GARCH(1,1) estimation and diagnostics
  gsadf.py          SADF / GSADF / BSADF with Monte Carlo critical values
  periods.py        Rally / correction dating and phase tests
  risk.py           Rule-based warning indicators
  charts.py         Plotly figures
  report.py         PDF (fpdf2 + matplotlib) and CSV export
views/              One module per dashboard page
tests/              Unit tests (ADF matches statsmodels, GSADF power, GARCH recovery, ...)
```

## References

- Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for multiple bubbles: Historical episodes of exuberance and
  collapse in the S&P 500. *International Economic Review*, 56(4), 1043–1078.
- Phillips, P. C. B., Wu, Y., & Yu, J. (2011). Explosive behavior in the 1990s Nasdaq. *International Economic Review*, 52(1), 201–226.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.

This dashboard is for academic research. It is not investment advice.
