"""Plotly figures for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#1f4e79", "#c0392b", "#2e86c1", "#d68910", "#239b56", "#7d3c98", "#17a589", "#a04000", "#5d6d7e", "#b7950b"]
PERIOD_COLOURS = {"Pre-rally": "rgba(46,134,193,0.12)", "Rally": "rgba(35,155,86,0.15)", "Correction": "rgba(192,57,43,0.15)"}


def _layout(fig: go.Figure, height: int = 460, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def price_volume(df: pd.DataFrame, name: str, candles: bool = True) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    if candles:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                     name=name, increasing_line_color="#c0392b", decreasing_line_color="#239b56"), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=name, line=dict(color=PALETTE[0])), 1, 1)
    for w, c in ((50, "#d68910"), (200, "#7d3c98")):
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(w).mean(), name=f"{w}-day MA",
                                 line=dict(color=c, width=1.2)), 1, 1)
    up = df["Close"].diff().fillna(0) >= 0
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=np.where(up, "#e6a1a1", "#a3d9b8")), 2, 1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _layout(fig, 560)


def normalised(panel: pd.DataFrame, labels: dict[str, str], log_scale: bool = False) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(panel.columns):
        s = panel[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=100 * s / s.iloc[0], name=labels.get(col, col),
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2 if i < 2 else 1.4)))
    fig.add_hline(y=100, line_dash="dot", line_color="grey")
    fig.update_yaxes(title_text="Rebased (first valid day = 100)", type="log" if log_scale else "linear")
    return _layout(fig, 480)


def returns_series(r: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=r.index, y=100 * r, name=name, marker_color=np.where(r >= 0, "#c0392b", "#239b56")))
    fig.update_yaxes(title_text="Daily log return (%)")
    return _layout(fig, 340)


def returns_hist(r: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=100 * r, nbinsx=80, histnorm="probability density", name="Empirical",
                                 marker_color=PALETTE[0], opacity=0.75))
    x = np.linspace(100 * r.min(), 100 * r.max(), 300)
    mu, sd = 100 * r.mean(), 100 * r.std()
    fig.add_trace(go.Scatter(x=x, y=np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                             name="Normal with same mean and std.", line=dict(color=PALETTE[1])))
    fig.update_xaxes(title_text="Daily log return (%)")
    fig.update_layout(hovermode="closest")
    return _layout(fig, 340)


def volume_compare(panel: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(panel.columns):
        s = panel[col].dropna()
        rel = s.rolling(20).mean() / s.rolling(120, min_periods=60).mean()
        fig.add_trace(go.Scatter(x=rel.index, y=rel, name=labels.get(col, col), line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.add_hline(y=1, line_dash="dot", line_color="grey")
    fig.update_yaxes(title_text="20-day avg volume / 120-day avg volume")
    return _layout(fig, 420)


def correlation(returns: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    c = returns.corr()
    names = [labels.get(x, x) for x in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(c.values, 2), texttemplate="%{text}"))
    fig.update_layout(hovermode="closest")
    return _layout(fig, 520)


def drawdowns(panel: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(panel.columns):
        s = panel[col].dropna()
        dd = s / s.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=100 * dd, name=labels.get(col, col), line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.update_yaxes(title_text="Drawdown from running peak (%)")
    return _layout(fig, 420)


def rolling_vol(panel: pd.DataFrame, labels: dict[str, str], window: int, ann: int) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(panel.columns):
        s = panel[col].dropna()
        v = np.log(s).diff().rolling(window).std() * np.sqrt(ann)
        fig.add_trace(go.Scatter(x=v.index, y=100 * v, name=labels.get(col, col), line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.update_yaxes(title_text=f"{window}-day annualised volatility (%)")
    return _layout(fig, 420)


def garch_vol(prices: pd.Series, cond_vol: pd.Series, forecast: pd.Series, ann: int) -> go.Figure:
    r = np.log(prices).diff().dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=cond_vol.index, y=100 * cond_vol, name="GARCH conditional volatility (ann.)",
                             line=dict(color=PALETTE[1])), 1, 1)
    fig.add_trace(go.Scatter(x=forecast.index, y=100 * forecast, name="Forecast", line=dict(color=PALETTE[1], dash="dot")), 1, 1)
    rv = r.rolling(20).std() * np.sqrt(ann)
    fig.add_trace(go.Scatter(x=rv.index, y=100 * rv, name="20-day realised volatility (ann.)",
                             line=dict(color=PALETTE[0], width=1)), 1, 1)
    fig.add_trace(go.Bar(x=r.index, y=100 * r.abs(), name="|Daily return| (%)", marker_color="#95a5a6"), 2, 1)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="|r| (%)", row=2, col=1)
    return _layout(fig, 560)


def bsadf_chart(prices: pd.Series, res, name: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        subplot_titles=(f"{name}: price with date-stamped explosive episodes", "BSADF sequence vs. critical values"))
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color=PALETTE[0])), 1, 1)
    fig.add_trace(go.Scatter(x=res.bsadf.index, y=res.bsadf, name="BSADF statistic", line=dict(color=PALETTE[0])), 2, 1)
    fig.add_trace(go.Scatter(x=res.cv95.index, y=res.cv95, name="95% critical value", line=dict(color=PALETTE[1], dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=res.cv90.index, y=res.cv90, name="90% critical value", line=dict(color="#d68910", dash="dot", width=1)), 2, 1)
    for _, ep in res.episodes.iterrows():
        for row in (1, 2):
            fig.add_vrect(x0=ep["Start"], x1=ep["End"], fillcolor="rgba(192,57,43,0.18)", line_width=0, row=row, col=1)
    return _layout(fig, 620)


def periods_chart(prices: pd.Series, spans: dict, name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=prices.index, y=prices, name=name, line=dict(color=PALETTE[0])))
    for label, span in spans.items():
        if span is None:
            continue
        fig.add_vrect(x0=span[0], x1=span[1], fillcolor=PERIOD_COLOURS[label], line_width=0,
                      annotation_text=label, annotation_position="top left")
    fig.update_yaxes(title_text="Price")
    return _layout(fig, 420)


def period_bars(table: pd.DataFrame, metric: str, labels_col: str = "Series") -> go.Figure:
    fig = go.Figure()
    for period, colour in (("Pre-rally", "#2e86c1"), ("Rally", "#239b56"), ("Correction", "#c0392b")):
        t = table[table["Period"] == period]
        if t.empty:
            continue
        fig.add_trace(go.Bar(x=t[labels_col], y=100 * t[metric], name=period, marker_color=colour))
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(title_text=f"{metric} (%)")
    return _layout(fig, 420)


def warning_timeline(hist: pd.DataFrame) -> go.Figure:
    z = hist.astype(float).T
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=[[0, "#f4f6f7"], [1, "#c0392b"]],
                               showscale=False, hovertemplate="%{y}<br>%{x|%Y-%m-%d}: %{z}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _layout(fig, 280)
