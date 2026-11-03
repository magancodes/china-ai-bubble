"""PDF and CSV report generation (matplotlib figures + fpdf2)."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
COLOURS = ["#1f4e79", "#c0392b", "#2e86c1", "#d68910", "#239b56", "#7d3c98", "#17a589", "#a04000", "#5d6d7e", "#b7950b"]


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: list[tuple[str, plt.Figure]] = field(default_factory=list)


def format_table(df: pd.DataFrame, pct_cols: set[str] = frozenset(), digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        def fmt(v, c=c):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "-"
            if isinstance(v, (float, np.floating)):
                return f"{100 * v:.2f}%" if c in pct_cols else (f"{v:,.{digits}f}" if abs(v) < 1e5 else f"{v:,.0f}")
            if isinstance(v, (bool, np.bool_)):
                return "Yes" if v else "No"
            return str(v)
        out[c] = out[c].map(fmt)
    return out


class _PDF(FPDF):
    title_text = ""

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(110)
        self.cell(0, 6, self.title_text, align="L")
        self.ln(8)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(110)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def build_pdf(title: str, subtitle: str, meta: list[tuple[str, str]], sections: list[Section]) -> bytes:
    pdf = _PDF(orientation="L", unit="mm", format="A4")
    pdf.title_text = title
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 20)
    pdf.ln(25)
    pdf.multi_cell(0, 10, title, align="C")
    pdf.set_font("DejaVu", "", 12)
    pdf.ln(2)
    pdf.multi_cell(0, 7, subtitle, align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 10)
    for k, v in meta:
        pdf.set_x(50)
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(55, 7, k)
        pdf.set_font("DejaVu", "", 10)
        pdf.multi_cell(140, 7, v)

    for sec in sections:
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 15)
        pdf.cell(0, 10, sec.heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", "", 9.5)
        for p in sec.paragraphs:
            pdf.multi_cell(0, 5.2, p)
            pdf.ln(1.5)
        for caption, df in sec.tables:
            _table(pdf, caption, df)
        for caption, fig in sec.figures:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            if pdf.get_y() > 95:
                pdf.add_page()
            pdf.set_font("DejaVu", "B", 10)
            pdf.cell(0, 7, caption, new_x="LMARGIN", new_y="NEXT")
            pdf.image(buf, w=250)
            pdf.ln(3)
    return bytes(pdf.output())


def _table(pdf: FPDF, caption: str, df: pd.DataFrame) -> None:
    df = df.reset_index() if df.index.name else df
    ncols = len(df.columns)
    font = 7.5 if ncols <= 8 else 6.5 if ncols <= 12 else 5.5
    pdf.set_font("DejaVu", "B", 10)
    pdf.cell(0, 7, caption, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", font)
    with pdf.table(text_align="LEFT", line_height=font * 0.55, first_row_as_headings=True,
                   headings_style=_heading_style()) as table:
        row = table.row()
        for c in df.columns:
            row.cell(str(c))
        for _, r in df.iterrows():
            row = table.row()
            for v in r:
                row.cell(str(v))
    pdf.ln(4)


def _heading_style():
    from fpdf.fonts import FontFace

    return FontFace(emphasis="BOLD", fill_color=(230, 236, 242))


def csv_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            keep_index = not isinstance(df.index, pd.RangeIndex)
            zf.writestr(f"{name}.csv", df.to_csv(index=keep_index))
    return buf.getvalue()


# ---- matplotlib figures for the PDF -------------------------------------------------

def _ax(figsize=(11, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def fig_normalised(panel: pd.DataFrame, labels: dict[str, str]) -> plt.Figure:
    fig, ax = _ax()
    for i, c in enumerate(panel.columns):
        s = panel[c].dropna()
        ax.plot(s.index, 100 * s / s.iloc[0], label=labels.get(c, c), color=COLOURS[i % 10], lw=1.8 if i < 2 else 1.1)
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("Rebased to 100 (log scale)")
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    return fig


def fig_drawdowns(panel: pd.DataFrame, labels: dict[str, str]) -> plt.Figure:
    fig, ax = _ax()
    for i, c in enumerate(panel.columns):
        s = panel[c].dropna()
        ax.plot(s.index, 100 * (s / s.cummax() - 1), label=labels.get(c, c), color=COLOURS[i % 10], lw=1.1)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=7, ncol=5, loc="lower left")
    return fig


def fig_garch(vols: dict[str, pd.Series]) -> plt.Figure:
    fig, ax = _ax()
    for i, (name, v) in enumerate(vols.items()):
        ax.plot(v.index, 100 * v, label=name, color=COLOURS[i % 10], lw=1.1)
    ax.set_ylabel("GARCH(1,1) conditional volatility, annualised (%)")
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    return fig


def fig_bsadf(prices: pd.Series, res, name: str) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    a1.plot(prices.index, prices, color=COLOURS[0], lw=1.2)
    a1.set_ylabel("Price")
    a1.set_title(f"{name}: price and BSADF date-stamping (shaded = explosive episodes at 95%)", fontsize=10)
    a2.plot(res.bsadf.index, res.bsadf, color=COLOURS[0], lw=1.1, label="BSADF")
    a2.plot(res.cv95.index, res.cv95, color=COLOURS[1], ls="--", lw=1, label="95% critical value")
    for _, ep in res.episodes.iterrows():
        for a in (a1, a2):
            a.axvspan(pd.Timestamp(ep["Start"]), pd.Timestamp(ep["End"]), color=COLOURS[1], alpha=0.15)
    a2.legend(fontsize=7)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    return fig


def fig_periods(prices: pd.Series, spans: dict, name: str) -> plt.Figure:
    fig, ax = _ax()
    ax.plot(prices.index, prices, color=COLOURS[0], lw=1.3)
    cols = {"Pre-rally": "#2e86c1", "Rally": "#239b56", "Correction": "#c0392b"}
    for label, span in spans.items():
        if span is not None:
            ax.axvspan(span[0], span[1], color=cols[label], alpha=0.15, label=label)
    ax.set_title(f"{name}: market phases used in the crash analysis", fontsize=10)
    ax.legend(fontsize=8)
    return fig
