"""Securities analysed in the case study."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str
    kind: str  # "index" or "equity"
    segment: str
    description: str
    proxy_etf: str | None = None


INDICES = [
    Security(
        "000688.SS",
        "STAR 50 Index",
        "index",
        "SSE STAR Market",
        "50 largest, most liquid STAR Market companies; heavily weighted to semiconductors and AI hardware.",
        proxy_etf="588000.SS",
    ),
    Security(
        "000300.SS",
        "CSI 300 Index",
        "index",
        "Shanghai + Shenzhen",
        "300 large-cap A-shares; the broad-market benchmark.",
        proxy_etf="510300.SS",
    ),
]

EQUITIES = [
    Security("688256.SS", "Cambricon Technologies", "equity", "STAR Market", "AI accelerator chips (MLU series)."),
    Security("688981.SS", "SMIC", "equity", "STAR Market", "China's largest contract chip foundry."),
    Security("688041.SS", "Hygon Information Technology", "equity", "STAR Market", "x86 CPUs and DCU AI accelerators."),
    Security("002230.SZ", "iFLYTEK", "equity", "SZSE Main Board", "Speech AI and the Spark large language model."),
    Security("688008.SS", "Montage Technology", "equity", "STAR Market", "Memory-interface and data-centre chips."),
    Security("002371.SZ", "NAURA Technology", "equity", "SZSE Main Board", "Semiconductor manufacturing equipment."),
    Security("000977.SZ", "Inspur Electronic Information", "equity", "SZSE Main Board", "AI servers and data-centre hardware."),
    Security("300308.SZ", "Zhongji Innolight", "equity", "ChiNext", "High-speed optical transceivers for AI clusters."),
]

UNIVERSE = INDICES + EQUITIES
BY_SYMBOL = {s.symbol: s for s in UNIVERSE}


def daily_price_limit(symbol: str) -> float | None:
    """Exchange daily price limit (as a fraction) for an A-share symbol, or None if unknown."""
    code, _, exch = symbol.partition(".")
    if exch not in {"SS", "SZ"} or len(code) != 6:
        return None
    if code.startswith(("688", "689", "300", "301")):
        return 0.20
    if code.startswith(("60", "00")):
        return 0.10
    return None
