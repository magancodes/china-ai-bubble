"""Generalised sup ADF (GSADF) bubble test of Phillips, Shi & Yu (2015).

For every window [r1, r2] of the (log) price series y the ADF regression

    dy_t = a + b y_{t-1} + sum_{i=1..k} phi_i dy_{t-i} + e_t

is estimated and ADF(r1, r2) = b_hat / se(b_hat) is recorded.

* BSADF(r2) = sup_{r1 in [0, r2 - r0]} ADF(r1, r2)   (backward sup ADF, used for date-stamping)
* SADF      = sup_{r2} ADF(0, r2)                     (Phillips, Wu & Yu 2011)
* GSADF     = sup_{r2 in [r0, 1]} BSADF(r2)

H0: unit root (martingale). H1: mildly explosive behaviour (b > 0). Right-tailed critical
values are obtained by Monte Carlo simulation of the null DGP
y_t = d T^{-eta} + y_{t-1} + eps_t with d = eta = 1, using the same sample size, minimum
window and lag order as the data.

All window regressions are computed from cumulative sums of cross-products so that each
end point solves every start point in one batched linear-algebra call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def default_min_window(n: int) -> int:
    """PSY (2015) rule: r0 = 0.01 + 1.8 / sqrt(T)."""
    return int(np.floor((0.01 + 1.8 / np.sqrt(n)) * n))


def _design(y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build regressors X (..., m, p) and regressand z (..., m) for the ADF regression."""
    y = y - y.mean(axis=-1, keepdims=True)
    dy = np.diff(y, axis=-1)
    n = y.shape[-1]
    m = n - 1 - k
    cols = [np.ones(y.shape[:-1] + (m,)), y[..., k : n - 1]]
    for i in range(1, k + 1):
        cols.append(dy[..., k - i : n - 1 - i])
    X = np.stack(cols, axis=-1)
    z = dy[..., k:]
    return X, z


def _bsadf_batch(y: np.ndarray, k: int, w0: int) -> tuple[np.ndarray, np.ndarray]:
    """BSADF and forward ADF(0, r2) sequences for a batch of series y with shape (R, n)."""
    X, z = _design(y, k)
    R, m, p = X.shape
    zeros = np.zeros((R, 1))
    XX = np.concatenate([np.zeros((R, 1, p, p)), np.cumsum(X[..., :, None] * X[..., None, :], axis=1)], axis=1)
    Xz = np.concatenate([np.zeros((R, 1, p)), np.cumsum(X * z[..., None], axis=1)], axis=1)
    zz = np.concatenate([zeros, np.cumsum(z * z, axis=1)], axis=1)
    e1 = np.zeros(p)
    e1[1] = 1.0
    bsadf = np.full((R, m), np.nan)
    sadf_seq = np.full((R, m), np.nan)
    for e in range(w0 - 1, m):
        S = e - w0 + 2
        XXw = XX[:, e + 1, None] - XX[:, :S]
        Xzw = Xz[:, e + 1, None] - Xz[:, :S]
        zzw = zz[:, e + 1, None] - zz[:, :S]
        nobs = (e + 1 - np.arange(S)).astype(float)
        rhs = np.stack([Xzw, np.broadcast_to(e1, Xzw.shape)], axis=-1)
        sol = np.linalg.solve(XXw, rhs)
        beta, inv_col = sol[..., 0], sol[..., 1]
        ssr = np.maximum(zzw - np.einsum("rsp,rsp->rs", beta, Xzw), 1e-300)
        sigma2 = ssr / (nobs - p)
        tstat = beta[..., 1] / np.sqrt(sigma2 * inv_col[..., 1])
        bsadf[:, e] = tstat.max(axis=1)
        sadf_seq[:, e] = tstat[:, 0]
    return bsadf, sadf_seq


def simulate_critical_values(n: int, k: int, w0: int, reps: int = 499, seed: int = 20260101, batch: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    bs_all, gsadf, sadf = [], [], []
    done = 0
    while done < reps:
        b = min(batch, reps - done)
        eps = rng.standard_normal((b, n))
        y = np.cumsum(eps + 1.0 / n, axis=1)
        bs, fwd = _bsadf_batch(y, k, w0)
        bs_all.append(bs)
        gsadf.append(np.nanmax(bs, axis=1))
        sadf.append(np.nanmax(fwd, axis=1))
        done += b
    bs_all = np.vstack(bs_all)
    gsadf = np.concatenate(gsadf)
    sadf = np.concatenate(sadf)
    qs = [0.90, 0.95, 0.99]
    return {
        "bsadf_cv": {q: np.nanquantile(bs_all, q, axis=0) for q in qs},
        "gsadf_cv": {q: float(np.quantile(gsadf, q)) for q in qs},
        "sadf_cv": {q: float(np.quantile(sadf, q)) for q in qs},
        "gsadf_dist": gsadf,
        "sadf_dist": sadf,
        "reps": reps,
        "seed": seed,
    }


@dataclass
class GSADFResult:
    bsadf: pd.Series
    cv90: pd.Series
    cv95: pd.Series
    cv99: pd.Series
    gsadf: float
    sadf: float
    gsadf_cv: dict
    sadf_cv: dict
    gsadf_pvalue: float
    sadf_pvalue: float
    episodes: pd.DataFrame
    n: int
    lag: int
    min_window: int
    min_duration: int
    reps: int
    seed: int

    @property
    def reject_95(self) -> bool:
        return self.gsadf > self.gsadf_cv[0.95]


def run_gsadf(
    prices: pd.Series,
    lag: int = 0,
    min_window: int | None = None,
    reps: int = 499,
    seed: int = 20260101,
    min_duration: int | None = None,
    cv: dict | None = None,
) -> GSADFResult:
    """Run SADF/GSADF on log prices and date-stamp explosive episodes at the 95% level."""
    prices = prices.dropna()
    y = np.log(prices.to_numpy(dtype=float))
    n = len(y)
    w0 = min_window or default_min_window(n)
    w0 = max(w0, lag + 6)
    if n - 1 - lag < w0 + 2:
        raise ValueError("Series too short for the chosen minimum window.")
    bs, fwd = _bsadf_batch(y[None, :], lag, w0)
    bs, fwd = bs[0], fwd[0]
    if cv is None:
        cv = simulate_critical_values(n, lag, w0, reps, seed)
    dates = prices.index[lag + 1 :]
    valid = ~np.isnan(bs)
    idx = dates[valid]
    bsadf = pd.Series(bs[valid], index=idx, name="BSADF")
    cvs = {q: pd.Series(cv["bsadf_cv"][q][valid], index=idx) for q in (0.90, 0.95, 0.99)}
    gs, sa = float(np.nanmax(bs)), float(np.nanmax(fwd))
    mdur = min_duration if min_duration is not None else int(np.ceil(np.log(n)))
    episodes = date_stamp(bsadf, cvs[0.95], prices, mdur)
    return GSADFResult(
        bsadf=bsadf,
        cv90=cvs[0.90],
        cv95=cvs[0.95],
        cv99=cvs[0.99],
        gsadf=gs,
        sadf=sa,
        gsadf_cv=cv["gsadf_cv"],
        sadf_cv=cv["sadf_cv"],
        gsadf_pvalue=float((1 + np.sum(cv["gsadf_dist"] >= gs)) / (1 + len(cv["gsadf_dist"]))),
        sadf_pvalue=float((1 + np.sum(cv["sadf_dist"] >= sa)) / (1 + len(cv["sadf_dist"]))),
        episodes=episodes,
        n=n,
        lag=lag,
        min_window=w0,
        min_duration=mdur,
        reps=cv["reps"],
        seed=cv["seed"],
    )


def date_stamp(bsadf: pd.Series, cv: pd.Series, prices: pd.Series, min_duration: int) -> pd.DataFrame:
    """Episodes where BSADF exceeds its critical value for at least `min_duration` observations."""
    above = (bsadf > cv).to_numpy()
    rows = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            length = j - i + 1
            if length >= min_duration:
                s, e = bsadf.index[i], bsadf.index[j]
                seg = bsadf.iloc[i : j + 1]
                rows.append(
                    {
                        "Start": s.date(),
                        "End": e.date(),
                        "Trading days": length,
                        "Peak BSADF": float(seg.max()),
                        "Peak BSADF date": seg.idxmax().date(),
                        "Price at start": float(prices.loc[s]),
                        "Price at end": float(prices.loc[e]),
                        "Return over episode": float(prices.loc[e] / prices.loc[s] - 1),
                    }
                )
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def illustrative_regimes(n: int = 600) -> dict:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {"bubble_start": idx[int(0.45 * n)], "peak": idx[int(0.60 * n) - 1], "collapse_end": idx[int(0.64 * n) - 1]}


def simulate_illustrative_bubble(n: int = 600, seed: int = 11) -> pd.Series:
    """ILLUSTRATIVE ONLY: simulated random walk with one explosive regime (P_t = 1.01 P_{t-1} + e_t)
    followed by a collapse. Used solely to demonstrate how the test behaves; never mixed with
    market results.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros(n)
    p[0] = 100.0
    start, peak, end = int(0.45 * n), int(0.60 * n), int(0.64 * n)
    for t in range(1, n):
        eps = 0.5 * rng.standard_normal()
        if start <= t < peak:
            p[t] = 1.01 * p[t - 1] + eps
        elif peak <= t < end:
            p[t] = p[t - 1] - (p[peak - 1] - p[start - 1]) / (end - peak) + eps
        else:
            p[t] = p[t - 1] + eps
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.maximum(p, 1.0), index=idx, name="Simulated price")
