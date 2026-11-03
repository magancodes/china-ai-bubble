"""GARCH(1,1) estimation.

r_t = mu + e_t,  e_t = sigma_t z_t,
sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2.
Persistence is alpha + beta; the half-life of a volatility shock is ln(0.5)/ln(alpha+beta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


@dataclass
class GarchResult:
    params: pd.DataFrame
    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life: float | None
    unconditional_vol_ann: float | None
    loglik: float
    aic: float
    bic: float
    converged: bool
    cond_vol_ann: pd.Series
    std_resid: pd.Series
    forecast_vol_ann: pd.Series
    arch_lm_pvalue: float
    ljung_box_sq_pvalue: float
    nobs: int
    dist: str


def fit_garch(prices: pd.Series, dist: str = "t", periods_per_year: int = 252, horizon: int = 20) -> GarchResult:
    r = 100 * np.log(prices).diff().dropna()
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    res = am.fit(disp="off", options={"maxiter": 1000})
    p = res.params
    omega, alpha, beta = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
    pers = alpha + beta
    half_life = float(np.log(0.5) / np.log(pers)) if 0 < pers < 1 else None
    uncond = float(np.sqrt(omega / (1 - pers) * periods_per_year) / 100) if pers < 1 else None
    table = pd.DataFrame(
        {"Estimate": res.params, "Std. error": res.std_err, "t-stat": res.tvalues, "p-value": res.pvalues}
    )
    fc = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1]
    last = r.index[-1]
    fc_index = pd.bdate_range(last + pd.Timedelta(days=1), periods=horizon)
    lm = het_arch(r - r.mean(), nlags=10)
    lb = acorr_ljungbox(res.std_resid.dropna() ** 2, lags=[10], return_df=True)
    return GarchResult(
        params=table,
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=pers,
        half_life=half_life,
        unconditional_vol_ann=uncond,
        loglik=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        converged=res.convergence_flag == 0,
        cond_vol_ann=res.conditional_volatility * np.sqrt(periods_per_year) / 100,
        std_resid=res.std_resid,
        forecast_vol_ann=pd.Series(np.sqrt(fc.values * periods_per_year) / 100, index=fc_index),
        arch_lm_pvalue=float(lm[1]),
        ljung_box_sq_pvalue=float(lb["lb_pvalue"].iloc[0]),
        nobs=len(r),
        dist=dist,
    )


def persistence_label(pers: float) -> str:
    if pers >= 1:
        return "Integrated / explosive (shocks never decay)"
    if pers >= 0.95:
        return "Very high persistence"
    if pers >= 0.85:
        return "High persistence"
    return "Moderate persistence"
