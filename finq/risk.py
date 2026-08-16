"""Risk structure metrics: how much risk a portfolio runs, and where it comes from.

Every volatility figure returned by this module is annualized unless the caller
passes ``freq=None``, using sqrt(PERIODS_PER_YEAR[freq]) -- sqrt(252) daily,
sqrt(52) weekly. The annualization convention travels with the number in every
report header, because an un-annualized vol and an annualized one differ by a
factor of sixteen and both look plausible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from finq.returns import PERIODS_PER_YEAR


class RiskError(Exception):
    """Raised when a risk metric cannot be computed."""


def _check(Sigma, w) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (covariance, weights) pair and return them as float arrays.

    This is the shared validation gate for every metric in this module,
    including the tail, market and stress metrics added later: shape,
    finiteness and weight-length agreement are enforced here in exactly one
    place. Extend this function rather than re-implementing the checks at each
    call site, so that a malformed input fails identically whichever entry
    point the caller reached for.

    Non-finite values are rejected rather than propagated. NaN flows silently
    through ``w @ Sigma @ w`` and comes back out as a NaN volatility, which
    reads downstream as "no risk computed" instead of "your covariance matrix
    is broken" -- and section 9's governing principle is that an analysis run
    on bad data must never look like a successful run.

    Symmetry is checked (within a float tolerance) for the same reason:
    ``eigh`` in ``effective_bets`` reads only the lower triangle, while
    ``w @ Sigma @ w`` reads all of it, so an asymmetric Sigma would make the
    two metrics silently disagree about the same portfolio instead of either
    one raising. No PSD check is made -- rmt_clean can legitimately hand back
    a matrix with faintly negative eigenvalues, and callers that care (like
    ``_vol``) already clamp variance at zero rather than trusting positivity.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(w, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise RiskError(f"Sigma must be a square matrix, got shape {Sigma.shape}")
    if w.ndim != 1 or w.shape[0] != Sigma.shape[0]:
        raise RiskError(
            f"weight length {w.shape[0] if w.ndim == 1 else w.shape} does not "
            f"match Sigma dimension {Sigma.shape[0]}"
        )
    if not np.isfinite(Sigma).all():
        raise RiskError("Sigma contains non-finite values (NaN or inf)")
    if not np.isfinite(w).all():
        raise RiskError("weights contain non-finite values (NaN or inf)")
    if not np.allclose(Sigma, Sigma.T, rtol=1e-8, atol=1e-10):
        raise RiskError("Sigma is not symmetric")
    return Sigma, w


def _check_returns(R, w) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (return matrix, weights) pair for metrics built on raw returns.

    ``_check`` validates a Sigma/w pair; tail, market and stress metrics (Task
    9) work from the T x N return matrix directly instead, so they need R's
    shape checked against w rather than a square Sigma's. Kept in this module,
    next to ``_check``, so both validation gates live in one place rather than
    each metric re-implementing its own.
    """
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float)
    if R.ndim != 2:
        raise RiskError(f"returns must be a 2-D T x N matrix, got shape {R.shape}")
    if w.ndim != 1 or w.shape[0] != R.shape[1]:
        raise RiskError(
            f"weight length {w.shape[0] if w.ndim == 1 else w.shape} does not "
            f"match the number of assets in R ({R.shape[1]})"
        )
    if not np.isfinite(R).all():
        raise RiskError("returns contain non-finite values (NaN or inf)")
    if not np.isfinite(w).all():
        raise RiskError("weights contain non-finite values (NaN or inf)")
    return R, w


def _annualize(value: float, freq: str | None) -> float:
    if freq is None:
        return value
    if freq not in PERIODS_PER_YEAR:
        raise RiskError(f"unknown freq {freq!r}; use 'daily', 'weekly', or None")
    return float(value * np.sqrt(PERIODS_PER_YEAR[freq]))


def _vol(Sigma: np.ndarray, w: np.ndarray, metric: str) -> float:
    """Un-annualized portfolio volatility, refusing to divide by zero later."""
    vol = float(np.sqrt(max(float(w @ Sigma @ w), 0.0)))
    if vol <= 0:
        raise RiskError(f"portfolio volatility is zero; {metric} undefined")
    return vol


def portfolio_vol(Sigma, w, freq: str | None = "daily") -> float:
    """Annualized portfolio volatility sqrt(w'Sigma w), or raw if freq is None."""
    Sigma, w = _check(Sigma, w)
    variance = float(w @ Sigma @ w)
    return _annualize(float(np.sqrt(max(variance, 0.0))), freq)


def risk_contributions(Sigma, w) -> tuple[np.ndarray, np.ndarray]:
    """Marginal contribution to risk, and each holding's share of total risk.

    Returns ``(mctr, pct_contribution)`` where ``mctr = Sigma w / sigma_p`` is
    the derivative of portfolio volatility with respect to each weight. Euler's
    theorem on the homogeneous-of-degree-one volatility function makes the
    decomposition exact: ``(w * mctr).sum() == sigma_p`` and therefore
    ``pct_contribution.sum() == 1``. That is what makes this the number to look
    at rather than the weights -- a 4% position in a high-vol, high-beta name
    can carry 15% of the risk, and the weight vector never shows it.
    """
    Sigma, w = _check(Sigma, w)
    vol = _vol(Sigma, w, "risk contributions")
    mctr = (Sigma @ w) / vol
    return mctr, (w * mctr) / vol


def diversification_ratio(Sigma, w) -> float:
    """Choueifaty diversification ratio: (w' sigma) / sqrt(w' Sigma w).

    The weighted average of the standalone volatilities over the portfolio
    volatility. For long-only weights it is >= 1, and equals 1 exactly when
    every pairwise correlation is 1 -- there is then nothing to diversify.
    """
    Sigma, w = _check(Sigma, w)
    vol = _vol(Sigma, w, "diversification ratio")
    return float((w @ np.sqrt(np.diag(Sigma))) / vol)


def effective_bets(Sigma, w) -> float:
    """Meucci: entropy of the variance shares of the principal portfolios.

    Rotate the weights into the eigenbasis of Sigma, so the portfolio becomes a
    combination of uncorrelated principal portfolios. Each contributes
    ``w_tilde_i^2 * lambda_i`` to the variance; normalizing those gives a
    probability distribution p, and ``exp(-sum p_i ln p_i)`` reports how many
    genuinely independent bets the portfolio holds. Ten names loading on one
    factor score near 1, not near 10.

    CAVEAT (Meucci, Santangelo & Deguest 2015): this PCA-basis form is
    basis-dependent when Sigma has repeated eigenvalues. ``Sigma = I`` (no
    correlation at all) is degenerate -- every orthonormal basis is a valid
    eigenbasis, and LAPACK's ``eigh`` happens to return the identity basis for
    an identity matrix, which is what makes an equal-weighted uncorrelated
    portfolio score N here. A basis aligned to ``w`` instead would score 1 for
    the same portfolio. This is a property of the naive PCA form, not a bug;
    minimum-torsion bases exist to remove the dependence but are out of scope.
    """
    Sigma, w = _check(Sigma, w)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    w_tilde = evecs.T @ w
    contrib = (w_tilde ** 2) * evals
    total = contrib.sum()
    if total <= 0:
        raise RiskError("portfolio variance is zero; effective bets undefined")
    p = contrib / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def concentration(w, pct_rc) -> dict[str, float]:
    """Herfindahl-Hirschman index on weights and on risk contributions.

    ``hhi_risk`` is the more informative of the two: weights can look evenly
    spread while the risk sits in two correlated names.
    """
    w = np.asarray(w, dtype=float)
    pct_rc = np.asarray(pct_rc, dtype=float)
    if w.ndim != 1 or pct_rc.ndim != 1:
        raise RiskError("weights and risk contributions must both be 1-D")
    if w.shape[0] != pct_rc.shape[0]:
        raise RiskError(
            f"weight length {w.shape[0]} does not match risk-contribution "
            f"length {pct_rc.shape[0]}"
        )
    if not np.isfinite(w).all() or not np.isfinite(pct_rc).all():
        raise RiskError("weights or risk contributions contain non-finite values")
    return {
        "hhi_weights": float((w ** 2).sum()),
        "hhi_risk": float((pct_rc ** 2).sum()),
    }


def _portfolio_series(R, w) -> np.ndarray:
    """Fixed-weight portfolio return series, validated via ``_check_returns``."""
    arr, w = _check_returns(R, w)
    return arr @ w


def var_cvar(R, w, level: float = 0.95, method: str = "historical") -> tuple[float, float]:
    """Value at risk and conditional VaR, reported as positive loss magnitudes.

    ``historical`` reads the (1 - level) empirical quantile of the portfolio
    return series directly. ``cornish_fisher`` instead adjusts the Gaussian
    quantile for sample skew and excess kurtosis, so it can report a deeper
    loss than the normal approximation when the return distribution is
    left-skewed or fat-tailed -- exactly the shape financial returns tend to
    have and a plain normal VaR misses.

    CVaR is the mean of the losses at or beyond VaR; it is floored at VaR
    itself so that a tail with no observations past the quantile (possible
    for ``cornish_fisher``, whose VaR is a parametric estimate rather than a
    literal data point) never reports a CVaR smaller than the VaR it sits
    beside.
    """
    if not 0.5 < level < 1.0:
        raise RiskError(f"level must be between 0.5 and 1, got {level}")
    p = _portfolio_series(R, w)
    if p.size < 30:
        raise RiskError(f"only {p.size} observations; too few for a tail estimate")

    if method == "historical":
        var = -float(np.quantile(p, 1.0 - level))
    elif method == "cornish_fisher":
        mu, sd = float(p.mean()), float(p.std(ddof=0))
        s, k = float(stats.skew(p)), float(stats.kurtosis(p, fisher=True))
        z = float(stats.norm.ppf(1.0 - level))
        z_cf = (z + (z ** 2 - 1) * s / 6
                + (z ** 3 - 3 * z) * k / 24
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        var = -float(mu + z_cf * sd)
    else:
        raise RiskError(f"unknown method {method!r}; use 'historical' or 'cornish_fisher'")

    tail = p[p <= -var]
    cvar = -float(tail.mean()) if tail.size else var
    return var, max(cvar, var)


def max_drawdown(R, w) -> float:
    """Largest peak-to-trough decline of the fixed-weight return series."""
    p = _portfolio_series(R, w)
    equity = np.cumprod(1.0 + p)
    peak = np.maximum.accumulate(equity)
    return float((1.0 - equity / peak).max())


def betas(R, w, benchmarks) -> dict[str, float]:
    """Univariate OLS beta of the portfolio against each benchmark separately.

    Each benchmark is regressed on its own -- not jointly -- so the numbers
    answer "how much does the portfolio move with WIG20" and "...with GSPC"
    independently, rather than a multi-factor decomposition that would need
    the benchmarks themselves to be reasonably uncorrelated to be readable.
    """
    if not isinstance(benchmarks, pd.DataFrame):
        raise RiskError("benchmarks must be a DataFrame of return series")
    if isinstance(R, pd.DataFrame):
        common = R.index.intersection(benchmarks.index)
        if len(common) < 30:
            raise RiskError(f"only {len(common)} overlapping dates with the benchmarks")
        p = _portfolio_series(R.loc[common], w)
        bm = benchmarks.loc[common]
    else:
        p = _portfolio_series(R, w)
        bm = benchmarks
        if len(bm) != len(p):
            raise RiskError("benchmark length does not match the return matrix")

    out = {}
    for name in bm.columns:
        b = bm[name].to_numpy(dtype=float)
        var_b = float(np.var(b, ddof=0))
        if var_b <= 0:
            raise RiskError(f"benchmark {name} has zero variance")
        out[name] = float(np.cov(p, b, ddof=0)[0, 1] / var_b)
    return out


def exceedance_correlation(R, threshold: float = 1.0,
                           min_obs: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Correlation conditional on both assets falling below -threshold sigma.

    Returns (downside_corr, unconditional_corr). This is the Longin-Solnik /
    Ang-Bekaert diagnostic: it answers whether diversification survives a selloff.
    """
    arr = R.to_numpy(dtype=float) if isinstance(R, pd.DataFrame) else np.asarray(R, dtype=float)
    z = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=0)
    N = arr.shape[1]

    uncond = np.corrcoef(arr, rowvar=False)
    down = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            mask = (z[:, i] < -threshold) & (z[:, j] < -threshold)
            n = int(mask.sum())
            if n < min_obs:
                raise RiskError(
                    f"only {n} joint observations below -{threshold} sigma for "
                    f"assets {i} and {j}; at least {min_obs} required. "
                    "Lower the threshold or use a longer history."
                )
            c = float(np.corrcoef(arr[mask, i], arr[mask, j])[0, 1])
            down[i, j] = down[j, i] = c
    return down, uncond


def fx_risk_share(R, fx_returns, w) -> float:
    """Fraction of portfolio variance attributable to currency moves.

    ``R`` is the total (local-asset-move + FX) return matrix and
    ``fx_returns`` is the FX-only leg on the same weights; the share is the
    covariance of the FX-only series with the total series, over the total
    variance -- the beta of the currency leg onto the whole portfolio, which
    sums to 1 across a currency/local decomposition of the same total.
    """
    total = _portfolio_series(R, w)
    currency = _portfolio_series(fx_returns, w)
    var_total = float(np.var(total, ddof=0))
    if var_total <= 0:
        raise RiskError("portfolio variance is zero; FX share undefined")
    if float(np.var(currency, ddof=0)) == 0.0:
        return 0.0
    # Share of variance explained by the FX component, via its covariance with the total.
    return float(np.cov(total, currency, ddof=0)[0, 1] / var_total)
