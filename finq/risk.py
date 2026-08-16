"""Risk structure metrics: how much risk a portfolio runs, and where it comes from.

Every volatility figure returned by this module is annualized unless the caller
passes ``freq=None``, using sqrt(PERIODS_PER_YEAR[freq]) -- sqrt(252) daily,
sqrt(52) weekly. The annualization convention travels with the number in every
report header, because an un-annualized vol and an annualized one differ by a
factor of sixteen and both look plausible.
"""
from __future__ import annotations

import numpy as np

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
