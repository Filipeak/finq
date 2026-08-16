from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ("sample", "ledoit_wolf", "rmt_clean")


class CovarianceError(Exception):
    """Raised when a covariance matrix cannot be estimated meaningfully."""


@dataclass(frozen=True)
class Diagnostics:
    method: str
    T: int
    N: int
    Q: float
    lambda_minus: float
    lambda_plus: float
    eigenvalues: np.ndarray          # of the correlation matrix, DESCENDING
    n_in_band: int                   # eigenvalues at or below lambda_plus, i.e. noise
    var_share_in_band: float
    shrinkage: float | None
    condition_number: float


def _as_array(R) -> np.ndarray:
    arr = R.to_numpy(dtype=float) if isinstance(R, pd.DataFrame) else np.asarray(R, dtype=float)
    if arr.ndim != 2:
        raise CovarianceError("returns must be a 2-D T x N matrix")
    if not np.isfinite(arr).all():
        raise CovarianceError("returns contain NaN or inf; align and clean first")
    return arr


def sample_cov(R) -> np.ndarray:
    """Sample covariance with the MLE (divide by T) convention.

    Divide by T, not T-1: the Ledoit-Wolf (2003) asymptotics assume this
    normalization, and mixing conventions would quietly bias the shrinkage
    intensity.
    """
    arr = _as_array(R)
    Y = arr - arr.mean(axis=0)
    return (Y.T @ Y) / arr.shape[0]


def mp_band(Q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur edges: sigma^2 (1 + 1/Q +/- 2 sqrt(1/Q))."""
    if Q <= 0:
        raise CovarianceError(f"Q must be positive, got {Q}")
    root = 2.0 * np.sqrt(1.0 / Q)
    return sigma2 * (1.0 + 1.0 / Q - root), sigma2 * (1.0 + 1.0 / Q + root)


def _fit_bulk_sigma2(eigenvalues: np.ndarray, Q: float) -> float:
    """Fixed point: for an MP bulk with variance sigma^2, the bulk mean is sigma^2.

    Laloux et al. fit sigma^2 to the bulk rather than fixing it at 1, because the
    market eigenvalue absorbs a large share of the trace and leaves the remaining
    bulk well below unit variance.
    """
    sigma2 = 1.0
    for _ in range(100):
        _, hi = mp_band(Q, sigma2)
        bulk = eigenvalues[eigenvalues <= hi]
        if bulk.size == 0:
            break
        new = float(bulk.mean())
        if abs(new - sigma2) < 1e-10:
            sigma2 = new
            break
        sigma2 = new
    return sigma2


def _diagnostics(method: str, Sigma: np.ndarray, T: int, N: int,
                 shrinkage: float | None) -> Diagnostics:
    sd = np.sqrt(np.diag(Sigma))
    if not np.all(sd > 0):
        dead = [int(i) for i in np.flatnonzero(sd <= 0)]
        raise CovarianceError(
            f"asset(s) at column index {dead} have zero variance (constant returns); "
            "the correlation matrix is undefined. Drop the stale series first."
        )
    corr = Sigma / np.outer(sd, sd)
    evals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    Q = T / N
    sigma2 = _fit_bulk_sigma2(evals, Q)
    lo, hi = mp_band(Q, sigma2)
    # "In band" means at or below lambda_plus: those modes are indistinguishable
    # from noise. Eigenvalues below lambda_minus are noise too, not signal.
    in_band = evals <= hi
    return Diagnostics(
        method=method,
        T=T,
        N=N,
        Q=Q,
        lambda_minus=lo,
        lambda_plus=hi,
        eigenvalues=evals,
        n_in_band=int(in_band.sum()),
        var_share_in_band=float(evals[in_band].sum() / evals.sum()),
        shrinkage=shrinkage,
        condition_number=float(np.linalg.cond(Sigma)),
    )


def ledoit_wolf_cc(R) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2003) shrinkage toward the constant-correlation target.

    This is the CONSTANT-CORRELATION estimator of the 2003 "Honey, I Shrunk the
    Sample Covariance Matrix" paper (Appendices A and B), not the 2004
    scaled-identity version that sklearn.covariance.LedoitWolf implements. The
    target keeps the sample variances and replaces every correlation with the
    average sample correlation r-bar:

        f_ii = s_ii,   f_ij = r-bar * sqrt(s_ii * s_jj)

    Returns ``(Sigma, delta_hat)`` where ``delta_hat`` is the estimated optimal
    shrinkage intensity in [0, 1]. Sigma is positive definite by construction
    even when T <= N, because F is positive definite and S is PSD.
    """
    arr = _as_array(R)
    T, N = arr.shape
    if T < 2:
        raise CovarianceError("need at least two observations")
    if N < 2:
        raise CovarianceError("need at least two assets")
    if T <= N:
        # Spec section 9: refuse 'sample' here, permit shrinkage with a prominent
        # warning. The estimate is well defined and positive definite, but delta
        # is doing nearly all the work and the correlations are barely observed.
        warnings.warn(
            f"T={T} <= N={N}: the sample covariance is singular, so this estimate "
            "rests almost entirely on the constant-correlation target. Treat the "
            "resulting correlations as indicative only.",
            UserWarning,
            stacklevel=2,
        )

    # sample_cov is the single definition of the divide-by-T (MLE) convention;
    # the Ledoit-Wolf asymptotics below assume it, so do not inline a T-1 variant.
    Y = arr - arr.mean(axis=0)
    S = sample_cov(arr)
    var = np.diag(S).copy()
    if (var <= 0).any():
        dead = [int(i) for i in np.flatnonzero(var <= 0)]
        raise CovarianceError(
            f"asset(s) at column index {dead} have zero variance (constant returns); "
            "the average correlation is undefined. Drop the stale series first."
        )
    s = np.sqrt(var)

    if N == 2:
        # With a single off-diagonal correlation, its "average" is itself, so
        # the constant-correlation target is algebraically identical to S --
        # not approximately, exactly, by construction. gamma_hat = ||F-S||^2
        # is then 0 up to floating-point noise alone (~1e-34, not exactly 0
        # for most inputs), which turns delta into a coin flip on the sign of
        # that noise instead of the mathematically forced 0.0. Short-circuit
        # before the noise-sensitive ratio is ever formed.
        return S, 0.0

    corr = S / np.outer(s, s)
    iu = np.triu_indices(N, k=1)
    rbar = float(corr[iu].mean())

    # Shrinkage target F: sample variances, one common correlation rbar.
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, var)

    # pi-hat: summed asymptotic variances of the sample covariance entries.
    # (1/T) sum_t (Y_it Y_jt - s_ij)^2 collapses to (1/T) sum_t (Y_it Y_jt)^2 - s_ij^2
    # because s_ij is itself the mean of Y_it Y_jt.
    Y2 = Y ** 2
    pi_mat = (Y2.T @ Y2) / T - S ** 2
    pi_hat = float(pi_mat.sum())

    # theta[i, j] = theta-hat_{ii,ij} = (1/T) sum_t (Y_it^2 - s_ii)(Y_it Y_jt - s_ij)
    theta = (Y ** 3).T @ Y / T - var[:, None] * S

    # rho-hat: the diagonal terms of pi, plus the off-diagonal covariance between
    # the sample entries and the estimated target. theta.T supplies theta_{jj,ij},
    # and np.outer(1/s, s)[i, j] is s_j / s_i = sqrt(s_jj / s_ii).
    off = (rbar / 2.0) * (np.outer(1.0 / s, s) * theta + np.outer(s, 1.0 / s) * theta.T)
    np.fill_diagonal(off, 0.0)
    rho_hat = float(np.diag(pi_mat).sum() + off.sum())

    gamma_hat = float(((F - S) ** 2).sum())
    if gamma_hat <= 0:
        return S, 0.0            # sample already equals the target; nothing to shrink

    delta = float(np.clip(((pi_hat - rho_hat) / gamma_hat) / T, 0.0, 1.0))
    Sigma = delta * F + (1.0 - delta) * S
    Sigma = (Sigma + Sigma.T) / 2.0          # kill float asymmetry
    return Sigma, delta


def estimate(R, method: str = "ledoit_wolf") -> tuple[np.ndarray, Diagnostics]:
    if method not in METHODS:
        raise CovarianceError(f"unknown method {method!r}; use one of {METHODS}")
    arr = _as_array(R)
    T, N = arr.shape
    if N < 2:
        raise CovarianceError("need at least two assets")

    if method == "sample":
        if T <= N:
            raise CovarianceError(
                f"T={T} <= N={N}: the sample covariance matrix is singular here. "
                "Use method='ledoit_wolf' or 'rmt_clean'."
            )
        Sigma = sample_cov(arr)
        return Sigma, _diagnostics("sample", Sigma, T, N, None)

    if method == "ledoit_wolf":
        Sigma, delta = ledoit_wolf_cc(arr)
        return Sigma, _diagnostics("ledoit_wolf", Sigma, T, N, delta)

    raise CovarianceError(f"method {method!r} not implemented yet")
