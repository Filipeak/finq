from __future__ import annotations

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

    raise CovarianceError(f"method {method!r} not implemented yet")
