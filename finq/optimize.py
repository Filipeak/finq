"""Risk-based portfolio construction: equal weight, minimum variance, risk parity.

No expected-returns optimizer lives here or ever will (see project scope):
every solver below only consumes a covariance matrix, never a return forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class OptimizeError(Exception):
    """Raised when a weight vector cannot be produced."""


@dataclass(frozen=True)
class Constraints:
    max_weight: float | None = None
    min_weight: float | None = None


def _validate(Sigma) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise OptimizeError("Sigma must be a square matrix")
    if Sigma.shape[0] < 1:
        raise OptimizeError("need at least one asset")
    return Sigma


def _bounds(n: int, cap: float | None) -> list[tuple[float, float]]:
    if cap is None:
        return [(0.0, 1.0)] * n
    if cap * n < 1.0 - 1e-12:
        raise OptimizeError(
            f"infeasible: max_weight={cap} across {n} assets cannot sum to 1"
        )
    return [(0.0, cap)] * n


def _solve(objective, n: int, cap: float | None) -> np.ndarray:
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=_bounds(n, cap),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimizeError(f"optimizer did not converge: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def _apply_min_weight(solver, Sigma: np.ndarray, constraints: Constraints) -> np.ndarray:
    """Drop holdings below the floor and re-solve on the reduced set until stable.

    CORRECTED per ruling R4 (recorded during pre-flight, applied here directly
    because an earlier draft of this reference code left the bug in place):
    when one or zero holdings clear the floor, collapse to 100% in the single
    largest holding rather than returning the unfiltered sub-floor weights --
    the naive `full[active] = w_sub` fallback silently violates the very floor
    the caller asked for (test_min_weight_drops_dust_and_resolves's
    `w[2] == 0.0` would fail against it: min_variance on
    diag([0.0001, 0.04, 4.0]) puts ~0.9975/0.0025/0.000025 on the three assets,
    only asset 0 clears a 0.05 floor, and the unfiltered fallback would leave
    w[2] at ~0.000025 instead of exactly 0.0).
    """
    n = Sigma.shape[0]
    active = np.arange(n)
    while True:
        w_sub = solver(Sigma[np.ix_(active, active)], constraints.max_weight)
        keep = w_sub >= constraints.min_weight
        if keep.all():
            full = np.zeros(n)
            full[active] = w_sub
            return full
        if keep.sum() <= 1:
            full = np.zeros(n)
            full[active[np.argmax(w_sub)]] = 1.0
            return full
        active = active[keep]


def _dispatch(solver, Sigma, constraints: Constraints | None) -> np.ndarray:
    Sigma = _validate(Sigma)
    constraints = constraints or Constraints()
    if constraints.min_weight:
        return _apply_min_weight(solver, Sigma, constraints)
    return solver(Sigma, constraints.max_weight)


def equal_weight(Sigma) -> np.ndarray:
    """The baseline every other method must justify itself against."""
    n = _validate(Sigma).shape[0]
    return np.full(n, 1.0 / n)


def _min_variance(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    return _solve(lambda w: float(w @ Sigma @ w), Sigma.shape[0], cap)


def min_variance(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    return _dispatch(_min_variance, Sigma, constraints)


def _risk_parity(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    n = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-24)))
        rc = w * (Sigma @ w) / vol
        return float(((rc - rc.mean()) ** 2).sum())

    return _solve(objective, n, cap)


def risk_parity(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Equal risk contribution: every holding supplies the same share of total risk."""
    return _dispatch(_risk_parity, Sigma, constraints)
