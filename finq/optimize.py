"""Risk-based portfolio construction: equal weight, minimum variance, risk parity.

No expected-returns optimizer lives here or ever will (see project scope):
every solver below only consumes a covariance matrix, never a return forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


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


def _max_diversification(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    sd = np.sqrt(np.diag(Sigma))

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-24)))
        return -float((w @ sd) / vol)

    return _solve(objective, Sigma.shape[0], cap)


def max_diversification(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Choueifaty: maximize (w'sigma) / sqrt(w'Sigma w)."""
    return _dispatch(_max_diversification, Sigma, constraints)


def _quasi_diag(node) -> list[int]:
    """Depth-first leaf order of the clustering tree."""
    if node.is_leaf():
        return [node.get_id()]
    return _quasi_diag(node.get_left()) + _quasi_diag(node.get_right())


def _cluster_var(Sigma: np.ndarray, idx: list[int]) -> float:
    sub = Sigma[np.ix_(idx, idx)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def _hrp(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    """
    ADDED during Task 12 planning review (not in the original brief draft): HRP
    is the one optimizer in this module that isn't SLSQP-based, so it has no
    numerical-solver failure to fall back on if Sigma is degenerate. Every
    other method here either doesn't touch per-asset variance directly
    (min_variance, risk_parity, max_diversification all only ever see w'Sigma
    w as a whole) or fails loudly through non-convergence. HRP instead divides
    by per-asset sd to build a correlation matrix -- a zero-variance asset
    (e.g. a stale/constant price series, entirely plausible input from the
    rest of this toolkit) makes that division 0/0, and NaN then flows silently
    through _quasi_diag/_cluster_var into the final weights with no error at
    all: an analysis on a broken asset must never look like a successful HRP
    run. This mirrors the zero-variance guards already in
    finq.covariance.ledoit_wolf_cc/_diagnostics and is added here for the same
    reason, not a hypothetical.
    """
    n = Sigma.shape[0]
    if n == 1:
        return np.array([1.0])
    if not np.isfinite(Sigma).all():
        raise OptimizeError("Sigma contains non-finite values (NaN or inf)")

    var = np.diag(Sigma)
    if (var <= 0).any():
        dead = [int(i) for i in np.flatnonzero(var <= 0)]
        raise OptimizeError(
            f"asset(s) at column index {dead} have zero variance (constant returns); "
            "HRP's correlation-based clustering is undefined. Drop the stale series first."
        )
    sd = np.sqrt(var)
    corr = Sigma / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    tree = to_tree(linkage(squareform(dist, checks=False), method="single"))
    order = _quasi_diag(tree)

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[half:] if side else c[:half]
                    for c in clusters
                    if len(c) > 1
                    for half in (len(c) // 2,)
                    for side in (0, 1)]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_l, var_r = _cluster_var(Sigma, left), _cluster_var(Sigma, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    weights = weights / weights.sum()

    if cap is not None:
        if cap * n < 1.0 - 1e-12:
            raise OptimizeError(
                f"infeasible: max_weight={cap} across {n} assets cannot sum to 1"
            )
        # Water-fill: trim the excess above the cap and redistribute it into the
        # remaining headroom. Clipping and renormalizing would push weights back
        # above the cap, so it must iterate.
        for _ in range(100):
            excess = float(np.clip(weights - cap, 0.0, None).sum())
            if excess <= 1e-12:
                break
            weights = np.minimum(weights, cap)
            room = cap - weights
            if room.sum() <= 1e-12:
                break
            weights = weights + excess * room / room.sum()
    return weights


def hrp(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado). Never inverts Sigma."""
    return _dispatch(_hrp, Sigma, constraints)


ALL_METHODS = {
    "equal_weight": lambda S, c: equal_weight(S),
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_diversification": max_diversification,
    "hrp": hrp,
}


def compare(Sigma, tickers: list[str], methods: list[str] | None = None,
            constraints: Constraints | None = None) -> pd.DataFrame:
    """Run every method side by side. Dispersion across methods is itself the finding:
    wide disagreement means the covariance estimate is unstable."""
    Sigma = _validate(Sigma)
    if len(tickers) != Sigma.shape[0]:
        raise OptimizeError(
            f"{len(tickers)} tickers but Sigma is {Sigma.shape[0]}x{Sigma.shape[0]}"
        )
    names = methods or list(ALL_METHODS)
    unknown = set(names) - set(ALL_METHODS)
    if unknown:
        raise OptimizeError(f"unknown method(s): {', '.join(sorted(unknown))}")

    out = pd.DataFrame(index=tickers)
    for name in names:
        out[name] = ALL_METHODS[name](Sigma, constraints)
    out["dispersion"] = out[names].max(axis=1) - out[names].min(axis=1)
    return out
