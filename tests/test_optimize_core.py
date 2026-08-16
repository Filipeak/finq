import numpy as np
import pytest
from finq.optimize import (equal_weight, min_variance, risk_parity,
                           Constraints, OptimizeError)
from finq.risk import risk_contributions, portfolio_vol


def test_equal_weight_is_one_over_N():
    np.testing.assert_allclose(equal_weight(np.eye(5)), np.full(5, 0.2))


def test_weights_are_long_only_and_sum_to_one():
    rng = np.random.default_rng(60)
    A = rng.normal(size=(6, 6))
    Sigma = A @ A.T / 6
    for solver in (min_variance, risk_parity):
        w = solver(Sigma)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()


def test_min_variance_two_asset_matches_closed_form():
    """w1 = (v2 - cov) / (v1 + v2 - 2 cov) for the unconstrained two-asset case."""
    v1, v2, cov = 0.04, 0.09, 0.01
    Sigma = np.array([[v1, cov], [cov, v2]])
    expected = (v2 - cov) / (v1 + v2 - 2 * cov)
    np.testing.assert_allclose(min_variance(Sigma), [expected, 1 - expected], atol=1e-6)


def test_min_variance_beats_equal_weight_on_variance():
    rng = np.random.default_rng(61)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    assert (portfolio_vol(Sigma, min_variance(Sigma), freq=None)
            <= portfolio_vol(Sigma, equal_weight(Sigma), freq=None) + 1e-12)


def test_min_variance_concentrates_in_the_low_vol_asset():
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = min_variance(Sigma)
    assert w[0] > 0.95


def test_risk_parity_equalizes_risk_contributions():
    rng = np.random.default_rng(62)
    A = rng.normal(size=(7, 7))
    Sigma = A @ A.T / 7
    _, pct = risk_contributions(Sigma, risk_parity(Sigma))
    np.testing.assert_allclose(pct, np.full(7, 1 / 7), atol=1e-4)


def test_risk_parity_on_uncorrelated_assets_is_inverse_volatility():
    Sigma = np.diag([0.01, 0.04, 0.09])
    inv_vol = 1 / np.sqrt(np.diag(Sigma))
    np.testing.assert_allclose(risk_parity(Sigma), inv_vol / inv_vol.sum(), atol=1e-5)


def test_max_weight_cap_is_respected():
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = min_variance(Sigma, Constraints(max_weight=0.5))
    assert w.max() <= 0.5 + 1e-8
    assert w.sum() == pytest.approx(1.0)


def test_min_weight_drops_dust_and_resolves():
    Sigma = np.diag([0.0001, 0.04, 4.0])
    w = min_variance(Sigma, Constraints(min_weight=0.05))
    assert w[2] == 0.0                      # dust dropped entirely
    assert w.sum() == pytest.approx(1.0)
    assert ((w == 0.0) | (w >= 0.05 - 1e-9)).all()


def test_infeasible_max_weight_fails_loudly():
    with pytest.raises(OptimizeError, match="infeasible"):
        min_variance(np.eye(5), Constraints(max_weight=0.1))   # 5 * 0.1 < 1


# --------------------------------------------------------------------------
# _apply_min_weight collapse boundary (ruling R4): keep.sum() <= 1 must
# collapse to 100% in the single largest holding, whether zero holdings clear
# the floor or exactly one does. A buggy `full[active] = w_sub` fallback would
# leave sub-floor dust in place instead of zeroing it out -- both cases below
# would catch that regression, but only from different starting keep-counts.
# --------------------------------------------------------------------------

def test_min_weight_collapses_to_single_holding_when_exactly_one_clears_floor():
    """keep.sum() == 1: same fixture as the brief's dust-drop test, but pinned
    to the exact hand-computed inverse-variance weights so a partial (not
    fully collapsed) result would be caught, not just the fact that w[2] == 0.
    Unconstrained min-variance on diag([0.0001, 0.04, 4.0]) is inverse-variance
    weighting: w = (1/var) / sum(1/var) = [0.99748136, 0.00249370, 0.00002494].
    Only asset 0 clears a 0.05 floor.
    """
    Sigma = np.diag([0.0001, 0.04, 4.0])
    w = min_variance(Sigma, Constraints(min_weight=0.05))
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0], atol=1e-9)


def test_min_weight_collapses_to_single_holding_when_zero_clear_floor():
    """keep.sum() == 0: min_weight set above the largest unconstrained weight
    (0.99748...), so NO holding clears the floor on the first pass. The
    corrected code must still collapse to the single largest holding at 100%,
    not return the tiny unfiltered sub-floor weights (the R4 bug) and not
    raise/return an all-zero vector that fails to sum to 1.
    """
    Sigma = np.diag([0.0001, 0.04, 4.0])
    w = min_variance(Sigma, Constraints(min_weight=0.998))
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0], atol=1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_min_weight_resolve_loop_reindexes_the_surviving_active_set():
    """Exercises the actual re-solve loop (not just single-pass collapse):
    the dropped dust asset sits in the MIDDLE of the index range, so a bug in
    mapping `active` back onto `full` (e.g. assuming the survivors are a
    contiguous prefix starting at 0) would misplace weight.

    variances = [0.01, 100.0, 0.01, 0.01] -> pass 1 inverse-variance weights
    are [0.33332, 0.0000333, 0.33332, 0.33332]; asset 1 (huge variance, tiny
    weight) is dropped by a 0.05 floor, leaving active = [0, 2, 3]. Re-solving
    on that reduced diag([0.01, 0.01, 0.01]) submatrix gives equal weights
    1/3 each, all above the floor -> stable. Expected final vector places
    1/3 at indices 0, 2, 3 and exactly 0 at index 1.
    """
    Sigma = np.diag([0.01, 100.0, 0.01, 0.01])
    w = min_variance(Sigma, Constraints(min_weight=0.05))
    np.testing.assert_allclose(w, [1 / 3, 0.0, 1 / 3, 1 / 3], atol=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_combined_max_and_min_weight_constraints_hold_together():
    Sigma = np.diag([0.0001, 0.01, 0.02, 0.5])
    w = min_variance(Sigma, Constraints(max_weight=0.6, min_weight=0.05))
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= 0.6 + 1e-8
    assert ((w == 0.0) | (w >= 0.05 - 1e-9)).all()


# --------------------------------------------------------------------------
# max_weight cap: exact water-filling literal, not just an inequality.
# --------------------------------------------------------------------------

def test_max_weight_cap_water_fills_the_remaining_budget_by_inverse_variance():
    """diag([0.0001, 0.04, 0.09]) capped at 0.5: asset 0 wants ~1.0 unconstrained
    and hits the 0.5 cap; the remaining 0.5 budget then splits between assets
    1 and 2 by the SAME inverse-variance rule (neither hits the cap):
        w1 = 0.5 * (1/0.04) / (1/0.04 + 1/0.09) = 0.34615384...
        w2 = 0.5 * (1/0.09) / (1/0.04 + 1/0.09) = 0.15384615...
    An inequality-only check (w.max() <= 0.5) cannot distinguish this from,
    say, an even 0.5/0.25/0.25 split.
    """
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = min_variance(Sigma, Constraints(max_weight=0.5))
    np.testing.assert_allclose(w, [0.5, 0.34615385, 0.15384615], atol=1e-6)


# --------------------------------------------------------------------------
# _bounds infeasibility boundary: `cap * n < 1.0 - 1e-12`. Test exactly on
# the boundary (must NOT raise), just outside the float-tolerance slack
# (must raise), and just inside it (must NOT raise) -- an interior-only test
# (e.g. cap=0.1, n=5) cannot distinguish the tolerance from a plain `<`.
# --------------------------------------------------------------------------

def test_bounds_at_exact_feasibility_boundary_does_not_raise():
    from finq.optimize import _bounds
    # 5 * 0.2 == 1.0 exactly: must be feasible.
    bounds = _bounds(5, 0.2)
    assert bounds == [(0.0, 0.2)] * 5


def test_bounds_just_outside_the_float_tolerance_raises():
    from finq.optimize import _bounds
    # 5 * (0.2 - 1e-10) = 0.9999999995, i.e. 1.0 - 5e-10 -- outside the
    # 1e-12 slack, so this must be rejected as infeasible.
    with pytest.raises(OptimizeError, match="infeasible"):
        _bounds(5, 0.2 - 1e-10)


def test_bounds_just_inside_the_float_tolerance_does_not_raise():
    from finq.optimize import _bounds
    # 3 * cap = 1.0 - 1e-13, which sits INSIDE the 1e-12 slack. A mutation
    # that dropped the "- 1e-12" tolerance entirely (bare `cap * n < 1.0`)
    # would wrongly raise here; the correct implementation must not.
    cap = (1.0 - 1e-13) / 3
    bounds = _bounds(3, cap)
    assert bounds == [(0.0, cap)] * 3


# --------------------------------------------------------------------------
# risk_parity: tight, hand-built asymmetric-vol-and-correlation fixture.
# The brief's own risk-parity test is a single RNG draw (seed 62) checked at
# atol=1e-4; that passes even if the optimizer stops somewhat short of the
# true equal-contribution optimum. This pins a second, deterministic fixture
# an order of magnitude tighter, with genuinely different vols and non-zero,
# non-uniform, sign-mixed correlations (so no accidental symmetry lets a
# wrong/partial solution look correct by luck).
# --------------------------------------------------------------------------

def test_risk_parity_equalizes_contributions_with_asymmetric_vols_and_correlation():
    vols = np.array([0.10, 0.20, 0.35])
    corr = np.array([
        [1.00, 0.30, -0.10],
        [0.30, 1.00, 0.40],
        [-0.10, 0.40, 1.00],
    ])
    Sigma = corr * np.outer(vols, vols)
    w = risk_parity(Sigma)
    _, pct = risk_contributions(Sigma, w)
    np.testing.assert_allclose(pct, np.full(3, 1 / 3), atol=1e-6)
    # Equal risk contribution with unequal vols implies unequal weights --
    # a degenerate equal-weight solution would trivially fail the pct check
    # above too, but this pins the qualitative shape as well.
    assert w.min() > 0.01
    assert not np.allclose(w, w[0])


def test_risk_parity_second_random_seed_holds_tightly():
    """A different RNG seed from the brief's (62), at a tighter tolerance, so
    a solver that only happens to converge well for one particular seed is
    still caught.
    """
    rng = np.random.default_rng(123)
    A = rng.normal(size=(5, 5))
    Sigma = A @ A.T / 5
    _, pct = risk_contributions(Sigma, risk_parity(Sigma))
    np.testing.assert_allclose(pct, np.full(5, 1 / 5), atol=1e-5)


# --------------------------------------------------------------------------
# validation gate
# --------------------------------------------------------------------------

def test_equal_weight_single_asset_is_fully_invested():
    np.testing.assert_allclose(equal_weight(np.array([[0.04]])), [1.0])


def test_every_solver_rejects_a_non_square_sigma():
    for fn in (equal_weight, min_variance, risk_parity):
        with pytest.raises(OptimizeError, match="square"):
            fn(np.ones((2, 3)))
