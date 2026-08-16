import numpy as np
import pandas as pd
import pytest
from finq.optimize import (max_diversification, hrp, compare, equal_weight,
                           min_variance, Constraints, OptimizeError)
from finq.risk import diversification_ratio


def test_max_diversification_beats_equal_weight_on_the_ratio():
    rng = np.random.default_rng(70)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    assert (diversification_ratio(Sigma, max_diversification(Sigma))
            >= diversification_ratio(Sigma, equal_weight(Sigma)) - 1e-9)


def test_max_diversification_weights_are_valid():
    rng = np.random.default_rng(71)
    A = rng.normal(size=(6, 6))
    w = max_diversification(A @ A.T / 6)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_hrp_weights_are_valid():
    rng = np.random.default_rng(72)
    A = rng.normal(size=(10, 10))
    w = hrp(A @ A.T / 10)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_hrp_tilts_toward_the_low_volatility_asset():
    """With an identity correlation matrix the leaf ORDER is arbitrary, so assert
    only what the allocation logic guarantees: least volatile gets the most."""
    Sigma = np.diag([0.01, 0.04, 0.09, 0.16])
    w = hrp(Sigma)
    assert w.argmax() == 0
    assert w[0] > w[3]


def test_hrp_splits_evenly_between_two_identical_clusters():
    """Two tight blocks of two assets each; HRP should give each block half."""
    block = np.array([[1.0, 0.95], [0.95, 1.0]])
    C = np.eye(4) * 0.0
    C[:2, :2] = block
    C[2:, 2:] = block
    Sigma = C * 0.04
    w = hrp(Sigma)
    assert w[:2].sum() == pytest.approx(0.5, abs=0.02)


def test_hrp_never_inverts_the_covariance_matrix():
    """A singular matrix breaks min_variance but must not break HRP."""
    rng = np.random.default_rng(73)
    R = rng.normal(size=(5, 12))
    Sigma = np.cov(R, rowvar=False)          # rank-deficient by construction
    w = hrp(Sigma)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)


def test_compare_returns_one_column_per_method_plus_dispersion():
    rng = np.random.default_rng(74)
    A = rng.normal(size=(5, 5))
    Sigma = A @ A.T / 5
    tickers = ["A", "B", "C", "D", "E"]
    df = compare(Sigma, tickers)
    assert list(df.index) == tickers
    for m in ("equal_weight", "min_variance", "risk_parity",
              "max_diversification", "hrp"):
        assert m in df.columns
        assert df[m].sum() == pytest.approx(1.0)
    assert "dispersion" in df.columns
    assert (df["dispersion"] >= 0).all()


def test_compare_dispersion_is_max_minus_min_across_methods():
    rng = np.random.default_rng(75)
    A = rng.normal(size=(4, 4))
    df = compare(A @ A.T / 4, ["A", "B", "C", "D"])
    methods = [c for c in df.columns if c != "dispersion"]
    expected = df[methods].max(axis=1) - df[methods].min(axis=1)
    np.testing.assert_allclose(df["dispersion"].to_numpy(), expected.to_numpy())


def test_compare_honors_constraints():
    rng = np.random.default_rng(76)
    A = rng.normal(size=(6, 6))
    df = compare(A @ A.T / 6, list("ABCDEF"), constraints=Constraints(max_weight=0.4))
    for m in ("min_variance", "risk_parity", "max_diversification"):
        assert df[m].max() <= 0.4 + 1e-6


# --------------------------------------------------------------------------
# HRP zero-variance / non-finite guard (added during Task 12 planning review):
# HRP is the only optimizer here that divides by per-asset sd to build a
# correlation matrix, so a stale/constant-price asset (variance == 0) or a
# NaN/inf entry must fail loudly, not silently produce NaN weights. This is a
# reachable spec-9 violation, not a hypothetical -- mirrors the guards already
# in finq.covariance.ledoit_wolf_cc/_diagnostics.
# --------------------------------------------------------------------------

def test_hrp_raises_on_zero_variance_asset():
    """A constant-price asset (zero variance) must be named in the error, not
    silently produce NaN weights via 0/0 in the correlation-matrix division."""
    Sigma = np.diag([0.01, 0.0, 0.09])
    with pytest.raises(OptimizeError, match=r"index \[1\]"):
        hrp(Sigma)


def test_hrp_raises_on_multiple_zero_variance_assets():
    Sigma = np.diag([0.0, 0.04, 0.0, 0.09])
    with pytest.raises(OptimizeError, match=r"index \[0, 2\]"):
        hrp(Sigma)


def test_hrp_raises_on_non_finite_sigma():
    Sigma = np.diag([0.01, 0.04, 0.09])
    Sigma[0, 1] = Sigma[1, 0] = np.nan
    with pytest.raises(OptimizeError, match="non-finite"):
        hrp(Sigma)


# --------------------------------------------------------------------------
# hrp() single-asset edge case: the n == 1 early return.
# --------------------------------------------------------------------------

def test_hrp_single_asset_is_fully_invested():
    np.testing.assert_allclose(hrp(np.array([[0.04]])), [1.0])


# --------------------------------------------------------------------------
# Odd-length cluster split: len(c) // 2 divides a 3-asset cluster asymmetrically
# (1 vs 2), unlike the brief's own fixtures which are all even (4 and 10
# assets). Hand-derived expected weights below (cross-checked independently
# with a standalone script implementing the same formulas, not by calling
# _hrp) pin the exact split, not just a qualitative "less risky gets more."
#
# Fixture: A, B tightly correlated (corr 0.9), C uncorrelated with both,
# variances [0.01, 0.04, 0.09]. Single-linkage clustering merges A,B first
# (distance sqrt(0.05) ~= 0.2236) and only then merges {A,B} with C (distance
# sqrt(0.5) ~= 0.7071); the to_tree/quasi-diag order comes out [C, A, B]
# (lower original leaf index sorts first in the linkage matrix's merge row).
# n == 3 -> half = 1, so the root split is left = [C] (1 asset) vs
# right = [A, B] (2 assets) -- exactly the asymmetric 1-vs-2 split this test
# targets.
#
#   var_l = Sigma[C,C] = 0.09
#   var_r = inverse-variance-weighted variance of {A,B}
#         = 0.8^2*0.01 + 0.2^2*0.04 + 2*0.8*0.2*0.018 = 0.01376
#   alpha = 1 - var_l/(var_l+var_r) = 1 - 0.09/0.10376 = 0.132613724...
#   weights after root split (unnormalized): C = alpha, {A,B} = 1 - alpha
#
#   Second split (of {A,B}): var_l = Sigma[A,A] = 0.01, var_r = Sigma[B,B] = 0.04
#   alpha2 = 1 - 0.01/0.05 = 0.8
#   A = (1 - alpha) * alpha2, B = (1 - alpha) * (1 - alpha2)
#
# giving final weights [A, B, C] = [0.6939090208, 0.1734772552, 0.132613724].
# --------------------------------------------------------------------------

def test_hrp_three_asset_odd_cluster_exact_split():
    v = np.array([0.01, 0.04, 0.09])
    corr = np.array([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    Sigma = corr * np.sqrt(np.outer(v, v))
    w = hrp(Sigma)
    np.testing.assert_allclose(
        w, [0.6939090208, 0.1734772552, 0.132613724], atol=1e-8
    )


def test_hrp_five_asset_simultaneous_sibling_splits_exact_weights():
    """5 assets, two internal clusters {0,1,2} (tight, corr 0.8) and {3,4}
    (tight, corr 0.8), independent of each other. This is a stronger check on
    the clusters-list adjacency-pairing assumption than the 3-asset test
    above: the root splits 2-vs-3 (another odd/asymmetric len(c)//2 split),
    and the SECOND round processes two sibling parent clusters in the very
    same iteration ([2,0] -> [[2],[0]] and [1,3,4] -> [[1],[3,4]] both split
    in one pass, giving clusters == [[2],[0],[1],[3,4]]). A pairing bug that
    mismatched clusters[i]/clusters[i+1] across different parents (rather
    than within one parent's own left/right halves) would corrupt exactly
    this shape and not the simpler single-split-per-round 3-asset case.
    Expected weights were cross-checked with a standalone reimplementation of
    the documented formulas (var_l, var_r via inverse-variance cluster
    weighting; alpha = 1 - var_l/(var_l+var_r)), not by calling _hrp.
    """
    v5 = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    corr5 = np.eye(5)
    for i in range(3):
        for j in range(3):
            if i != j:
                corr5[i, j] = 0.8
    corr5[3, 4] = corr5[4, 3] = 0.8
    Sigma5 = corr5 * np.sqrt(np.outer(v5, v5))
    w = hrp(Sigma5)
    np.testing.assert_allclose(
        w,
        [0.3997508575, 0.31104645, 0.1332502858, 0.0866402259, 0.0693121808],
        atol=1e-8,
    )


# --------------------------------------------------------------------------
# HRP's own cap-infeasibility check and water-fill loop. This duplicates
# (rather than reuses) the `_bounds` infeasibility check from Task 11, so it
# needs its own boundary coverage: exactly on the boundary must not raise,
# just outside the float tolerance must raise, just inside must not.
# --------------------------------------------------------------------------

def test_hrp_cap_at_exact_feasibility_boundary_forces_equal_weights():
    """cap * n == 1.0 exactly leaves only one feasible point: every weight
    pinned at cap. This is a stronger check than an inequality -- it pins the
    water-fill loop's fixed point exactly, regardless of what the uncapped
    HRP split would have been.
    """
    Sigma = np.diag([0.01, 0.04, 0.09, 0.16])
    w = hrp(Sigma, Constraints(max_weight=0.25))
    np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25], atol=1e-9)


def test_hrp_cap_just_outside_the_float_tolerance_raises():
    Sigma = np.diag([0.01, 0.04, 0.09, 0.16, 0.25])
    with pytest.raises(OptimizeError, match="infeasible"):
        hrp(Sigma, Constraints(max_weight=0.2 - 1e-10))


def test_hrp_cap_just_inside_the_float_tolerance_does_not_raise():
    cap = (1.0 - 1e-13) / 3
    Sigma = np.diag([0.01, 0.04, 0.09])
    w = hrp(Sigma, Constraints(max_weight=cap))
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= cap + 1e-9


def test_hrp_cap_water_fill_redistributes_excess_from_a_dominant_asset():
    """One asset would dominate uncapped (near-zero variance vs the rest);
    capping it must push the excess into the others while respecting the cap
    everywhere, and weights must still sum to 1."""
    Sigma = np.diag([0.0001, 0.04, 0.09, 0.16])
    w = hrp(Sigma, Constraints(max_weight=0.4))
    assert w.max() <= 0.4 + 1e-8
    assert w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# max_diversification: an exact closed-form case (identity Sigma -> minimizing
# ||w||_2 subject to sum(w)==1, w>=0 is uniform by Cauchy-Schwarz) so this
# pins a scale, not just an inequality against equal_weight.
# --------------------------------------------------------------------------

def test_max_diversification_on_identity_matches_equal_weight_exactly():
    n = 5
    w = max_diversification(np.eye(n))
    np.testing.assert_allclose(w, np.full(n, 1.0 / n), atol=1e-6)


def test_max_diversification_respects_max_weight_cap():
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = max_diversification(Sigma, Constraints(max_weight=0.5))
    assert w.max() <= 0.5 + 1e-8
    assert w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# compare(): unknown-method error path, and methods=[] / single-method edges.
# --------------------------------------------------------------------------

ALL_METHOD_NAMES = ("equal_weight", "min_variance", "risk_parity",
                     "max_diversification", "hrp")


def test_compare_unknown_method_raises():
    Sigma = np.eye(3)
    with pytest.raises(OptimizeError, match="unknown method"):
        compare(Sigma, list("ABC"), methods=["not_a_real_method"])


def test_compare_single_method_has_zero_dispersion():
    """Dispersion is max-minus-min ACROSS the selected methods; with only one
    method selected there is nothing to disperse against, so it must be
    exactly zero everywhere, not merely small."""
    rng = np.random.default_rng(77)
    A = rng.normal(size=(4, 4))
    df = compare(A @ A.T / 4, list("ABCD"), methods=["min_variance"])
    assert list(df.columns) == ["min_variance", "dispersion"]
    np.testing.assert_allclose(df["dispersion"].to_numpy(), np.zeros(4))


def test_compare_empty_methods_list_falls_back_to_default_all_methods():
    """DOCUMENTS current behavior: `methods or list(ALL_METHODS)` treats an
    explicitly-passed empty list the same as methods=None, because [] is
    falsy in Python. This is a real ambiguity worth flagging (an empty
    selection silently becomes 'run everything' instead of 'run nothing' or
    raising) -- pinned here as a regression guard on the documented behavior,
    not an endorsement that it is the ideal API.
    """
    rng = np.random.default_rng(78)
    A = rng.normal(size=(3, 3))
    df = compare(A @ A.T / 3, list("ABC"), methods=[])
    assert set(df.columns) == set(ALL_METHOD_NAMES) | {"dispersion"}


def test_compare_row_index_matches_tickers_not_default_range():
    df = compare(np.eye(3), ["X", "Y", "Z"], methods=["equal_weight"])
    assert list(df.index) == ["X", "Y", "Z"]
    assert not isinstance(df.index, pd.RangeIndex)


def test_compare_ticker_count_mismatch_raises():
    with pytest.raises(OptimizeError):
        compare(np.eye(3), ["X", "Y"])
