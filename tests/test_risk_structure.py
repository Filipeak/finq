import numpy as np
import pytest
from finq.risk import (portfolio_vol, risk_contributions, diversification_ratio,
                       effective_bets, concentration, RiskError)

# A deliberately asymmetric fixture: three different variances, three different
# weights, one negative covariance. Every expected value below is arithmetic done
# by hand off these numbers, not a value read back out of the implementation.
#
#   Sigma @ W   = [0.0214, 0.0324, 0.0346]
#   W' Sigma W  = 0.02734
#   sqrt(diag)  = [0.2, 0.3, 0.4]        W . sqrt(diag) = 0.27
SIGMA = np.array([[0.040, 0.006, -0.002],
                  [0.006, 0.090, 0.012],
                  [-0.002, 0.012, 0.160]])
W = np.array([0.5, 0.3, 0.2])
SIGMA_W = np.array([0.0214, 0.0324, 0.0346])
QUAD = 0.02734


# --------------------------------------------------------------------------
# portfolio_vol
# --------------------------------------------------------------------------

def test_equal_weight_uncorrelated_unit_vol_gives_one_over_sqrt_N():
    """The canonical analytic identity for diversification."""
    for N in (4, 9, 16, 25):
        Sigma = np.eye(N)
        w = np.full(N, 1.0 / N)
        assert portfolio_vol(Sigma, w, freq=None) == pytest.approx(1.0 / np.sqrt(N))


def test_perfectly_correlated_assets_give_no_diversification():
    Sigma = np.full((5, 5), 0.04)          # every pairwise correlation is 1
    w = np.full(5, 0.2)
    assert portfolio_vol(Sigma, w, freq=None) == pytest.approx(0.2)


def test_portfolio_vol_uses_the_full_quadratic_form():
    """Binds on every off-diagonal term and on each individual weight.

    The equal-weight/identity cases above still pass if Sigma's off-diagonals
    are ignored, or if the weights are replaced by 1/N. This one does not.
    """
    assert portfolio_vol(SIGMA, W, freq=None) == pytest.approx(np.sqrt(QUAD))
    # diag-only would give sqrt(0.5^2*0.04 + 0.3^2*0.09 + 0.2^2*0.16) = sqrt(0.0243)
    assert portfolio_vol(SIGMA, W, freq=None) != pytest.approx(np.sqrt(0.0243))
    # equal weights would give sqrt(0.0356)
    assert portfolio_vol(SIGMA, np.full(3, 1 / 3), freq=None) != pytest.approx(np.sqrt(QUAD))


def test_annualization_uses_sqrt_252_daily_and_sqrt_52_weekly():
    Sigma = np.eye(2) * 0.0001
    w = np.array([0.5, 0.5])
    raw = portfolio_vol(Sigma, w, freq=None)
    assert portfolio_vol(Sigma, w, freq="daily") == pytest.approx(raw * np.sqrt(252))
    assert portfolio_vol(Sigma, w, freq="weekly") == pytest.approx(raw * np.sqrt(52))


def test_daily_is_the_default_frequency():
    assert portfolio_vol(SIGMA, W) == pytest.approx(np.sqrt(QUAD) * np.sqrt(252))


def test_rejects_unknown_frequency():
    with pytest.raises(RiskError, match="freq"):
        portfolio_vol(SIGMA, W, freq="monthly")


# --------------------------------------------------------------------------
# risk_contributions
# --------------------------------------------------------------------------

def test_risk_contributions_sum_to_portfolio_volatility():
    rng = np.random.default_rng(30)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    w = rng.random(8)
    w = w / w.sum()
    mctr, pct = risk_contributions(Sigma, w)
    assert (w * mctr).sum() == pytest.approx(portfolio_vol(Sigma, w, freq=None))
    assert pct.sum() == pytest.approx(1.0)


def test_percentage_risk_contributions_sum_to_exactly_one():
    """The Euler property must hold for wildly heterogeneous inputs too."""
    rng = np.random.default_rng(31)
    for _ in range(5):
        A = rng.normal(size=(12, 12)) * rng.uniform(0.5, 4.0, size=(1, 12))
        Sigma = A @ A.T / 12
        w = rng.random(12) ** 3            # heavily skewed, far from equal weight
        w = w / w.sum()
        _, pct = risk_contributions(Sigma, w)
        assert pct.sum() == pytest.approx(1.0, abs=1e-12)


def test_equal_weight_identity_gives_equal_risk_contributions():
    Sigma = np.eye(6)
    w = np.full(6, 1 / 6)
    _, pct = risk_contributions(Sigma, w)
    np.testing.assert_allclose(pct, np.full(6, 1 / 6))


def test_mctr_is_sigma_w_over_portfolio_vol():
    """Pins the MCTR formula itself, not just the sum property.

    pct summing to 1 is invariant to any mctr that is proportional to Sigma@w,
    so it alone does not fix the scale. These literals do.
    """
    mctr, _ = risk_contributions(SIGMA, W)
    np.testing.assert_allclose(mctr, SIGMA_W / np.sqrt(QUAD), rtol=1e-12)
    np.testing.assert_allclose(mctr, [0.12942391, 0.19595022, 0.20925548], rtol=1e-7)


def test_risk_share_diverges_from_weight_share():
    """The whole point of the metric: the 20% position carries 25% of the risk."""
    _, pct = risk_contributions(SIGMA, W)
    np.testing.assert_allclose(pct, (W * SIGMA_W) / QUAD, rtol=1e-12)
    np.testing.assert_allclose(pct, [0.39136796, 0.35552304, 0.25310900], rtol=1e-7)
    assert pct[2] > W[2]        # smallest weight, largest risk share
    assert pct[0] < W[0]        # largest weight, smaller risk share
    assert pct.sum() == pytest.approx(1.0)


def test_risk_contributions_reject_a_zero_volatility_portfolio():
    with pytest.raises(RiskError, match="zero"):
        risk_contributions(np.zeros((3, 3)), W)


# --------------------------------------------------------------------------
# diversification_ratio
# --------------------------------------------------------------------------

def test_diversification_ratio_is_one_for_a_single_asset():
    assert diversification_ratio(np.array([[0.04]]), np.array([1.0])) == pytest.approx(1.0)


def test_diversification_ratio_exceeds_one_for_imperfect_correlation():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.04]])
    assert diversification_ratio(Sigma, np.array([0.5, 0.5])) > 1.0


def test_diversification_ratio_is_one_when_perfectly_correlated():
    Sigma = np.full((3, 3), 0.04)
    assert diversification_ratio(Sigma, np.full(3, 1 / 3)) == pytest.approx(1.0)


def test_diversification_ratio_weights_standalone_vols_not_variances():
    """(w . sd) / sigma_p, on a fixture where sd and variance differ visibly.

    Using diag(Sigma) instead of sqrt(diag(Sigma)) gives 0.0574/0.16535 = 0.347,
    so this literal catches the most likely single-character error.
    """
    assert diversification_ratio(SIGMA, W) == pytest.approx(0.27 / np.sqrt(QUAD))
    assert diversification_ratio(SIGMA, W) == pytest.approx(1.63291850, abs=1e-7)


def test_diversification_ratio_is_at_least_one_for_long_only_weights():
    rng = np.random.default_rng(32)
    for _ in range(20):
        A = rng.normal(size=(10, 10)) * rng.uniform(0.3, 3.0, size=(1, 10))
        Sigma = A @ A.T / 10
        w = rng.random(10)
        w = w / w.sum()
        assert diversification_ratio(Sigma, w) >= 1.0 - 1e-12


def test_diversification_ratio_rejects_a_zero_volatility_portfolio():
    with pytest.raises(RiskError, match="zero"):
        diversification_ratio(np.zeros((3, 3)), W)


# --------------------------------------------------------------------------
# effective_bets
# --------------------------------------------------------------------------

def test_effective_bets_equals_N_for_equal_weighted_uncorrelated():
    Sigma = np.eye(10)
    assert effective_bets(Sigma, np.full(10, 0.1)) == pytest.approx(10.0)


def test_effective_bets_is_one_for_a_single_factor():
    Sigma = np.full((8, 8), 0.04) + np.eye(8) * 1e-12
    assert effective_bets(Sigma, np.full(8, 0.125)) == pytest.approx(1.0, abs=1e-4)


def test_effective_bets_weights_each_principal_portfolio_by_its_eigenvalue():
    """Diagonal Sigma with three different variances and three different weights.

    The eigenbasis is then the standard basis and the answer is analytic:
    contributions (0.0100, 0.0081, 0.0100) -> p = (0.35587, 0.28826, 0.35587)
    -> exp(entropy) = 2.98597. Dropping the eigenvalue factor, or the square on
    w_tilde, moves this number; N=3 and equal weights would not.
    """
    Sigma = np.diag([0.04, 0.09, 0.25])
    assert effective_bets(Sigma, W) == pytest.approx(2.98597442, abs=1e-7)
    assert 1.0 < effective_bets(Sigma, W) < 3.0


def test_effective_bets_uses_the_eigenbasis_not_the_asset_basis():
    """The rotation is the whole method, so it needs a fixture that feels it.

    Every other effective-bets fixture here is diagonal or equally weighted, and
    for those the eigenbasis IS the standard basis -- so `w_tilde` and `w` agree
    and the rotation could be skipped entirely without any test noticing.

    A 2x2 equicorrelation matrix has analytically known eigenvectors that are
    NOT standard basis vectors: (1,1)/sqrt(2) with eigenvalue s2(1+rho) and
    (1,-1)/sqrt(2) with s2(1-rho). With s2=0.04, rho=0.5, w=(0.8, 0.2):
        w_tilde = (1.0/sqrt2, 0.6/sqrt2) -> contributions (0.03, 0.0036)
        p       = (0.892857, 0.107143)   -> exp(entropy) = 1.4056499
    Using the un-rotated weights instead would give 1.5467599.
    """
    Sigma = np.array([[0.04, 0.02], [0.02, 0.04]])
    w = np.array([0.8, 0.2])
    assert effective_bets(Sigma, w) == pytest.approx(1.40564993, abs=1e-7)
    assert effective_bets(Sigma, w) != pytest.approx(1.54675985, abs=1e-3)


def test_effective_bets_survives_a_rank_deficient_covariance():
    """T < N is permitted for the shrunk estimators, so this input is reachable.

    A sample covariance from three observations on five assets has three
    eigenvalues that are zero up to rounding -- some of them faintly negative.
    Those modes contribute nothing and must be dropped, not turned into NaN by
    taking log(0).
    """
    rng = np.random.default_rng(34)
    X = rng.normal(size=(3, 5))
    X = X - X.mean(axis=0)
    Sigma = (X.T @ X) / 3
    assert np.linalg.matrix_rank(Sigma) < 5
    nb = effective_bets(Sigma, np.array([0.4, 0.25, 0.15, 0.1, 0.1]))
    assert np.isfinite(nb)
    assert nb == pytest.approx(1.16397141, abs=1e-7)


def test_effective_bets_never_reports_less_than_one_bet():
    """A non-PSD Sigma must not produce a sub-unit -- i.e. nonsensical -- count.

    [[0.04, 0.06], [0.06, 0.04]] has eigenvalues 0.10 and -0.02. Left unclipped,
    the negative mode makes one variance share exceed 1, the entropy goes
    negative, and the metric reports 0.967 bets: fewer than holding one asset.
    """
    Sigma = np.array([[0.04, 0.06], [0.06, 0.04]])
    assert np.linalg.eigvalsh(Sigma).min() < 0
    assert effective_bets(Sigma, np.array([0.7, 0.3])) == pytest.approx(1.0)


def test_effective_bets_falls_as_correlation_rises():
    sd = np.array([0.2, 0.35])
    w = np.array([0.6, 0.4])
    previous = np.inf
    for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        Sigma = corr * np.outer(sd, sd)
        nb = effective_bets(Sigma, w)
        assert nb < previous
        previous = nb
    assert previous == pytest.approx(1.0, abs=0.05)


def test_effective_bets_never_exceeds_the_number_of_assets():
    rng = np.random.default_rng(33)
    for _ in range(20):
        A = rng.normal(size=(7, 7)) * rng.uniform(0.3, 3.0, size=(1, 7))
        Sigma = A @ A.T / 7
        w = rng.random(7)
        w = w / w.sum()
        nb = effective_bets(Sigma, w)
        assert 1.0 <= nb <= 7.0 + 1e-9


def test_effective_bets_rejects_a_zero_variance_portfolio():
    with pytest.raises(RiskError, match="zero"):
        effective_bets(np.zeros((3, 3)), W)


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------

def test_concentration_reports_both_hhi_measures():
    w = np.array([0.7, 0.1, 0.1, 0.1])
    pct = np.array([0.9, 0.05, 0.03, 0.02])
    c = concentration(w, pct)
    assert c["hhi_weights"] == pytest.approx(0.52)
    assert c["hhi_risk"] > c["hhi_weights"]


def test_concentration_hhi_risk_is_the_sum_of_squared_risk_shares():
    w = np.array([0.7, 0.1, 0.1, 0.1])
    pct = np.array([0.9, 0.05, 0.03, 0.02])
    # 0.81 + 0.0025 + 0.0009 + 0.0004
    assert concentration(w, pct)["hhi_risk"] == pytest.approx(0.8138)


def test_concentration_distinguishes_evenly_weighted_but_risk_concentrated():
    """Equal weights hide the concentration; the risk HHI exposes it.

    A one-factor covariance, PSD by construction, where the first name carries a
    1.5 beta and the rest carry 0.3. Five equal weights, one risk exposure.
    """
    beta = np.array([1.5, 0.3, 0.3, 0.3, 0.3])
    Sigma = np.outer(beta, beta) * 0.04 + np.diag([0.01, 0.02, 0.02, 0.02, 0.02])
    assert np.linalg.eigvalsh(Sigma).min() > 0        # the fixture really is PSD
    w = np.full(5, 0.2)
    _, pct = risk_contributions(Sigma, w)
    np.testing.assert_allclose(pct, [0.45073, 0.13732, 0.13732, 0.13732, 0.13732],
                               atol=1e-5)
    c = concentration(w, pct)
    assert c["hhi_weights"] == pytest.approx(0.2)      # perfectly even by weight
    assert c["hhi_risk"] == pytest.approx(0.27858427, abs=1e-7)
    assert c["hhi_risk"] > c["hhi_weights"]            # but not by risk


def test_concentration_rejects_a_length_mismatch():
    with pytest.raises(RiskError, match="length"):
        concentration(np.array([0.5, 0.5]), np.array([0.4, 0.3, 0.3]))


# --------------------------------------------------------------------------
# validation (_check)
# --------------------------------------------------------------------------

def test_rejects_weight_length_mismatch():
    with pytest.raises(RiskError, match="length"):
        portfolio_vol(np.eye(3), np.array([0.5, 0.5]))


def test_rejects_non_square_sigma():
    with pytest.raises(RiskError, match="square"):
        portfolio_vol(np.ones((3, 4)), np.array([0.4, 0.3, 0.3]))


def test_rejects_non_finite_sigma():
    Sigma = SIGMA.copy()
    Sigma[1, 2] = np.nan
    with pytest.raises(RiskError, match="non-finite"):
        portfolio_vol(Sigma, W)


def test_rejects_non_finite_weights():
    with pytest.raises(RiskError, match="non-finite"):
        portfolio_vol(SIGMA, np.array([0.5, np.inf, 0.2]))


def test_rejects_an_asymmetric_sigma():
    # eigh (used by effective_bets) reads only the lower triangle while
    # w @ Sigma @ w reads all of it -- an asymmetric Sigma would make the two
    # metrics silently disagree about the same portfolio instead of raising.
    bad = SIGMA.copy()
    bad[0, 1] += 0.05
    with pytest.raises(RiskError, match="symmetric"):
        portfolio_vol(bad, W)


def test_accepts_symmetry_within_float_tolerance():
    # Float noise (e.g. from an eigendecomposition reconstruction) must not
    # trip the symmetry gate -- only a genuine asymmetry should. 1e-12 is well
    # inside the gate's tolerance; SIGMA's own entries are O(1e-2) to O(1e-1).
    noisy = SIGMA.copy()
    noisy[0, 1] += 1e-12
    portfolio_vol(noisy, W)  # must not raise


def test_every_metric_goes_through_the_same_validation_gate():
    """A malformed input must fail the same way whichever entry point is used."""
    bad = SIGMA.copy()
    bad[0, 0] = np.nan
    for fn in (portfolio_vol, risk_contributions, diversification_ratio, effective_bets):
        with pytest.raises(RiskError, match="non-finite"):
            fn(bad, W)
        with pytest.raises(RiskError, match="length"):
            fn(SIGMA, np.array([0.5, 0.5]))


# --------------------------------------------------------------------------
# validation (_check_returns) -- the (R, w) counterpart used by return-matrix
# metrics (tail/market/stress), which don't have a square Sigma to validate.
# --------------------------------------------------------------------------

def test_check_returns_accepts_a_matching_pair():
    from finq.risk import _check_returns
    R = np.zeros((100, 3))
    R_out, w_out = _check_returns(R, W)
    np.testing.assert_array_equal(R_out, R)
    np.testing.assert_array_equal(w_out, W)


def test_check_returns_rejects_weight_length_mismatch():
    from finq.risk import _check_returns
    with pytest.raises(RiskError, match="length"):
        _check_returns(np.zeros((100, 3)), np.array([0.5, 0.5]))


def test_check_returns_rejects_non_2d_input():
    from finq.risk import _check_returns
    with pytest.raises(RiskError, match="2-D"):
        _check_returns(np.zeros(100), W)


def test_check_returns_rejects_non_finite_values():
    from finq.risk import _check_returns
    R = np.zeros((100, 3))
    R[5, 1] = np.nan
    with pytest.raises(RiskError, match="non-finite"):
        _check_returns(R, W)
    with pytest.raises(RiskError, match="non-finite"):
        _check_returns(np.zeros((100, 3)), np.array([0.5, np.inf, 0.2]))
