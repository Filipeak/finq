import numpy as np
import pandas as pd
import pytest
from finq.covariance import estimate, sample_cov, mp_band, CovarianceError


def test_sample_cov_divides_by_T_not_T_minus_1():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 4))
    S = sample_cov(R)
    Y = R - R.mean(axis=0)
    np.testing.assert_allclose(S, (Y.T @ Y) / 500, rtol=1e-12)


def test_sample_cov_is_not_the_unbiased_T_minus_1_estimator():
    # Guards the MLE convention from the other direction: np.cov's default ddof=1
    # is exactly the mistake that would silently break Ledoit-Wolf's asymptotics.
    rng = np.random.default_rng(100)
    R = rng.normal(size=(50, 3))
    S = sample_cov(R)
    unbiased = np.cov(R, rowvar=False, ddof=1)
    np.testing.assert_allclose(S, unbiased * (49 / 50), rtol=1e-12)
    assert not np.allclose(S, unbiased, rtol=1e-6)


def test_sample_cov_recovers_known_diagonal_covariance():
    rng = np.random.default_rng(1)
    true_sd = np.array([0.01, 0.02, 0.03])
    R = rng.normal(size=(200_000, 3)) * true_sd
    S = sample_cov(R)
    np.testing.assert_allclose(np.sqrt(np.diag(S)), true_sd, rtol=0.02)
    assert abs(S[0, 1]) < 1e-4


def test_sample_cov_recovers_a_known_off_diagonal_covariance():
    rng = np.random.default_rng(101)
    T = 200_000
    z = rng.normal(size=(T, 2))
    R = np.column_stack([z[:, 0], 0.6 * z[:, 0] + np.sqrt(1 - 0.36) * z[:, 1]])
    S = sample_cov(R)
    assert S[0, 1] == pytest.approx(0.6, abs=0.01)
    assert S[0, 1] == S[1, 0]


def test_sample_cov_accepts_a_dataframe():
    rng = np.random.default_rng(102)
    arr = rng.normal(size=(120, 4))
    df = pd.DataFrame(arr, columns=list("abcd"))
    np.testing.assert_allclose(sample_cov(df), sample_cov(arr), rtol=1e-12)


def test_mp_band_matches_analytic_formula():
    for Q in (2.0, 5.0, 33.0):
        lo, hi = mp_band(Q, sigma2=1.0)
        assert lo == pytest.approx(1 + 1 / Q - 2 * np.sqrt(1 / Q))
        assert hi == pytest.approx(1 + 1 / Q + 2 * np.sqrt(1 / Q))


def test_mp_band_scales_with_sigma2():
    lo1, hi1 = mp_band(4.0, sigma2=1.0)
    lo2, hi2 = mp_band(4.0, sigma2=0.5)
    assert lo2 == pytest.approx(lo1 * 0.5)
    assert hi2 == pytest.approx(hi1 * 0.5)


def test_mp_band_rejects_nonpositive_Q():
    for bad in (0.0, -1.0):
        with pytest.raises(CovarianceError, match="Q must be positive"):
            mp_band(bad)


def test_diagnostics_report_shape_and_Q():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(600, 20))
    _, d = estimate(R, method="sample")
    assert (d.T, d.N) == (600, 20)
    assert d.Q == pytest.approx(30.0)
    assert d.method == "sample"
    assert d.shrinkage is None
    assert len(d.eigenvalues) == 20
    assert np.all(np.diff(d.eigenvalues) <= 1e-12)     # descending


def test_estimate_returns_the_sample_covariance_matrix():
    rng = np.random.default_rng(103)
    R = rng.normal(size=(300, 6)) * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Sigma, _ = estimate(R, method="sample")
    np.testing.assert_allclose(Sigma, sample_cov(R), rtol=1e-12)


def test_estimate_accepts_a_dataframe():
    rng = np.random.default_rng(104)
    arr = rng.normal(size=(300, 5))
    df = pd.DataFrame(arr, columns=list("vwxyz"))
    Sigma_df, d_df = estimate(df, method="sample")
    Sigma_arr, d_arr = estimate(arr, method="sample")
    np.testing.assert_allclose(Sigma_df, Sigma_arr, rtol=1e-12)
    assert (d_df.T, d_df.N) == (d_arr.T, d_arr.N)


def test_eigenvalues_are_of_the_correlation_matrix_not_the_covariance():
    # Correlation has unit diagonal, so its eigenvalues sum to N whatever the
    # per-asset scales are. Eigenvalues of a badly scaled covariance would not.
    rng = np.random.default_rng(105)
    N = 8
    R = rng.normal(size=(400, N)) * np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e-3, 1e-2, 1e-1])
    _, d = estimate(R, method="sample")
    assert d.eigenvalues.sum() == pytest.approx(float(N), rel=1e-10)


def test_pure_noise_puts_essentially_all_eigenvalues_inside_the_band():
    rng = np.random.default_rng(3)
    R = rng.normal(size=(1000, 50))
    _, d = estimate(R, method="sample")
    assert d.n_in_band >= 48


def test_one_injected_common_factor_escapes_the_band():
    rng = np.random.default_rng(4)
    T, N = 1000, 50
    factor = rng.normal(size=(T, 1))
    R = 0.7 * factor + 0.7 * rng.normal(size=(T, N))
    _, d = estimate(R, method="sample")
    assert d.N - d.n_in_band == 1


def test_bulk_sigma2_is_fitted_to_the_bulk_not_fixed_at_one():
    # A dominant market eigenvalue absorbs a large share of the trace, so the
    # remaining bulk sits well below unit variance. Laloux et al. fit sigma^2 to
    # that bulk; a band built with sigma2=1.0 would be far too wide to be useful.
    rng = np.random.default_rng(11)
    T, N = 2000, 50
    market = rng.normal(size=(T, 1))
    R = market + rng.normal(size=(T, N))          # equicorrelated, rho = 0.5
    _, d = estimate(R, method="sample")

    _, hi_unfitted = mp_band(d.Q, sigma2=1.0)
    assert d.lambda_plus < 0.65 * hi_unfitted

    bulk_mean = float(d.eigenvalues[1:].mean())   # everything but the market mode
    expected_hi = bulk_mean * (1 + 1 / d.Q + 2 * np.sqrt(1 / d.Q))
    assert d.lambda_plus == pytest.approx(expected_hi, rel=0.05)
    assert d.lambda_minus == pytest.approx(
        bulk_mean * (1 + 1 / d.Q - 2 * np.sqrt(1 / d.Q)), rel=0.05
    )


def test_var_share_in_band_counts_only_the_bulk():
    rng = np.random.default_rng(12)
    T, N = 1000, 50
    factor = rng.normal(size=(T, 1))
    R = 0.7 * factor + 0.7 * rng.normal(size=(T, N))
    _, d = estimate(R, method="sample")
    in_band = d.eigenvalues <= d.lambda_plus
    assert d.var_share_in_band == pytest.approx(
        d.eigenvalues[in_band].sum() / d.eigenvalues.sum(), rel=1e-12
    )
    # The market mode carries roughly half the trace here, so the noise share
    # must be well under 1 -- a var_share hard-wired to 1.0 would be a lie.
    assert 0.3 < d.var_share_in_band < 0.7


def test_pure_noise_var_share_in_band_is_essentially_all_of_it():
    rng = np.random.default_rng(13)
    R = rng.normal(size=(1000, 50))
    _, d = estimate(R, method="sample")
    assert d.var_share_in_band > 0.9


def test_condition_number_is_of_the_returned_matrix():
    rng = np.random.default_rng(14)
    R = rng.normal(size=(400, 5)) * np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    Sigma, d = estimate(R, method="sample")
    assert d.condition_number == pytest.approx(float(np.linalg.cond(Sigma)), rel=1e-10)
    assert d.condition_number > 1_000     # the 100x scaled asset dominates


def test_rejects_T_less_than_N_for_sample_method():
    rng = np.random.default_rng(5)
    R = rng.normal(size=(10, 30))
    with pytest.raises(CovarianceError, match="sample"):
        estimate(R, method="sample")


def test_rejects_T_equal_to_N_for_sample_method():
    rng = np.random.default_rng(15)
    R = rng.normal(size=(20, 20))
    with pytest.raises(CovarianceError, match="singular"):
        estimate(R, method="sample")


def test_rejects_unknown_method():
    rng = np.random.default_rng(6)
    R = rng.normal(size=(100, 5))
    with pytest.raises(CovarianceError, match="unknown"):
        estimate(R, method="nonsense")


def test_known_but_unimplemented_methods_say_so_rather_than_unknown():
    # A method that is on METHODS but not yet built must say "not implemented",
    # not "unknown" -- the two mean very different things to a caller.
    # 'ledoit_wolf' graduated out of this list in Task 6; 'rmt_clean' is next.
    rng = np.random.default_rng(16)
    R = rng.normal(size=(100, 5))
    for method in ("rmt_clean",):
        with pytest.raises(CovarianceError, match="not implemented"):
            estimate(R, method=method)


def test_implemented_methods_do_not_report_themselves_unimplemented():
    # The counterpart to the test above: everything not listed there must work.
    rng = np.random.default_rng(116)
    R = rng.normal(size=(100, 5))
    for method in ("sample", "ledoit_wolf"):
        Sigma, d = estimate(R, method=method)
        assert d.method == method
        assert Sigma.shape == (5, 5)


def test_rejects_non_finite_returns():
    rng = np.random.default_rng(7)
    R = rng.normal(size=(100, 5))
    R[3, 2] = np.nan
    with pytest.raises(CovarianceError, match="NaN or inf"):
        estimate(R, method="sample")
    R[3, 2] = np.inf
    with pytest.raises(CovarianceError, match="NaN or inf"):
        estimate(R, method="sample")


def test_rejects_non_2d_input():
    with pytest.raises(CovarianceError, match="2-D"):
        sample_cov(np.zeros(10))
    with pytest.raises(CovarianceError, match="2-D"):
        estimate(np.zeros((4, 4, 4)), method="sample")


def test_rejects_fewer_than_two_assets():
    rng = np.random.default_rng(8)
    with pytest.raises(CovarianceError, match="two assets"):
        estimate(rng.normal(size=(100, 1)), method="sample")


def test_rejects_a_zero_variance_asset_instead_of_returning_nan_diagnostics():
    # A completely stale price series has zero return variance; standardizing to a
    # correlation matrix would divide by zero and hand back NaN eigenvalues without
    # raising anything. An analysis on bad data must never look successful.
    rng = np.random.default_rng(9)
    R = rng.normal(size=(200, 4))
    R[:, 2] = 0.0
    with pytest.raises(CovarianceError, match="zero variance"):
        estimate(R, method="sample")
