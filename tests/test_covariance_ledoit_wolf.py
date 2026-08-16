import numpy as np
import pandas as pd
import pytest
from finq.covariance import CovarianceError, estimate, ledoit_wolf_cc, sample_cov


def blocky_returns(T: int, seed: int = 99) -> np.ndarray:
    """Two blocks with very different internal correlation.

    This matters: on i.i.d. Gaussian noise the constant-correlation target is
    almost exactly right, gamma collapses, and delta saturates at 1.0 — which
    makes any monotonicity assertion meaningless. Heterogeneous block structure
    misspecifies the target, so delta lands strictly inside (0, 1) and the
    T-dependence is actually observable.
    """
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=(T, 1)), rng.normal(size=(T, 1))
    tight = 0.95 * f1 + 0.15 * rng.normal(size=(T, 10))
    loose = 0.20 * f2 + 0.98 * rng.normal(size=(T, 10))
    return np.column_stack([tight, loose]) * np.linspace(0.5, 2.0, 20)


def test_shrinkage_intensity_is_a_valid_fraction():
    rng = np.random.default_rng(10)
    for T in (80, 300, 2000):
        _, delta = ledoit_wolf_cc(rng.normal(size=(T, 20)))
        assert 0.0 <= delta <= 1.0


def test_shrinkage_intensity_is_interior_for_misspecified_target():
    _, delta = ledoit_wolf_cc(blocky_returns(400))
    assert 0.0 < delta < 1.0


def test_shrinkage_rises_as_sample_shrinks():
    base = blocky_returns(6000)
    deltas = [ledoit_wolf_cc(base[:T])[1] for T in (150, 400, 1200, 6000)]
    assert deltas == sorted(deltas, reverse=True), deltas
    assert deltas[0] > deltas[-1] * 5


def test_result_is_symmetric_and_positive_definite():
    rng = np.random.default_rng(12)
    Sigma, _ = ledoit_wolf_cc(rng.normal(size=(300, 15)))
    np.testing.assert_allclose(Sigma, Sigma.T, rtol=1e-12)
    assert np.linalg.eigvalsh(Sigma).min() > 0


def test_positive_definite_even_when_T_less_than_N():
    rng = np.random.default_rng(13)
    R = rng.normal(size=(20, 40))
    assert np.linalg.eigvalsh(sample_cov(R)).min() < 1e-10      # sample is singular
    with pytest.warns(UserWarning, match="T=20 <= N=40"):
        Sigma, _ = ledoit_wolf_cc(R)
    assert np.linalg.eigvalsh(Sigma).min() > 0


def test_diagonal_variances_are_preserved_exactly():
    """F and S share a diagonal, so any convex combination must too."""
    rng = np.random.default_rng(14)
    R = rng.normal(size=(250, 12))
    Sigma, _ = ledoit_wolf_cc(R)
    np.testing.assert_allclose(np.diag(Sigma), np.diag(sample_cov(R)), rtol=1e-12)


def test_shrinks_toward_constant_correlation_not_identity():
    """Off-diagonal correlations must move toward r-bar, not toward zero."""
    rng = np.random.default_rng(15)
    T, N = 100, 20
    factor = rng.normal(size=(T, 1))
    R = 0.8 * factor + 0.4 * rng.normal(size=(T, N))     # strongly positive rbar
    S = sample_cov(R)
    Sigma, delta = ledoit_wolf_cc(R)
    sd_s, sd_t = np.sqrt(np.diag(S)), np.sqrt(np.diag(Sigma))
    corr_s = S / np.outer(sd_s, sd_s)
    corr_t = Sigma / np.outer(sd_t, sd_t)
    iu = np.triu_indices(N, 1)
    rbar = corr_s[iu].mean()
    assert delta > 0.0
    assert rbar > 0.3
    # shrunk correlations sit strictly between the sample value and r-bar
    spread_s = np.abs(corr_s[iu] - rbar).mean()
    spread_t = np.abs(corr_t[iu] - rbar).mean()
    assert spread_t < spread_s
    assert corr_t[iu].mean() == pytest.approx(rbar, abs=1e-8)


def test_estimate_wires_shrinkage_into_diagnostics():
    rng = np.random.default_rng(16)
    Sigma, d = estimate(rng.normal(size=(400, 20)), method="ledoit_wolf")
    assert d.method == "ledoit_wolf"
    assert d.shrinkage is not None and 0.0 <= d.shrinkage <= 1.0
    assert Sigma.shape == (20, 20)


def test_default_method_is_ledoit_wolf():
    rng = np.random.default_rng(17)
    _, d = estimate(rng.normal(size=(400, 10)))
    assert d.method == "ledoit_wolf"


# ---------------------------------------------------------------------------
# An independent reference, transcribed from spec 6.4 in its UN-simplified form.
#
# The implementation works with Gram-matrix algebra and the algebraic collapse
#   (1/T) sum_t (Y_it Y_jt - s_ij)^2  ==  (1/T) sum_t (Y_it Y_jt)^2 - s_ij^2
# This reference instead loops over (i, j) and evaluates the bracketed
# expressions literally, so it checks the collapse as well as the formula.
# ---------------------------------------------------------------------------
def reference_ledoit_wolf(R, cross_terms: bool = True):
    R = np.asarray(R, dtype=float)
    Y = R - R.mean(axis=0)
    T, N = Y.shape

    S = np.empty((N, N))
    pi = np.empty((N, N))
    th = np.empty((N, N))                 # th[i, j] = theta-hat_{ii,ij}
    for i in range(N):
        for j in range(N):
            S[i, j] = np.mean(Y[:, i] * Y[:, j])
    for i in range(N):
        for j in range(N):
            pi[i, j] = np.mean((Y[:, i] * Y[:, j] - S[i, j]) ** 2)
            th[i, j] = np.mean((Y[:, i] ** 2 - S[i, i]) * (Y[:, i] * Y[:, j] - S[i, j]))

    rbar = float(np.mean([
        S[i, j] / np.sqrt(S[i, i] * S[j, j]) for i in range(N) for j in range(i + 1, N)
    ]))

    F = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            F[i, j] = S[i, i] if i == j else rbar * np.sqrt(S[i, i] * S[j, j])

    pi_hat = float(sum(pi[i, j] for i in range(N) for j in range(N)))
    rho_hat = float(sum(pi[i, i] for i in range(N)))
    if cross_terms:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rho_hat += (rbar / 2.0) * (
                    np.sqrt(S[j, j] / S[i, i]) * th[i, j]
                    + np.sqrt(S[i, i] / S[j, j]) * th[j, i]
                )
    gamma_hat = float(sum((F[i, j] - S[i, j]) ** 2 for i in range(N) for j in range(N)))

    raw = ((pi_hat - rho_hat) / gamma_hat) / T
    delta = max(0.0, min(raw, 1.0))
    return delta * F + (1.0 - delta) * S, delta, raw


def factor_returns(T: int = 260, N: int = 14, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 1))
    return (0.7 * f + 0.6 * rng.normal(size=(T, N))) * np.linspace(0.4, 2.5, N)


def test_matches_an_independent_transcription_of_the_spec_formula():
    cases = [
        blocky_returns(400),
        factor_returns(),
        np.random.default_rng(31).normal(size=(120, 9)),
    ]
    for R in cases:
        Sigma, delta = ledoit_wolf_cc(R)
        ref_Sigma, ref_delta, _ = reference_ledoit_wolf(R)
        assert delta == pytest.approx(ref_delta, rel=1e-11, abs=1e-13)
        np.testing.assert_allclose(Sigma, ref_Sigma, rtol=1e-10, atol=1e-14)


def test_rho_hat_includes_the_rbar_over_two_cross_terms():
    # rho-hat is not just the diagonal of pi: it also carries the covariance
    # between the sample entries and the estimated target, weighted by rbar/2.
    # Dropping those terms is a silent, plausible-looking bug, so pin it by
    # checking the two variants are far apart AND that we match the right one.
    # The cross terms are driven by third moments, so they only bite on skewed
    # data. Two lognormal blocks with different factor loadings give a target
    # that is misspecified enough to keep delta interior while the cross terms
    # still shift it by ~31%.
    rng = np.random.default_rng(56)
    z, e = rng.normal(size=(1000, 1)), rng.normal(size=(1000, 20))
    R = np.column_stack([
        np.exp(0.9 * z + 0.4 * e[:, :10]),
        np.exp(0.2 * z + 0.9 * e[:, 10:]),
    ]) * np.linspace(0.5, 2.0, 20)

    _, delta = ledoit_wolf_cc(R)
    _, with_cross, raw_with = reference_ledoit_wolf(R, cross_terms=True)
    _, _, raw_without = reference_ledoit_wolf(R, cross_terms=False)
    assert 0.0 < delta < 1.0
    assert delta == pytest.approx(with_cross, rel=1e-11)
    assert abs(raw_with - raw_without) > 0.25 * abs(raw_with)
    assert delta != pytest.approx(raw_without, rel=1e-3)


def test_sigma_is_delta_F_plus_one_minus_delta_S_and_not_the_other_way_round():
    R = blocky_returns(500, seed=22)
    S = sample_cov(R)
    s = np.sqrt(np.diag(S))
    iu = np.triu_indices(R.shape[1], 1)
    rbar = (S / np.outer(s, s))[iu].mean()
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, np.diag(S))

    Sigma, delta = ledoit_wolf_cc(R)
    np.testing.assert_allclose(Sigma, delta * F + (1.0 - delta) * S, rtol=1e-12, atol=1e-15)
    # and the swapped combination is materially different, so the check has teeth
    swapped = (1.0 - delta) * F + delta * S
    assert not np.allclose(Sigma, swapped, rtol=1e-6)


def test_shrunk_correlations_are_the_sample_ones_pulled_toward_rbar_by_delta():
    # Direction and magnitude in one assertion: each off-diagonal correlation
    # must retain exactly (1 - delta) of its distance from rbar. Swapping F and
    # S would leave delta of it instead.
    R = blocky_returns(500, seed=23)
    N = R.shape[1]
    S = sample_cov(R)
    Sigma, delta = ledoit_wolf_cc(R)
    sd_s, sd_t = np.sqrt(np.diag(S)), np.sqrt(np.diag(Sigma))
    corr_s, corr_t = S / np.outer(sd_s, sd_s), Sigma / np.outer(sd_t, sd_t)
    iu = np.triu_indices(N, 1)
    rbar = corr_s[iu].mean()
    assert 0.01 < delta < 0.99
    np.testing.assert_allclose(
        corr_t[iu] - rbar, (1.0 - delta) * (corr_s[iu] - rbar), rtol=1e-10, atol=1e-14
    )


def test_saturated_intensity_returns_the_target_exactly():
    # On i.i.d. Gaussian noise the constant-correlation target is essentially
    # right, the unclipped intensity comes out above 1 (~1.087 here), and delta
    # saturates. delta == 1 must then reproduce F exactly.
    R = np.random.default_rng(41).normal(size=(300, 20))
    Sigma, delta = ledoit_wolf_cc(R)
    assert delta == 1.0
    S = sample_cov(R)
    s = np.sqrt(np.diag(S))
    iu = np.triu_indices(20, 1)
    rbar = (S / np.outer(s, s))[iu].mean()
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, np.diag(S))
    np.testing.assert_allclose(Sigma, F, rtol=1e-12, atol=1e-15)
    # every off-diagonal correlation is now exactly the same number
    corr = Sigma / np.outer(s, s)
    np.testing.assert_allclose(corr[iu], np.full(iu[0].size, rbar), rtol=1e-10)


def test_intensity_is_clipped_at_zero_when_the_estimator_goes_negative():
    # Fat-tailed returns can drive rho-hat above pi-hat, making the unclipped
    # intensity negative (-0.298 for this seed). A negative delta would
    # EXTRAPOLATE away from the target rather than shrink toward it, so it must
    # be clipped to 0 and the sample matrix returned untouched.
    rng = np.random.default_rng(812)
    z = rng.standard_t(3, size=(40, 1))
    R = 2.0 * z + rng.standard_t(3, size=(40, 8))
    Sigma, delta = ledoit_wolf_cc(R)
    assert delta == 0.0
    np.testing.assert_allclose(Sigma, sample_cov(R), rtol=1e-12, atol=1e-15)


def test_two_assets_leave_nothing_to_shrink():
    # With N = 2 there is a single off-diagonal correlation, so rbar equals it
    # and F == S algebraically -- exactly, not approximately. gamma_hat =
    # ||F-S||^2 therefore reduces to pure floating-point noise (~1e-34) whose
    # SIGN is essentially random, which turned delta into a coin flip between
    # 0.0 and 1.0 rather than the mathematically forced 0.0. Seed 42 alone
    # happened to land on exact-zero noise and would not have caught this --
    # sweep many seeds so a regression can't hide behind a lucky one.
    for seed in range(50):
        rng = np.random.default_rng(seed)
        R = rng.normal(size=(200, 2)) * np.array([0.5, 3.0])
        Sigma, delta = ledoit_wolf_cc(R)
        assert delta == 0.0, f"seed {seed}: delta={delta}"
        assert np.isfinite(Sigma).all()
        np.testing.assert_allclose(Sigma, sample_cov(R), rtol=1e-12, atol=1e-15)


def test_shrinkage_repairs_a_singular_sample_matrix():
    rng = np.random.default_rng(43)
    R = rng.normal(size=(30, 60))
    with pytest.warns(UserWarning, match="T=30 <= N=60"):
        Sigma, delta = ledoit_wolf_cc(R)
    assert delta > 0.0
    assert np.linalg.cond(sample_cov(R)) > 1e12          # singular
    assert np.linalg.cond(Sigma) < 1e4                   # usable


def test_does_not_mutate_its_input():
    R = blocky_returns(200, seed=44)
    before = R.copy()
    ledoit_wolf_cc(R)
    np.testing.assert_array_equal(R, before)


def test_accepts_a_dataframe():
    arr = blocky_returns(300, seed=45)
    df = pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
    Sigma_df, d_df = ledoit_wolf_cc(df)
    Sigma_arr, d_arr = ledoit_wolf_cc(arr)
    np.testing.assert_allclose(Sigma_df, Sigma_arr, rtol=1e-12)
    assert d_df == d_arr


def test_rejects_degenerate_inputs():
    rng = np.random.default_rng(46)
    with pytest.raises(CovarianceError, match="two observations"):
        ledoit_wolf_cc(rng.normal(size=(1, 5)))
    with pytest.raises(CovarianceError, match="two assets"):
        ledoit_wolf_cc(rng.normal(size=(50, 1)))
    with pytest.raises(CovarianceError, match="2-D"):
        ledoit_wolf_cc(rng.normal(size=50))
    bad = rng.normal(size=(50, 4))
    bad[7, 1] = np.nan
    with pytest.raises(CovarianceError, match="NaN or inf"):
        ledoit_wolf_cc(bad)


def test_rejects_a_zero_variance_asset():
    # rbar is undefined when an asset never moves: the correlation column is 0/0.
    rng = np.random.default_rng(47)
    R = rng.normal(size=(200, 5))
    R[:, 3] = 0.0
    with pytest.raises(CovarianceError, match="zero variance"):
        ledoit_wolf_cc(R)
    with pytest.raises(CovarianceError, match="zero variance"):
        estimate(R, method="ledoit_wolf")


def test_estimate_returns_exactly_what_ledoit_wolf_cc_computed():
    R = blocky_returns(350, seed=48)
    Sigma_direct, delta_direct = ledoit_wolf_cc(R)
    Sigma, d = estimate(R, method="ledoit_wolf")
    np.testing.assert_allclose(Sigma, Sigma_direct, rtol=1e-12)
    assert d.shrinkage == delta_direct
    assert 0.0 < d.shrinkage < 1.0            # not a hard-wired 0.0 or 1.0
    assert d.method == "ledoit_wolf"
    assert (d.T, d.N) == R.shape


def test_estimate_permits_T_less_than_N_for_ledoit_wolf():
    # 'sample' refuses this case as singular; shrinkage is the reason the method
    # exists, so it must go through -- with a warning, since it is a thin sample.
    rng = np.random.default_rng(49)
    R = rng.normal(size=(25, 40))
    with pytest.raises(CovarianceError, match="singular"):
        estimate(R, method="sample")
    with pytest.warns(UserWarning, match="T=25 <= N=40"):
        Sigma, d = estimate(R, method="ledoit_wolf")
    assert d.Q == pytest.approx(25 / 40)
    assert np.linalg.eigvalsh(Sigma).min() > 0
    assert np.isfinite(d.condition_number)


def test_estimate_accepts_a_dataframe():
    arr = blocky_returns(300, seed=50)
    df = pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
    Sigma_df, d_df = estimate(df, method="ledoit_wolf")
    Sigma_arr, d_arr = estimate(arr, method="ledoit_wolf")
    np.testing.assert_allclose(Sigma_df, Sigma_arr, rtol=1e-12)
    assert d_df.shrinkage == d_arr.shrinkage
