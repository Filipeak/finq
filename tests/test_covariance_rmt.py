import operator

import numpy as np
import pytest
from finq.covariance import _fit_bulk_sigma2, estimate, mp_band, rmt_clean, sample_cov


def _corr(M):
    sd = np.sqrt(np.diag(M))
    return M / np.outer(sd, sd)


def test_diagonal_is_exactly_the_sample_variance():
    rng = np.random.default_rng(20)
    R = rng.normal(size=(800, 30)) * np.linspace(0.01, 0.03, 30)
    Sigma = rmt_clean(R)
    np.testing.assert_allclose(np.diag(Sigma), np.diag(sample_cov(R)), rtol=1e-10)


def test_correlation_trace_is_preserved():
    rng = np.random.default_rng(21)
    R = rng.normal(size=(600, 25))
    C = _corr(rmt_clean(R))
    assert np.trace(C) == pytest.approx(25.0, rel=1e-9)


def test_result_is_symmetric_and_positive_definite():
    rng = np.random.default_rng(22)
    Sigma = rmt_clean(rng.normal(size=(500, 20)))
    np.testing.assert_allclose(Sigma, Sigma.T, rtol=1e-10)
    assert np.linalg.eigvalsh(Sigma).min() > 0


def test_pure_noise_collapses_to_near_identity_correlation():
    """With no real structure, cleaning should erase the spurious correlations."""
    rng = np.random.default_rng(23)
    R = rng.normal(size=(400, 40))
    off_before = np.abs(_corr(sample_cov(R))[np.triu_indices(40, 1)]).mean()
    off_after = np.abs(_corr(rmt_clean(R))[np.triu_indices(40, 1)]).mean()
    assert off_after < off_before / 5


def test_real_factor_structure_survives_cleaning():
    rng = np.random.default_rng(24)
    T, N = 800, 30
    factor = rng.normal(size=(T, 1))
    R = 0.9 * factor + 0.4 * rng.normal(size=(T, N))
    C = _corr(rmt_clean(R))
    iu = np.triu_indices(N, 1)
    assert C[iu].mean() > 0.5           # the common factor is retained


def test_estimate_wires_rmt_clean():
    rng = np.random.default_rng(25)
    Sigma, d = estimate(rng.normal(size=(500, 20)), method="rmt_clean")
    assert d.method == "rmt_clean"
    assert d.shrinkage is None
    assert Sigma.shape == (20, 20)


def test_works_when_T_less_than_N():
    rng = np.random.default_rng(26)
    Sigma = rmt_clean(rng.normal(size=(30, 45)))
    assert np.isfinite(Sigma).all()
    assert np.linalg.eigvalsh(Sigma).min() > 0


# ---------------------------------------------------------------------------
# An independent reference, transcribed from the brief's algorithm description
# in its UN-vectorized form: explicit (i, j) loops for the correlation and
# rescale steps, a Python-loop mean for the bulk fit, and an explicit
# outer-product accumulation for the eigen-reconstruction instead of the
# @ np.diag(...) @ matrix form. This exercises the same formulas through a
# different code path, so a stray constant multiplier or a swapped operand
# anywhere in the vectorized implementation (replacement formula, diagonal
# renormalization, final sd rescale) would show up as a large mismatch here,
# not just a plausible-looking one.
# ---------------------------------------------------------------------------
def reference_rmt_clean(R):
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    Y = R - R.mean(axis=0)
    S = (Y.T @ Y) / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)
    C = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = S[i, j] / (sd[i] * sd[j])

    evals, evecs = np.linalg.eigh(C)
    Q = T / N

    sigma2 = 1.0
    for _ in range(100):
        hi = sigma2 * (1 + 1 / Q + 2 * np.sqrt(1 / Q))
        bulk_vals = [e for e in evals if e <= hi]
        if not bulk_vals:
            break
        new = sum(bulk_vals) / len(bulk_vals)
        if abs(new - sigma2) < 1e-10:
            sigma2 = new
            break
        sigma2 = new
    hi = sigma2 * (1 + 1 / Q + 2 * np.sqrt(1 / Q))

    is_noise = [e <= hi for e in evals]
    n_noise = sum(is_noise)
    if n_noise == 0:
        cleaned = list(evals)
    else:
        signal_sum = sum(e for e, noisy in zip(evals, is_noise) if not noisy)
        repl = max((N - signal_sum) / n_noise, 1e-12)
        cleaned = [repl if noisy else e for e, noisy in zip(evals, is_noise)]

    C_clean = np.zeros((N, N))
    for k in range(N):
        v = evecs[:, k]
        C_clean += cleaned[k] * np.outer(v, v)

    d = np.sqrt(np.diag(C_clean))
    for i in range(N):
        for j in range(N):
            C_clean[i, j] /= d[i] * d[j]
    np.fill_diagonal(C_clean, 1.0)

    Sigma = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            Sigma[i, j] = C_clean[i, j] * sd[i] * sd[j]
    return (Sigma + Sigma.T) / 2.0


def test_matches_an_independent_transcription_of_the_algorithm():
    rng = np.random.default_rng(31)
    plain = rng.normal(size=(700, 25))

    rng = np.random.default_rng(32)
    factor = rng.normal(size=(600, 1))
    factor_case = (0.8 * factor + 0.5 * rng.normal(size=(600, 18))) * np.linspace(0.5, 2, 18)

    rng = np.random.default_rng(33)
    thin = rng.normal(size=(40, 60))          # T < N

    for R in (plain, factor_case, thin):
        Sigma = rmt_clean(R)
        Sigma_ref = reference_rmt_clean(R)
        np.testing.assert_allclose(Sigma, Sigma_ref, rtol=1e-9, atol=1e-12)


def test_factor_correlation_matches_the_population_value_within_a_tight_band():
    # Direction-only assertions (">0.5") can't tell a correct cleaning from one
    # that is scaled by a stray constant factor -- 0.6 or 1.3 times the right
    # answer would still clear that bar. Pin it to the population correlation
    # instead: with R_i = 0.9*factor + 0.4*noise_i (factor, noise_i iid unit
    # normal), Corr(R_i, R_j) = 0.9^2 / (0.9^2 + 0.4^2) = 0.81/0.97 exactly.
    rho_pop = 0.81 / 0.97
    diffs = []
    for seed in range(24, 30):
        rng = np.random.default_rng(seed)
        T, N = 800, 30
        factor = rng.normal(size=(T, 1))
        R = 0.9 * factor + 0.4 * rng.normal(size=(T, N))
        C = _corr(rmt_clean(R))
        iu = np.triu_indices(N, 1)
        diffs.append(C[iu].mean() - rho_pop)
    assert max(abs(d) for d in diffs) < 0.03, diffs


def test_boundary_eigenvalue_at_the_mp_edge_is_classified_as_noise_not_signal(monkeypatch):
    # The noise/signal split is `evals <= hi`: an eigenvalue sitting exactly on
    # lambda_plus must be cleaned (treated as noise), matching the "in band"
    # convention _diagnostics uses elsewhere. A `<` mutant would instead keep
    # it as signal. To pin the operator itself -- not just its neighborhood --
    # place one eigenvalue bit-exactly at hi and its immediate float successor
    # (np.nextafter) just above it, and fix the bulk-fit sigma^2 so `hi` is a
    # known closed-form constant rather than something re-derived from this
    # same engineered spectrum (which would be a circular, unstable target).
    rng = np.random.default_rng(99)
    T, N = 500, 20
    R = rng.normal(size=(T, N)) * np.linspace(0.01, 0.02, N)
    Q = T / N

    fixed_sigma2 = 1.0
    _, hi = mp_band(Q, fixed_sigma2)

    at_edge = hi                                  # bit-exact on the boundary
    just_above = np.nextafter(hi, np.inf)          # the very next float
    rest = np.full(N - 3, 0.1)                     # ordinary, unambiguous bulk
    market = N - at_edge - just_above - rest.sum()  # soaks up the trace, >> hi
    assert market > hi * 3

    evals = np.sort(np.concatenate([[market, just_above, at_edge], rest]))
    evecs, _ = np.linalg.qr(rng.normal(size=(N, N)))

    monkeypatch.setattr(np.linalg, "eigh", lambda _C: (evals, evecs))
    monkeypatch.setattr(
        "finq.covariance._fit_bulk_sigma2", lambda _evals_desc, _Q: fixed_sigma2
    )

    Sigma = rmt_clean(R)
    sd = np.sqrt(np.diag(sample_cov(R)))
    C_result = Sigma / np.outer(sd, sd)

    def expected_under(op):
        is_noise = op(evals, hi)
        n_noise = int(is_noise.sum())
        signal_sum = float(evals[~is_noise].sum())
        repl = (N - signal_sum) / n_noise
        cleaned = np.where(is_noise, repl, evals)
        Cc = evecs @ np.diag(cleaned) @ evecs.T
        d = np.sqrt(np.diag(Cc))
        Cc = Cc / np.outer(d, d)
        np.fill_diagonal(Cc, 1.0)
        return Cc

    C_le = expected_under(operator.le)   # <=, the spec
    C_lt = expected_under(operator.lt)   # <,  the mutant

    np.testing.assert_allclose(C_result, C_le, rtol=1e-9, atol=1e-12)
    assert np.abs(C_result - C_lt).max() > 0.1     # the mutant is not a near-miss


def test_empty_noise_set_leaves_the_sample_correlation_essentially_untouched(monkeypatch):
    # If the fitted band is degenerate and no eigenvalue qualifies as noise,
    # the n_noise == 0 branch must fall back to the unmodified eigenvalues --
    # not raise, not zero anything out. Reconstructing C from its own
    # untouched eigendecomposition must reproduce the original sample
    # covariance (up to the final symmetrization), which pins the branch
    # against silently returning something scaled or shifted.
    rng = np.random.default_rng(77)
    T, N = 300, 15
    R = rng.normal(size=(T, N)) * np.linspace(0.01, 0.02, N)

    monkeypatch.setattr("finq.covariance._fit_bulk_sigma2", lambda _evals_desc, _Q: 1e-8)

    Sigma = rmt_clean(R)
    np.testing.assert_allclose(Sigma, sample_cov(R), rtol=1e-8, atol=1e-14)
