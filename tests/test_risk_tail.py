import numpy as np
import pandas as pd
import pytest
from finq.risk import (var_cvar, max_drawdown, betas, exceedance_correlation,
                       fx_risk_share, RiskError)


# --------------------------------------------------------------------------
# var_cvar
# --------------------------------------------------------------------------

def test_historical_var_is_the_empirical_quantile_loss():
    r = np.linspace(-0.10, 0.09, 100).reshape(-1, 1)
    var, cvar = var_cvar(r, np.array([1.0]), level=0.95, method="historical")
    assert var == pytest.approx(0.0905, abs=2e-3)
    assert cvar > var


def test_historical_var_and_cvar_pin_the_exact_tail_mean():
    """Pins both numbers exactly, not just their ordering.

    ``cvar > var`` above is satisfied by any correct-looking tail mean, so a
    mutant that averages the wrong slice (e.g. ``p < -var`` dropping the
    quantile point itself, or averaging the whole series) would still pass
    it. With p = linspace(-0.10, 0.09, 100), the 5% quantile (linear
    interpolation) is exactly -0.0905, and the tail p <= -0.0905 is the five
    points {-0.10, -0.098081, -0.096162, -0.094242, -0.092323} (i = 0..4),
    whose mean is 0.09616161616161616 -- computed by hand off np.linspace's
    known step size (0.19 / 99), not read back out of the implementation.
    """
    p = np.linspace(-0.10, 0.09, 100).reshape(-1, 1)
    var, cvar = var_cvar(p, np.array([1.0]), level=0.95, method="historical")
    assert var == pytest.approx(0.0905, abs=1e-12)
    assert cvar == pytest.approx(0.09616161616161616, abs=1e-9)


def test_cvar_tail_includes_the_var_quantile_point_itself():
    """Pins the ``<=`` vs ``<`` boundary in the tail slice, which the other
    tests here cannot: floating point makes ``1.0 - 0.95`` land at
    0.050000000000000044, not the mathematical 0.05, so np.quantile's
    virtual index (5.000000000000004 on 101 points) is never *exactly* an
    integer and the interpolated quantile never bit-exactly matches a plain
    data point -- a ``p <= -var`` vs ``p < -var`` mutant is invisible on any
    fixture built by just picking a "round" level and count.

    Forcing p[5] and p[6] to the identical value sidesteps that: whatever
    fractional weight the interpolation lands on, blending a value with
    itself returns that value exactly (``lo + (hi - lo) * frac == lo`` when
    ``hi == lo``), so the quantile is bit-identical to both p[5] and p[6].
    The correct tail then includes both duplicate points: indices 0..6 (7
    points, mean -0.09457142857142857). Excluding the boundary
    (``p < -var``) drops both, leaving only indices 0..4 (5 points, mean
    -0.0962) -- a different, wrong number this test catches.
    """
    p = np.linspace(-0.10, 0.09, 101)
    p[6] = p[5]                      # duplicate straddling the quantile index
    var, cvar = var_cvar(p.reshape(-1, 1), np.array([1.0]), level=0.95,
                         method="historical")
    assert var == pytest.approx(0.09050000000000001, abs=1e-14)
    assert cvar == pytest.approx(0.09457142857142857, abs=1e-9)


def test_cvar_is_never_below_var():
    rng = np.random.default_rng(40)
    R = rng.standard_t(df=4, size=(2000, 3)) * 0.01
    w = np.full(3, 1 / 3)
    for level in (0.95, 0.99):
        var, cvar = var_cvar(R, w, level=level)
        assert cvar >= var


def test_var_cvar_boundary_at_minimum_observation_count():
    """The p.size < 30 gate: 30 observations must pass, 29 must raise.

    A test with only interior sizes (2000, 5000, ...) can't distinguish a
    correct ``< 30`` check from an off-by-one (``<= 30`` or ``< 29``).
    """
    rng = np.random.default_rng(49)
    p_ok = rng.normal(scale=0.01, size=(30, 1))
    var, cvar = var_cvar(p_ok, np.array([1.0]), level=0.95)
    assert np.isfinite(var) and np.isfinite(cvar)

    p_short = rng.normal(scale=0.01, size=(29, 1))
    with pytest.raises(RiskError, match="observations"):
        var_cvar(p_short, np.array([1.0]), level=0.95)


def test_cornish_fisher_exceeds_normal_var_for_left_skewed_returns():
    rng = np.random.default_rng(41)
    x = -np.abs(rng.normal(size=(5000, 1))) ** 2 * 0.01     # strong left skew
    w = np.array([1.0])
    cf, _ = var_cvar(x, w, level=0.99, method="cornish_fisher")
    hist, _ = var_cvar(x, w, level=0.99, method="historical")
    assert cf > 0 and hist > 0


def test_cornish_fisher_var_exceeds_and_pins_the_hand_computed_expansion():
    """Strengthens the brief's same-fixture test, which only checks cf > 0.

    ``cf > 0 and hist > 0`` is satisfied by nearly any formula, correct or
    not -- it doesn't even check the Cornish-Fisher expansion actually
    inflates VaR for negative skew, despite the test name's promise. Pin the
    comparison directly, and pin both values to what the reference
    expansion (skew/excess-kurtosis Cornish-Fisher, computed independently
    with scipy.stats.skew/kurtosis) produces on this exact seeded fixture.
    """
    from scipy import stats
    rng = np.random.default_rng(41)
    x = -np.abs(rng.normal(size=(5000, 1))) ** 2 * 0.01
    w = np.array([1.0])
    cf, _ = var_cvar(x, w, level=0.99, method="cornish_fisher")
    hist, _ = var_cvar(x, w, level=0.99, method="historical")

    p = x[:, 0]
    mu, sd = float(p.mean()), float(p.std(ddof=0))
    s, k = float(stats.skew(p)), float(stats.kurtosis(p, fisher=True))
    z = float(stats.norm.ppf(0.01))
    z_cf = (z + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    expected_cf = -(mu + z_cf * sd)

    assert cf == pytest.approx(expected_cf, rel=1e-9)
    assert cf > hist
    assert cf == pytest.approx(0.06753084263152487, abs=1e-6)
    assert hist == pytest.approx(0.06482903097193626, abs=1e-6)


def test_var_cvar_rejects_an_out_of_range_level():
    with pytest.raises(RiskError, match="level"):
        var_cvar(np.zeros((50, 1)), np.array([1.0]), level=1.0)
    with pytest.raises(RiskError, match="level"):
        var_cvar(np.zeros((50, 1)), np.array([1.0]), level=0.5)


def test_var_cvar_rejects_an_unknown_method():
    with pytest.raises(RiskError, match="method"):
        var_cvar(np.zeros((50, 1)), np.array([1.0]), method="bogus")


# --------------------------------------------------------------------------
# max_drawdown
# --------------------------------------------------------------------------

def test_max_drawdown_of_a_known_path():
    # +10%, then -50%, then +10%  ->  peak 1.10, trough 0.55, drawdown = 0.5
    r = np.array([[0.10], [-0.50], [0.10]])
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.5)


def test_max_drawdown_is_zero_for_a_monotone_rise():
    r = np.full((10, 1), 0.01)
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.0)


def test_max_drawdown_recovery_to_prior_peak_does_not_erase_the_earlier_trough():
    """A boundary the brief's tests never exercise: dd == 0 at the *last* point.

    +20%, -50%, +100% takes equity from 1.20 to 0.60 back to exactly 1.20 --
    the final point sits exactly on the running peak again, so its own
    drawdown is 0. A mutant that reports the drawdown at the final index
    (or "the most recent trough vs the most recent peak") instead of
    max()-ing over the whole path would return 0.0 here; the correct
    answer is still 0.5, from the middle point.
    """
    r = np.array([[0.20], [-0.50], [1.00]])
    equity = np.cumprod(1.0 + r[:, 0])
    assert equity[-1] == pytest.approx(equity[0])   # last point == first peak
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.5)


def test_max_drawdown_single_observation_is_zero():
    """A single period is trivially its own peak; drawdown == 0, not NaN."""
    r = np.array([[-0.05]])
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# betas
# --------------------------------------------------------------------------

def test_beta_of_one_when_portfolio_equals_the_benchmark():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    mkt = pd.Series(rng.normal(scale=0.01, size=300), index=idx)
    R = pd.DataFrame({"A": mkt}, index=idx)
    b = betas(R, np.array([1.0]), pd.DataFrame({"MKT": mkt}))
    assert b["MKT"] == pytest.approx(1.0)


def test_betas_to_two_benchmarks_are_reported_separately():
    rng = np.random.default_rng(43)
    idx = pd.date_range("2025-01-01", periods=500, freq="B")
    wig = pd.Series(rng.normal(scale=0.01, size=500), index=idx)
    spx = pd.Series(rng.normal(scale=0.01, size=500), index=idx)
    R = pd.DataFrame({"PL": 1.5 * wig, "US": 0.5 * spx}, index=idx)
    b = betas(R, np.array([0.5, 0.5]), pd.DataFrame({"WIG20": wig, "GSPC": spx}))
    assert set(b) == {"WIG20", "GSPC"}
    assert b["WIG20"] == pytest.approx(0.75, abs=0.05)
    assert b["GSPC"] == pytest.approx(0.25, abs=0.05)


def test_betas_closed_form_pins_the_exact_ols_coefficient():
    """A deterministic fixture where beta is algebraically exact, not just close.

    p = 1.5*b + 0.002 (a pure linear relationship plus a constant offset).
    cov(p, b) = 1.5 * var(b) exactly regardless of b's own values, and the
    additive constant drops out of the covariance entirely, so
    beta = cov(p, b) / var(b) = 1.5 to machine precision. This pins the OLS
    formula itself -- swapping cov and var, using corr instead of cov, or
    forgetting ddof=0 would all move this off 1.5 by more than float noise.
    """
    b = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
    p = 1.5 * b + 0.002
    R = p.reshape(-1, 1)
    w = np.array([1.0])
    bm = pd.DataFrame({"BM": b})
    out = betas(R, w, bm)
    assert out["BM"] == pytest.approx(1.5, rel=1e-10)


def test_betas_rejects_too_few_overlapping_dates_boundary():
    """len(common) < 30 gate: 30 overlapping dates pass, 29 must raise.

    Constructs the benchmark index as a strict prefix of R's own date index,
    so the intersection size is exactly controlled rather than incidental.
    """
    rng = np.random.default_rng(50)
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    R = pd.DataFrame({"A": rng.normal(scale=0.01, size=60)}, index=idx)

    bm_ok = pd.DataFrame({"BM": rng.normal(scale=0.01, size=30)}, index=idx[:30])
    out = betas(R, np.array([1.0]), bm_ok)
    assert np.isfinite(out["BM"])

    bm_short = pd.DataFrame({"BM": rng.normal(scale=0.01, size=29)}, index=idx[:29])
    with pytest.raises(RiskError, match="overlapping"):
        betas(R, np.array([1.0]), bm_short)


def test_betas_rejects_a_zero_variance_benchmark():
    R = np.random.default_rng(51).normal(scale=0.01, size=(50, 1))
    bm = pd.DataFrame({"FLAT": np.full(50, 0.003)})
    with pytest.raises(RiskError, match="zero"):
        betas(R, np.array([1.0]), bm)


def test_betas_rejects_a_non_dataframe_benchmark():
    R = np.random.default_rng(52).normal(scale=0.01, size=(50, 1))
    with pytest.raises(RiskError, match="DataFrame"):
        betas(R, np.array([1.0]), np.zeros((50, 1)))


def test_betas_rejects_array_benchmark_length_mismatch():
    rng = np.random.default_rng(53)
    R = rng.normal(scale=0.01, size=(50, 1))
    bm = pd.DataFrame({"BM": rng.normal(scale=0.01, size=49)})
    with pytest.raises(RiskError, match="match"):
        betas(R, np.array([1.0]), bm)


# --------------------------------------------------------------------------
# exceedance_correlation
# --------------------------------------------------------------------------

def test_exceedance_correlation_detects_downside_dependence():
    """Two assets independent in calm times but crashing together."""
    rng = np.random.default_rng(44)
    T = 6000
    a = rng.normal(size=T)
    b = rng.normal(size=T)
    crash = rng.random(T) < 0.08
    shock = rng.normal(size=T) - 2.5
    a = np.where(crash, shock, a)
    b = np.where(crash, shock, b)
    R = np.column_stack([a, b])
    down, uncond = exceedance_correlation(R, threshold=1.0)
    assert down[0, 1] > uncond[0, 1]


def test_exceedance_correlation_needs_enough_joint_observations():
    rng = np.random.default_rng(45)
    R = rng.normal(size=(40, 3))
    with pytest.raises(RiskError, match="observations"):
        exceedance_correlation(R, threshold=2.5, min_obs=100)


def _crash_column(T, k):
    """T observations, the first k deep in the tail, the rest calm near zero.

    Deterministic (no randomness in the split itself): every one of the
    first k points standardizes to well below -1 sigma, and every one of
    the remaining T-k calm points standardizes to well above -1 sigma, so
    the joint exceedance mask count is exactly k by construction -- verified
    against threshold=1.0 for the exact k values used below.
    """
    x = np.empty(T)
    x[:k] = -5.0 - 0.001 * np.arange(k)
    x[k:] = 0.01 * np.sin(np.arange(T - k))
    return x


def test_exceedance_correlation_excludes_points_exactly_at_the_threshold():
    """Pins the ``<`` (strict) vs ``<=`` convention at the z == -threshold edge.

    The min-obs boundary test below controls the *count* of points beyond
    the threshold but keeps every one of them comfortably inside it (z well
    below -1), so it cannot see whether the comparison is strict. This
    fixture makes it observable: 100 points at +1.0 and 100 at -1.0 give an
    exact mean of 0 and an exact std of 1.0 (ddof=0), so every z-score is
    bit-exactly +-1.0 -- the -1.0 points sit exactly *on* -threshold, not
    beyond it. Under the documented "falling below" (strict <) convention,
    none of them qualify, so the joint count is 0 and the call must raise.
    A ``<=`` mutant would instead count all 100 as joint exceedances and
    return normally.
    """
    x = np.array([1.0] * 100 + [-1.0] * 100)
    R = np.column_stack([x, x.copy()])
    with pytest.raises(RiskError, match="observations"):
        exceedance_correlation(R, threshold=1.0, min_obs=1)


def test_exceedance_correlation_min_obs_boundary_exact_and_off_by_one():
    """n < min_obs gate: n == min_obs must pass, n == min_obs - 1 must raise.

    Both assets are built from the same deterministic crash/calm split (see
    ``_crash_column``), so the joint mask count equals the crash count k
    exactly -- not approximately, so this pins the boundary precisely rather
    than relying on a random draw to happen to land on it.
    """
    T = 200
    x_ok, y_ok = _crash_column(T, 20), _crash_column(T, 20)
    down, uncond = exceedance_correlation(np.column_stack([x_ok, y_ok]),
                                          threshold=1.0, min_obs=20)
    assert np.isfinite(down[0, 1])

    x_short, y_short = _crash_column(T, 19), _crash_column(T, 19)
    with pytest.raises(RiskError, match="observations"):
        exceedance_correlation(np.column_stack([x_short, y_short]),
                               threshold=1.0, min_obs=20)


# --------------------------------------------------------------------------
# fx_risk_share
# --------------------------------------------------------------------------

def test_fx_risk_share_is_zero_for_an_all_pln_portfolio():
    rng = np.random.default_rng(46)
    R = rng.normal(scale=0.01, size=(500, 3))
    fxr = np.zeros((500, 3))
    assert fx_risk_share(R, fxr, np.full(3, 1 / 3)) == pytest.approx(0.0)


def test_fx_risk_share_is_positive_when_fx_moves():
    rng = np.random.default_rng(47)
    asset = rng.normal(scale=0.005, size=(1000, 2))
    fxr = np.repeat(rng.normal(scale=0.02, size=(1000, 1)), 2, axis=1)
    R = (1 + asset) * (1 + fxr) - 1
    share = fx_risk_share(R, fxr, np.array([0.5, 0.5]))
    assert 0.5 < share <= 1.0


def test_fx_risk_share_pins_the_exact_covariance_share():
    """Pins a scale, not just an ordering.

    ``0.5 < share <= 1.0`` above admits any constant-multiplier bug that
    keeps the number in range (e.g. using corr instead of cov/var, which
    would land near 1.0 regardless of the true share). Recompute the
    expected share independently with bare ``np.cov``/``np.var`` calls on
    the same fixture and pin it exactly.
    """
    rng = np.random.default_rng(48)
    T = 60
    fx = rng.normal(scale=0.01, size=T)
    fxr = np.column_stack([fx, fx])
    asset_noise = rng.normal(scale=0.003, size=(T, 2))
    R = asset_noise + fxr
    w = np.array([0.5, 0.5])

    total = R @ w
    currency = fxr @ w
    expected = np.cov(total, currency, ddof=0)[0, 1] / np.var(total, ddof=0)

    share = fx_risk_share(R, fxr, w)
    assert share == pytest.approx(expected, rel=1e-10)
    assert share == pytest.approx(0.9268842658820938, abs=1e-9)


def test_fx_risk_share_rejects_a_zero_variance_portfolio():
    R = np.zeros((50, 2))
    fxr = np.zeros((50, 2))
    with pytest.raises(RiskError, match="zero"):
        fx_risk_share(R, fxr, np.array([0.5, 0.5]))
