import numpy as np
import pandas as pd
import pytest
from finq.data import PriceData
from finq.returns import aligned, PERIODS_PER_YEAR, ReturnsError


def make_prices(dates_pl, dates_us):
    idx = sorted(set(dates_pl) | set(dates_us))
    close = pd.DataFrame(index=pd.DatetimeIndex(idx), columns=["PL.WA", "US"], dtype=float)
    close.loc[pd.DatetimeIndex(dates_pl), "PL.WA"] = [100.0, 110.0, 121.0][:len(dates_pl)]
    close.loc[pd.DatetimeIndex(dates_us), "US"] = [50.0, 55.0, 60.5][:len(dates_us)]
    vol = close * 0 + 1000.0
    return PriceData(close, vol, {"PL.WA": "PLN", "US": "USD"}, [])


def test_pln_asset_return_is_plain_price_return():
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    pdta = make_prices(d, d)
    rates = {"PLN": pd.Series(1.0, index=d), "USD": pd.Series(4.0, index=d)}
    rm = aligned(pdta, rates)
    np.testing.assert_allclose(rm.R["PL.WA"].to_numpy(), [0.10, 0.10])


def test_usd_asset_return_compounds_asset_and_fx():
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    pdta = make_prices(d, d)
    rates = {"PLN": pd.Series(1.0, index=d), "USD": pd.Series([4.0, 4.4, 4.4], index=d)}
    rm = aligned(pdta, rates)
    # day 1: asset +10%, fx +10% -> 1.1*1.1-1 = 0.21 ; day 2: asset +10%, fx 0% -> 0.10
    np.testing.assert_allclose(rm.R["US"].to_numpy(), [0.21, 0.10], rtol=1e-12)


def test_fx_component_is_isolated_and_zero_for_pln():
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    pdta = make_prices(d, d)
    rates = {"PLN": pd.Series(1.0, index=d), "USD": pd.Series([4.0, 4.4, 4.4], index=d)}
    rm = aligned(pdta, rates)
    np.testing.assert_allclose(rm.fx_returns["PL.WA"].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [0.10, 0.0], rtol=1e-12)


def test_inner_join_drops_non_common_days_and_counts_them():
    pl = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    us = pd.to_datetime(["2025-01-02", "2025-01-06"])       # US closed on the 3rd
    pdta = make_prices(pl, us)
    rates = {"PLN": pd.Series(1.0, index=pl), "USD": pd.Series(4.0, index=pl)}
    rm = aligned(pdta, rates)
    assert len(rm.R) == 1            # two common dates -> one return
    assert rm.dropped_days == 1


def test_no_forward_fill_leaves_no_zero_return_artifacts():
    pl = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    us = pd.to_datetime(["2025-01-02", "2025-01-06"])
    pdta = make_prices(pl, us)
    rates = {"PLN": pd.Series(1.0, index=pl), "USD": pd.Series(4.0, index=pl)}
    rm = aligned(pdta, rates)
    assert not (rm.R == 0.0).any().any()


def test_weekly_frequency_reduces_row_count():
    d = pd.bdate_range("2025-01-01", "2025-03-31")
    close = pd.DataFrame({"A.WA": np.linspace(100, 130, len(d))}, index=d)
    pdta = PriceData(close, close * 0 + 1000, {"A.WA": "PLN"}, [])
    rates = {"PLN": pd.Series(1.0, index=d)}
    daily = aligned(pdta, rates, freq="daily")
    weekly = aligned(pdta, rates, freq="weekly")
    assert len(weekly.R) < len(daily.R) / 4
    assert weekly.freq == "weekly"


def test_periods_per_year_constants():
    assert PERIODS_PER_YEAR == {"daily": 252, "weekly": 52}


def test_too_few_observations_fails_loudly():
    d = pd.to_datetime(["2025-01-02", "2025-01-03"])
    close = pd.DataFrame({"A.WA": [100.0, 101.0]}, index=d)
    pdta = PriceData(close, close * 0 + 1, {"A.WA": "PLN"}, [])
    rates = {"PLN": pd.Series(1.0, index=d)}
    with pytest.raises(ReturnsError, match="60"):
        aligned(pdta, rates, min_obs=60)


def test_missing_currency_raises_returns_error_naming_ticker_and_currency():
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    close = pd.DataFrame({"EU.WA": [100.0, 101.0, 102.0]}, index=d)
    pdta = PriceData(close, close * 0 + 1000.0, {"EU.WA": "EUR"}, [])
    rates = {"PLN": pd.Series(1.0, index=d)}  # no EUR series supplied
    with pytest.raises(ReturnsError) as exc_info:
        aligned(pdta, rates)
    msg = str(exc_info.value)
    assert "EU.WA" in msg
    assert "EUR" in msg


def test_fx_interior_gap_forward_fills_to_last_official_rate():
    # NBP simply doesn't publish on 2025-01-03; the last official rate (4.0) still
    # governs that day, so a 0% FX return there is correct, not fabricated.
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    pdta = make_prices(d, d)
    fx_dates = pd.to_datetime(["2025-01-02", "2025-01-06"])
    rates = {
        "PLN": pd.Series(1.0, index=d),
        "USD": pd.Series([4.0, 4.84], index=fx_dates),
    }
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 0
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [0.0, 0.21], rtol=1e-12)
    # asset +10% both days; day1 fx flat -> 0.10, day2 fx +21% -> 1.10*1.21-1
    np.testing.assert_allclose(rm.R["US"].to_numpy(), [0.10, 1.10 * 1.21 - 1.0], rtol=1e-12)


def test_fx_leading_gap_drops_day_and_counts_it_without_fabricating_a_rate():
    # USD's first published rate is 2025-01-03; 2025-01-02 predates any known rate,
    # so there is nothing to forward-fill from -- that day must be dropped, not
    # back-filled from the later rate.
    d = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    pdta = make_prices(d, d)
    fx_dates = pd.to_datetime(["2025-01-03", "2025-01-06"])
    rates = {
        "PLN": pd.Series(1.0, index=d),
        "USD": pd.Series([4.4, 4.84], index=fx_dates),
    }
    rm = aligned(pdta, rates)
    assert pd.Timestamp("2025-01-02") not in rm.R.index
    assert rm.dropped_days == 1
    assert len(rm.R) == 1
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [4.84 / 4.4 - 1.0], rtol=1e-12)
    # No fabricated rate anywhere: the surviving FX return is the true 10% move,
    # never a manufactured 0.0 standing in for the unobserved 2025-01-02 rate.
    assert not (rm.fx_returns["US"] == 0.0).any()


# --- FX staleness bound (Task 4b) -------------------------------------------------
#
# The leading-gap tests above cover "no rate yet". These cover the opposite edge: a
# rate series that stops. `fx()` serves a stale cache on a failed fetch with no
# staleness channel, so an unbounded forward-fill turns a dead feed into a run of
# exactly-0.0 FX returns and a report that claims success.


def make_shared_prices(dates):
    """One PLN and one USD ticker on a single, gap-free calendar, both +10%/day.

    Gap-free on the price side on purpose: every dropped day in the tests below is
    then attributable to FX alone.
    """
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    close = pd.DataFrame(
        {"PL.WA": 100.0 * 1.1 ** np.arange(n), "US": 50.0 * 1.1 ** np.arange(n)},
        index=idx,
    )
    return PriceData(close, close * 0 + 1000.0, {"PL.WA": "PLN", "US": "USD"}, [])


def test_fx_trailing_gap_beyond_bound_drops_stale_days_and_counts_them():
    # USD stops publishing after 2025-03-05; trading resumes a month later. Carrying
    # 4.84 across those five days would manufacture five 0.0 FX returns.
    trading = pd.to_datetime([
        "2025-03-03", "2025-03-04", "2025-03-05",
        "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11",
    ])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-05"])
    rates = {"USD": pd.Series([4.0, 4.4, 4.84], index=fx_dates)}
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 5
    for stale in trading[3:]:
        assert stale not in rm.R.index
    assert list(rm.R.index) == [pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-05")]
    # No fabricated flat-FX day survived anywhere in the panel.
    assert not (rm.fx_returns["US"] == 0.0).any()
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [0.10, 0.10], rtol=1e-12)


def test_fx_stale_across_whole_window_fails_loudly_instead_of_reporting_success():
    # The pure "network down, week-old cache" case: nothing is salvageable, and the
    # run must not come back looking successful with every FX return equal to 0.0.
    trading = pd.to_datetime(["2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10"])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    rates = {"USD": pd.Series([4.0, 4.4], index=fx_dates)}
    with pytest.raises(ReturnsError, match="fewer than two"):
        aligned(pdta, rates)


def test_fx_trailing_gap_inside_the_bound_is_still_forward_filled():
    # A holiday cluster: NBP publishes nothing for three days, so no new official
    # rate exists and a 0.0 FX return on those days is the truth, not a fabrication.
    # This is the regression guard against "fixing" staleness by dropping every
    # carried day.
    trading = pd.to_datetime([
        "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07",
    ])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    rates = {"USD": pd.Series([4.0, 4.4], index=fx_dates)}
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 0
    assert len(rm.R) == 4
    np.testing.assert_allclose(
        rm.fx_returns["US"].to_numpy(), [0.10, 0.0, 0.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(
        rm.R["US"].to_numpy(), [1.1 * 1.1 - 1.0, 0.10, 0.10, 0.10], rtol=1e-12)


def test_fx_carry_of_exactly_max_staleness_days_is_kept():
    # 2025-03-11 sits exactly 7 calendar days after the last published rate.
    assert (pd.Timestamp("2025-03-11") - pd.Timestamp("2025-03-04")).days == 7
    trading = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-11"])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    rates = {"USD": pd.Series([4.0, 4.4], index=fx_dates)}
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 0
    assert pd.Timestamp("2025-03-11") in rm.R.index
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [0.10, 0.0], rtol=1e-12)


def test_fx_carry_of_one_day_past_max_staleness_is_dropped():
    # Identical to the test above, one calendar day further out: 8 > 7.
    assert (pd.Timestamp("2025-03-12") - pd.Timestamp("2025-03-04")).days == 8
    trading = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-12"])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    rates = {"USD": pd.Series([4.0, 4.4], index=fx_dates)}
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 1
    assert pd.Timestamp("2025-03-12") not in rm.R.index
    assert len(rm.R) == 1
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy(), [0.10], rtol=1e-12)


def test_custom_max_fx_staleness_days_tightens_the_bound():
    # Staleness runs 1, 2, 3 calendar days; a bound of 2 must take only the last day.
    trading = pd.to_datetime([
        "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07",
    ])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    rates = {"USD": pd.Series([4.0, 4.4], index=fx_dates)}
    rm = aligned(pdta, rates, max_fx_staleness_days=2)
    assert rm.dropped_days == 1
    assert pd.Timestamp("2025-03-07") not in rm.R.index
    assert len(rm.R) == 3


def test_custom_max_fx_staleness_days_can_widen_the_bound():
    # Same input as the beyond-bound test; 60 days admits the whole 33-day carry, so
    # the bound is genuinely read from the argument rather than hardcoded to 7.
    trading = pd.to_datetime([
        "2025-03-03", "2025-03-04", "2025-03-05",
        "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11",
    ])
    pdta = make_shared_prices(trading)
    fx_dates = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-05"])
    rates = {"USD": pd.Series([4.0, 4.4, 4.84], index=fx_dates)}
    rm = aligned(pdta, rates, max_fx_staleness_days=60)
    assert rm.dropped_days == 0
    assert len(rm.R) == 7
    np.testing.assert_allclose(rm.fx_returns["US"].to_numpy()[2:], 0.0)


def test_fx_staleness_is_evaluated_per_currency():
    # USD keeps publishing, EUR stops. EUR's staleness must take the day out of the
    # whole panel, not just out of the EUR-denominated column.
    trading = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-14"])
    close = pd.DataFrame(
        {"US": [50.0, 55.0, 60.5], "EU": [20.0, 22.0, 24.2]}, index=trading)
    pdta = PriceData(close, close * 0 + 1000.0, {"US": "USD", "EU": "EUR"}, [])
    rates = {
        "USD": pd.Series([4.0, 4.4, 4.84], index=trading),          # never stale
        "EUR": pd.Series([4.0, 4.4], index=trading[:2]),            # stale by 10 days
    }
    rm = aligned(pdta, rates)
    assert rm.dropped_days == 1
    assert pd.Timestamp("2025-03-14") not in rm.R.index
    assert len(rm.R) == 1
