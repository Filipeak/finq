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
