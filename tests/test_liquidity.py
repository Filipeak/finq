"""Liquidity metrics (spec 6.6) -- fully offline, deterministic, fixture-backed.

Every RNG here is seeded. The fixture-backed tests read frozen CSVs from
tests/fixtures/ and never touch the network.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finq.data import PriceData
from finq.liquidity import (adv, amihud, days_to_liquidate, stale_price_flag,
                            LiquidityError)

FIX = Path(__file__).parent / "fixtures"


def _panel(*tickers: str) -> PriceData:
    """Build a PriceData from frozen fixture CSVs -- no cache, no network."""
    closes, volumes = {}, {}
    for t in tickers:
        name = t.replace("^", "IDX_")
        df = pd.read_csv(FIX / f"{name}.csv", parse_dates=["date"]).set_index("date")
        closes[t] = df["close"]
        volumes[t] = df["volume"]
    close = pd.DataFrame(closes).dropna(how="any")
    volume = pd.DataFrame(volumes).reindex(close.index)
    return PriceData(close=close, volume=volume,
                     currency={t: "PLN" for t in tickers}, stale=[])


# --------------------------------------------------------------------------
# adv -- average daily traded VALUE, not share count
# --------------------------------------------------------------------------

def test_adv_is_mean_traded_value_over_the_window():
    idx = pd.date_range("2025-01-01", periods=40, freq="B")
    close = pd.DataFrame({"A": np.full(40, 10.0)}, index=idx)
    volume = pd.DataFrame({"A": np.full(40, 1000.0)}, index=idx)
    assert adv(close, volume, window=20)["A"] == pytest.approx(10_000.0)


def test_adv_uses_only_the_trailing_window():
    idx = pd.date_range("2025-01-01", periods=40, freq="B")
    vol = np.concatenate([np.full(20, 1.0), np.full(20, 1000.0)])
    close = pd.DataFrame({"A": np.full(40, 10.0)}, index=idx)
    volume = pd.DataFrame({"A": vol}, index=idx)
    assert adv(close, volume, window=20)["A"] == pytest.approx(10_000.0)


def test_adv_weights_by_price_so_two_names_with_equal_share_volume_differ():
    """Guards against averaging share volume: CHEAP and DEAR trade the same number
    of shares, so a share-volume average would report them identical."""
    idx = pd.date_range("2025-01-01", periods=25, freq="B")
    close = pd.DataFrame({"CHEAP": np.full(25, 2.0), "DEAR": np.full(25, 200.0)},
                         index=idx)
    volume = pd.DataFrame({"CHEAP": np.full(25, 500.0), "DEAR": np.full(25, 500.0)},
                          index=idx)
    a = adv(close, volume, window=20)
    assert a["CHEAP"] == pytest.approx(1_000.0)
    assert a["DEAR"] == pytest.approx(100_000.0)


def test_adv_rejects_a_history_shorter_than_the_window():
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    close = pd.DataFrame({"A": np.full(10, 10.0)}, index=idx)
    volume = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    with pytest.raises(LiquidityError, match="at least 20"):
        adv(close, volume, window=20)


def test_adv_rejects_mismatched_close_and_volume():
    idx = pd.date_range("2025-01-01", periods=25, freq="B")
    close = pd.DataFrame({"A": np.full(25, 10.0), "B": np.full(25, 10.0)}, index=idx)
    volume = pd.DataFrame({"A": np.full(25, 100.0)}, index=idx)
    with pytest.raises(LiquidityError, match="same shape"):
        adv(close, volume, window=20)


# --------------------------------------------------------------------------
# days_to_liquidate
# --------------------------------------------------------------------------

def test_days_to_liquidate_scales_with_position_size():
    assert days_to_liquidate(100_000, 50_000, participation=0.10) == pytest.approx(20.0)
    assert days_to_liquidate(200_000, 50_000, participation=0.10) == pytest.approx(40.0)


def test_days_to_liquidate_scales_inversely_with_participation():
    assert days_to_liquidate(100_000, 50_000, participation=0.20) == pytest.approx(10.0)
    assert days_to_liquidate(100_000, 50_000, participation=1.0) == pytest.approx(2.0)


def test_days_to_liquidate_rejects_zero_volume():
    with pytest.raises(LiquidityError, match="zero"):
        days_to_liquidate(100_000, 0.0)


def test_days_to_liquidate_rejects_participation_outside_the_unit_interval():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(LiquidityError, match="participation"):
            days_to_liquidate(100_000, 50_000, participation=bad)


# --------------------------------------------------------------------------
# amihud
# --------------------------------------------------------------------------

def test_amihud_is_higher_for_the_less_liquid_asset():
    idx = pd.date_range("2025-01-01", periods=50, freq="B")
    rng = np.random.default_rng(50)
    r = rng.normal(scale=0.01, size=50)
    returns = pd.DataFrame({"THIN": r, "DEEP": r}, index=idx)
    traded = pd.DataFrame({"THIN": np.full(50, 1e4), "DEEP": np.full(50, 1e8)}, index=idx)
    a = amihud(returns, traded)
    assert a["THIN"] > a["DEEP"]


def test_amihud_matches_the_closed_form_with_the_1e6_scale():
    """|r| = 0.02 against 1e6 of value every day: 0.02/1e6 * 1e6 = 0.02 exactly.

    Pins the reporting scale, which the ordering test above cannot see.
    """
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    returns = pd.DataFrame({"A": np.full(30, -0.02)}, index=idx)
    traded = pd.DataFrame({"A": np.full(30, 1e6)}, index=idx)
    assert amihud(returns, traded)["A"] == pytest.approx(0.02)


def test_amihud_skips_zero_traded_value_days_rather_than_diverging():
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    traded = pd.DataFrame({"A": [1e6, 0.0, 1e6]}, index=idx)
    got = amihud(returns, traded)["A"]
    assert np.isfinite(got)
    assert got == pytest.approx(0.02)     # mean of 0.01 and 0.03, zero day dropped


def test_amihud_rejects_mismatched_inputs():
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    returns = pd.DataFrame({"A": np.full(5, 0.01), "B": np.full(5, 0.01)}, index=idx)
    traded = pd.DataFrame({"A": np.full(5, 1e6)}, index=idx)
    with pytest.raises(LiquidityError, match="same shape"):
        amihud(returns, traded)


# --------------------------------------------------------------------------
# stale_price_flag
# --------------------------------------------------------------------------

def test_stale_price_flag_counts_unchanged_runs():
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    close = pd.DataFrame({
        "STALE": [10, 10, 10, 10, 11, 12, 13, 14, 15, 16],
        "LIVE": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    }, index=idx, dtype=float)
    flags = stale_price_flag(close, min_run=3)
    assert flags["STALE"] == 4
    assert flags["LIVE"] == 0


def test_stale_price_flag_ignores_short_runs():
    idx = pd.date_range("2025-01-01", periods=6, freq="B")
    close = pd.DataFrame({"A": [10, 10, 11, 12, 13, 14]}, index=idx, dtype=float)
    assert stale_price_flag(close, min_run=3)["A"] == 0


def test_stale_price_flag_counts_a_run_of_exactly_min_run():
    """The boundary: three equal closes IS a run of three, two is not.

    min_run counts OBSERVATIONS, not diffs, so an implementation comparing the
    diff count against min_run is off by one and fails here.
    """
    idx = pd.date_range("2025-01-01", periods=6, freq="B")
    close = pd.DataFrame({
        "EXACT": [10, 10, 10, 11, 12, 13],   # run of 3 observations -> counted
        "SHORT": [10, 10, 11, 12, 13, 14],   # run of 2 observations -> not counted
    }, index=idx, dtype=float)
    flags = stale_price_flag(close, min_run=3)
    assert flags["EXACT"] == 3
    assert flags["SHORT"] == 0


def test_stale_price_flag_counts_a_run_that_ends_the_series():
    idx = pd.date_range("2025-01-01", periods=6, freq="B")
    close = pd.DataFrame({"A": [10, 11, 12, 13, 13, 13]}, index=idx, dtype=float)
    assert stale_price_flag(close, min_run=3)["A"] == 3


def test_stale_price_flag_sums_multiple_runs_in_one_series():
    idx = pd.date_range("2025-01-01", periods=11, freq="B")
    close = pd.DataFrame(
        {"A": [10, 10, 10, 11, 12, 20, 20, 20, 20, 21, 22]}, index=idx, dtype=float
    )
    assert stale_price_flag(close, min_run=3)["A"] == 7     # 3 + 4


def test_stale_price_flag_reports_every_ticker_including_clean_ones():
    """A clean ticker must appear with 0, not be absent -- callers iterate the dict
    to build the report, and a missing key would silently drop the ticker."""
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    flags = stale_price_flag(close, min_run=3)
    assert set(flags) == {"A"}
    assert flags["A"] == 0


# --------------------------------------------------------------------------
# Fixture-backed: real frozen WSE data, offline
# --------------------------------------------------------------------------

def test_thin_etf_takes_far_longer_to_exit_than_a_blue_chip():
    pdata = _panel("PKO.WA", "ETFBW20TR.WA")
    a = adv(pdata.close, pdata.volume, window=20)
    position = 1_000_000.0
    blue_chip = days_to_liquidate(position, a["PKO.WA"])
    thin_etf = days_to_liquidate(position, a["ETFBW20TR.WA"])
    assert thin_etf > 10 * blue_chip


def test_fixture_panel_flags_the_thin_etf_and_clears_the_blue_chip():
    """ETFBW20TR.WA has a genuine three-day run of unchanged closes in the frozen
    panel; PKO.WA's longest run is two days, which must NOT be flagged."""
    pdata = _panel("PKO.WA", "ETFBW20TR.WA")
    flags = stale_price_flag(pdata.close, min_run=3)
    assert flags["ETFBW20TR.WA"] > 0
    assert flags["PKO.WA"] == 0
