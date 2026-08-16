from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finq.data import PriceData

PERIODS_PER_YEAR: dict[str, int] = {"daily": 252, "weekly": 52}


class ReturnsError(Exception):
    """Raised when a usable return matrix cannot be built."""


@dataclass(frozen=True)
class ReturnMatrix:
    R: pd.DataFrame
    freq: str
    # Trading days lost to non-common price calendars AND to trading days that
    # precede a currency's first available FX rate (see aligned()).
    dropped_days: int
    tickers: list[str]
    fx_returns: pd.DataFrame


def aligned(price_data: PriceData, fx_rates: dict[str, pd.Series],
            freq: str = "daily", min_obs: int = 0) -> ReturnMatrix:
    if freq not in PERIODS_PER_YEAR:
        raise ReturnsError(f"unknown freq {freq!r}; use 'daily' or 'weekly'")

    close = price_data.close
    union_days = len(close.index)
    common = close.dropna(how="any")           # inner join, never forward-fill
    dropped_days = union_days - len(common)

    if freq == "weekly":
        common = common.resample("W-WED").last().dropna(how="any")

    if len(common) < 2:
        raise ReturnsError("fewer than two common trading days across all tickers")

    tickers = list(common.columns)

    # Align FX onto the trading calendar. NBP publishes on business days only; a
    # missing publication means no new official rate exists, so forward-filling an
    # *interior* gap is correct -- it carries forward the last official rate, and an
    # unchanged official rate genuinely produces a 0% FX return that day. But a
    # trading day that precedes a currency's first published rate has no official
    # rate to carry forward at all: back-filling it from a later rate would fabricate
    # a value and manufacture a return exactly like forward-filling a closed price
    # day would. So that trading day is dropped from the whole panel instead, and
    # folded into dropped_days -- never fabricated.
    fx_on_dates: dict[str, pd.Series] = {}
    for t in tickers:
        code = price_data.currency[t]
        if code == "PLN" or code in fx_on_dates:
            continue
        if code not in fx_rates:
            raise ReturnsError(f"{t}: no FX rate series provided for currency {code!r}")
        s = fx_rates[code].sort_index()
        fx_on_dates[code] = s.reindex(s.index.union(common.index)).ffill().reindex(common.index)

    missing_fx_dates: set = set()
    for s in fx_on_dates.values():
        missing_fx_dates |= set(s.index[s.isna()])

    if missing_fx_dates:
        keep = ~common.index.isin(missing_fx_dates)
        dropped_days += int((~keep).sum())
        common = common.loc[keep]
        fx_on_dates = {code: s.loc[keep] for code, s in fx_on_dates.items()}

    if len(common) < 2:
        raise ReturnsError(
            "fewer than two common trading days remain after dropping days "
            "with no available FX rate"
        )

    asset_ret = common.pct_change().dropna(how="any")

    fx_ret = pd.DataFrame(index=asset_ret.index, columns=tickers, dtype=float)
    for t in tickers:
        code = price_data.currency[t]
        if code == "PLN":
            fx_ret[t] = 0.0
        else:
            fx_ret[t] = fx_on_dates[code].pct_change().reindex(asset_ret.index)

    if fx_ret.isna().any().any():
        raise ReturnsError("FX return series has gaps after alignment; refusing to substitute data")

    pln_ret = (1.0 + asset_ret) * (1.0 + fx_ret) - 1.0

    if len(pln_ret) < min_obs:
        raise ReturnsError(
            f"only {len(pln_ret)} observations after alignment; "
            f"at least {min_obs} required for any estimator here to be meaningful"
        )

    return ReturnMatrix(pln_ret, freq, dropped_days, tickers, fx_ret)
