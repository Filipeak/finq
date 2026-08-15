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

    # Align FX onto the same dates. NBP publishes on business days only; a missing
    # publication means no new official rate exists, so forward-fill is correct here.
    fx_on_dates = {}
    for code, series in fx_rates.items():
        s = series.sort_index()
        joined = s.reindex(s.index.union(common.index)).ffill().reindex(common.index)
        if joined.isna().any():
            joined = joined.bfill()
        if joined.isna().any():
            raise ReturnsError(f"FX {code}: no rate available on some trading days")
        fx_on_dates[code] = joined

    asset_ret = common.pct_change().dropna(how="any")

    fx_ret = pd.DataFrame(index=asset_ret.index, columns=tickers, dtype=float)
    for t in tickers:
        code = price_data.currency[t]
        if code == "PLN":
            fx_ret[t] = 0.0
        else:
            fx_ret[t] = fx_on_dates[code].pct_change().reindex(asset_ret.index).fillna(0.0)

    pln_ret = (1.0 + asset_ret) * (1.0 + fx_ret) - 1.0

    if len(pln_ret) < min_obs:
        raise ReturnsError(
            f"only {len(pln_ret)} observations after alignment; "
            f"at least {min_obs} required for any estimator here to be meaningful"
        )

    return ReturnMatrix(pln_ret, freq, dropped_days, tickers, fx_ret)
