"""Liquidity metrics (design spec 6.6).

Liquidity is *reported*, never imposed as an optimizer constraint: at 10-30 held
names the useful question is "could I exit this?", not "constrain the solver".

Everything here is denominated in traded VALUE (close * volume), not share count.
Share counts are not comparable across names -- 500 shares of a 2 PLN stock and
500 shares of a 200 PLN stock are two orders of magnitude apart in exit risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LiquidityError(Exception):
    """Raised when a liquidity metric cannot be computed."""


def adv(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average daily traded value over the trailing window, in quote currency.

    Returns one number per column: mean(close_t * volume_t) over the last
    `window` rows. Value, not share count -- see the module docstring.
    """
    if close.shape != volume.shape:
        raise LiquidityError("close and volume must have the same shape")
    if len(close) < window:
        raise LiquidityError(f"only {len(close)} rows; need at least {window}")
    traded = close * volume
    return traded.tail(window).mean()


def days_to_liquidate(position_value: float, adv_value: float,
                      participation: float = 0.10) -> float:
    """Trading days to exit at a given share of average daily traded value."""
    if adv_value <= 0:
        raise LiquidityError("average daily volume is zero; position cannot be exited")
    if not 0 < participation <= 1:
        raise LiquidityError(f"participation must be in (0, 1], got {participation}")
    return float(position_value / (adv_value * participation))


def amihud(returns: pd.DataFrame, traded_value: pd.DataFrame) -> pd.Series:
    """Amihud illiquidity: mean |return| per unit of traded value, scaled by 1e6.

    Larger means LESS liquid: more price impact per zloty traded. The 1e6 factor
    is the conventional reporting scale -- raw values are ~1e-8 and unreadable.

    Days with zero traded value carry no information about price impact (there was
    no trade to have impact), so they are excluded rather than being allowed to
    divide by zero and poison the mean with an infinity.
    """
    if returns.shape != traded_value.shape:
        raise LiquidityError("returns and traded_value must have the same shape")
    tv = traded_value.replace(0.0, np.nan)
    return (returns.abs() / tv).mean() * 1e6


def stale_price_flag(close: pd.DataFrame, min_run: int = 3) -> dict[str, int]:
    """Count observations inside runs of unchanged closes of at least min_run days.

    Stale prices bias correlation estimates downward, so any non-zero count is a
    warning that this ticker's covariance entries understate its true co-movement
    -- the portfolio will look better diversified than it is (spec section 9).

    `min_run` counts OBSERVATIONS, not price changes: three equal closes in a row
    is a run of three. Every column appears in the result, with 0 for clean
    series, so callers can iterate the dict without losing a ticker.
    """
    flags: dict[str, int] = {}
    for col in close.columns:
        s = close[col].dropna()
        unchanged = s.diff() == 0
        total = 0
        run = 0
        for flag in unchanged:
            if flag:
                run += 1
            else:
                if run + 1 >= min_run:
                    total += run + 1
                run = 0
        if run + 1 >= min_run:
            total += run + 1
        flags[col] = int(total)
    return flags
