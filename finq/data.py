from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
DEFAULT_CACHE = Path("cache")
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"


class DataError(Exception):
    """Raised when market data cannot be obtained or is unusable."""


@dataclass(frozen=True)
class PriceData:
    close: pd.DataFrame
    volume: pd.DataFrame
    currency: dict[str, str]
    stale: list[str]


def _cache_file(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('^', 'IDX_')}.csv"


def _fetch_yahoo(ticker: str) -> tuple[pd.DataFrame, str]:
    url = f"{_YAHOO}{urllib.parse.quote(ticker)}?range=3y&interval=1d"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        raise DataError(f"{ticker}: fetch failed ({exc})") from exc

    result = payload.get("chart", {}).get("result")
    if not result:
        raise DataError(f"{ticker}: Yahoo returned no data for this symbol")
    res = result[0]
    quote = res.get("indicators", {}).get("quote", [{}])[0]
    if "close" not in quote:
        raise DataError(f"{ticker}: symbol resolves but carries no price series")
    currency = res.get("meta", {}).get("currency")
    if not currency:
        raise DataError(f"{ticker}: no currency reported; symbol is not usable")

    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": quote["close"],
        "volume": quote.get("volume"),
    }).dropna(subset=["close"])
    return df, currency


def _is_cache_fresh(path: Path) -> bool:
    """Check if cache file is fresh (< 1 day old)."""
    if not path.exists():
        return False
    age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
    return age_days < 1


def _load_one(ticker: str, cache_dir: Path) -> tuple[pd.DataFrame, str, bool]:
    """Return (frame, currency, stale). Fresh cache first; refetch when stale,
    falling back to the cache with stale=True only if the refetch itself fails.
    """
    path = _cache_file(cache_dir, ticker)
    cur_path = cache_dir / "currencies.json"
    currencies = json.loads(cur_path.read_text()) if cur_path.exists() else {}

    if path.exists() and _is_cache_fresh(path) and ticker in currencies:
        return pd.read_csv(path, parse_dates=["date"]), currencies[ticker], False

    try:
        df, currency = _fetch_yahoo(ticker)
    except DataError:
        if path.exists() and ticker in currencies:
            return pd.read_csv(path, parse_dates=["date"]), currencies[ticker], True
        raise

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    currencies[ticker] = currency
    cur_path.write_text(json.dumps(currencies, indent=2))
    time.sleep(0.3)
    return df, currency, False


def prices(tickers: list[str], start: str, end: str,
           cache_dir: Path | None = None) -> PriceData:
    """Daily close and volume per ticker. Calendars are NOT aligned here."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    closes, volumes, currency, stale = {}, {}, {}, []
    for t in tickers:
        df, cur, is_stale = _load_one(t, cache_dir)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)].set_index("date")
        if df.empty:
            raise DataError(f"{t}: no observations between {start} and {end}")
        closes[t] = df["close"]
        volumes[t] = df["volume"] if "volume" in df else pd.Series(index=df.index, dtype=float)
        currency[t] = cur
        if is_stale:
            stale.append(t)

    close_df = pd.DataFrame(closes)
    volume_df = pd.DataFrame(volumes).reindex(index=close_df.index, columns=close_df.columns)
    return PriceData(close_df, volume_df, currency, stale)


_NBP = "https://api.nbp.pl/api/exchangerates/rates/a/"


def _fetch_nbp(code: str) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    rows, cursor = [], start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=90), end)
        url = f"{_NBP}{code.lower()}/{cursor.date()}/{stop.date()}/?format=json"
        try:
            r = requests.get(url, timeout=30)
        except Exception as exc:
            raise DataError(f"FX {code}: fetch failed ({exc})") from exc
        if r.status_code == 404 and not rows:
            raise DataError(f"FX {code}: NBP does not publish this currency")
        if r.status_code == 200:
            rows += [(d["effectiveDate"], d["mid"]) for d in r.json()["rates"]]
        elif r.status_code != 404:
            # 404 is tolerated (no publication days in range), but any other non-200 is a hard failure
            raise DataError(f"FX {code}: fetch failed (status {r.status_code} for {cursor.date()} to {stop.date()})")
        cursor = stop + pd.Timedelta(days=1)
        time.sleep(0.3)
    if not rows:
        raise DataError(f"FX {code}: NBP returned no rates")
    return pd.DataFrame(rows, columns=["date", "mid"]).drop_duplicates("date")


def fx(code: str, start: str, end: str, cache_dir: Path | None = None) -> pd.Series:
    """PLN per one unit of `code`, on NBP publication days.

    The returned Series carries an out-of-band staleness flag in
    ``s.attrs["stale"]`` (bool): True only when a refetch was attempted and
    failed, and the cache was served as a fallback -- mirroring
    ``PriceData.stale`` from ``prices()``. Attached via ``.attrs`` rather than
    a wrapper return type so every existing ``isinstance(s, pd.Series)``
    caller keeps working unchanged; callers that care about staleness (the
    report header) read ``s.attrs["stale"]`` explicitly. Without this, a
    network-down fallback to a months-old cache is indistinguishable from a
    fresh fetch to every downstream consumer.
    """
    code = code.upper()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    if code == "PLN":
        idx = pd.date_range(lo, hi, freq="D")
        s = pd.Series(1.0, index=idx, name="PLN")
        s.attrs["stale"] = False
        return s

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    path = cache_dir / f"FX_{code}.csv"
    is_stale = False

    if path.exists() and _is_cache_fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        try:
            df = _fetch_nbp(code)
            df["date"] = pd.to_datetime(df["date"])
        except DataError:
            if path.exists():
                df = pd.read_csv(path, parse_dates=["date"])
                s = df.set_index("date")["mid"].sort_index()
                s = s[(s.index >= lo) & (s.index <= hi)]
                if s.empty:
                    raise DataError(f"FX {code}: no rates between {start} and {end}")
                s = s.rename(code)
                s.attrs["stale"] = True
                return s
            else:
                raise
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    s = df.set_index("date")["mid"].sort_index()
    s = s[(s.index >= lo) & (s.index <= hi)]
    if s.empty:
        raise DataError(f"FX {code}: no rates between {start} and {end}")
    s = s.rename(code)
    s.attrs["stale"] = is_stale
    return s
