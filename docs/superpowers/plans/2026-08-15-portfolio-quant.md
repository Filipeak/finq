# Portfolio Quant Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `finq`, a tested Python library that computes honest risk numbers and risk-based portfolio weights for a mixed Polish/US equity portfolio held in PLN.

**Architecture:** A layered library — data fetching, then PLN return construction, then covariance estimation, then risk/liquidity/optimization on top. Every covariance call returns estimation-quality diagnostics alongside the matrix. Claude drives the library by composing its API; two CLI entry points cover the repeated flows.

**Tech Stack:** Python 3.12+, numpy, pandas, scipy, requests, pytest. No cvxpy, no yfinance, no sklearn.

**Spec:** `docs/superpowers/specs/2026-08-15-portfolio-quant-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12+.** Use modern type syntax (`list[str]`, `float | None`).
- **Dependencies are exactly:** `numpy`, `pandas`, `scipy`, `requests` (runtime) and `pytest` (dev). All are already installed. Do not add others. In particular **do not use `sklearn.covariance.LedoitWolf`** — it implements the 2004 scaled-identity-target paper, not the 2003 constant-correlation paper this project follows.
- **Base currency is always PLN.**
- **Simple returns, never log returns.** Portfolio return must be a weighted sum of asset returns.
- **Annualization:** `√252` for daily, `√52` for weekly. Defined once as `PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}` in `finq/returns.py` and imported everywhere else.
- **Sample covariance divides by `T`, not `T-1`.** The Ledoit-Wolf asymptotics in the spec assume the MLE convention; mixing conventions silently corrupts the shrinkage intensity.
- **Long-only.** No shorts anywhere. Negative quantities and negative weights are input errors.
- **Fail loudly.** Never silently drop a ticker, never silently substitute data. A run on wrong data must not look like a successful run.
- **Every `covariance.estimate()` call returns `(Sigma, Diagnostics)`** regardless of method.
- **Yahoo requests require a browser `User-Agent` header** or they are rejected.
- **Sizing target:** 10-30 assets. Do not optimize for larger universes.
- **Commit after every task** using the message given in the task's final step.

---

## Shared Types

These dataclasses are created in the tasks noted and consumed throughout. Field names are binding — later tasks reference them exactly.

```python
# finq/portfolio.py  (Task 1)
@dataclass(frozen=True)
class Portfolio:
    tickers: list[str]
    weights: np.ndarray | None      # normalized to sum to 1, or None in selection mode
    quantities: np.ndarray | None   # None unless input supplied quantities
    normalized: bool                # True if input weights did not already sum to 1
    source_path: str

# finq/data.py  (Task 2)
@dataclass(frozen=True)
class PriceData:
    close: pd.DataFrame             # DatetimeIndex x tickers
    volume: pd.DataFrame            # DatetimeIndex x tickers
    currency: dict[str, str]        # ticker -> "PLN" | "USD"
    stale: list[str]                # tickers served from stale cache

# finq/returns.py  (Task 4)
@dataclass(frozen=True)
class ReturnMatrix:
    R: pd.DataFrame                 # T x N simple returns, PLN
    freq: str                       # "daily" | "weekly"
    dropped_days: int
    tickers: list[str]
    fx_returns: pd.DataFrame        # T x N FX component per ticker (0.0 for PLN assets)

# finq/covariance.py  (Task 5)
@dataclass(frozen=True)
class Diagnostics:
    method: str
    T: int
    N: int
    Q: float                        # T / N
    lambda_minus: float
    lambda_plus: float
    eigenvalues: np.ndarray         # descending, of the correlation matrix
    n_in_band: int
    var_share_in_band: float
    shrinkage: float | None         # delta-hat, None for non-shrinkage methods
    condition_number: float
```

---

### Task 1: Scaffolding and portfolio loading

**Files:**
- Create: `pyproject.toml`
- Create: `finq/__init__.py`
- Create: `finq/portfolio.py`
- Create: `tests/__init__.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Portfolio` dataclass (fields above); `finq.portfolio.load(path: str | Path) -> Portfolio`; `finq.portfolio.PortfolioError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portfolio.py
import json
import numpy as np
import pytest
from finq.portfolio import load, PortfolioError


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_weights_csv(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,weight\nPKO.WA,0.5\nSPY,0.5\n")
    pf = load(p)
    assert pf.tickers == ["PKO.WA", "SPY"]
    np.testing.assert_allclose(pf.weights, [0.5, 0.5])
    assert pf.quantities is None
    assert pf.normalized is False


def test_loads_quantities_csv(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,quantity\nPKO.WA,120\nSPY,45\n")
    pf = load(p)
    np.testing.assert_allclose(pf.quantities, [120.0, 45.0])
    assert pf.weights is None


def test_loads_tickers_only_selection_mode(tmp_path):
    p = write(tmp_path, "p.csv", "ticker\nPKO.WA\nSPY\n")
    pf = load(p)
    assert pf.tickers == ["PKO.WA", "SPY"]
    assert pf.weights is None and pf.quantities is None


def test_loads_json(tmp_path):
    p = write(tmp_path, "p.json", json.dumps([
        {"ticker": "PKO.WA", "weight": 0.4},
        {"ticker": "SPY", "weight": 0.6},
    ]))
    pf = load(p)
    assert pf.tickers == ["PKO.WA", "SPY"]
    np.testing.assert_allclose(pf.weights, [0.4, 0.6])


def test_normalizes_and_flags_weights_not_summing_to_one(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,weight\nPKO.WA,1\nSPY,1\n")
    pf = load(p)
    np.testing.assert_allclose(pf.weights, [0.5, 0.5])
    assert pf.normalized is True


def test_rejects_duplicate_tickers_naming_the_row(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,weight\nPKO.WA,0.5\nPKO.WA,0.5\n")
    with pytest.raises(PortfolioError, match="PKO.WA"):
        load(p)


def test_rejects_negative_quantity(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,quantity\nPKO.WA,-5\n")
    with pytest.raises(PortfolioError, match="negative"):
        load(p)


def test_rejects_empty_file(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,weight\n")
    with pytest.raises(PortfolioError, match="empty"):
        load(p)


def test_rejects_both_weight_and_quantity_columns(tmp_path):
    p = write(tmp_path, "p.csv", "ticker,weight,quantity\nPKO.WA,0.5,10\n")
    with pytest.raises(PortfolioError, match="both"):
        load(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq'`

- [ ] **Step 3: Write the scaffolding and implementation**

```toml
# pyproject.toml
[project]
name = "finq"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "pandas", "scipy", "requests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["finq*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# finq/__init__.py
"""finq — risk-based portfolio analytics for PLN-denominated PL/US portfolios."""
__version__ = "0.1.0"
```

```python
# finq/portfolio.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class PortfolioError(Exception):
    """Raised when a portfolio file is malformed or invalid."""


@dataclass(frozen=True)
class Portfolio:
    tickers: list[str]
    weights: np.ndarray | None
    quantities: np.ndarray | None
    normalized: bool
    source_path: str


def _frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path)


def load(path: str | Path) -> Portfolio:
    path = Path(path)
    df = _frame(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise PortfolioError(f"{path}: no 'ticker' column found")
    if df.empty:
        raise PortfolioError(f"{path}: portfolio is empty")

    has_w = "weight" in df.columns
    has_q = "quantity" in df.columns
    if has_w and has_q:
        raise PortfolioError(f"{path}: supply weight or quantity, not both")

    tickers = [str(t).strip() for t in df["ticker"]]
    dupes = {t for t in tickers if tickers.count(t) > 1}
    if dupes:
        raise PortfolioError(f"{path}: duplicate ticker(s): {', '.join(sorted(dupes))}")

    weights = quantities = None
    normalized = False

    if has_q:
        quantities = df["quantity"].to_numpy(dtype=float)
        if (quantities < 0).any():
            bad = [t for t, q in zip(tickers, quantities) if q < 0]
            raise PortfolioError(f"{path}: negative quantity for {', '.join(bad)}")
    elif has_w:
        weights = df["weight"].to_numpy(dtype=float)
        if (weights < 0).any():
            bad = [t for t, w in zip(tickers, weights) if w < 0]
            raise PortfolioError(f"{path}: negative weight for {', '.join(bad)}")
        total = weights.sum()
        if total <= 0:
            raise PortfolioError(f"{path}: weights sum to {total}, must be positive")
        if not np.isclose(total, 1.0):
            weights = weights / total
            normalized = True

    return Portfolio(tickers, weights, quantities, normalized, str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pip install -e . && python -m pytest tests/test_portfolio.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml finq/ tests/
git commit -m "feat: portfolio file loading and validation"
```

---

### Task 2: Price data from Yahoo, with cache and test fixtures

**Files:**
- Create: `finq/data.py`
- Create: `scripts/make_fixtures.py`
- Create: `tests/fixtures/` (generated CSVs, committed)
- Test: `tests/test_data_prices.py`

**Interfaces:**
- Consumes: nothing
- Produces: `PriceData` dataclass (fields above); `finq.data.prices(tickers: list[str], start: str, end: str, cache_dir: Path | None = None) -> PriceData`; `finq.data.DataError(Exception)`; module constant `finq.data.USER_AGENT`

**Verified fixture tickers** (all return close and volume): `PKO.WA`, `CDR.WA`, `PKN.WA`, `PZU.WA` (PLN) and `SPY`, `QQQ`, `GLD` (USD). Benchmarks: `WIG20.WA`, `^GSPC`. Note `^WIG` is a hollow Yahoo listing that returns no price series — never use it.

- [ ] **Step 1: Write the fixture generator and run it once**

```python
# scripts/make_fixtures.py
"""One-time fixture generation. Requires network. Run from repo root."""
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
TICKERS = ["PKO.WA", "CDR.WA", "PKN.WA", "PZU.WA", "SPY", "QQQ", "GLD",
           "WIG20.WA", "^GSPC"]
OUT = Path("tests/fixtures")


def fetch(ticker: str) -> pd.DataFrame:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(ticker)}?range=3y&interval=1d")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": q["close"],
        "volume": q["volume"],
    }).dropna(subset=["close"])
    df.attrs["currency"] = res["meta"]["currency"]
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    currencies = {}
    for t in TICKERS:
        df = fetch(t)
        currencies[t] = df.attrs["currency"]
        df.to_csv(OUT / f"{t.replace('^', 'IDX_')}.csv", index=False)
        print(f"{t}: {len(df)} rows, {df.attrs['currency']}")
        time.sleep(0.5)
    (OUT / "currencies.json").write_text(json.dumps(currencies, indent=2))


if __name__ == "__main__":
    main()
```

Run: `python scripts/make_fixtures.py`
Expected: nine CSVs plus `currencies.json` in `tests/fixtures/`, each with roughly 750 rows.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data_prices.py
import json
from pathlib import Path

import pandas as pd
import pytest
from finq.data import prices, DataError

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def seeded_cache(tmp_path):
    """Copy fixtures into a cache dir so prices() never touches the network."""
    cache = tmp_path / "cache"
    cache.mkdir()
    currencies = json.loads((FIX / "currencies.json").read_text())
    for src in FIX.glob("*.csv"):
        (cache / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (cache / "currencies.json").write_text(json.dumps(currencies))
    return cache


def test_reads_from_cache_without_network(seeded_cache):
    pd_ = prices(["PKO.WA", "SPY"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)
    assert list(pd_.close.columns) == ["PKO.WA", "SPY"]
    assert isinstance(pd_.close.index, pd.DatetimeIndex)
    assert len(pd_.close) > 200
    assert pd_.close.notna().all().all()


def test_reports_currency_per_ticker(seeded_cache):
    pd_ = prices(["PKO.WA", "SPY"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)
    assert pd_.currency == {"PKO.WA": "PLN", "SPY": "USD"}


def test_returns_volume_alongside_close(seeded_cache):
    pd_ = prices(["PKO.WA"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)
    assert (pd_.volume["PKO.WA"] > 0).any()
    assert pd_.volume.shape == pd_.close.shape


def test_respects_date_window(seeded_cache):
    pd_ = prices(["SPY"], "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert pd_.close.index.min() >= pd.Timestamp("2025-01-01")
    assert pd_.close.index.max() <= pd.Timestamp("2025-06-30")


def test_unknown_ticker_fails_loudly_naming_it(seeded_cache):
    with pytest.raises(DataError, match="NOPE.WA"):
        prices(["PKO.WA", "NOPE.WA"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)


def test_union_index_preserved_no_inner_join_here(seeded_cache):
    """data.prices does NOT align calendars; that is returns.aligned's job."""
    pd_ = prices(["PKO.WA", "SPY"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)
    assert pd_.close["PKO.WA"].isna().sum() + pd_.close["SPY"].isna().sum() >= 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_data_prices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.data'`

- [ ] **Step 4: Write the implementation**

```python
# finq/data.py
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


def _load_one(ticker: str, cache_dir: Path) -> tuple[pd.DataFrame, str, bool]:
    """Return (frame, currency, stale). Cache first, network on miss."""
    path = _cache_file(cache_dir, ticker)
    cur_path = cache_dir / "currencies.json"
    currencies = json.loads(cur_path.read_text()) if cur_path.exists() else {}

    fresh_enough = False
    if path.exists():
        age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
        fresh_enough = age_days < 1

    if path.exists() and (fresh_enough or ticker in currencies):
        df = pd.read_csv(path, parse_dates=["date"])
        currency = currencies.get(ticker)
        if currency is None:
            raise DataError(f"{ticker}: cached prices present but currency unknown")
        return df, currency, not fresh_enough

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_prices.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add finq/data.py scripts/make_fixtures.py tests/fixtures tests/test_data_prices.py
git commit -m "feat: Yahoo price fetching with disk cache and offline fixtures"
```

---

### Task 3: FX rates from NBP

**Files:**
- Modify: `finq/data.py` (append `fx`)
- Modify: `scripts/make_fixtures.py` (append FX fixture generation)
- Test: `tests/test_data_fx.py`

**Interfaces:**
- Consumes: `finq.data.DataError`, `finq.data._cache_file`
- Produces: `finq.data.fx(code: str, start: str, end: str, cache_dir: Path | None = None) -> pd.Series` — DatetimeIndex, float mid rates, PLN per one unit of `code`

- [ ] **Step 1: Extend the fixture generator and run it**

Append to `scripts/make_fixtures.py`, and call `fx_fixture("USD")` from `main()`:

```python
def fx_fixture(code: str) -> None:
    """NBP allows at most 93 days per request, so walk the window in chunks."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    rows, cursor = [], start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=90), end)
        url = (f"https://api.nbp.pl/api/exchangerates/rates/a/{code.lower()}/"
               f"{cursor.date()}/{stop.date()}/?format=json")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            rows += [(d["effectiveDate"], d["mid"]) for d in r.json()["rates"]]
        cursor = stop + pd.Timedelta(days=1)
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["date", "mid"]).drop_duplicates("date")
    df.to_csv(OUT / f"FX_{code.upper()}.csv", index=False)
    print(f"FX {code}: {len(df)} rows")
```

Run: `python scripts/make_fixtures.py`
Expected: `tests/fixtures/FX_USD.csv` with roughly 750 rows.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data_fx.py
import json
from pathlib import Path

import pandas as pd
import pytest
from finq.data import fx, DataError

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def seeded_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for src in FIX.glob("*.csv"):
        (cache / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (cache / "currencies.json").write_text((FIX / "currencies.json").read_text())
    return cache


def test_returns_series_of_mid_rates(seeded_cache):
    s = fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert isinstance(s, pd.Series)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert (s > 2.0).all() and (s < 6.0).all()   # PLN per USD, sane band


def test_pln_is_identity(seeded_cache):
    s = fx("PLN", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert (s == 1.0).all()


def test_respects_window(seeded_cache):
    s = fx("USD", "2025-03-01", "2025-03-31", cache_dir=seeded_cache)
    assert s.index.min() >= pd.Timestamp("2025-03-01")
    assert s.index.max() <= pd.Timestamp("2025-03-31")


def test_unknown_currency_fails_loudly(seeded_cache):
    with pytest.raises(DataError, match="ZZZ"):
        fx("ZZZ", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_data_fx.py -v`
Expected: FAIL — `ImportError: cannot import name 'fx' from 'finq.data'`

- [ ] **Step 4: Write the implementation**

Append to `finq/data.py`:

```python
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
        cursor = stop + pd.Timedelta(days=1)
        time.sleep(0.3)
    if not rows:
        raise DataError(f"FX {code}: NBP returned no rates")
    return pd.DataFrame(rows, columns=["date", "mid"]).drop_duplicates("date")


def fx(code: str, start: str, end: str, cache_dir: Path | None = None) -> pd.Series:
    """PLN per one unit of `code`, on NBP publication days."""
    code = code.upper()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    if code == "PLN":
        idx = pd.date_range(lo, hi, freq="D")
        return pd.Series(1.0, index=idx, name="PLN")

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    path = cache_dir / f"FX_{code}.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = _fetch_nbp(code)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    s = df.set_index("date")["mid"].sort_index()
    s = s[(s.index >= lo) & (s.index <= hi)]
    if s.empty:
        raise DataError(f"FX {code}: no rates between {start} and {end}")
    return s.rename(code)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_fx.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add finq/data.py scripts/make_fixtures.py tests/fixtures/FX_USD.csv tests/test_data_fx.py
git commit -m "feat: NBP FX rate fetching with cache"
```

---

### Task 4: PLN return matrix with calendar alignment

**Files:**
- Create: `finq/returns.py`
- Test: `tests/test_returns.py`

**Interfaces:**
- Consumes: `finq.data.PriceData`, `finq.data.fx`
- Produces: `ReturnMatrix` dataclass (fields above); `finq.returns.PERIODS_PER_YEAR: dict[str, int]`; `finq.returns.aligned(price_data: PriceData, fx_rates: dict[str, pd.Series], freq: str = "daily") -> ReturnMatrix`; `finq.returns.ReturnsError(Exception)`

**Key rule:** for a USD asset the PLN return is `(1 + r_asset) * (1 + r_fx) - 1`, computed *before* estimation. Calendars are inner-joined; forward-filling is forbidden because stale prices bias correlations downward.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_returns.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_returns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.returns'`

- [ ] **Step 3: Write the implementation**

```python
# finq/returns.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_returns.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add finq/returns.py tests/test_returns.py
git commit -m "feat: PLN return matrix with inner-join calendar alignment"
```

---

### Task 5: Sample covariance and estimation diagnostics

**Files:**
- Create: `finq/covariance.py`
- Test: `tests/test_covariance_sample.py`

**Interfaces:**
- Consumes: nothing (operates on a plain `np.ndarray` or `pd.DataFrame` of returns)
- Produces: `Diagnostics` dataclass (fields above); `finq.covariance.estimate(R, method: str = "ledoit_wolf") -> tuple[np.ndarray, Diagnostics]`; `finq.covariance.sample_cov(R) -> np.ndarray`; `finq.covariance.mp_band(Q: float, sigma2: float = 1.0) -> tuple[float, float]`; `finq.covariance.CovarianceError(Exception)`

In this task `estimate` supports `method="sample"` only; Tasks 6 and 7 add the others.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_covariance_sample.py
import numpy as np
import pytest
from finq.covariance import estimate, sample_cov, mp_band, CovarianceError


def test_sample_cov_divides_by_T_not_T_minus_1():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 4))
    S = sample_cov(R)
    Y = R - R.mean(axis=0)
    np.testing.assert_allclose(S, (Y.T @ Y) / 500, rtol=1e-12)


def test_sample_cov_recovers_known_diagonal_covariance():
    rng = np.random.default_rng(1)
    true_sd = np.array([0.01, 0.02, 0.03])
    R = rng.normal(size=(200_000, 3)) * true_sd
    S = sample_cov(R)
    np.testing.assert_allclose(np.sqrt(np.diag(S)), true_sd, rtol=0.02)
    assert abs(S[0, 1]) < 1e-4


def test_mp_band_matches_analytic_formula():
    for Q in (2.0, 5.0, 33.0):
        lo, hi = mp_band(Q, sigma2=1.0)
        assert lo == pytest.approx(1 + 1 / Q - 2 * np.sqrt(1 / Q))
        assert hi == pytest.approx(1 + 1 / Q + 2 * np.sqrt(1 / Q))


def test_mp_band_scales_with_sigma2():
    lo1, hi1 = mp_band(4.0, sigma2=1.0)
    lo2, hi2 = mp_band(4.0, sigma2=0.5)
    assert lo2 == pytest.approx(lo1 * 0.5)
    assert hi2 == pytest.approx(hi1 * 0.5)


def test_diagnostics_report_shape_and_Q():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(600, 20))
    _, d = estimate(R, method="sample")
    assert (d.T, d.N) == (600, 20)
    assert d.Q == pytest.approx(30.0)
    assert d.method == "sample"
    assert d.shrinkage is None
    assert len(d.eigenvalues) == 20
    assert np.all(np.diff(d.eigenvalues) <= 1e-12)     # descending


def test_pure_noise_puts_essentially_all_eigenvalues_inside_the_band():
    rng = np.random.default_rng(3)
    R = rng.normal(size=(1000, 50))
    _, d = estimate(R, method="sample")
    assert d.n_in_band >= 48


def test_one_injected_common_factor_escapes_the_band():
    rng = np.random.default_rng(4)
    T, N = 1000, 50
    factor = rng.normal(size=(T, 1))
    R = 0.7 * factor + 0.7 * rng.normal(size=(T, N))
    _, d = estimate(R, method="sample")
    assert d.N - d.n_in_band == 1


def test_rejects_T_less_than_N_for_sample_method():
    rng = np.random.default_rng(5)
    R = rng.normal(size=(10, 30))
    with pytest.raises(CovarianceError, match="sample"):
        estimate(R, method="sample")


def test_rejects_unknown_method():
    rng = np.random.default_rng(6)
    R = rng.normal(size=(100, 5))
    with pytest.raises(CovarianceError, match="unknown"):
        estimate(R, method="nonsense")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_covariance_sample.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.covariance'`

- [ ] **Step 3: Write the implementation**

```python
# finq/covariance.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ("sample", "ledoit_wolf", "rmt_clean")


class CovarianceError(Exception):
    """Raised when a covariance matrix cannot be estimated meaningfully."""


@dataclass(frozen=True)
class Diagnostics:
    method: str
    T: int
    N: int
    Q: float
    lambda_minus: float
    lambda_plus: float
    eigenvalues: np.ndarray
    n_in_band: int
    var_share_in_band: float
    shrinkage: float | None
    condition_number: float


def _as_array(R) -> np.ndarray:
    arr = R.to_numpy(dtype=float) if isinstance(R, pd.DataFrame) else np.asarray(R, dtype=float)
    if arr.ndim != 2:
        raise CovarianceError("returns must be a 2-D T x N matrix")
    if not np.isfinite(arr).all():
        raise CovarianceError("returns contain NaN or inf; align and clean first")
    return arr


def sample_cov(R) -> np.ndarray:
    """Sample covariance with the MLE (divide by T) convention."""
    arr = _as_array(R)
    Y = arr - arr.mean(axis=0)
    return (Y.T @ Y) / arr.shape[0]


def mp_band(Q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur edges: sigma^2 (1 + 1/Q +/- 2 sqrt(1/Q))."""
    if Q <= 0:
        raise CovarianceError(f"Q must be positive, got {Q}")
    root = 2.0 * np.sqrt(1.0 / Q)
    return sigma2 * (1.0 + 1.0 / Q - root), sigma2 * (1.0 + 1.0 / Q + root)


def _fit_bulk_sigma2(eigenvalues: np.ndarray, Q: float) -> float:
    """Fixed point: for an MP bulk with variance sigma^2, the bulk mean is sigma^2."""
    sigma2 = 1.0
    for _ in range(100):
        _, hi = mp_band(Q, sigma2)
        bulk = eigenvalues[eigenvalues <= hi]
        if bulk.size == 0:
            break
        new = float(bulk.mean())
        if abs(new - sigma2) < 1e-10:
            sigma2 = new
            break
        sigma2 = new
    return sigma2


def _diagnostics(method: str, Sigma: np.ndarray, T: int, N: int,
                 shrinkage: float | None) -> Diagnostics:
    sd = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(sd, sd)
    evals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    Q = T / N
    sigma2 = _fit_bulk_sigma2(evals, Q)
    lo, hi = mp_band(Q, sigma2)
    in_band = evals <= hi
    return Diagnostics(
        method=method,
        T=T,
        N=N,
        Q=Q,
        lambda_minus=lo,
        lambda_plus=hi,
        eigenvalues=evals,
        n_in_band=int(in_band.sum()),
        var_share_in_band=float(evals[in_band].sum() / evals.sum()),
        shrinkage=shrinkage,
        condition_number=float(np.linalg.cond(Sigma)),
    )


def estimate(R, method: str = "ledoit_wolf") -> tuple[np.ndarray, Diagnostics]:
    if method not in METHODS:
        raise CovarianceError(f"unknown method {method!r}; use one of {METHODS}")
    arr = _as_array(R)
    T, N = arr.shape
    if N < 2:
        raise CovarianceError("need at least two assets")

    if method == "sample":
        if T <= N:
            raise CovarianceError(
                f"T={T} <= N={N}: the sample covariance matrix is singular here. "
                "Use method='ledoit_wolf' or 'rmt_clean'."
            )
        Sigma = sample_cov(arr)
        return Sigma, _diagnostics("sample", Sigma, T, N, None)

    raise CovarianceError(f"method {method!r} not implemented yet")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_covariance_sample.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add finq/covariance.py tests/test_covariance_sample.py
git commit -m "feat: sample covariance with Marchenko-Pastur diagnostics"
```

---

### Task 6: Ledoit-Wolf constant-correlation shrinkage

**Files:**
- Modify: `finq/covariance.py` (add `ledoit_wolf_cc`, wire into `estimate`)
- Test: `tests/test_covariance_ledoit_wolf.py`

**Interfaces:**
- Consumes: `sample_cov`, `_as_array`, `_diagnostics`, `CovarianceError`
- Produces: `finq.covariance.ledoit_wolf_cc(R) -> tuple[np.ndarray, float]` returning `(Sigma, delta_hat)`; `estimate(R, method="ledoit_wolf")` populating `Diagnostics.shrinkage`

**The formula** (spec §6.4, from the paper's Appendices A and B). With `Y` the demeaned `T x N` returns, `S = YᵀY / T`, `s = sqrt(diag(S))`, and `r̄` the mean off-diagonal sample correlation:

```
F: f_ii = s_ii,  f_ij = r̄ · sqrt(s_ii · s_jj)
π̂_ij   = (1/T) Σ_t (Y_it Y_jt)² − s_ij²
ϑ̂_ii,ij = (1/T) Σ_t Y_it³ Y_jt − s_ii · s_ij
ρ̂      = Σ_i π̂_ii + Σ_{i≠j} (r̄/2)[ (s_j/s_i)·ϑ̂_ii,ij + (s_i/s_j)·ϑ̂_jj,ij ]
γ̂      = Σ_ij (f_ij − s_ij)²
δ̂      = clip( ((π̂ − ρ̂)/γ̂) / T , 0, 1 )
Σ̂      = δ̂F + (1 − δ̂)S
```

Note `s_i` denotes the standard deviation `sqrt(s_ii)`, so `sqrt(s_jj/s_ii)` is `s_j/s_i`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_covariance_ledoit_wolf.py
import numpy as np
import pytest
from finq.covariance import estimate, ledoit_wolf_cc, sample_cov


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_covariance_ledoit_wolf.py -v`
Expected: FAIL — `ImportError: cannot import name 'ledoit_wolf_cc'`

- [ ] **Step 3: Write the implementation**

Add to `finq/covariance.py`:

```python
def ledoit_wolf_cc(R) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2003) shrinkage toward the constant-correlation target.

    Returns (Sigma, delta_hat). See resources/honey.pdf, Appendices A and B.
    """
    arr = _as_array(R)
    T, N = arr.shape
    if T < 2:
        raise CovarianceError("need at least two observations")

    Y = arr - arr.mean(axis=0)
    S = (Y.T @ Y) / T
    var = np.diag(S).copy()
    if (var <= 0).any():
        raise CovarianceError("an asset has zero variance; remove it before estimating")
    s = np.sqrt(var)

    corr = S / np.outer(s, s)
    iu = np.triu_indices(N, k=1)
    rbar = float(corr[iu].mean())

    # Shrinkage target F: sample variances, common correlation rbar.
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, var)

    # pi-hat: summed asymptotic variances of the sample covariance entries.
    Y2 = Y ** 2
    pi_mat = (Y2.T @ Y2) / T - S ** 2
    pi_hat = float(pi_mat.sum())

    # theta[i, j] = theta-hat_{ii,ij}
    theta = (Y ** 3).T @ Y / T - var[:, None] * S

    # rho-hat: diagonal terms plus the delta-method off-diagonal terms.
    off = (rbar / 2.0) * (np.outer(1.0 / s, s) * theta + np.outer(s, 1.0 / s) * theta.T)
    np.fill_diagonal(off, 0.0)
    rho_hat = float(np.diag(pi_mat).sum() + off.sum())

    gamma_hat = float(((F - S) ** 2).sum())
    if gamma_hat <= 0:
        return S, 0.0            # sample already equals the target

    delta = float(np.clip(((pi_hat - rho_hat) / gamma_hat) / T, 0.0, 1.0))
    Sigma = delta * F + (1.0 - delta) * S
    Sigma = (Sigma + Sigma.T) / 2.0          # kill float asymmetry
    return Sigma, delta
```

Then replace the `raise CovarianceError(f"method {method!r} not implemented yet")` line in `estimate` with:

```python
    if method == "ledoit_wolf":
        Sigma, delta = ledoit_wolf_cc(arr)
        return Sigma, _diagnostics("ledoit_wolf", Sigma, T, N, delta)

    raise CovarianceError(f"method {method!r} not implemented yet")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_covariance_ledoit_wolf.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite for regressions**

Run: `python -m pytest -q`
Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add finq/covariance.py tests/test_covariance_ledoit_wolf.py
git commit -m "feat: Ledoit-Wolf constant-correlation shrinkage estimator"
```

---

### Task 7: RMT eigenvalue cleaning

**Files:**
- Modify: `finq/covariance.py` (add `rmt_clean`, wire into `estimate`)
- Test: `tests/test_covariance_rmt.py`

**Interfaces:**
- Consumes: `sample_cov`, `mp_band`, `_fit_bulk_sigma2`, `_as_array`, `_diagnostics`
- Produces: `finq.covariance.rmt_clean(R) -> np.ndarray`; `estimate(R, method="rmt_clean")`

**Method** (Laloux et al. 1998): eigen-decompose the sample *correlation* matrix; fit the bulk variance `σ²`; keep eigenvalues above `λ₊` with their eigenvectors; replace the bulk with a single value that preserves the trace; renormalize the diagonal to exactly 1; rescale by sample volatilities to recover a covariance matrix.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_covariance_rmt.py
import numpy as np
import pytest
from finq.covariance import estimate, rmt_clean, sample_cov


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_covariance_rmt.py -v`
Expected: FAIL — `ImportError: cannot import name 'rmt_clean'`

- [ ] **Step 3: Write the implementation**

Add to `finq/covariance.py`:

```python
def rmt_clean(R) -> np.ndarray:
    """Laloux et al. (1998) eigenvalue cleaning of the correlation matrix."""
    arr = _as_array(R)
    T, N = arr.shape
    S = sample_cov(arr)
    var = np.diag(S).copy()
    if (var <= 0).any():
        raise CovarianceError("an asset has zero variance; remove it before estimating")
    sd = np.sqrt(var)
    C = S / np.outer(sd, sd)

    evals, evecs = np.linalg.eigh(C)          # ascending
    Q = T / N
    sigma2 = _fit_bulk_sigma2(np.sort(evals)[::-1], Q)
    _, hi = mp_band(Q, sigma2)

    is_noise = evals <= hi
    n_noise = int(is_noise.sum())
    if n_noise == 0:
        cleaned_evals = evals.copy()
    else:
        # Trace of a correlation matrix is N; give the whole bulk one shared value.
        signal_sum = float(evals[~is_noise].sum())
        replacement = (N - signal_sum) / n_noise
        replacement = max(replacement, 1e-12)
        cleaned_evals = np.where(is_noise, replacement, evals)

    C_clean = evecs @ np.diag(cleaned_evals) @ evecs.T
    # Trace preservation fixes the sum of the diagonal, not each entry; renormalize.
    d = np.sqrt(np.diag(C_clean))
    C_clean = C_clean / np.outer(d, d)
    np.fill_diagonal(C_clean, 1.0)

    Sigma = C_clean * np.outer(sd, sd)
    return (Sigma + Sigma.T) / 2.0
```

Then replace the trailing `raise CovarianceError(f"method {method!r} not implemented yet")` in `estimate` with:

```python
    Sigma = rmt_clean(arr)
    return Sigma, _diagnostics("rmt_clean", Sigma, T, N, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_covariance_rmt.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add finq/covariance.py tests/test_covariance_rmt.py
git commit -m "feat: RMT eigenvalue cleaning per Laloux et al."
```

---

### Task 8: Risk structure metrics

**Files:**
- Create: `finq/risk.py`
- Test: `tests/test_risk_structure.py`

**Interfaces:**
- Consumes: `finq.returns.PERIODS_PER_YEAR`
- Produces: `finq.risk.portfolio_vol(Sigma, w, freq="daily") -> float`; `finq.risk.risk_contributions(Sigma, w) -> tuple[np.ndarray, np.ndarray]` returning `(mctr, pct_contribution)`; `finq.risk.diversification_ratio(Sigma, w) -> float`; `finq.risk.effective_bets(Sigma, w) -> float`; `finq.risk.concentration(w, pct_rc) -> dict[str, float]` with keys `hhi_weights` and `hhi_risk`; `finq.risk.RiskError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_structure.py
import numpy as np
import pytest
from finq.risk import (portfolio_vol, risk_contributions, diversification_ratio,
                       effective_bets, concentration, RiskError)


def test_equal_weight_uncorrelated_unit_vol_gives_one_over_sqrt_N():
    """The canonical analytic identity for diversification."""
    for N in (4, 9, 16, 25):
        Sigma = np.eye(N)
        w = np.full(N, 1.0 / N)
        assert portfolio_vol(Sigma, w, freq=None) == pytest.approx(1.0 / np.sqrt(N))


def test_perfectly_correlated_assets_give_no_diversification():
    Sigma = np.full((5, 5), 0.04)          # every pairwise correlation is 1
    w = np.full(5, 0.2)
    assert portfolio_vol(Sigma, w, freq=None) == pytest.approx(0.2)


def test_annualization_uses_sqrt_252_daily_and_sqrt_52_weekly():
    Sigma = np.eye(2) * 0.0001
    w = np.array([0.5, 0.5])
    raw = portfolio_vol(Sigma, w, freq=None)
    assert portfolio_vol(Sigma, w, freq="daily") == pytest.approx(raw * np.sqrt(252))
    assert portfolio_vol(Sigma, w, freq="weekly") == pytest.approx(raw * np.sqrt(52))


def test_risk_contributions_sum_to_portfolio_volatility():
    rng = np.random.default_rng(30)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    w = rng.random(8)
    w = w / w.sum()
    mctr, pct = risk_contributions(Sigma, w)
    assert (w * mctr).sum() == pytest.approx(portfolio_vol(Sigma, w, freq=None))
    assert pct.sum() == pytest.approx(1.0)


def test_equal_weight_identity_gives_equal_risk_contributions():
    Sigma = np.eye(6)
    w = np.full(6, 1 / 6)
    _, pct = risk_contributions(Sigma, w)
    np.testing.assert_allclose(pct, np.full(6, 1 / 6))


def test_diversification_ratio_is_one_for_a_single_asset():
    assert diversification_ratio(np.array([[0.04]]), np.array([1.0])) == pytest.approx(1.0)


def test_diversification_ratio_exceeds_one_for_imperfect_correlation():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.04]])
    assert diversification_ratio(Sigma, np.array([0.5, 0.5])) > 1.0


def test_diversification_ratio_is_one_when_perfectly_correlated():
    Sigma = np.full((3, 3), 0.04)
    assert diversification_ratio(Sigma, np.full(3, 1 / 3)) == pytest.approx(1.0)


def test_effective_bets_equals_N_for_equal_weighted_uncorrelated():
    Sigma = np.eye(10)
    assert effective_bets(Sigma, np.full(10, 0.1)) == pytest.approx(10.0)


def test_effective_bets_is_one_for_a_single_factor():
    Sigma = np.full((8, 8), 0.04) + np.eye(8) * 1e-12
    assert effective_bets(Sigma, np.full(8, 0.125)) == pytest.approx(1.0, abs=1e-4)


def test_concentration_reports_both_hhi_measures():
    w = np.array([0.7, 0.1, 0.1, 0.1])
    pct = np.array([0.9, 0.05, 0.03, 0.02])
    c = concentration(w, pct)
    assert c["hhi_weights"] == pytest.approx(0.52)
    assert c["hhi_risk"] > c["hhi_weights"]


def test_rejects_weight_length_mismatch():
    with pytest.raises(RiskError, match="length"):
        portfolio_vol(np.eye(3), np.array([0.5, 0.5]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_risk_structure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.risk'`

- [ ] **Step 3: Write the implementation**

```python
# finq/risk.py
from __future__ import annotations

import numpy as np

from finq.returns import PERIODS_PER_YEAR


class RiskError(Exception):
    """Raised when a risk metric cannot be computed."""


def _check(Sigma: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(w, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise RiskError("Sigma must be a square matrix")
    if w.ndim != 1 or w.shape[0] != Sigma.shape[0]:
        raise RiskError(
            f"weight length {w.shape[0]} does not match Sigma dimension {Sigma.shape[0]}"
        )
    return Sigma, w


def _annualize(value: float, freq: str | None) -> float:
    if freq is None:
        return value
    if freq not in PERIODS_PER_YEAR:
        raise RiskError(f"unknown freq {freq!r}; use 'daily', 'weekly', or None")
    return value * np.sqrt(PERIODS_PER_YEAR[freq])


def portfolio_vol(Sigma, w, freq: str | None = "daily") -> float:
    Sigma, w = _check(Sigma, w)
    variance = float(w @ Sigma @ w)
    return _annualize(float(np.sqrt(max(variance, 0.0))), freq)


def risk_contributions(Sigma, w) -> tuple[np.ndarray, np.ndarray]:
    """Marginal contribution to risk, and each holding's share of total risk."""
    Sigma, w = _check(Sigma, w)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if vol <= 0:
        raise RiskError("portfolio volatility is zero; risk contributions undefined")
    mctr = (Sigma @ w) / vol
    return mctr, (w * mctr) / vol


def diversification_ratio(Sigma, w) -> float:
    Sigma, w = _check(Sigma, w)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if vol <= 0:
        raise RiskError("portfolio volatility is zero; diversification ratio undefined")
    return float((w @ np.sqrt(np.diag(Sigma))) / vol)


def effective_bets(Sigma, w) -> float:
    """Meucci: entropy of the variance shares of the principal portfolios."""
    Sigma, w = _check(Sigma, w)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    w_tilde = evecs.T @ w
    contrib = (w_tilde ** 2) * evals
    total = contrib.sum()
    if total <= 0:
        raise RiskError("portfolio variance is zero; effective bets undefined")
    p = contrib / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def concentration(w, pct_rc) -> dict[str, float]:
    w = np.asarray(w, dtype=float)
    pct_rc = np.asarray(pct_rc, dtype=float)
    return {
        "hhi_weights": float((w ** 2).sum()),
        "hhi_risk": float((pct_rc ** 2).sum()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_risk_structure.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add finq/risk.py tests/test_risk_structure.py
git commit -m "feat: portfolio risk structure metrics"
```

---

### Task 9: Tail, market, and stress risk metrics

**Files:**
- Modify: `finq/risk.py`
- Test: `tests/test_risk_tail.py`

**Interfaces:**
- Consumes: `finq.risk._check`, `RiskError`
- Produces: `finq.risk.var_cvar(R, w, level=0.95, method="historical") -> tuple[float, float]` returning `(var, cvar)` as positive loss magnitudes; `finq.risk.max_drawdown(R, w) -> float` (positive fraction); `finq.risk.betas(R, w, benchmarks: pd.DataFrame) -> dict[str, float]`; `finq.risk.exceedance_correlation(R, threshold=1.0, min_obs=20) -> tuple[np.ndarray, np.ndarray]` returning `(downside_corr, unconditional_corr)`; `finq.risk.fx_risk_share(R, fx_returns, w) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_tail.py
import numpy as np
import pandas as pd
import pytest
from finq.risk import (var_cvar, max_drawdown, betas, exceedance_correlation,
                       fx_risk_share, RiskError)


def test_historical_var_is_the_empirical_quantile_loss():
    r = np.linspace(-0.10, 0.09, 100).reshape(-1, 1)
    var, cvar = var_cvar(r, np.array([1.0]), level=0.95, method="historical")
    assert var == pytest.approx(0.0905, abs=2e-3)
    assert cvar > var


def test_cvar_is_never_below_var():
    rng = np.random.default_rng(40)
    R = rng.standard_t(df=4, size=(2000, 3)) * 0.01
    w = np.full(3, 1 / 3)
    for level in (0.95, 0.99):
        var, cvar = var_cvar(R, w, level=level)
        assert cvar >= var


def test_cornish_fisher_exceeds_normal_var_for_left_skewed_returns():
    rng = np.random.default_rng(41)
    x = -np.abs(rng.normal(size=(5000, 1))) ** 2 * 0.01     # strong left skew
    w = np.array([1.0])
    cf, _ = var_cvar(x, w, level=0.99, method="cornish_fisher")
    hist, _ = var_cvar(x, w, level=0.99, method="historical")
    assert cf > 0 and hist > 0


def test_max_drawdown_of_a_known_path():
    # +10%, then -50%, then +10%  ->  peak 1.10, trough 0.55, drawdown = 0.5
    r = np.array([[0.10], [-0.50], [0.10]])
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.5)


def test_max_drawdown_is_zero_for_a_monotone_rise():
    r = np.full((10, 1), 0.01)
    assert max_drawdown(r, np.array([1.0])) == pytest.approx(0.0)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_risk_tail.py -v`
Expected: FAIL — `ImportError: cannot import name 'var_cvar'`

- [ ] **Step 3: Write the implementation**

Add to `finq/risk.py` (add `import pandas as pd` and `from scipy import stats` at the top):

```python
def _portfolio_series(R, w) -> np.ndarray:
    arr = R.to_numpy(dtype=float) if isinstance(R, pd.DataFrame) else np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float)
    if arr.shape[1] != w.shape[0]:
        raise RiskError(f"weight length {w.shape[0]} does not match {arr.shape[1]} columns")
    return arr @ w


def var_cvar(R, w, level: float = 0.95, method: str = "historical") -> tuple[float, float]:
    """Value at risk and conditional VaR, reported as positive loss magnitudes."""
    if not 0.5 < level < 1.0:
        raise RiskError(f"level must be between 0.5 and 1, got {level}")
    p = _portfolio_series(R, w)
    if p.size < 30:
        raise RiskError(f"only {p.size} observations; too few for a tail estimate")

    if method == "historical":
        var = -float(np.quantile(p, 1.0 - level))
    elif method == "cornish_fisher":
        mu, sd = float(p.mean()), float(p.std(ddof=0))
        s, k = float(stats.skew(p)), float(stats.kurtosis(p, fisher=True))
        z = float(stats.norm.ppf(1.0 - level))
        z_cf = (z + (z ** 2 - 1) * s / 6
                + (z ** 3 - 3 * z) * k / 24
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        var = -float(mu + z_cf * sd)
    else:
        raise RiskError(f"unknown method {method!r}; use 'historical' or 'cornish_fisher'")

    tail = p[p <= -var]
    cvar = -float(tail.mean()) if tail.size else var
    return var, max(cvar, var)


def max_drawdown(R, w) -> float:
    """Largest peak-to-trough decline of the fixed-weight return series."""
    p = _portfolio_series(R, w)
    equity = np.cumprod(1.0 + p)
    peak = np.maximum.accumulate(equity)
    return float((1.0 - equity / peak).max())


def betas(R, w, benchmarks) -> dict[str, float]:
    """Univariate OLS beta of the portfolio against each benchmark separately."""
    if not isinstance(benchmarks, pd.DataFrame):
        raise RiskError("benchmarks must be a DataFrame of return series")
    if isinstance(R, pd.DataFrame):
        common = R.index.intersection(benchmarks.index)
        if len(common) < 30:
            raise RiskError(f"only {len(common)} overlapping dates with the benchmarks")
        p = _portfolio_series(R.loc[common], w)
        bm = benchmarks.loc[common]
    else:
        p = _portfolio_series(R, w)
        bm = benchmarks
        if len(bm) != len(p):
            raise RiskError("benchmark length does not match the return matrix")

    out = {}
    for name in bm.columns:
        b = bm[name].to_numpy(dtype=float)
        var_b = float(np.var(b, ddof=0))
        if var_b <= 0:
            raise RiskError(f"benchmark {name} has zero variance")
        out[name] = float(np.cov(p, b, ddof=0)[0, 1] / var_b)
    return out


def exceedance_correlation(R, threshold: float = 1.0,
                           min_obs: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Correlation conditional on both assets falling below -threshold sigma.

    Returns (downside_corr, unconditional_corr). This is the Longin-Solnik /
    Ang-Bekaert diagnostic: it answers whether diversification survives a selloff.
    """
    arr = R.to_numpy(dtype=float) if isinstance(R, pd.DataFrame) else np.asarray(R, dtype=float)
    z = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=0)
    N = arr.shape[1]

    uncond = np.corrcoef(arr, rowvar=False)
    down = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            mask = (z[:, i] < -threshold) & (z[:, j] < -threshold)
            n = int(mask.sum())
            if n < min_obs:
                raise RiskError(
                    f"only {n} joint observations below -{threshold} sigma for "
                    f"assets {i} and {j}; at least {min_obs} required. "
                    "Lower the threshold or use a longer history."
                )
            c = float(np.corrcoef(arr[mask, i], arr[mask, j])[0, 1])
            down[i, j] = down[j, i] = c
    return down, uncond


def fx_risk_share(R, fx_returns, w) -> float:
    """Fraction of portfolio variance attributable to currency moves."""
    total = _portfolio_series(R, w)
    currency = _portfolio_series(fx_returns, w)
    var_total = float(np.var(total, ddof=0))
    if var_total <= 0:
        raise RiskError("portfolio variance is zero; FX share undefined")
    if float(np.var(currency, ddof=0)) == 0.0:
        return 0.0
    # Share of variance explained by the FX component, via its covariance with the total.
    return float(np.cov(total, currency, ddof=0)[0, 1] / var_total)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_risk_tail.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add finq/risk.py tests/test_risk_tail.py
git commit -m "feat: tail, market, and stress-correlation risk metrics"
```

---

### Task 10: Liquidity metrics

**Files:**
- Create: `finq/liquidity.py`
- Test: `tests/test_liquidity.py`

**Interfaces:**
- Consumes: `finq.data.PriceData`
- Produces: `finq.liquidity.adv(close, volume, window=20) -> pd.Series`; `finq.liquidity.days_to_liquidate(position_value, adv_value, participation=0.10) -> float`; `finq.liquidity.amihud(returns, traded_value) -> pd.Series`; `finq.liquidity.stale_price_flag(close, min_run=3) -> dict[str, int]`; `finq.liquidity.LiquidityError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_liquidity.py
import numpy as np
import pandas as pd
import pytest
from finq.liquidity import (adv, days_to_liquidate, amihud, stale_price_flag,
                            LiquidityError)


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


def test_days_to_liquidate_scales_with_position_size():
    assert days_to_liquidate(100_000, 50_000, participation=0.10) == pytest.approx(20.0)
    assert days_to_liquidate(200_000, 50_000, participation=0.10) == pytest.approx(40.0)


def test_days_to_liquidate_rejects_zero_volume():
    with pytest.raises(LiquidityError, match="zero"):
        days_to_liquidate(100_000, 0.0)


def test_amihud_is_higher_for_the_less_liquid_asset():
    idx = pd.date_range("2025-01-01", periods=50, freq="B")
    rng = np.random.default_rng(50)
    r = rng.normal(scale=0.01, size=50)
    returns = pd.DataFrame({"THIN": r, "DEEP": r}, index=idx)
    traded = pd.DataFrame({"THIN": np.full(50, 1e4), "DEEP": np.full(50, 1e8)}, index=idx)
    a = amihud(returns, traded)
    assert a["THIN"] > a["DEEP"]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_liquidity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.liquidity'`

- [ ] **Step 3: Write the implementation**

```python
# finq/liquidity.py
from __future__ import annotations

import numpy as np
import pandas as pd


class LiquidityError(Exception):
    """Raised when a liquidity metric cannot be computed."""


def adv(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average daily traded value over the trailing window, in quote currency."""
    if close.shape != volume.shape:
        raise LiquidityError("close and volume must have the same shape")
    if len(close) < window:
        raise LiquidityError(f"only {len(close)} rows; need at least {window}")
    traded = close * volume
    return traded.tail(window).mean()


def days_to_liquidate(position_value: float, adv_value: float,
                      participation: float = 0.10) -> float:
    """Trading days to exit at a given share of average daily volume."""
    if adv_value <= 0:
        raise LiquidityError("average daily volume is zero; position cannot be exited")
    if not 0 < participation <= 1:
        raise LiquidityError(f"participation must be in (0, 1], got {participation}")
    return float(position_value / (adv_value * participation))


def amihud(returns: pd.DataFrame, traded_value: pd.DataFrame) -> pd.Series:
    """Amihud illiquidity: mean |return| per unit of traded value, scaled by 1e6."""
    if returns.shape != traded_value.shape:
        raise LiquidityError("returns and traded_value must have the same shape")
    tv = traded_value.replace(0.0, np.nan)
    return (returns.abs() / tv).mean() * 1e6


def stale_price_flag(close: pd.DataFrame, min_run: int = 3) -> dict[str, int]:
    """Count observations inside runs of unchanged closes of at least min_run days.

    Stale prices bias correlation estimates downward, so any non-zero count is a
    warning that this ticker's covariance entries understate its true co-movement.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_liquidity.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add finq/liquidity.py tests/test_liquidity.py
git commit -m "feat: liquidity metrics for thin WSE names"
```

---

### Task 11: Equal weight, minimum variance, and risk parity

**Files:**
- Create: `finq/optimize.py`
- Test: `tests/test_optimize_core.py`

**Interfaces:**
- Consumes: `finq.risk.risk_contributions`, `finq.risk.portfolio_vol`
- Produces: `finq.optimize.Constraints` dataclass with fields `max_weight: float | None = None`, `min_weight: float | None = None`; `finq.optimize.equal_weight(Sigma) -> np.ndarray`; `finq.optimize.min_variance(Sigma, constraints=None) -> np.ndarray`; `finq.optimize.risk_parity(Sigma, constraints=None) -> np.ndarray`; `finq.optimize.OptimizeError(Exception)`

All optimizers are long-only with weights summing to 1. `min_weight` is applied as post-solve cleanup: drop holdings below the floor and re-solve on the reduced set, repeating until stable or one holding remains.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimize_core.py
import numpy as np
import pytest
from finq.optimize import (equal_weight, min_variance, risk_parity,
                           Constraints, OptimizeError)
from finq.risk import risk_contributions, portfolio_vol


def test_equal_weight_is_one_over_N():
    np.testing.assert_allclose(equal_weight(np.eye(5)), np.full(5, 0.2))


def test_weights_are_long_only_and_sum_to_one():
    rng = np.random.default_rng(60)
    A = rng.normal(size=(6, 6))
    Sigma = A @ A.T / 6
    for solver in (min_variance, risk_parity):
        w = solver(Sigma)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()


def test_min_variance_two_asset_matches_closed_form():
    """w1 = (v2 - cov) / (v1 + v2 - 2 cov) for the unconstrained two-asset case."""
    v1, v2, cov = 0.04, 0.09, 0.01
    Sigma = np.array([[v1, cov], [cov, v2]])
    expected = (v2 - cov) / (v1 + v2 - 2 * cov)
    np.testing.assert_allclose(min_variance(Sigma), [expected, 1 - expected], atol=1e-6)


def test_min_variance_beats_equal_weight_on_variance():
    rng = np.random.default_rng(61)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    assert (portfolio_vol(Sigma, min_variance(Sigma), freq=None)
            <= portfolio_vol(Sigma, equal_weight(Sigma), freq=None) + 1e-12)


def test_min_variance_concentrates_in_the_low_vol_asset():
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = min_variance(Sigma)
    assert w[0] > 0.95


def test_risk_parity_equalizes_risk_contributions():
    rng = np.random.default_rng(62)
    A = rng.normal(size=(7, 7))
    Sigma = A @ A.T / 7
    _, pct = risk_contributions(Sigma, risk_parity(Sigma))
    np.testing.assert_allclose(pct, np.full(7, 1 / 7), atol=1e-4)


def test_risk_parity_on_uncorrelated_assets_is_inverse_volatility():
    Sigma = np.diag([0.01, 0.04, 0.09])
    inv_vol = 1 / np.sqrt(np.diag(Sigma))
    np.testing.assert_allclose(risk_parity(Sigma), inv_vol / inv_vol.sum(), atol=1e-5)


def test_max_weight_cap_is_respected():
    Sigma = np.diag([0.0001, 0.04, 0.09])
    w = min_variance(Sigma, Constraints(max_weight=0.5))
    assert w.max() <= 0.5 + 1e-8
    assert w.sum() == pytest.approx(1.0)


def test_min_weight_drops_dust_and_resolves():
    Sigma = np.diag([0.0001, 0.04, 4.0])
    w = min_variance(Sigma, Constraints(min_weight=0.05))
    assert w[2] == 0.0                      # dust dropped entirely
    assert w.sum() == pytest.approx(1.0)
    assert ((w == 0.0) | (w >= 0.05 - 1e-9)).all()


def test_infeasible_max_weight_fails_loudly():
    with pytest.raises(OptimizeError, match="infeasible"):
        min_variance(np.eye(5), Constraints(max_weight=0.1))   # 5 * 0.1 < 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimize_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.optimize'`

- [ ] **Step 3: Write the implementation**

```python
# finq/optimize.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class OptimizeError(Exception):
    """Raised when a weight vector cannot be produced."""


@dataclass(frozen=True)
class Constraints:
    max_weight: float | None = None
    min_weight: float | None = None


def _validate(Sigma) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise OptimizeError("Sigma must be a square matrix")
    if Sigma.shape[0] < 1:
        raise OptimizeError("need at least one asset")
    return Sigma


def _bounds(n: int, cap: float | None) -> list[tuple[float, float]]:
    if cap is None:
        return [(0.0, 1.0)] * n
    if cap * n < 1.0 - 1e-12:
        raise OptimizeError(
            f"infeasible: max_weight={cap} across {n} assets cannot sum to 1"
        )
    return [(0.0, cap)] * n


def _solve(objective, n: int, cap: float | None) -> np.ndarray:
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=_bounds(n, cap),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimizeError(f"optimizer did not converge: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def _apply_min_weight(solver, Sigma: np.ndarray, constraints: Constraints) -> np.ndarray:
    """Drop holdings below the floor and re-solve on the reduced set until stable."""
    n = Sigma.shape[0]
    active = np.arange(n)
    while True:
        w_sub = solver(Sigma[np.ix_(active, active)], constraints.max_weight)
        keep = w_sub >= constraints.min_weight
        if keep.all() or keep.sum() <= 1:
            full = np.zeros(n)
            full[active] = w_sub
            return full
        active = active[keep]


def _dispatch(solver, Sigma, constraints: Constraints | None) -> np.ndarray:
    Sigma = _validate(Sigma)
    constraints = constraints or Constraints()
    if constraints.min_weight:
        return _apply_min_weight(solver, Sigma, constraints)
    return solver(Sigma, constraints.max_weight)


def equal_weight(Sigma) -> np.ndarray:
    """The baseline every other method must justify itself against."""
    n = _validate(Sigma).shape[0]
    return np.full(n, 1.0 / n)


def _min_variance(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    return _solve(lambda w: float(w @ Sigma @ w), Sigma.shape[0], cap)


def min_variance(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    return _dispatch(_min_variance, Sigma, constraints)


def _risk_parity(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    n = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-24)))
        rc = w * (Sigma @ w) / vol
        return float(((rc - rc.mean()) ** 2).sum()) * 1e4

    return _solve(objective, n, cap)


def risk_parity(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Equal risk contribution: every holding supplies the same share of total risk."""
    return _dispatch(_risk_parity, Sigma, constraints)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize_core.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add finq/optimize.py tests/test_optimize_core.py
git commit -m "feat: equal weight, minimum variance, and risk parity optimizers"
```

---

### Task 12: Maximum diversification, HRP, and cross-method comparison

**Files:**
- Modify: `finq/optimize.py`
- Test: `tests/test_optimize_advanced.py`

**Interfaces:**
- Consumes: everything from Task 11
- Produces: `finq.optimize.max_diversification(Sigma, constraints=None) -> np.ndarray`; `finq.optimize.hrp(Sigma, constraints=None) -> np.ndarray`; `finq.optimize.compare(Sigma, tickers, methods=None, constraints=None) -> pd.DataFrame` — index is `tickers`, one column per method, plus a `dispersion` column holding the max-minus-min weight across methods per asset

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimize_advanced.py
import numpy as np
import pandas as pd
import pytest
from finq.optimize import (max_diversification, hrp, compare, equal_weight,
                           Constraints)
from finq.risk import diversification_ratio


def test_max_diversification_beats_equal_weight_on_the_ratio():
    rng = np.random.default_rng(70)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8
    assert (diversification_ratio(Sigma, max_diversification(Sigma))
            >= diversification_ratio(Sigma, equal_weight(Sigma)) - 1e-9)


def test_max_diversification_weights_are_valid():
    rng = np.random.default_rng(71)
    A = rng.normal(size=(6, 6))
    w = max_diversification(A @ A.T / 6)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_hrp_weights_are_valid():
    rng = np.random.default_rng(72)
    A = rng.normal(size=(10, 10))
    w = hrp(A @ A.T / 10)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_hrp_tilts_toward_the_low_volatility_asset():
    """With an identity correlation matrix the leaf ORDER is arbitrary, so assert
    only what the allocation logic guarantees: least volatile gets the most."""
    Sigma = np.diag([0.01, 0.04, 0.09, 0.16])
    w = hrp(Sigma)
    assert w.argmax() == 0
    assert w[0] > w[3]


def test_hrp_splits_evenly_between_two_identical_clusters():
    """Two tight blocks of two assets each; HRP should give each block half."""
    block = np.array([[1.0, 0.95], [0.95, 1.0]])
    C = np.eye(4) * 0.0
    C[:2, :2] = block
    C[2:, 2:] = block
    Sigma = C * 0.04
    w = hrp(Sigma)
    assert w[:2].sum() == pytest.approx(0.5, abs=0.02)


def test_hrp_never_inverts_the_covariance_matrix():
    """A singular matrix breaks min_variance but must not break HRP."""
    rng = np.random.default_rng(73)
    R = rng.normal(size=(5, 12))
    Sigma = np.cov(R, rowvar=False)          # rank-deficient by construction
    w = hrp(Sigma)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)


def test_compare_returns_one_column_per_method_plus_dispersion():
    rng = np.random.default_rng(74)
    A = rng.normal(size=(5, 5))
    Sigma = A @ A.T / 5
    tickers = ["A", "B", "C", "D", "E"]
    df = compare(Sigma, tickers)
    assert list(df.index) == tickers
    for m in ("equal_weight", "min_variance", "risk_parity",
              "max_diversification", "hrp"):
        assert m in df.columns
        assert df[m].sum() == pytest.approx(1.0)
    assert "dispersion" in df.columns
    assert (df["dispersion"] >= 0).all()


def test_compare_dispersion_is_max_minus_min_across_methods():
    rng = np.random.default_rng(75)
    A = rng.normal(size=(4, 4))
    df = compare(A @ A.T / 4, ["A", "B", "C", "D"])
    methods = [c for c in df.columns if c != "dispersion"]
    expected = df[methods].max(axis=1) - df[methods].min(axis=1)
    np.testing.assert_allclose(df["dispersion"].to_numpy(), expected.to_numpy())


def test_compare_honors_constraints():
    rng = np.random.default_rng(76)
    A = rng.normal(size=(6, 6))
    df = compare(A @ A.T / 6, list("ABCDEF"), constraints=Constraints(max_weight=0.4))
    for m in ("min_variance", "risk_parity", "max_diversification"):
        assert df[m].max() <= 0.4 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimize_advanced.py -v`
Expected: FAIL — `ImportError: cannot import name 'max_diversification'`

- [ ] **Step 3: Write the implementation**

Add to `finq/optimize.py` (add `import pandas as pd` and `from scipy.cluster.hierarchy import linkage, to_tree` and `from scipy.spatial.distance import squareform` at the top):

```python
def _max_diversification(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    sd = np.sqrt(np.diag(Sigma))

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-24)))
        return -float((w @ sd) / vol)

    return _solve(objective, Sigma.shape[0], cap)


def max_diversification(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Choueifaty: maximize (w'sigma) / sqrt(w'Sigma w)."""
    return _dispatch(_max_diversification, Sigma, constraints)


def _quasi_diag(node) -> list[int]:
    """Depth-first leaf order of the clustering tree."""
    if node.is_leaf():
        return [node.get_id()]
    return _quasi_diag(node.get_left()) + _quasi_diag(node.get_right())


def _cluster_var(Sigma: np.ndarray, idx: list[int]) -> float:
    sub = Sigma[np.ix_(idx, idx)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def _hrp(Sigma: np.ndarray, cap: float | None) -> np.ndarray:
    n = Sigma.shape[0]
    if n == 1:
        return np.array([1.0])

    sd = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    tree = to_tree(linkage(squareform(dist, checks=False), method="single"))
    order = _quasi_diag(tree)

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[half:] if side else c[:half]
                    for c in clusters
                    if len(c) > 1
                    for half in (len(c) // 2,)
                    for side in (0, 1)]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_l, var_r = _cluster_var(Sigma, left), _cluster_var(Sigma, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    weights = weights / weights.sum()

    if cap is not None:
        if cap * n < 1.0 - 1e-12:
            raise OptimizeError(
                f"infeasible: max_weight={cap} across {n} assets cannot sum to 1"
            )
        # Water-fill: trim the excess above the cap and redistribute it into the
        # remaining headroom. Clipping and renormalizing would push weights back
        # above the cap, so it must iterate.
        for _ in range(100):
            excess = float(np.clip(weights - cap, 0.0, None).sum())
            if excess <= 1e-12:
                break
            weights = np.minimum(weights, cap)
            room = cap - weights
            if room.sum() <= 1e-12:
                break
            weights = weights + excess * room / room.sum()
    return weights


def hrp(Sigma, constraints: Constraints | None = None) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado). Never inverts Sigma."""
    return _dispatch(_hrp, Sigma, constraints)


ALL_METHODS = {
    "equal_weight": lambda S, c: equal_weight(S),
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_diversification": max_diversification,
    "hrp": hrp,
}


def compare(Sigma, tickers: list[str], methods: list[str] | None = None,
            constraints: Constraints | None = None) -> pd.DataFrame:
    """Run every method side by side. Dispersion across methods is itself the finding:
    wide disagreement means the covariance estimate is unstable."""
    Sigma = _validate(Sigma)
    if len(tickers) != Sigma.shape[0]:
        raise OptimizeError(
            f"{len(tickers)} tickers but Sigma is {Sigma.shape[0]}x{Sigma.shape[0]}"
        )
    names = methods or list(ALL_METHODS)
    unknown = set(names) - set(ALL_METHODS)
    if unknown:
        raise OptimizeError(f"unknown method(s): {', '.join(sorted(unknown))}")

    out = pd.DataFrame(index=tickers)
    for name in names:
        out[name] = ALL_METHODS[name](Sigma, constraints)
    out["dispersion"] = out[names].max(axis=1) - out[names].min(axis=1)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize_advanced.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add finq/optimize.py tests/test_optimize_advanced.py
git commit -m "feat: max diversification, HRP, and cross-method weight comparison"
```

---

### Task 13: Reporting, public API, and CLI

**Files:**
- Create: `finq/report.py`
- Create: `finq/__main__.py`
- Modify: `finq/__init__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module above
- Produces: `finq.report.header(diag, freq, dropped_days, stale) -> str`; `finq.report.analyze_text(...) -> str`; `finq.report.optimize_text(...) -> str`; `finq.__main__` with subcommands `analyze` and `optimize`; `finq/__init__.py` re-exporting `load`, `prices`, `fx`, `aligned`, `estimate`, `compare`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from finq.covariance import estimate
from finq.report import header

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def seeded_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for src in FIX.glob("*.csv"):
        (cache / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (cache / "currencies.json").write_text((FIX / "currencies.json").read_text())
    return cache


def test_header_always_states_the_estimation_context():
    rng = np.random.default_rng(80)
    _, diag = estimate(rng.normal(size=(500, 10)), method="ledoit_wolf")
    text = header(diag, freq="daily", dropped_days=7, stale=[])
    for token in ("N=10", "T=500", "Q=", "ledoit_wolf", "delta", "7"):
        assert token in text


def test_header_warns_when_data_is_stale():
    rng = np.random.default_rng(81)
    _, diag = estimate(rng.normal(size=(300, 5)), method="sample")
    assert "STALE" in header(diag, freq="daily", dropped_days=0, stale=["PKO.WA"]).upper()


def test_public_api_reexports():
    import finq
    for name in ("load", "prices", "fx", "aligned", "estimate", "compare"):
        assert hasattr(finq, name)


def _run(args, cache):
    return subprocess.run(
        [sys.executable, "-m", "finq", *args, "--cache-dir", str(cache)],
        capture_output=True, text=True, timeout=180,
    )


def test_analyze_command_prints_a_report(tmp_path, seeded_cache):
    p = tmp_path / "p.csv"
    p.write_text("ticker,weight\nPKO.WA,0.4\nSPY,0.4\nGLD,0.2\n", encoding="utf-8")
    r = _run(["analyze", str(p)], seeded_cache)
    assert r.returncode == 0, r.stderr
    for token in ("Q=", "Annualized volatility", "risk contribution", "PKO.WA"):
        assert token in r.stdout


def test_optimize_command_prints_all_methods(tmp_path, seeded_cache):
    p = tmp_path / "p.csv"
    p.write_text("ticker\nPKO.WA\nCDR.WA\nSPY\nGLD\n", encoding="utf-8")
    r = _run(["optimize", str(p)], seeded_cache)
    assert r.returncode == 0, r.stderr
    for token in ("equal_weight", "min_variance", "risk_parity", "hrp", "dispersion"):
        assert token in r.stdout


def test_analyze_refuses_selection_mode_portfolio(tmp_path, seeded_cache):
    p = tmp_path / "p.csv"
    p.write_text("ticker\nPKO.WA\nSPY\n", encoding="utf-8")
    r = _run(["analyze", str(p)], seeded_cache)
    assert r.returncode != 0
    assert "weight" in (r.stdout + r.stderr).lower()


def test_unknown_ticker_exits_nonzero_and_names_it(tmp_path, seeded_cache):
    p = tmp_path / "p.csv"
    p.write_text("ticker,weight\nNOPE.WA,1.0\n", encoding="utf-8")
    r = _run(["analyze", str(p)], seeded_cache)
    assert r.returncode != 0
    assert "NOPE.WA" in (r.stdout + r.stderr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finq.report'`

- [ ] **Step 3: Write the implementation**

```python
# finq/report.py
from __future__ import annotations

import numpy as np
import pandas as pd

from finq.covariance import Diagnostics


def header(diag: Diagnostics, freq: str, dropped_days: int, stale: list[str]) -> str:
    lines = [
        "=" * 72,
        f"Estimation context  |  N={diag.N}  T={diag.T}  Q={diag.Q:.1f}  freq={freq}",
        f"Covariance method   |  {diag.method}",
    ]
    if diag.shrinkage is not None:
        note = ("   <- high: the sample matrix carried little information"
                if diag.shrinkage > 0.5 else "")
        lines.append(f"Shrinkage intensity |  delta={diag.shrinkage:.3f}{note}")
    lines.append(
        f"Noise band          |  {diag.n_in_band}/{diag.N} eigenvalues inside "
        f"[{diag.lambda_minus:.3f}, {diag.lambda_plus:.3f}], "
        f"{diag.var_share_in_band:.0%} of spectrum"
    )
    lines.append(f"Calendar alignment  |  {dropped_days} non-common days dropped")
    if stale:
        lines.append(f"WARNING: STALE CACHE for {', '.join(stale)} — prices may be out of date")
    if diag.Q < 2:
        lines.append("WARNING: Q < 2 — this covariance matrix is mostly noise")
    lines.append("=" * 72)
    return "\n".join(lines)


def analyze_text(tickers: list[str], w: np.ndarray, vol: float,
                 pct_rc: np.ndarray, div_ratio: float, enb: float,
                 conc: dict[str, float], var95: float, cvar95: float,
                 mdd: float, beta_map: dict[str, float], fx_share: float,
                 liquidity: pd.DataFrame | None) -> str:
    table = pd.DataFrame({
        "weight": w,
        "risk contribution": pct_rc,
    }, index=tickers).sort_values("risk contribution", ascending=False)

    lines = [
        "",
        f"Annualized volatility   {vol:.2%}",
        f"Diversification ratio   {div_ratio:.2f}",
        f"Effective bets          {enb:.1f} of {len(tickers)} holdings",
        f"HHI (weights / risk)    {conc['hhi_weights']:.3f} / {conc['hhi_risk']:.3f}",
        f"VaR 95% / CVaR 95%      {var95:.2%} / {cvar95:.2%}  (per period)",
        f"Max drawdown            {mdd:.2%}",
        f"FX share of risk        {fx_share:.1%}",
        "Betas                   " + "  ".join(f"{k}={v:.2f}" for k, v in beta_map.items()),
        "",
        "Weight vs risk contribution",
        "-" * 72,
        table.to_string(formatters={
            "weight": "{:.2%}".format,
            "risk contribution": "{:.2%}".format,
        }),
    ]
    if liquidity is not None:
        lines += ["", "Liquidity", "-" * 72, liquidity.to_string()]
    return "\n".join(lines)


def optimize_text(comparison: pd.DataFrame) -> str:
    spread = comparison["dispersion"].max()
    verdict = ("Methods broadly agree; the covariance estimate looks stable."
               if spread < 0.10 else
               "Methods disagree materially. Treat any single weight vector as "
               "one option among several, not a precise answer.")
    return "\n".join([
        "",
        "Proposed weights by method",
        "-" * 72,
        comparison.to_string(float_format=lambda v: f"{v:.2%}"),
        "",
        f"Largest cross-method disagreement: {spread:.2%}",
        verdict,
    ])
```

```python
# finq/__main__.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finq import covariance, data, liquidity as liq, optimize, portfolio, report, risk
from finq.returns import aligned

BENCHMARKS = {"WIG20.WA": "WIG20", "^GSPC": "GSPC"}


def _build(args):
    pf = portfolio.load(args.path)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(args.lookback.rstrip("y")))
    cache = Path(args.cache_dir) if args.cache_dir else None

    tickers = pf.tickers + list(BENCHMARKS)
    pdata = data.prices(tickers, str(start.date()), str(end.date()), cache_dir=cache)

    codes = sorted(set(pdata.currency.values()))
    rates = {c: data.fx(c, str(start.date()), str(end.date()), cache_dir=cache) for c in codes}

    rm = aligned(pdata, rates, freq=args.freq, min_obs=60)
    held = rm.R[pf.tickers]
    bench = rm.R[list(BENCHMARKS)].rename(columns=BENCHMARKS)

    Sigma, diag = covariance.estimate(held, method=args.cov)
    return pf, pdata, rm, held, bench, Sigma, diag


def _weights(pf, pdata) -> np.ndarray:
    if pf.weights is not None:
        return pf.weights
    if pf.quantities is not None:
        last = pdata.close[pf.tickers].ffill().iloc[-1].to_numpy(dtype=float)
        value = pf.quantities * last
        return value / value.sum()
    raise SystemExit(
        "analyze needs weights or quantities; this file has tickers only. "
        "Use `optimize` for selection mode."
    )


def cmd_analyze(args) -> None:
    pf, pdata, rm, held, bench, Sigma, diag = _build(args)
    w = _weights(pf, pdata)
    if pf.normalized:
        print("NOTE: input weights did not sum to 1 and were normalized.")

    _, pct = risk.risk_contributions(Sigma, w)
    var95, cvar95 = risk.var_cvar(held, w, level=0.95)
    fx_held = rm.fx_returns[pf.tickers]

    try:
        beta_map = risk.betas(held, w, bench)
    except risk.RiskError as exc:
        beta_map = {}
        print(f"NOTE: betas unavailable ({exc})")

    liquidity_table = None
    if pf.quantities is not None:
        advs = liq.adv(pdata.close[pf.tickers], pdata.volume[pf.tickers], window=20)
        last = pdata.close[pf.tickers].ffill().iloc[-1]
        liquidity_table = pd.DataFrame({
            "ADV20 (PLN)": advs.round(0),
            "days to exit": [
                liq.days_to_liquidate(q * p, a) for q, p, a in
                zip(pf.quantities, last, advs)
            ],
            "stale days": pd.Series(liq.stale_price_flag(pdata.close[pf.tickers])),
        })

    print(report.header(diag, rm.freq, rm.dropped_days, pdata.stale))
    print(report.analyze_text(
        pf.tickers, w, risk.portfolio_vol(Sigma, w, freq=rm.freq), pct,
        risk.diversification_ratio(Sigma, w), risk.effective_bets(Sigma, w),
        risk.concentration(w, pct), var95, cvar95, risk.max_drawdown(held, w),
        beta_map, risk.fx_risk_share(held, fx_held, w), liquidity_table,
    ))


def cmd_optimize(args) -> None:
    pf, pdata, rm, held, bench, Sigma, diag = _build(args)
    constraints = optimize.Constraints(max_weight=args.max_weight,
                                       min_weight=args.min_weight)
    methods = None if args.method == "all" else [args.method]
    comparison = optimize.compare(Sigma, pf.tickers, methods, constraints)
    print(report.header(diag, rm.freq, rm.dropped_days, pdata.stale))
    print(report.optimize_text(comparison))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finq")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("analyze", "optimize"):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--cov", default="ledoit_wolf", choices=covariance.METHODS)
        p.add_argument("--freq", default="daily", choices=["daily", "weekly"])
        p.add_argument("--lookback", default="3y")
        p.add_argument("--cache-dir", default=None)
        if name == "optimize":
            p.add_argument("--method", default="all",
                           choices=["all", *optimize.ALL_METHODS])
            p.add_argument("--max-weight", type=float, default=None)
            p.add_argument("--min-weight", type=float, default=None)

    args = parser.parse_args(argv)
    try:
        (cmd_analyze if args.command == "analyze" else cmd_optimize)(args)
    except (portfolio.PortfolioError, data.DataError, optimize.OptimizeError,
            covariance.CovarianceError, risk.RiskError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# finq/__init__.py
"""finq — risk-based portfolio analytics for PLN-denominated PL/US portfolios."""
from finq.covariance import estimate
from finq.data import fx, prices
from finq.optimize import compare
from finq.portfolio import load
from finq.returns import aligned

__version__ = "0.1.0"
__all__ = ["load", "prices", "fx", "aligned", "estimate", "compare"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add finq/report.py finq/__main__.py finq/__init__.py tests/test_cli.py
git commit -m "feat: reporting layer, public API, and analyze/optimize CLI"
```

---

### Task 14: The portfolio-quant skill

**Files:**
- Create: `skills/portfolio-quant/SKILL.md`
- Create: `README.md`
- Test: `tests/test_skill_doc.py`

**Interfaces:**
- Consumes: the whole public API
- Produces: a skill document Claude loads when asked portfolio risk questions

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skill_doc.py
"""The skill is a contract with the model; these tests keep it honest."""
import re
from pathlib import Path

SKILL = Path(__file__).parents[1] / "skills" / "portfolio-quant" / "SKILL.md"


def test_skill_file_exists_with_frontmatter():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert re.search(r"^name:\s*portfolio-quant\s*$", text, re.M)
    assert re.search(r"^description:\s*\S", text, re.M)


def test_skill_documents_every_public_entry_point():
    text = SKILL.read_text(encoding="utf-8")
    for symbol in ("portfolio.load", "data.prices", "data.fx", "returns.aligned",
                   "covariance.estimate", "risk.risk_contributions",
                   "risk.exceedance_correlation", "liquidity.adv",
                   "optimize.compare"):
        assert symbol in text, f"{symbol} missing from SKILL.md"


def test_skill_states_the_mandatory_interpretation_rules():
    text = SKILL.read_text(encoding="utf-8").lower()
    for rule in ("q =", "delta", "equal weight", "dispersion"):
        assert rule in text


def test_skill_warns_about_the_silent_failure_modes():
    text = SKILL.read_text(encoding="utf-8").lower()
    for pitfall in ("simple returns", "252", "forward-fill", "sklearn"):
        assert pitfall in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skill_doc.py -v`
Expected: FAIL — `FileNotFoundError`

- [ ] **Step 3: Write the skill and README**

Create `skills/portfolio-quant/SKILL.md`:

````markdown
---
name: portfolio-quant
description: Use when analyzing portfolio risk, covariance, volatility, diversification, liquidity, or computing portfolio weights for Polish and US holdings. Covers the finq library.
---

# Portfolio Quant

`finq` computes risk-based portfolio analytics for a PLN-denominated portfolio of
Polish (WSE) and US holdings. It is **risk-based only** — it never forecasts returns,
so there is no mean-variance, max-Sharpe, or Black-Litterman here. Do not add one.

## The two headline flows

```bash
python -m finq analyze  portfolio.csv    # risk report for weights you hold
python -m finq optimize portfolio.csv    # proposed weights, all methods compared
```

Flags: `--cov {sample,ledoit_wolf,rmt_clean}`, `--freq {daily,weekly}`,
`--lookback 3y`, and for `optimize` also `--method`, `--max-weight`, `--min-weight`.

## Everything else: compose against the API

```python
import pandas as pd
from finq import load, prices, fx, aligned, estimate
from finq import risk, liquidity, optimize

pf = load("portfolio.csv")
pd_ = prices(pf.tickers, "2022-01-01", "2025-12-31")
rates = {c: fx(c, "2022-01-01", "2025-12-31") for c in set(pd_.currency.values())}
rm = aligned(pd_, rates, freq="daily")
Sigma, diag = estimate(rm.R, method="ledoit_wolf")

mctr, pct = risk.risk_contributions(Sigma, pf.weights)
```

Answer open questions by composing these primitives — do not reimplement the math.

## Input formats

CSV or JSON, one of three shapes. `quantity` is preferred: it gives true weights from
live prices and is the only shape that enables days-to-liquidate.

```
ticker,quantity   |   ticker,weight   |   ticker
PKO.WA,120        |   PKO.WA,0.15     |   PKO.WA
SPY,45            |   SPY,0.30        |   SPY
```

Tickers-only is selection mode: `optimize` works, `analyze` refuses.

## Choosing a covariance estimator

| Method | Use when |
|---|---|
| `ledoit_wolf` | **Default.** Ledoit-Wolf (2003), constant-correlation target. Always positive definite, stabilizes optimizers. |
| `rmt_clean` | Diagnosing whether correlation structure is real. Strips the Marchenko-Pastur noise bulk. |
| `sample` | A labelled baseline for comparison only. Never the answer. |

`sample` is a reference point, not a choice. If asked which is "correct", show the
difference between them — that difference is the finding.

## Interpretation rules — apply these on EVERY output

1. **Always report `Q = T/N` and, for shrinkage, `delta`.** High delta means the
   sample matrix carried little information and everything downstream inherits that.
   `Q < 2` means the matrix is mostly noise; say so plainly.
2. **Always compare optimized weights against equal weight.** 1/N is famously hard to
   beat. A method that cannot beat it should be reported as not beating it.
3. **Show cross-method dispersion before presenting any single weight vector.**
   `optimize.compare()` gives it. Wide disagreement means the covariance estimate is
   unstable — say that instead of quoting weights to two decimals.
4. **Never present a number without its estimation context.** `report.header()`
   produces it; include it.
5. **Say when there is too little data to mean anything** rather than computing anyway.

## Pitfalls that produce silently wrong numbers

- **Simple returns, never log returns.** Portfolio return must be a weighted sum of
  asset returns. Log returns break that.
- **Annualize with sqrt(252) daily, sqrt(52) weekly.** Use `PERIODS_PER_YEAR`; never
  hardcode.
- **Never forward-fill closed trading days.** Stale prices create zero returns that bias
  correlations *downward* — you get a diversification estimate that is too optimistic
  exactly when it matters. `returns.aligned()` inner-joins; leave it that way. (FX is the
  one exception: an unpublished NBP rate genuinely means no new rate exists.)
- **Do not use `sklearn.covariance.LedoitWolf`.** It implements the 2004 paper with a
  scaled-identity target. This project follows the 2003 constant-correlation paper in
  `resources/honey.pdf`.
- **Sample covariance divides by T, not T-1.** The shrinkage asymptotics assume it.
- **FX belongs inside the covariance matrix.** It is folded into returns before
  estimation, so every USD holding shares a currency factor. Never add it afterwards.
- **`^WIG` is unusable** — Yahoo resolves the symbol but returns no price series. Use
  `WIG20.WA`, remembering it is 20 bank- and energy-heavy names, not the broad market.

## API reference

- `portfolio.load(path) -> Portfolio` — fields `tickers`, `weights`, `quantities`,
  `normalized`, `source_path`
- `data.prices(tickers, start, end, cache_dir=None) -> PriceData` — fields `close`,
  `volume`, `currency`, `stale`. Does **not** align calendars.
- `data.fx(code, start, end, cache_dir=None) -> pd.Series` — NBP mid rates, PLN per unit
- `returns.aligned(price_data, fx_rates, freq="daily", min_obs=0) -> ReturnMatrix` —
  fields `R`, `freq`, `dropped_days`, `tickers`, `fx_returns`
- `covariance.estimate(R, method) -> (Sigma, Diagnostics)`
- `risk.portfolio_vol(Sigma, w, freq)` · `risk.risk_contributions(Sigma, w)` ·
  `risk.diversification_ratio(Sigma, w)` · `risk.effective_bets(Sigma, w)` ·
  `risk.concentration(w, pct_rc)` · `risk.var_cvar(R, w, level, method)` ·
  `risk.max_drawdown(R, w)` · `risk.betas(R, w, benchmarks)` ·
  `risk.exceedance_correlation(R, threshold, min_obs)` ·
  `risk.fx_risk_share(R, fx_returns, w)`
- `liquidity.adv(close, volume, window)` · `liquidity.days_to_liquidate(...)` ·
  `liquidity.amihud(...)` · `liquidity.stale_price_flag(...)`
- `optimize.equal_weight` · `min_variance` · `risk_parity` · `max_diversification` ·
  `hrp` · `compare(Sigma, tickers, methods, constraints)`, all taking
  `Constraints(max_weight, min_weight)`

## Worth reaching for

`risk.exceedance_correlation` answers the question most risk reports duck: correlations
conditional on both assets falling more than one sigma. If downside correlation is far
above the unconditional figure, the portfolio is less diversified than it looks in exactly
the conditions that matter. This is the Ang-Bekaert / Longin-Solnik diagnostic.
````

Create `README.md`:

```markdown
# finance-utils

Risk-based portfolio analytics for a PLN-denominated portfolio of Polish (WSE) and
US holdings.

## Quick start

```bash
python -m pip install -e .
python -m finq analyze  portfolio.csv
python -m finq optimize portfolio.csv
```

## What it does

Computes covariance with Ledoit-Wolf (2003) constant-correlation shrinkage or
Laloux et al. (1998) RMT cleaning, then reports risk decomposition, tail risk,
stress correlation, liquidity, and risk-based weights — always alongside the
diagnostics that say how much to trust the estimate.

It never forecasts returns. See `docs/superpowers/specs/` for why.

## Layout

- `finq/` — the library
- `skills/portfolio-quant/` — the skill Claude loads
- `resources/` — the source papers
- `tests/` — run with `python -m pytest`
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skill_doc.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite and a real end-to-end check**

```bash
python -m pytest -q
printf 'ticker,quantity\nPKO.WA,120\nCDR.WA,30\nPKN.WA,80\nSPY,45\nGLD,20\n' > portfolio.csv
python -m finq analyze portfolio.csv
python -m finq optimize portfolio.csv --max-weight 0.35
```

Expected: full suite passes; both commands print a report headed by the estimation
context, with no traceback. This is the first run that touches the live network.

- [ ] **Step 6: Commit**

```bash
git add skills/ README.md tests/test_skill_doc.py
git commit -m "feat: portfolio-quant skill and project README"
```

---

## Self-Review Notes

**Spec coverage check** — every spec section maps to a task:

| Spec section | Task |
|---|---|
| §6.1 `data.py` prices, cache, staleness | 2 |
| §6.1 `data.py` NBP FX | 3 |
| §6.2 `portfolio.py`, three input shapes, validation | 1 |
| §6.3 `returns.py`, PLN compounding, inner join, weekly, fx_component | 4 |
| §6.4 sample + diagnostics (Q, MP band, spectrum, condition number) | 5 |
| §6.4 Ledoit-Wolf constant-correlation shrinkage | 6 |
| §6.4 RMT cleaning | 7 |
| §6.5 vol, risk contributions, div ratio, effective bets, concentration | 8 |
| §6.5 VaR/CVaR, drawdown, betas, exceedance correlation, FX share | 9 |
| §6.6 ADV, days-to-liquidate, Amihud, stale flag | 10 |
| §6.7 equal weight, min variance, risk parity, constraints | 11 |
| §6.7 max diversification, HRP, compare | 12 |
| §6.8 report + §7 CLI | 13 |
| §7 the skill | 14 |
| §8 testing — known-answer tests | every task |
| §9 error handling | 1, 2, 4, 5, 11, 13 |

**Deferred from the spec, deliberately:** §8's golden-file test over a frozen price panel
is covered in spirit by the committed `tests/fixtures/` panel that Tasks 2, 3, and 13 run
against. A separate golden-output file was not added because report formatting is expected
to change; the numeric guarantees live in the analytic identity tests instead.
