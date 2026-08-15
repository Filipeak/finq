import json
from pathlib import Path

import pandas as pd
import pytest
import finq.data as data_module
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


def test_unknown_ticker_fails_loudly_naming_it(seeded_cache, monkeypatch):
    """NOPE.WA has no cache entry, so prices() must fall through to a fetch. Mock
    that fetch instead of hitting live Yahoo, so the suite stays fully offline.
    """
    def fake_fetch(ticker):
        raise DataError(f"{ticker}: Yahoo returned no data for this symbol")

    monkeypatch.setattr(data_module, "_fetch_yahoo", fake_fetch)
    with pytest.raises(DataError, match="NOPE.WA"):
        prices(["PKO.WA", "NOPE.WA"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)


def test_union_index_preserved_no_inner_join_here(seeded_cache):
    """data.prices does NOT align calendars; that is returns.aligned's job."""
    pd_ = prices(["PKO.WA", "SPY"], "2024-01-01", "2025-12-31", cache_dir=seeded_cache)
    both_present = pd_.close["PKO.WA"].notna() & pd_.close["SPY"].notna()
    na_count = pd_.close["PKO.WA"].isna().sum() + pd_.close["SPY"].isna().sum()
    assert na_count >= 1, "PKO.WA and SPY have different holiday calendars; some NaNs are expected"
    assert len(pd_.close) > both_present.sum(), "union index must be longer than the inner-join overlap"
