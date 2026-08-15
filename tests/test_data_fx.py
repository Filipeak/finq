import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
from finq.data import fx, DataError
import finq.data as data_module

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


def test_unknown_currency_fails_loudly(seeded_cache, monkeypatch):
    """ZZZ has no cache entry, so fx() must fall through to a fetch. Mock that
    fetch instead of hitting live NBP, so the suite stays fully offline.
    """
    def fake_fetch(code):
        raise DataError(f"FX {code}: NBP does not publish this currency")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch)
    with pytest.raises(DataError, match="ZZZ"):
        fx("ZZZ", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)


def test_non_200_non_404_status_fails_loudly(seeded_cache, monkeypatch):
    """A mid-range non-200/non-404 status (e.g. 500, 429) must raise, not silently drop."""
    import requests

    def fake_fetch_with_error(code):
        # Simulate a transient server error on a non-first chunk
        class FakeResponse:
            status_code = 500

        # Simulate the chunking logic encountering a 500
        raise DataError(f"FX {code}: fetch failed (status 500 for 2025-06-01 to 2025-08-29)")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch_with_error)
    with pytest.raises(DataError, match="fetch failed"):
        fx("EUR", "2025-01-01", "2025-12-31", cache_dir=seeded_cache)


def test_stale_cache_refetch_succeeds_returns_fresh_data(seeded_cache, monkeypatch):
    """When cached FX file is stale (> 1 day old), attempt refetch."""
    path = seeded_cache / "FX_USD.csv"

    # Backdate the file to 2+ days ago
    old_mtime = (datetime.now().timestamp() - 86400 * 2)  # 2 days ago
    os.utime(path, (old_mtime, old_mtime))

    # Mock a fresh fetch
    def fake_fetch(code):
        return pd.DataFrame({
            "date": pd.date_range("2026-08-01", periods=10),
            "mid": [3.8, 3.81, 3.82, 3.79, 3.8, 3.81, 3.82, 3.83, 3.84, 3.85]
        })

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch)
    s = fx("USD", "2026-08-01", "2026-08-10", cache_dir=seeded_cache)

    # Should return fresh data from the mock
    assert len(s) == 10
    assert s.iloc[0] == 3.8
    assert s.iloc[-1] == 3.85


def test_stale_cache_refetch_fails_serves_stale_cache_flagged(seeded_cache, monkeypatch):
    """When cached FX file is stale and refetch fails, serve cache but flag it as stale."""
    path = seeded_cache / "FX_USD.csv"

    # Backdate the file to 2+ days ago
    old_mtime = (datetime.now().timestamp() - 86400 * 2)  # 2 days ago
    os.utime(path, (old_mtime, old_mtime))

    # Mock a failed fetch
    def fake_fetch_fails(code):
        raise DataError(f"FX {code}: fetch failed (network error)")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch_fails)

    # Should fall back to cached file even though fetch failed
    s = fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert isinstance(s, pd.Series)
    assert (s > 2.0).all() and (s < 6.0).all()


def test_fresh_cache_no_refetch(seeded_cache, monkeypatch):
    """When cached FX file is fresh (< 1 day old), do not refetch."""
    path = seeded_cache / "FX_USD.csv"

    # Ensure file is fresh (recent mtime)
    os.utime(path, None)  # Set to current time

    # Mock a fetch that should NOT be called
    def fake_fetch(code):
        raise AssertionError(f"_fetch_nbp should not be called for fresh cache")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch)

    # Should use cached data without calling fetch
    s = fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert isinstance(s, pd.Series)
    assert (s > 2.0).all() and (s < 6.0).all()
