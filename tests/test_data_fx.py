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
    """A mid-range non-200/non-404 status (e.g. 500, 429) must raise, not silently drop.
    First chunk returns 200, second chunk returns 500 to test the mid-range error case.
    """
    import requests
    import json

    call_count = [0]  # Closure counter for chunk requests

    def fake_get(url, **kwargs):
        call_count[0] += 1

        class FakeResponse:
            def __init__(self, status, data=None):
                self.status_code = status
                self._data = data

            def json(self):
                if self._data is None:
                    raise ValueError("No JSON data")
                return json.loads(self._data)

        if call_count[0] == 1:
            # First chunk returns 200 with some data
            return FakeResponse(200, b'{"rates": [{"effectiveDate": "2025-06-01", "mid": 4.0}]}')
        else:
            # Second chunk returns 500
            return FakeResponse(500)

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(DataError) as exc_info:
        # EUR not in cache, so fx() calls _fetch_nbp which calls requests.get
        fx("EUR", "2025-01-01", "2025-12-31", cache_dir=seeded_cache)

    error_msg = str(exc_info.value)
    assert "EUR" in error_msg
    assert "fetch failed" in error_msg
    assert "500" in error_msg


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


def test_stale_cache_refetch_fails_serves_cache(seeded_cache, monkeypatch):
    """When cached FX file is stale and refetch fails, serve cache without rewriting it."""
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


def test_stale_cache_refetch_fails_preserves_mtime(seeded_cache, monkeypatch):
    """When refetch fails, the cached file's mtime must not be updated (no re-write).
    This ensures subsequent calls still recognize the cache as stale.
    """
    path = seeded_cache / "FX_USD.csv"

    # Backdate the file to 3 days ago
    old_mtime = datetime.now().timestamp() - 86400 * 3
    os.utime(path, (old_mtime, old_mtime))

    # Capture the original mtime
    original_mtime = path.stat().st_mtime

    # Mock a failed fetch
    def fake_fetch_fails(code):
        raise DataError(f"FX {code}: fetch failed (network error)")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch_fails)

    # Call fx() once (will fail to refetch but fall back to cache)
    fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)

    # mtime must not have changed
    new_mtime = path.stat().st_mtime
    assert new_mtime == original_mtime, f"mtime changed from {original_mtime} to {new_mtime}"


def test_404_mid_range_is_tolerated(seeded_cache, monkeypatch):
    """A 404 response mid-range (no publication days) must NOT raise, only if first chunk 404s."""
    import json

    call_count = [0]

    def fake_get(url, **kwargs):
        call_count[0] += 1

        class FakeResponse:
            def __init__(self, status, data=None):
                self.status_code = status
                self._data = data

            def json(self):
                if self._data is None:
                    raise ValueError("No JSON data")
                return json.loads(self._data)

        if call_count[0] == 1:
            # First chunk returns 200 with some data
            return FakeResponse(200, b'{"rates": [{"effectiveDate": "2025-06-01", "mid": 4.0}]}')
        else:
            # Second chunk returns 404 (no publication days, legitimate)
            return FakeResponse(404)

    monkeypatch.setattr("requests.get", fake_get)

    # Should NOT raise despite 404 on second chunk
    s = fx("EUR", "2025-01-01", "2025-12-31", cache_dir=seeded_cache)
    assert isinstance(s, pd.Series)
    # Should have at least the 200 chunk's data
    assert len(s) > 0


def test_fresh_cache_is_not_stale(seeded_cache):
    """A fresh cache read must report attrs['stale'] == False, not just be truthy-absent."""
    s = fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert s.attrs["stale"] is False


def test_successful_refetch_is_not_stale(seeded_cache, monkeypatch):
    path = seeded_cache / "FX_USD.csv"
    old_mtime = datetime.now().timestamp() - 86400 * 2
    os.utime(path, (old_mtime, old_mtime))

    def fake_fetch(code):
        return pd.DataFrame({
            "date": pd.date_range("2026-08-01", periods=10),
            "mid": [3.8] * 10,
        })

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch)
    s = fx("USD", "2026-08-01", "2026-08-10", cache_dir=seeded_cache)
    assert s.attrs["stale"] is False


def test_failed_refetch_fallback_is_stale(seeded_cache, monkeypatch):
    """This is the reachable spec-9 gap Task 4b flagged: a network-down fallback
    to cache must be visible to callers (mirrors PriceData.stale), or a report
    built on months-old FX rates can look identical to a report built on fresh
    ones."""
    path = seeded_cache / "FX_USD.csv"
    old_mtime = datetime.now().timestamp() - 86400 * 2
    os.utime(path, (old_mtime, old_mtime))

    def fake_fetch_fails(code):
        raise DataError(f"FX {code}: fetch failed (network error)")

    monkeypatch.setattr(data_module, "_fetch_nbp", fake_fetch_fails)
    s = fx("USD", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert s.attrs["stale"] is True


def test_pln_identity_is_not_stale(seeded_cache):
    s = fx("PLN", "2025-01-01", "2025-06-30", cache_dir=seeded_cache)
    assert s.attrs["stale"] is False


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
