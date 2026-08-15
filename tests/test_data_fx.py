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
