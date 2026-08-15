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
