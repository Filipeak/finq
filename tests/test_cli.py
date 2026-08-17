import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from finq.covariance import estimate
from finq.report import header, optimize_text

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


# ---------------------------------------------------------------------------
# Additional tests beyond the brief.
#
# The brief's own 8 tests only ever exercise the *weights* portfolio path
# (ticker,weight CSVs), so pf.quantities is always None there and the entire
# liquidity table / PLN-conversion block in cmd_analyze (finq/__main__.py) is
# never executed by the brief's own suite. The tests below close that gap,
# plus a handful of other standing-rule gaps: header()'s combined
# price+FX staleness path, optimize_text()'s exact spread==0.10 boundary, the
# betas() except-fallback path, and argparse's --method validation.
# ---------------------------------------------------------------------------


def test_header_warns_with_both_price_and_fx_staleness_named_together():
    """The brief's own stale test only ever passes a single stale ticker. Pin
    that BOTH a price-stale ticker and an FX-stale currency marker show up
    together in the same warning line, since _build() merges pdata.stale with
    an "FX:<code>"-prefixed list and a naive implementation could show one
    and silently drop the other.
    """
    rng = np.random.default_rng(82)
    _, diag = estimate(rng.normal(size=(300, 5)), method="sample")
    text = header(diag, freq="daily", dropped_days=0, stale=["PKO.WA", "FX:USD"])
    warn_line = next(l for l in text.splitlines() if l.startswith("WARNING: STALE CACHE"))
    assert "PKO.WA" in warn_line
    assert "FX:USD" in warn_line


def test_optimize_text_verdict_flips_exactly_at_spread_point_10():
    """optimize_text splits on `spread < 0.10 else ...` -- pin both sides of
    that exact literal boundary, not just values comfortably away from it.
    """
    df_at = pd.DataFrame(
        {"equal_weight": [0.50, 0.50], "min_variance": [0.60, 0.40],
         "dispersion": [0.10, 0.10]},
        index=["A", "B"],
    )
    text_at = optimize_text(df_at)
    assert "disagree materially" in text_at
    assert "broadly agree" not in text_at

    df_under = pd.DataFrame(
        {"equal_weight": [0.50, 0.50], "min_variance": [0.599, 0.401],
         "dispersion": [0.099, 0.099]},
        index=["A", "B"],
    )
    text_under = optimize_text(df_under)
    assert "broadly agree" in text_under
    assert "disagree materially" not in text_under


def test_optimize_rejects_unknown_method_with_a_clean_argparse_error(tmp_path, seeded_cache):
    """argparse's `choices=` must reject a bad --method value with a normal
    usage error (exit 2, message on stderr), not let it fall through to a
    Python traceback deep inside optimize.compare().
    """
    p = tmp_path / "p.csv"
    p.write_text("ticker\nPKO.WA\nSPY\n", encoding="utf-8")
    r = _run(["optimize", str(p), "--method", "not_a_real_method"], seeded_cache)
    assert r.returncode == 2
    assert "invalid choice" in r.stderr.lower()
    assert "Traceback" not in r.stderr


def test_analyze_betas_failure_falls_back_gracefully(tmp_path, seeded_cache, monkeypatch, capsys):
    """risk.betas() only ever raises RiskError when a benchmark has zero
    variance or the portfolio/benchmark overlap is under 30 rows. The CLI's
    own aligned() call always gives `held` and `bench` the exact same index
    (>= min_obs=60 rows), so that failure is not reachable through the
    fixture data alone -- there is no natural way to trip it end to end.
    Force it directly so the `except risk.RiskError` branch in cmd_analyze is
    proven to degrade gracefully (prints a NOTE, keeps going, still emits a
    complete report) rather than being untested dead code.
    """
    from finq import __main__ as main_mod
    from finq import risk as risk_mod

    def boom(R, w, benchmarks):
        raise risk_mod.RiskError("synthetic: benchmark has zero variance")

    monkeypatch.setattr(main_mod.risk, "betas", boom)

    p = tmp_path / "p.csv"
    p.write_text("ticker,weight\nPKO.WA,0.5\nSPY,0.5\n", encoding="utf-8")
    rc = main_mod.main(["analyze", str(p), "--cache-dir", str(seeded_cache)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "betas unavailable" in out
    assert "synthetic: benchmark has zero variance" in out
    assert "Annualized volatility" in out   # the rest of the report still ran


def test_header_shows_both_price_and_fx_staleness_from_a_real_run(
    tmp_path, seeded_cache, monkeypatch, capsys
):
    """End-to-end version of the header-merge test: backdate PKO.WA's price
    cache AND the FX_USD cache so both go through the "stale refetch
    fallback" path in finq.data, with the refetch itself mocked to fail
    (kept fully offline, no live Yahoo/NBP call). Confirms _build() actually
    performs the pdata.stale + fx_stale merge end to end, not just that
    report.header() can format a hand-built list correctly.
    """
    import os
    from datetime import datetime
    from finq import __main__ as main_mod
    from finq import data as data_mod

    old = datetime.now().timestamp() - 86400 * 2
    os.utime(seeded_cache / "PKO.WA.csv", (old, old))
    os.utime(seeded_cache / "FX_USD.csv", (old, old))

    def fail_yahoo(ticker):
        raise data_mod.DataError(f"{ticker}: fetch failed (offline)")

    def fail_nbp(code):
        raise data_mod.DataError(f"FX {code}: fetch failed (offline)")

    monkeypatch.setattr(data_mod, "_fetch_yahoo", fail_yahoo)
    monkeypatch.setattr(data_mod, "_fetch_nbp", fail_nbp)

    p = tmp_path / "p.csv"
    p.write_text("ticker,weight\nPKO.WA,0.5\nSPY,0.5\n", encoding="utf-8")
    rc = main_mod.main(["analyze", str(p), "--cache-dir", str(seeded_cache)])
    out = capsys.readouterr().out
    assert rc == 0
    warn_line = next(l for l in out.splitlines() if l.startswith("WARNING: STALE CACHE"))
    assert "PKO.WA" in warn_line
    assert "FX:USD" in warn_line


def test_liquidity_table_converts_usd_holdings_to_pln(tmp_path, seeded_cache):
    """CRITICAL GAP the brief's own tests never cover: a ticker,quantity
    portfolio (quantities mode) mixing PLN and USD names. finq.liquidity.adv()
    is currency-blind by design, so cmd_analyze must convert the USD close
    panel to PLN (via the FX rates already fetched in _build()) before
    calling adv() -- otherwise the "ADV20 (PLN)" column for SPY would just be
    the raw USD figure under a mislabeled PLN header.

    This recomputes the expected PLN-converted ADV, and the expected
    days-to-exit, directly from the same finq.data / finq.liquidity building
    blocks cmd_analyze itself uses (mirroring _build()'s own pipeline), and
    cross-checks the CLI's printed numbers against them -- while also pinning
    that the number is materially different from (and not literally equal
    to) the raw, unconverted USD ADV.
    """
    from finq import data as data_mod
    from finq import liquidity as liq_mod

    p = tmp_path / "p.csv"
    p.write_text(
        "ticker,quantity\nPKO.WA,1000\nCDR.WA,500\nSPY,300\nGLD,200\n",
        encoding="utf-8",
    )
    r = _run(["analyze", str(p)], seeded_cache)
    assert r.returncode == 0, r.stderr
    assert "ADV20 (PLN)" in r.stdout
    assert "Liquidity" in r.stdout

    tickers = ["PKO.WA", "CDR.WA", "SPY", "GLD"]
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    pdata = data_mod.prices(tickers, str(start.date()), str(end.date()), cache_dir=seeded_cache)
    usd_rate = data_mod.fx("USD", str(start.date()), str(end.date()), cache_dir=seeded_cache)

    # Raw, currency-blind ADV -- what a BROKEN (unconverted) implementation
    # would have printed under the same "ADV20 (PLN)" label.
    raw_usd_adv = liq_mod.adv(pdata.close[["SPY"]], pdata.volume[["SPY"]], window=20)["SPY"]

    # The correct, PLN-converted figure: convert the close panel first, same
    # as cmd_analyze's own pln_close construction.
    pln_close = pdata.close[["SPY"]].copy()
    pln_close["SPY"] = pln_close["SPY"] * usd_rate.reindex(pln_close.index).ffill()
    expected_adv = liq_mod.adv(pln_close, pdata.volume[["SPY"]], window=20)["SPY"]

    last_pln = pln_close.ffill().iloc[-1]["SPY"]
    quantity = 300.0
    expected_days = liq_mod.days_to_liquidate(quantity * last_pln, expected_adv)

    # pandas right-justifies this column in scientific notation when the
    # magnitudes across rows span several orders (PLN-converted ADV easily
    # does, PKO.WA vs SPY), so the number pattern must accept "1.33e+11".
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    m = re.search(rf"^SPY\s+({num})\s+({num})\s+(\d+)", r.stdout, re.MULTILINE)
    assert m, r.stdout
    printed_adv, printed_days, printed_stale = float(m.group(1)), float(m.group(2)), int(m.group(3))

    assert printed_adv == pytest.approx(expected_adv, rel=1e-6)
    # to_string() prints "days to exit" at a fixed 6 decimal places, and this
    # position is small enough (huge SPY ADV, modest quantity) that 6 decimal
    # digits carries only ~2 significant figures -- allow for that print
    # truncation rather than the underlying float precision.
    assert printed_days == pytest.approx(expected_days, rel=1e-3, abs=1e-6)

    # Materially different from (not the raw USD number): USD/PLN over the
    # fixture window sits roughly 3.5-4.5, so a converted value must be at
    # least double the raw one.
    assert printed_adv > raw_usd_adv * 2
    assert printed_adv != pytest.approx(raw_usd_adv, rel=0.5)

    # stale_price_flag stays on the ORIGINAL local close, not the
    # PLN-converted panel -- cross-check against the unconverted series.
    expected_stale = liq_mod.stale_price_flag(pdata.close[["SPY"]])["SPY"]
    assert printed_stale == expected_stale


def test_analyze_weights_convert_quantity_holdings_to_pln(tmp_path, seeded_cache):
    """CRITICAL GAP: _weights() (finq/__main__.py) computed
    `quantity * last_close` and normalized directly, summing a ~776 USD SPY
    close against a ~111 PLN PKO.WA close as if they were the same currency.
    That weight array feeds every risk.* call in cmd_analyze (portfolio_vol,
    risk_contributions, var_cvar, betas, ...), so a USD/EUR-heavy quantity
    portfolio's entire risk report -- not just this one printed column -- was
    built on wrong weights.

    Recomputes the expected PLN-converted weight for SPY directly from
    finq.data (mirroring _build()'s own pipeline) and cross-checks the
    printed "weight" column in the "Weight vs risk contribution" table
    against it, while also pinning that it is NOT the naive, unconverted
    weight a broken implementation would print.
    """
    from finq import data as data_mod

    p = tmp_path / "p.csv"
    p.write_text("ticker,quantity\nPKO.WA,10\nSPY,1\n", encoding="utf-8")
    r = _run(["analyze", str(p)], seeded_cache)
    assert r.returncode == 0, r.stderr

    tickers = ["PKO.WA", "SPY"]
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    pdata = data_mod.prices(tickers, str(start.date()), str(end.date()), cache_dir=seeded_cache)
    usd_rate = data_mod.fx("USD", str(start.date()), str(end.date()), cache_dir=seeded_cache)

    pko_last = pdata.close["PKO.WA"].ffill().iloc[-1]
    spy_last_usd = pdata.close["SPY"].ffill().iloc[-1]
    spy_last_pln = spy_last_usd * usd_rate.reindex(pdata.close.index).ffill().iloc[-1]

    pko_value = 10 * pko_last
    spy_value_pln = 1 * spy_last_pln
    expected_spy_weight = spy_value_pln / (pko_value + spy_value_pln)

    # What a BROKEN (currency-unconverted) implementation would compute.
    spy_value_naive = 1 * spy_last_usd
    naive_spy_weight = spy_value_naive / (pko_value + spy_value_naive)

    num = r"[-+]?\d*\.?\d+"
    m = re.search(rf"^SPY\s+({num})%\s+({num})%", r.stdout, re.MULTILINE)
    assert m, r.stdout
    printed_spy_weight = float(m.group(1)) / 100.0

    assert printed_spy_weight == pytest.approx(expected_spy_weight, abs=1e-3)
    assert printed_spy_weight != pytest.approx(naive_spy_weight, abs=0.05)
