from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finq import covariance, data, liquidity as liq, optimize, portfolio, report, risk
from finq.returns import aligned

BENCHMARKS = {"ETFBW20TR.WA": "WIG20TR", "^GSPC": "GSPC"}


def _build(args):
    """
    NOTE (added during Task 13 planning review, after the Task 4b follow-up
    landed): finq.data.fx() now carries a staleness flag in
    ``s.attrs["stale"]`` (mirroring PriceData.stale), added specifically so
    the report header could surface it here -- a network-down fallback to a
    months-old FX cache must not be indistinguishable from a fresh fetch.
    Collect it into `fx_stale` and merge it with `pdata.stale` before calling
    `report.header()` in both cmd_analyze and cmd_optimize below; don't just
    pass `pdata.stale` alone, which was the brief's original (incomplete)
    draft and would silently drop FX staleness from every report.
    """
    pf = portfolio.load(args.path)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(args.lookback.rstrip("y")))
    cache = Path(args.cache_dir) if args.cache_dir else None

    tickers = pf.tickers + list(BENCHMARKS)
    pdata = data.prices(tickers, str(start.date()), str(end.date()), cache_dir=cache)

    codes = sorted(set(pdata.currency.values()))
    rates = {c: data.fx(c, str(start.date()), str(end.date()), cache_dir=cache) for c in codes}
    fx_stale = sorted(c for c, s in rates.items() if s.attrs.get("stale", False))

    rm = aligned(pdata, rates, freq=args.freq, min_obs=60)
    held = rm.R[pf.tickers]
    bench = rm.R[list(BENCHMARKS)].rename(columns=BENCHMARKS)

    Sigma, diag = covariance.estimate(held, method=args.cov)
    stale = pdata.stale + [f"FX:{c}" for c in fx_stale]
    return pf, pdata, rm, held, bench, Sigma, diag, stale, rates


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
    pf, pdata, rm, held, bench, Sigma, diag, stale, rates = _build(args)
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
        # PLN conversion (added during Task 13 planning review -- flagged as a
        # MUST-WIRE-EXPLICITLY concern back in Task 10): finq.liquidity.adv()
        # is currency-blind by design ("in quote currency", per its own
        # docstring) -- it multiplies whatever close/volume it's handed. On a
        # mixed PL/US panel, calling it directly on raw pdata.close would give
        # PLN for WSE names and USD for US names under one shared "(PLN)"
        # label, which spec 6.6 requires to genuinely be PLN. The
        # days_to_liquidate RATIO would still be numerically self-consistent
        # per asset (both position value and ADV in the same unconverted local
        # currency), but the displayed ADV figures would be incommensurate
        # across currencies and mislabeled. Convert the whole close-price
        # panel to PLN first, date-aligned against the FX rates already
        # fetched in _build(), and convert the position-value side (`last`)
        # the same way so the ratio stays consistent once ADV is genuinely in
        # PLN. stale_price_flag stays on the ORIGINAL local pdata.close, not
        # the PLN-converted panel -- staleness is about the underlying price
        # feed not updating, and FX movement would mask that if applied to
        # the converted panel.
        pln_close = pdata.close[pf.tickers].copy()
        for t in pf.tickers:
            code = pdata.currency[t]
            if code != "PLN":
                pln_close[t] = pln_close[t] * rates[code].reindex(pln_close.index).ffill()

        advs = liq.adv(pln_close, pdata.volume[pf.tickers], window=20)
        last_pln = pln_close.ffill().iloc[-1]
        liquidity_table = pd.DataFrame({
            "ADV20 (PLN)": advs.round(0),
            "days to exit": [
                liq.days_to_liquidate(q * p, a) for q, p, a in
                zip(pf.quantities, last_pln, advs)
            ],
            "stale days": pd.Series(liq.stale_price_flag(pdata.close[pf.tickers])),
        })

    print(report.header(diag, rm.freq, rm.dropped_days, stale))
    print(report.analyze_text(
        pf.tickers, w, risk.portfolio_vol(Sigma, w, freq=rm.freq), pct,
        risk.diversification_ratio(Sigma, w), risk.effective_bets(Sigma, w),
        risk.concentration(w, pct), var95, cvar95, risk.max_drawdown(held, w),
        beta_map, risk.fx_risk_share(held, fx_held, w), liquidity_table,
    ))


def cmd_optimize(args) -> None:
    pf, pdata, rm, held, bench, Sigma, diag, stale, rates = _build(args)
    constraints = optimize.Constraints(max_weight=args.max_weight,
                                       min_weight=args.min_weight)
    methods = None if args.method == "all" else [args.method]
    comparison = optimize.compare(Sigma, pf.tickers, methods, constraints)
    print(report.header(diag, rm.freq, rm.dropped_days, stale))
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
