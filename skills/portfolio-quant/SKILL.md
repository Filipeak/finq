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
`--lookback 3y`, `--cache-dir`, and for `optimize` also `--method`, `--max-weight`,
`--min-weight`.

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

# pf.weights is None for a quantity- or ticker-only portfolio; resolve weights from
# quantities * last close first in that case (see finq/__main__.py's _weights()).
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
  one exception: an unpublished NBP rate genuinely means no new rate exists, carried
  forward up to `max_fx_staleness_days`; a longer gap is dropped from the panel, not
  fabricated.)
- **Do not use `sklearn.covariance.LedoitWolf`.** It implements the 2004 paper with a
  scaled-identity target. This project follows the 2003 constant-correlation paper in
  `resources/honey.pdf`.
- **Sample covariance divides by T, not T-1.** The shrinkage asymptotics assume it.
- **FX belongs inside the covariance matrix.** It is folded into returns before
  estimation, so every USD holding shares a currency factor. Never add it afterwards.
- **`liquidity.adv()` is currency-blind.** It reports "in quote currency" per its own
  docstring — it multiplies whatever close/volume it is handed, with no FX conversion.
  On a mixed PL/US portfolio, calling it directly on raw prices silently mixes PLN and
  USD figures under one column. Convert close prices to PLN first (multiply by
  `data.fx()` rates, date-aligned to the price index) before calling `adv()` or
  `days_to_liquidate()` — see how `finq/__main__.py`'s `cmd_analyze` does it for the
  worked example.
- **Raw WSE index symbols are unusable as benchmarks.** `^WIG`, `WIG20.WA`, `MWIG40.WA`
  and `SWIG80.WA` are all `instrumentType: INDEX` on Yahoo — the symbol resolves and
  returns a live quote, but the chart endpoint gives one bar and no history, so there is
  no return series to regress against. Use `ETFBW20TR.WA` (Beta ETF WIG20TR), a
  PLN-denominated WSE-listed total-return tracker. Remember it follows WIG20 — 20 bank-
  and energy-heavy names, not the broad Polish market — and being total-return it
  includes dividends while holdings' price returns do not.

## API reference

- `portfolio.load(path) -> Portfolio` — fields `tickers`, `weights`, `quantities`,
  `normalized`, `source_path`
- `data.prices(tickers, start, end, cache_dir=None) -> PriceData` — fields `close`,
  `volume`, `currency`, `stale`. Does **not** align calendars.
- `data.fx(code, start, end, cache_dir=None) -> pd.Series` — NBP mid rates, PLN per unit.
  Carries a staleness flag in `s.attrs["stale"]` (bool): True only when a refetch was
  attempted and failed and the cache was served as a fallback, mirroring
  `PriceData.stale`. Check it if composing a script directly against `fx()` rather than
  going through the CLI (which already merges it into the report header).
- `returns.aligned(price_data, fx_rates, freq="daily", min_obs=0,
  max_fx_staleness_days=7) -> ReturnMatrix` — fields `R`, `freq`, `dropped_days`,
  `tickers`, `fx_returns`
- `covariance.estimate(R, method="ledoit_wolf") -> (Sigma, Diagnostics)`
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
