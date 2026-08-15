# Portfolio Quant Toolkit — Design

**Date:** 2026-08-15
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project 1 of the `finance-utils` project

---

## 1. Context

The user holds a personal portfolio of Polish (WSE) and American equities and ETFs,
denominated in PLN. The goal is a toolkit that Claude drives to answer quantitative
questions about that portfolio: risk, covariance structure, volatility, liquidity, and
allocation.

The `resources/` folder contains three papers that define the intellectual core:

| Paper | Contribution used here |
|---|---|
| Ledoit & Wolf (2003), *Honey, I Shrunk the Sample Covariance Matrix* | Shrinkage estimator with constant-correlation target |
| Laloux, Cizeau, Bouchaud & Potters (1998), *Noise Dressing of Financial Correlation Matrices* | Marchenko-Pastur noise band, eigenvalue cleaning |
| Ang & Bekaert (2002), *International Asset Allocation With Regime Shifts* | Asymmetric/exceedance correlation as a diagnostic |

All three make the same argument: the sample covariance matrix is mostly estimation
error, and optimizers amplify that error. This toolkit treats the covariance matrix as
its central object and never presents a number derived from it without also reporting
how trustworthy that matrix is.

## 2. Goal

One sentence: **given a portfolio file, compute honest risk numbers and propose
risk-based weights, with the estimation quality always visible.**

Two modes:

- **Analysis** — tickers *and* weights (or quantities) supplied. Report risk.
- **Selection** — tickers only. Propose weights.

## 3. Scope

### In scope

- Portfolio input from CSV or JSON
- Price and FX data fetching with an on-disk cache
- PLN-based return construction for mixed PLN/USD holdings
- Three covariance estimators plus estimation-quality diagnostics
- Risk decomposition, concentration, tail risk, stress correlation
- Liquidity profiling
- Five risk-based weighting methods
- A skill teaching Claude the API, the method, and the interpretation rules
- Two CLI entry points for the headline flows

### Out of scope (deliberately, each a possible later sub-project)

- **Expected returns in any form.** No mean-variance, no max-Sharpe, no
  Black-Litterman. The system is risk-based only. This is the single largest
  scope decision and it is intentional: expected returns are far harder to
  estimate than covariances, and admitting them would reintroduce exactly the
  error-maximization problem the three papers exist to solve.
- Transaction ledger, cost basis, tax (PIT-38), realized P&L
- Investment thesis tracking and monitoring loops
- Fundamental analysis, EDGAR ingestion, Polish report parsing
- Backtesting, screening, alpha research
- Universes larger than ~30 assets
- Published HTML dashboards

### Sizing assumption

10-30 assets, daily returns, 2-3 years of history. This makes Q = T/N comfortable
(roughly 15-50) for daily data. Shrinkage therefore serves optimizer *stability*
more than noise filtering, and RMT cleaning serves primarily as a *diagnostic*.
Both are still implemented in full, because Q collapses under weekly or monthly
returns (2 years weekly with 30 assets gives Q = 3.5) where the noise problem is
severe.

## 4. Decisions and rationale

| Decision | Rationale |
|---|---|
| Risk-based only, no return forecasts | Expected returns are the least estimable input; excluding them removes the dominant error source and keeps every output defensible |
| Tested library + composable API, not a fixed CLI | Portfolio math must be deterministic and test-covered; portfolio *questions* are open-ended and cannot be enumerated as flags in advance |
| FX folded into returns before estimation | Currency risk is real correlation shared across all USD holdings; it belongs inside the covariance matrix, not as a bolted-on line item |
| Inner-join trading calendars | Forward-filling closed days creates spurious zero returns that bias correlations *downward*, producing diversification estimates that are too optimistic precisely where optimism is most dangerous |
| Ledoit-Wolf implemented, not imported | `sklearn.covariance.LedoitWolf` implements the 2004 paper (scaled-identity target). The 2003 paper in `resources/` uses the constant-correlation target. We follow the paper the user has. |
| Equal weight is a permanent baseline | 1/N is notoriously hard to beat; any method that cannot should have to say so on every run |
| Fail loudly on data problems | A silently dropped ticker means a *different portfolio* was analyzed, and the output looks completely normal |

## 5. Architecture

```
finance-utils/
├── resources/                     # the three papers
├── docs/superpowers/specs/        # this document
├── finq/
│   ├── __init__.py                # curated public API surface
│   ├── data.py                    # Yahoo prices, NBP FX, disk cache
│   ├── portfolio.py               # load + validate CSV/JSON
│   ├── returns.py                 # prices -> aligned PLN return matrix
│   ├── covariance.py              # estimators + diagnostics
│   ├── risk.py                    # risk metrics
│   ├── liquidity.py               # liquidity metrics
│   ├── optimize.py                # weighting methods
│   ├── report.py                  # terminal-table formatting
│   └── __main__.py                # `analyze` and `optimize` entry points
├── tests/
│   ├── fixtures/                  # frozen price panels
│   └── test_*.py
├── skills/portfolio-quant/SKILL.md
└── cache/                         # gitignored
```

### Data flow

```
portfolio.csv/json
      │  portfolio.load()
      ▼
Portfolio(tickers, weights|quantities|None)
      │  data.prices(tickers, lookback)   ── Yahoo chart API ──┐
      │  data.fx(currencies)              ── NBP API ──────────┤── cache/
      ▼                                                        │
returns.aligned(prices, fx, base="PLN")  ──►  R  (T × N)  ◄────┘
      │
      ├──► covariance.estimate(R, method) ──► (Sigma, Diagnostics)
      │            │
      │            ├──► risk.report(Sigma, w, R)
      │            └──► optimize.weights(Sigma, method, constraints)
      │
      └──► liquidity.profile(prices, volumes, positions)
```

## 6. Module specifications

### 6.1 `data.py`

Fetches and caches raw market data. Never computes anything.

- `prices(tickers, start, end) -> DataFrame` — daily OHLCV per ticker from the Yahoo
  chart endpoint (`query1.finance.yahoo.com/v8/finance/chart/{ticker}`). Verified working
  for both `PKO.WA` (WSE, PLN) and `SPY` (NYSE Arca, USD) without an API key.
  Requires a browser User-Agent header.
- `fx(pair, start, end) -> Series` — NBP table A mid rates
  (`api.nbp.pl/api/exchangerates/rates/a/{code}/{start}/{end}`). Verified working.
  NBP publishes on Polish business days only; gaps are forward-filled *for FX only*,
  which is correct because an unpublished rate means no new official rate exists,
  not a missing observation.
- `currency_of(ticker) -> str` — from the Yahoo chart `meta.currency` field, cached.
- Cache: one CSV per ticker per source under `cache/`, keyed by ticker and date range.
  Reads check cache first. A cache older than one trading day triggers a refetch; if
  the refetch fails, cached data is served **with an explicit staleness warning in the
  return value**, never silently.
- Rate limiting: sequential requests with a small delay. At 10-30 tickers this is a
  few seconds on a cold cache and near-instant afterwards.

### 6.2 `portfolio.py`

- `load(path) -> Portfolio` — accepts CSV or JSON, detected by extension.
- Three accepted shapes:
  - `ticker,quantity` — **preferred.** Yields true weights from live prices and is the
    only shape that enables days-to-liquidate.
  - `ticker,weight` — accepted. Normalized to sum to 1, and the normalization is
    reported if the input did not already sum to 1.
  - `ticker` alone — selection mode; analysis functions requiring weights will refuse.
- Validation: duplicate tickers rejected; negative quantities rejected (no shorts in
  scope); empty file rejected. All rejections name the offending row.

### 6.3 `returns.py`

- `aligned(prices, fx, base="PLN", freq="daily") -> ReturnMatrix`
- For a USD-denominated holding, the PLN return is
  `(1 + r_asset) * (1 + r_fx) - 1`, applied **before** any estimation.
- Calendar alignment is an inner join across all tickers' trading days. The number of
  dropped days is recorded on the result and surfaced in every report.
- `freq` supports `daily` and `weekly` (Wednesday-to-Wednesday, to reduce the effect of
  weekend-adjacent holidays). Weekly is offered because it reduces nonsynchronous-trading
  bias between WSE and NYSE, at the cost of collapsing Q.
- Simple returns throughout, not log returns, because portfolio return is a weighted
  sum of simple asset returns and this is the property the risk math depends on. The
  choice is documented in the skill so it is never silently reversed.
- `fx_component(...) -> DataFrame` — isolates the FX contribution per holding, enabling
  the "how much of my risk is just the złoty" decomposition.

### 6.4 `covariance.py`

`estimate(R, method) -> (Sigma, Diagnostics)` where `method` is one of:

**`sample`** — the plain sample covariance matrix. Implemented as a labelled baseline
and reference point, never as a default.

**`ledoit_wolf`** — Ledoit & Wolf (2003), constant-correlation target, per the paper's
Appendices A and B.

Shrinkage target `F`, from the average sample correlation `r̄`:

```
f_ii = s_ii
f_ij = r̄ * sqrt(s_ii * s_jj)      (i ≠ j)
```

Optimal intensity `δ̂ = max(0, min(κ̂/T, 1))` with `κ̂ = (π̂ - ρ̂) / γ̂`:

```
π̂_ij = (1/T) Σ_t [ (y_it - ȳ_i)(y_jt - ȳ_j) - s_ij ]²
π̂    = Σ_i Σ_j π̂_ij

ϑ̂_ii,ij = (1/T) Σ_t [ (y_it - ȳ_i)² - s_ii ] [ (y_it - ȳ_i)(y_jt - ȳ_j) - s_ij ]

ρ̂ = Σ_i π̂_ii
    + Σ_i Σ_{j≠i} (r̄/2) [ sqrt(s_jj/s_ii) · ϑ̂_ii,ij + sqrt(s_ii/s_jj) · ϑ̂_jj,ij ]

γ̂ = Σ_i Σ_j (f_ij - s_ij)²
```

Result: `Σ̂ = δ̂F + (1 - δ̂)S`. Positive definite by construction even when N > T, since
F is positive definite and S is positive semi-definite.

**`rmt_clean`** — Laloux et al. (1998). With `Q = T/N` and returns standardized to unit
variance, the Marchenko-Pastur noise band edges are:

```
λ± = σ² (1 + 1/Q ± 2·sqrt(1/Q))
```

Eigen-decompose the correlation matrix. Eigenvalues above `λ₊` are treated as signal and
kept with their eigenvectors. The bulk at or below `λ₊` is replaced by a single common
value chosen to preserve the trace. The cleaned correlation matrix is re-scaled by the
sample volatilities to recover a covariance matrix. Following the paper, `σ²` is fitted
to the bulk rather than fixed at 1, since the market eigenvalue absorbs a large share of
the total variance.

**Diagnostics** — returned on *every* call, regardless of method:

- `Q = T/N`, and the raw T and N
- Marchenko-Pastur band edges `λ₋`, `λ₊`
- Full eigenvalue spectrum, and the count and variance share falling inside the band
- Selected shrinkage intensity `δ̂` (for the `ledoit_wolf` method)
- Condition number of the resulting matrix
- Days dropped by calendar alignment

`δ̂` functions as an honesty meter: a high value means the sample matrix carried little
usable information, and every downstream number inherits that weakness.

### 6.5 `risk.py`

All volatility figures annualized (√252 daily, √52 weekly), with the convention stated
in output.

- `portfolio_vol(Sigma, w)`
- `risk_contributions(Sigma, w)` — marginal (MCTR) and percentage contributions. These
  must sum to total portfolio volatility, which is also a test. Reveals the small
  position carrying outsized risk.
- `diversification_ratio(Sigma, w)` — Choueifaty: `(w'σ) / sqrt(w'Σw)`
- `effective_bets(Sigma, w)` — Meucci: entropy of the variance shares of the principal
  portfolios, `exp(-Σ p_i ln p_i)`
- `concentration(w, rc)` — HHI on weights and, more informatively, on risk contributions
- `var_cvar(R, w, level)` — historical and Cornish-Fisher, at 95% and 99%
- `max_drawdown(R, w)` — on the fixed-weight return series
- `betas(R, w, benchmarks)` — separate betas to the Polish and US markets, since the
  portfolio spans two and a single blended beta would hide which one drives it.
  Verified benchmark tickers: **`ETFBW20TR.WA`** (PLN) and **`^GSPC`** (USD).

  Raw WSE index symbols cannot serve as benchmarks. `^WIG`, `WIG20.WA`, `MWIG40.WA`
  and `SWIG80.WA` are all `instrumentType: INDEX` on Yahoo: the symbol resolves and
  returns a current quote, but the chart endpoint returns a single bar and no history,
  so no return series exists to regress against. (An earlier draft of this spec listed
  `WIG20.WA` as verified — that check confirmed a quote, not a series, and was wrong.)
  The Polish benchmark is therefore `ETFBW20TR.WA`, the Beta ETF WIG20TR: PLN-denominated,
  WSE-listed, trading on the same calendar as the Polish holdings, with history from
  2019-01-07. Two caveats travel with it: it tracks WIG20, so it represents 20 bank- and
  energy-heavy names rather than the broad market, and it is a total-return tracker while
  holdings carry price returns, so it includes dividends they do not. Any further
  benchmark ticker must be verified to return an actual multi-bar price series — not
  merely a resolvable quote — before use.
- `exceedance_correlation(R, threshold)` — Longin-Solnik / Ang-Bekaert: correlation
  conditional on both standardized returns falling below `-threshold` σ, compared against
  the unconditional correlation. Answers whether diversification survives a drawdown,
  which is the only correlation question that matters for risk.
- `fx_risk_share(R, w)` — fraction of portfolio variance attributable to currency

### 6.6 `liquidity.py`

Uses the volume series already returned by the Yahoo chart endpoint.

- `adv(prices, volumes, window)` — average daily traded value in PLN, 20d and 60d
- `days_to_liquidate(position_value, adv, participation=0.10)` — trading days to exit at
  a given share of daily volume. Requires quantities, hence the preference for that input.
- `amihud(returns, traded_value)` — mean of `|r_t| / traded_value_t`
- `stale_price_flag(prices)` — consecutive unchanged closes, a genuine occurrence on thin
  WSE names and a signal that the asset's correlation estimates are biased low

Liquidity is reported, not imposed as an optimizer constraint. At 10-30 held names the
useful question is "could I exit this?", not "constrain the solver."

### 6.7 `optimize.py`

Common constraints: long-only, weights sum to 1, optional maximum weight cap, optional
minimum weight. Minimum weight is applied as a post-solve cleanup: holdings falling below
the floor are dropped and the problem is **re-solved on the reduced set**, repeating until
no holding is below the floor or only one remains. Solved with `scipy.optimize` (SLSQP);
`cvxpy` is not required and is not installed.

- `equal_weight` — the permanent baseline
- `min_variance`
- `risk_parity` — equal risk contribution
- `max_diversification` — Choueifaty, maximizes the diversification ratio
- `hrp` — Hierarchical Risk Parity: correlation distance `d = sqrt(0.5(1-ρ))`, hierarchical
  clustering, quasi-diagonalization, recursive bisection. Included specifically because it
  never inverts the covariance matrix, which is the right instinct when that matrix is noisy.

`compare(Sigma, methods) -> DataFrame` runs all methods side by side. The dispersion
across methods is itself a reported finding: wide disagreement means the covariance
estimate is unstable and none of the weight vectors should be trusted precisely.

### 6.8 `report.py`

Terminal-oriented table formatting only. Every report header carries the estimation
context: N, T, Q, the estimator used, `δ̂` if applicable, days dropped in alignment, and
cache staleness if any.

## 7. Interaction design

### The skill: `skills/portfolio-quant/SKILL.md`

Triggers on portfolio risk, covariance, volatility, liquidity, diversification, and
weighting questions. Contains:

- The public API surface with signatures and short examples
- Which estimator suits which situation, and why `sample` is a baseline rather than a choice
- Pitfalls that produce silently wrong numbers: annualization factors, simple vs log
  returns, calendar alignment, currency conversion ordering
- **Interpretation rules, mandatory on every output:** report Q and δ̂; compare optimized
  weights against equal weight; show cross-method dispersion before presenting any single
  weight vector; state when a result rests on too little data to mean anything

### CLI entry points

```
python -m finq analyze  portfolio.csv [--cov ledoit_wolf] [--freq daily] [--lookback 3y]
python -m finq optimize portfolio.csv [--cov ledoit_wolf] [--freq daily] [--lookback 3y]
                                      [--method all] [--max-weight 0.25] [--min-weight 0.02]
```

These cover the two flows that will run repeatedly. Everything else is Claude composing
against the API — for example, "what happens to my risk if I halve PKO and add gold",
"which two holdings are secretly the same bet", "how much of my vol is USD/PLN", or
"show me the same report with the naive covariance so I can see the difference". None of
these can be anticipated as CLI flags; all of them need the primitives to be correct.

## 8. Testing strategy

TDD. Portfolio math fails silently — a wrong covariance function does not raise, it just
returns plausible numbers forever — so every function ships with a known-answer test.

**Analytic identities:**
- N uncorrelated unit-variance assets, equal-weighted → portfolio vol = 1/√N
- Marchenko-Pastur edges match `σ²(1 + 1/Q ± 2√(1/Q))` for several values of Q
- Risk contributions sum to portfolio volatility
- ERC solution → all risk contributions equal within tolerance
- Two-asset minimum-variance weights match the closed form
- Diversification ratio = 1 for a single asset, > 1 for any imperfectly correlated pair

**Estimator properties:**
- `δ̂ ∈ [0, 1]` always; rises as T falls
- Ledoit-Wolf output positive definite when T < N, where sample is singular
- On synthetic i.i.d. noise, essentially all eigenvalues fall inside the MP band
- On synthetic data with one injected common factor, exactly one eigenvalue escapes it
- Shrinkage of an already-perfect constant-correlation matrix leaves it unchanged

**Integration:**
- Golden-file test over a frozen price panel in `tests/fixtures/`, so results stay
  reproducible as the code changes
- The data layer is tested against saved fixtures, never the live network

## 9. Error handling

The governing principle: an analysis run on the wrong data must never look like a
successful run.

| Condition | Behavior |
|---|---|
| Unknown or delisted ticker | Fail, naming the ticker. Never silently drop — that analyzes a different portfolio. |
| Insufficient history (recent listing) | Truncate the whole panel to the common window and report exactly how many observations were lost and which ticker caused it. Fail outright only if the common window leaves T < 60 observations, which is too little for any estimator here to mean anything. |
| T < N | Refuse `sample`; permit `ledoit_wolf` and `rmt_clean` with a prominent warning |
| Network failure | Serve cache with an explicit staleness warning in the returned object and every report header |
| Weights not summing to 1 | Normalize, and report that normalization occurred |
| Stale prices (unchanged closes) | Flag the ticker and warn that its correlations are biased low |
| Quantities absent, liquidity requested | Refuse with a message explaining that position size is required |
| Optimizer fails to converge | Report failure with the solver status; never return the last iterate as if it were a solution |

## 10. References

- Ledoit, O. & Wolf, M. (2003). *Honey, I Shrunk the Sample Covariance Matrix.* `resources/honey.pdf`
- Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1998). *Noise Dressing of Financial Correlation Matrices.* `resources/9810255v1.pdf`
- Ang, A. & Bekaert, G. (2002). *International Asset Allocation With Regime Shifts.* `resources/1137.pdf`
- Choueifaty, Y. & Coignard, Y. (2008). *Toward Maximum Diversification.* (diversification ratio)
- Meucci, A. (2009). *Managing Diversification.* (effective number of bets)
- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out of Sample.* (HRP)
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive Diversification.* (the 1/N baseline)
