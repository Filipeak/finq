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
- `.claude/skills/portfolio-quant/` — the skill Claude loads
- `resources/` — the source papers
- `tests/` — run with `python -m pytest`
