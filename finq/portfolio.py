from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class PortfolioError(Exception):
    """Raised when a portfolio file is malformed or invalid."""


@dataclass(frozen=True)
class Portfolio:
    tickers: list[str]
    weights: np.ndarray | None
    quantities: np.ndarray | None
    normalized: bool
    source_path: str


def _frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path)


def load(path: str | Path) -> Portfolio:
    path = Path(path)
    df = _frame(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        raise PortfolioError(f"{path}: no 'ticker' column found")
    if df.empty:
        raise PortfolioError(f"{path}: portfolio is empty")

    has_w = "weight" in df.columns
    has_q = "quantity" in df.columns
    if has_w and has_q:
        raise PortfolioError(f"{path}: supply weight or quantity, not both")

    tickers = [str(t).strip() for t in df["ticker"]]
    dupes = {t for t in tickers if tickers.count(t) > 1}
    if dupes:
        raise PortfolioError(f"{path}: duplicate ticker(s): {', '.join(sorted(dupes))}")

    weights = quantities = None
    normalized = False

    if has_q:
        quantities = df["quantity"].to_numpy(dtype=float)
        if (quantities < 0).any():
            bad = [t for t, q in zip(tickers, quantities) if q < 0]
            raise PortfolioError(f"{path}: negative quantity for {', '.join(bad)}")
    elif has_w:
        weights = df["weight"].to_numpy(dtype=float)
        if (weights < 0).any():
            bad = [t for t, w in zip(tickers, weights) if w < 0]
            raise PortfolioError(f"{path}: negative weight for {', '.join(bad)}")
        total = weights.sum()
        if total <= 0:
            raise PortfolioError(f"{path}: weights sum to {total}, must be positive")
        if not np.isclose(total, 1.0):
            weights = weights / total
            normalized = True

    return Portfolio(tickers, weights, quantities, normalized, str(path))
