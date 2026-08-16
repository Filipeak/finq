from __future__ import annotations

import numpy as np
import pandas as pd

from finq.covariance import Diagnostics


def header(diag: Diagnostics, freq: str, dropped_days: int, stale: list[str]) -> str:
    lines = [
        "=" * 72,
        f"Estimation context  |  N={diag.N}  T={diag.T}  Q={diag.Q:.1f}  freq={freq}",
        f"Covariance method   |  {diag.method}",
    ]
    if diag.shrinkage is not None:
        note = ("   <- high: the sample matrix carried little information"
                if diag.shrinkage > 0.5 else "")
        lines.append(f"Shrinkage intensity |  delta={diag.shrinkage:.3f}{note}")
    lines.append(
        f"Noise band          |  {diag.n_in_band}/{diag.N} eigenvalues inside "
        f"[{diag.lambda_minus:.3f}, {diag.lambda_plus:.3f}], "
        f"{diag.var_share_in_band:.0%} of spectrum"
    )
    lines.append(f"Calendar alignment  |  {dropped_days} non-common days dropped")
    if stale:
        lines.append(f"WARNING: STALE CACHE for {', '.join(stale)} — prices may be out of date")
    if diag.Q < 2:
        lines.append("WARNING: Q < 2 — this covariance matrix is mostly noise")
    lines.append("=" * 72)
    return "\n".join(lines)


def analyze_text(tickers: list[str], w: np.ndarray, vol: float,
                 pct_rc: np.ndarray, div_ratio: float, enb: float,
                 conc: dict[str, float], var95: float, cvar95: float,
                 mdd: float, beta_map: dict[str, float], fx_share: float,
                 liquidity: pd.DataFrame | None) -> str:
    table = pd.DataFrame({
        "weight": w,
        "risk contribution": pct_rc,
    }, index=tickers).sort_values("risk contribution", ascending=False)

    lines = [
        "",
        f"Annualized volatility   {vol:.2%}",
        f"Diversification ratio   {div_ratio:.2f}",
        f"Effective bets          {enb:.1f} of {len(tickers)} holdings",
        f"HHI (weights / risk)    {conc['hhi_weights']:.3f} / {conc['hhi_risk']:.3f}",
        f"VaR 95% / CVaR 95%      {var95:.2%} / {cvar95:.2%}  (per period)",
        f"Max drawdown            {mdd:.2%}",
        f"FX share of risk        {fx_share:.1%}",
        "Betas                   " + "  ".join(f"{k}={v:.2f}" for k, v in beta_map.items()),
        "",
        "Weight vs risk contribution",
        "-" * 72,
        table.to_string(formatters={
            "weight": "{:.2%}".format,
            "risk contribution": "{:.2%}".format,
        }),
    ]
    if liquidity is not None:
        lines += ["", "Liquidity", "-" * 72, liquidity.to_string()]
    return "\n".join(lines)


def optimize_text(comparison: pd.DataFrame) -> str:
    spread = comparison["dispersion"].max()
    verdict = ("Methods broadly agree; the covariance estimate looks stable."
               if spread < 0.10 else
               "Methods disagree materially. Treat any single weight vector as "
               "one option among several, not a precise answer.")
    return "\n".join([
        "",
        "Proposed weights by method",
        "-" * 72,
        comparison.to_string(float_format=lambda v: f"{v:.2%}"),
        "",
        f"Largest cross-method disagreement: {spread:.2%}",
        verdict,
    ])
