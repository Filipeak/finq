"""One-time fixture generation. Requires network. Run from repo root."""
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
# NOTE: WIG20.WA (the spec's nominal Polish benchmark) is a real, currency-bearing
# Yahoo listing but Yahoo's chart API returns only a single row for it regardless of
# range/interval (validRanges=['1d','5d'], firstTradeDate=null) -- confirmed the same
# holds for WIG.WA, MWIG40.WA, SWIG80.WA, i.e. Yahoo provides no chart history for
# WSE INDEX-type instruments generally. ETFBW20TR.WA (Beta ETF WIG20TR, a WSE-traded
# fund tracking the WIG20 total-return index) is used instead: it returns full PLN
# daily history and closely tracks WIG20 performance. See task-2-report.md.
TICKERS = ["PKO.WA", "CDR.WA", "PKN.WA", "PZU.WA", "SPY", "QQQ", "GLD",
           "ETFBW20TR.WA", "^GSPC"]
OUT = Path("tests/fixtures")


def fetch(ticker: str) -> pd.DataFrame:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(ticker)}?range=3y&interval=1d")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": q["close"],
        "volume": q["volume"],
    }).dropna(subset=["close"])
    df.attrs["currency"] = res["meta"]["currency"]
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    currencies = {}
    for t in TICKERS:
        df = fetch(t)
        currencies[t] = df.attrs["currency"]
        df.to_csv(OUT / f"{t.replace('^', 'IDX_')}.csv", index=False)
        print(f"{t}: {len(df)} rows, {df.attrs['currency']}")
        time.sleep(0.5)
    (OUT / "currencies.json").write_text(json.dumps(currencies, indent=2))


if __name__ == "__main__":
    main()
