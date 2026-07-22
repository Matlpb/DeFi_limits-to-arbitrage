"""
Thin client for the Kraken public REST API (spot USD prices).

The public OHLC endpoint only serves the most recent ~720 candles, so a historical
window is rebuilt from the Trades endpoint, which paginates back to a pair's inception
via the ``since`` cursor. :func:`usd_prices_1min` returns a 1-minute close-price series
for a Kraken pair over a chosen window - used to build the volatility control.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

_TRADES_URL = "https://api.kraken.com/0/public/Trades"


def fetch_trades(pair: str, start_ts: str, end_ts: str, sleep: float = 1.0) -> pd.DataFrame:
    """All trades for a Kraken ``pair`` within [start_ts, end_ts] (UTC), via since-pagination.

    Pages the Trades endpoint forward from ``start_ts`` (1000 trades per call) until the
    window end is passed, respecting the public rate limit with ``sleep`` seconds between
    calls. Returns a DataFrame ``[time, price, volume]`` sorted by time.
    """
    start = pd.Timestamp(start_ts, tz="UTC")
    end = pd.Timestamp(end_ts, tz="UTC")
    since = int(start.value)
    end_sec = end.timestamp()

    rows = []
    while True:
        resp = requests.get(_TRADES_URL, params={"pair": pair, "since": since}, timeout=30).json()
        if resp.get("error"):
            raise RuntimeError(f"Kraken Trades error for {pair}: {resp['error']}")
        result = resp["result"]
        key = next(k for k in result if k != "last")
        batch = result[key]
        if not batch:
            break
        rows.extend(batch)
        last = int(result["last"])
        if float(batch[-1][2]) >= end_sec or last <= since:
            break
        since = last
        time.sleep(sleep)

    df = pd.DataFrame(rows, columns=["price", "volume", "time", "side", "type", "misc", "id"])
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df[(df["time"] >= start) & (df["time"] <= end)]
    return df[["time", "price", "volume"]].sort_values("time").reset_index(drop=True)


def usd_prices_1min(pair: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    """1-minute close USD price series for a Kraken ``pair`` over the window (from trades).

    Trades are resampled to 1-minute bars - the last trade price in each minute,
    forward-filled over empty minutes. Returns ``[time, price]``.
    """
    trades = fetch_trades(pair, start_ts, end_ts)
    if trades.empty:
        print(f"{pair}: no trades in window")
        return pd.DataFrame(columns=["time", "price"])

    px = trades.set_index("time")["price"].resample("1min").last().ffill()
    print(f"{pair}: {len(trades)} trades -> {len(px)} 1-min bars "
          f"({px.index.min()} -> {px.index.max()})")
    return px.reset_index()
