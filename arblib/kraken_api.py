"""
Thin client for the Kraken public REST API (spot USD prices).

Two granularities, each from the endpoint that serves it best:

* :func:`usd_prices_1min` - a 1-minute close series over an arbitrary historical window, rebuilt
  from the **Trades** endpoint (paginated back via the ``since`` cursor), since the OHLC endpoint
  only serves the most recent ~720 candles. Used to build the block-level volatility control.
* :func:`usd_prices_daily` - a daily close series over a multi-month lookback, from the **OHLC**
  endpoint: at the daily interval its ~720-candle limit is ~2 years, so one call covers 6-12 months.
  Used for the volatility-regime detection.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

_BASE_URL = "https://api.kraken.com/0/public"


def _to_utc(ts) -> pd.Timestamp:
    """Coerce any date/time input to a tz-aware UTC ``Timestamp`` (naive is assumed UTC)."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _kraken_get(endpoint: str, params: dict) -> dict:
    """GET a Kraken public ``endpoint`` and return its ``result``, raising on an API error."""
    resp = requests.get(f"{_BASE_URL}/{endpoint}", params=params, timeout=30).json()
    if resp.get("error"):
        raise RuntimeError(f"Kraken {endpoint} error for {params.get('pair')}: {resp['error']}")
    return resp["result"]


def _result_series_key(result: dict) -> str:
    """The single data key in a Kraken result payload (everything except the ``last`` cursor)."""
    return next(k for k in result if k != "last")


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
        result = _kraken_get("Trades", {"pair": pair, "since": since})
        batch = result[_result_series_key(result)]
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


def usd_prices_daily(pair: str, days: int = 365, start=None, end=None) -> pd.DataFrame:
    """Daily close USD price series for a Kraken ``pair`` (OHLC endpoint).

    One OHLC call at the daily interval (1440 min) returns up to ~720 committed candles - about two
    years - so any 6-12 month lookback comes back in a single request. Returns ``[time, price]`` with
    a tz-aware UTC daily ``time`` and the daily close as ``price``.

    By default returns the most recent ``days`` days (relative to now). Pass ``start`` and/or ``end``
    (UTC-parseable) to fetch a **fixed, reproducible** window instead - the way the market-regime
    notebook should pull, so the classified year does not shift between runs.
    """
    start = _to_utc(start) if start is not None else None
    end = _to_utc(end) if end is not None else None
    since = int(start.timestamp()) if start is not None \
        else int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp())
    result = _kraken_get("OHLC", {"pair": pair, "interval": 1440, "since": since})
    ohlc = pd.DataFrame(result[_result_series_key(result)],
                        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    ohlc["time"] = pd.to_datetime(ohlc["time"].astype(float), unit="s", utc=True)
    ohlc["price"] = ohlc["close"].astype(float)
    out = ohlc[["time", "price"]]
    if start is not None:
        out = out[out["time"] >= start]
    if end is not None:
        out = out[out["time"] <= end]
    out = out.reset_index(drop=True)
    print(f"{pair}: {len(out)} daily candles ({out['time'].min().date()} -> {out['time'].max().date()})")
    return out
