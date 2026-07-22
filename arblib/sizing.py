"""
Trade-size distribution in USD.

Turns the raw per-swap extracts into the pooled, USD-denominated trade-size
distribution and its quantile reference sizes, which the arbitrage step then uses as
the notionals at which to price executable round trips.

    swaps = pooled_swaps(dfs)
    swaps = add_usd_amounts(swaps, x_usd, y_usd, window_min)
    x_in, y_in = split_in_legs(swaps)
    sizes = pooled_trade_sizes(x_in, y_in)
    quantiles = trade_size_quantiles(sizes, [0.2, 0.4, 0.6, 0.8])
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import formulas

_INFO_COLS = ["evt_tx_hash", "dex", "pool", "evt_block_number", "hour"]


def pooled_swaps(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concat every DEX / pool extract into one frame of swaps."""
    swaps = pd.concat(dfs.values(), ignore_index=True)
    print(f"Total swaps across all pools / DEXes: {len(swaps)}")
    print(f"Distinct pools: {swaps['pool'].nunique()} | DEX tags: {list(swaps['dex'].unique())}")
    return swaps


def _trailing_mean(price_series: pd.DataFrame, window_min: int) -> pd.DataFrame:
    """Trailing ``window_min``-minute mean of a 1-minute ``[time, price]`` series, shifted one
    bar so the value at minute t uses only bars strictly before t (no intra-minute or future
    leak). Returns ``[time, rate]``."""
    s = price_series.sort_values("time").set_index("time")["price"]
    rate = s.rolling(f"{window_min}min").mean().shift(1)
    return rate.rename("rate").reset_index()


def add_usd_amounts(swaps: pd.DataFrame, x_usd: pd.DataFrame, y_usd: pd.DataFrame,
                    window_min: int) -> pd.DataFrame:
    """Add human-unit (``amount*_h``) and USD (``amount*_usd``) amounts to ``swaps``.

    Each token is valued with a *causal* CEX exchange rate: the trailing ``window_min``-minute
    mean of its Kraken 1-minute USD price (``x_usd`` = token0/USD, ``y_usd`` = token1/USD),
    shifted so a swap only ever sees past minutes, then matched to each swap by time with a
    backward ``merge_asof``. This smooths the raw 1-minute quotes into a stable hourly-style
    rate while never averaging in future prices. Adds ``amount0_h`` / ``amount1_h``,
    ``amount0_usd`` / ``amount1_usd``, ``time``, ``hour``; returns a copy. Raises if any swap
    predates the available CEX window (widen ``S.cex_start_ts``).
    """
    swaps = swaps.copy()
    swaps["amount0_h"] = formulas.to_human_units(swaps["amount0"], swaps["token0_decimals"])
    swaps["amount1_h"] = formulas.to_human_units(swaps["amount1"], swaps["token1_decimals"])
    swaps["time"] = pd.to_datetime(
        swaps["evt_block_time"].str.replace(" UTC", "", regex=False), utc=True
    )
    swaps["hour"] = swaps["time"].dt.floor("h")
    swaps = swaps.sort_values("time").reset_index(drop=True)

    x_rate = _trailing_mean(x_usd, window_min).rename(columns={"rate": "x_rate"})
    y_rate = _trailing_mean(y_usd, window_min).rename(columns={"rate": "y_rate"})
    swaps = pd.merge_asof(swaps, x_rate, on="time", direction="backward")
    swaps = pd.merge_asof(swaps, y_rate, on="time", direction="backward")

    swaps["amount0_usd"] = formulas.to_usd(swaps["amount0_h"], swaps["x_rate"].to_numpy())
    swaps["amount1_usd"] = formulas.to_usd(swaps["amount1_h"], swaps["y_rate"].to_numpy())
    assert swaps["amount0_usd"].notna().all() and swaps["amount1_usd"].notna().all(), \
        "some swaps predate the CEX price window - widen S.cex_start_ts"
    return swaps


def split_in_legs(swaps: pd.DataFrame, min_token0: float = 1.0,
                  min_token1: float = 0.0001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split swaps into their two in-legs, dust-filtered and sorted largest-first.

    Each swap has one positive and one negative amount, so ``amount0 > 0`` (token0 in)
    and ``amount1 > 0`` (token1 in) partition the swaps. Returns ``(x_in, y_in)`` - the
    token0-in and token1-in legs, each with ``_INFO_COLS`` plus the leg's human/USD size,
    sorted by USD value descending. Requires :func:`add_usd_amounts` first.
    """
    kept = swaps[
        (swaps["amount0_h"].abs() >= min_token0) &
        (swaps["amount1_h"].abs() >= min_token1)
    ].copy()
    print(f"Kept {len(kept)} of {len(swaps)} swaps after dust filter")

    x_in = (
        kept.loc[kept["amount0_h"] > 0, _INFO_COLS + ["amount0_h", "amount0_usd"]]
        .sort_values("amount0_usd", ascending=False)
        .reset_index(drop=True)
    )
    y_in = (
        kept.loc[kept["amount1_h"] > 0, _INFO_COLS + ["amount1_h", "amount1_usd"]]
        .sort_values("amount1_usd", ascending=False)
        .reset_index(drop=True)
    )

    print(f"token0-in: {len(x_in)} swaps | token1-in: {len(y_in)} swaps")
    return x_in, y_in


def pooled_trade_sizes(x_in: pd.DataFrame, y_in: pd.DataFrame) -> pd.Series:
    """Pool both in-legs' USD sizes into one trade-size distribution."""
    return pd.concat([x_in["amount0_usd"], y_in["amount1_usd"]], ignore_index=True)


def trade_size_quantiles(trade_sizes: pd.Series,
                         qs: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> pd.Series:
    """Reference trade sizes = the given quantiles of the pooled USD distribution."""
    return trade_sizes.quantile(list(qs))


def constant_usd_prices(x_usd: pd.DataFrame, y_usd: pd.DataFrame) -> tuple[float, float]:
    """Constant USD price for token0 / token1 = the window mean of each Kraken 1-minute series.

    Holds the arbitrage notional fixed regardless of intraday moves. Returns ``(price0, price1)``.
    """
    return float(x_usd["price"].mean()), float(y_usd["price"].mean())
