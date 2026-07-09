"""
Trade-size distribution in USD.

Turns the raw per-swap extracts into the pooled, USD-denominated trade-size
distribution and its quantile reference sizes, which the arbitrage step then uses as
the notionals at which to price executable round trips.

    swaps = pooled_swaps(dfs)
    swaps = add_usd_amounts(swaps, prices)
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
    """Concat every DEX / pool extract into one frame, keeping static-fee pools only.

    A pool whose ``fee`` field is not constant is dropped, since the constant-fee AMM
    math used downstream would not apply.
    """
    swaps = pd.concat(dfs.values(), ignore_index=True)

    fee_per_pool = swaps.groupby("pool")["fee"].nunique()
    static_pools = fee_per_pool[fee_per_pool == 1].index
    dropped_pools = fee_per_pool[fee_per_pool > 1].index
    swaps = swaps[swaps["pool"].isin(static_pools)].reset_index(drop=True)

    print(f"Total swaps across all pools / DEXes: {len(swaps)}")
    print(f"Distinct pools: {swaps['pool'].nunique()} | DEX tags: {list(swaps['dex'].unique())}")
    print(f"Dropped {len(dropped_pools)} dynamic-fee pool(s): {list(dropped_pools)}")
    return swaps


def add_usd_amounts(swaps: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add human-unit (``amount*_h``) and USD (``amount*_usd``) amounts to ``swaps``.

    ``prices`` is the hourly USD price table (``hour`` parsed to datetime). Each swap is
    matched to its block-hour and each token priced by its own hourly rate. Adds a
    ``hour`` column and returns a copy. Raises if any swap has no matching hourly price.
    """
    swaps = swaps.copy()
    swaps["amount0_h"] = formulas.to_human_units(swaps["amount0"], swaps["token0_decimals"])
    swaps["amount1_h"] = formulas.to_human_units(swaps["amount1"], swaps["token1_decimals"])

    price_lookup = prices.set_index(["hour", "contract_address"])["price"]
    swaps["hour"] = pd.to_datetime(
        swaps["evt_block_time"].str.replace(" UTC", "", regex=False)
    ).dt.floor("h")

    rate0 = price_lookup.reindex(pd.MultiIndex.from_arrays([swaps["hour"], swaps["token0"]])).to_numpy()
    rate1 = price_lookup.reindex(pd.MultiIndex.from_arrays([swaps["hour"], swaps["token1"]])).to_numpy()
    swaps["amount0_usd"] = formulas.to_usd(swaps["amount0_h"], rate0)
    swaps["amount1_usd"] = formulas.to_usd(swaps["amount1_h"], rate1)

    assert swaps["amount0_usd"].notna().all() and swaps["amount1_usd"].notna().all(), \
        "some swaps have no matching hourly USD price"
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


def constant_usd_prices(prices: pd.DataFrame) -> pd.Series:
    """Earliest-hour USD price per token (a Series indexed by contract address).

    Holds the USD notional constant over the study window regardless of intraday moves.
    ``prices`` is the parsed hourly price table.
    """
    return prices.sort_values("hour").groupby("contract_address")["price"].first()
