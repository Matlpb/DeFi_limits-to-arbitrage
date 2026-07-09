"""
Trade-size distribution in USD.

Turns the raw per-swap extracts into the pooled, USD-denominated trade-size
distribution and its quantile "reference sizes", which the arbitrage step then
uses as the notionals at which to price executable round trips.

    swaps = pooled_swaps(dfs)                 # concat all pools, static-fee only
    swaps = add_usd_amounts(swaps, prices)    # human amounts + hourly USD value
    x_in, y_in = split_in_legs(swaps)         # USDC-in / WETH-in legs (dust-filtered)
    sizes = pooled_trade_sizes(x_in, y_in)
    quantiles = trade_size_quantiles(sizes, [0.2, 0.4, 0.6, 0.8])
"""

import pandas as pd

from . import formulas

# Metadata carried alongside each in-leg size so an outlier can be traced back.
_INFO_COLS = ["evt_tx_hash", "dex", "pool", "evt_block_number", "hour"]


def pooled_swaps(dfs):
    """Concat every DEX / pool extract into one frame, keeping static-fee pools only.

    A pool whose ``fee`` field is not constant is dropped (dynamic-fee pools break
    the constant-fee AMM math used downstream).
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


def add_usd_amounts(swaps, prices):
    """Add human-unit (`amount*_h`) and USD (`amount*_usd`) amounts to ``swaps``.

    ``prices`` is the hourly USD price table (``hour`` parsed to datetime, see
    ``data_io.load_usd_prices``). Each swap is matched to its block-hour and each
    token priced by its own hourly rate. Adds a ``hour`` column and returns ``swaps``.
    Raises if any swap has no matching hourly price for either token.
    """
    swaps = swaps.copy()
    swaps["amount0_h"] = formulas.to_human_units(swaps["amount0"], swaps["token0_decimals"])
    swaps["amount1_h"] = formulas.to_human_units(swaps["amount1"], swaps["token1_decimals"])

    price_lookup = prices.set_index(["hour", "contract_address"])["price"]
    swaps["hour"] = pd.to_datetime(
        swaps["evt_block_time"].str.replace(" UTC", "", regex=False)
    ).dt.floor("h")

    rate0 = price_lookup.reindex(
        pd.MultiIndex.from_arrays([swaps["hour"], swaps["token0"]])
    ).to_numpy()
    rate1 = price_lookup.reindex(
        pd.MultiIndex.from_arrays([swaps["hour"], swaps["token1"]])
    ).to_numpy()
    swaps["amount0_usd"] = formulas.to_usd(swaps["amount0_h"], rate0)
    swaps["amount1_usd"] = formulas.to_usd(swaps["amount1_h"], rate1)

    assert swaps["amount0_usd"].notna().all() and swaps["amount1_usd"].notna().all(), \
        "some swaps have no matching hourly USD price"
    return swaps


def split_in_legs(swaps, min_usdc=1.0, min_weth=0.0001):
    """Split swaps into their two in-legs, dust-filtered and sorted largest-first.

    Each swap has one positive and one negative amount, so ``amount0 > 0`` (USDC in)
    and ``amount1 > 0`` (WETH in) partition the swaps. Returns ``(x_in, y_in)`` -
    the USDC-in and WETH-in legs, each with ``_INFO_COLS`` + the leg's human/USD
    size, sorted by USD value descending. Requires :func:`add_usd_amounts` first.
    """
    kept = swaps[
        (swaps["amount0_h"].abs() >= min_usdc) &
        (swaps["amount1_h"].abs() >= min_weth)
    ].copy()
    print(f"Kept {len(kept)} of {len(swaps)} swaps after dust filter")

    x_in = (
        kept.loc[kept["amount0_h"] > 0, _INFO_COLS + ["amount0_h", "amount0_usd"]]
        .sort_values("amount0_usd", ascending=False)
        .reset_index(drop=True)
    )   # token0 (USDC) sent in
    y_in = (
        kept.loc[kept["amount1_h"] > 0, _INFO_COLS + ["amount1_h", "amount1_usd"]]
        .sort_values("amount1_usd", ascending=False)
        .reset_index(drop=True)
    )   # token1 (WETH) sent in

    print(f"X-in (USDC in): {len(x_in)} swaps | Y-in (WETH in): {len(y_in)} swaps")
    return x_in, y_in


def pooled_trade_sizes(x_in, y_in):
    """Pool both in-legs' USD sizes into one trade-size distribution (a Series)."""
    return pd.concat([x_in["amount0_usd"], y_in["amount1_usd"]], ignore_index=True)


def trade_size_quantiles(trade_sizes, qs=(0.2, 0.4, 0.6, 0.8)):
    """Reference trade sizes = the given quantiles of the pooled USD distribution."""
    return trade_sizes.quantile(list(qs))


def constant_usd_prices(prices):
    """Earliest-hour USD price per token (a Series indexed by contract address).

    Used to hold the USD notional constant over the study window regardless of
    intraday price moves. ``prices`` is the parsed hourly price table.
    """
    return prices.sort_values("hour").groupby("contract_address")["price"].first()
