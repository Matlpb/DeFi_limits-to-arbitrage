"""
Cross-DEX price comparison.

Given the aligned per-pool price series, compute the absolute mid-price difference
for every pair of pools that live on different DEXes (intra-DEX pairs are skipped:
arbitrage is about price gaps across venues).
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from . import formulas


def add_mid_price(pools: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add a ``mid_price`` column to every processed pool frame (in place).

    The mid price (token0 per token1) is computed from ``sqrtPriceX96`` via
    :func:`arblib.formulas.mid_price`, since the processed CSVs store ``sqrtPriceX96``
    but not the price. Returns ``pools``.
    """
    for df in pools.values():
        df["mid_price"] = formulas.mid_price(
            df["sqrtPriceX96"], df["token0_decimals"], df["token1_decimals"]
        )
    return pools


def _group_pools_by_dex(filtered_pools: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """Group ``{pool_name: df}`` by DEX -> ``{dex: {pool_name: df}}``."""
    dex_pools: dict[str, dict[str, pd.DataFrame]] = {}
    for pool_name, df in filtered_pools.items():
        dex_name = pool_name.split("_")[0]
        dex_pools.setdefault(dex_name, {})[pool_name] = df
    return dex_pools


def compute_pairwise_differences(
    filtered_pools: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Absolute mid-price difference for every cross-DEX pool pair.

    Returns ``(price_differences, all_diffs_df)``: ``price_differences`` maps
    ``"<poolA>_vs_<poolB>"`` to a frame with both prices and ``abs_diff``;
    ``all_diffs_df`` is a wide summary holding ``abs_diff`` for every pair by block.
    """
    dex_pools = _group_pools_by_dex(filtered_pools)
    print(f"DEXes found: {list(dex_pools.keys())}")
    print(f"Pools per DEX: {[(d, len(p)) for d, p in dex_pools.items()]}\n")

    price_differences = {}
    for dex1, dex2 in combinations(dex_pools, 2):
        for pool1, df1 in dex_pools[dex1].items():
            for pool2, df2 in dex_pools[dex2].items():
                left = df1[["evt_block_number", "evt_block_time", "mid_price"]].rename(
                    columns={"mid_price": f"{pool1}_price"}
                )
                right = df2[["evt_block_number", "mid_price"]].rename(
                    columns={"mid_price": f"{pool2}_price"}
                )

                merged = left.merge(right, on="evt_block_number", how="inner")
                merged["abs_diff"] = (merged[f"{pool1}_price"] - merged[f"{pool2}_price"]).abs()

                price_differences[f"{pool1}_vs_{pool2}"] = merged

    print(f"Computed {len(price_differences)} pairwise comparisons:\n")
    for pair_name in sorted(price_differences):
        df = price_differences[pair_name]
        print(f"{pair_name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Max diff: {df['abs_diff'].max():.6f}")
        print(f"  Mean diff: {df['abs_diff'].mean():.6f}")
        print(f"  Min diff: {df['abs_diff'].min():.6f}\n")

    first = next(iter(price_differences.values()))
    all_diffs_df = first[["evt_block_number", "evt_block_time"]].copy()
    for pair_name in sorted(price_differences):
        all_diffs_df[pair_name] = price_differences[pair_name]["abs_diff"].values

    print(f"Combined difference dataframe shape: {all_diffs_df.shape}")
    return price_differences, all_diffs_df
