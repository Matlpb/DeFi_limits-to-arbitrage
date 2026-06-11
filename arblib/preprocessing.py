"""
Price time-series reconstruction pipeline.

Raw Dune extracts contain one row per swap event. To compare pools against each
other we turn those irregular event streams into one aligned, gap-free price
series per pool. The pipeline is:

    1. keep_latest_swap_per_block   - one (latest) price per pool & block
    2. split_by_pool                - one DataFrame per pool address
    3. filter_pools_by_swap_gap     - drop pools that trade too rarely
    4. reconstruct_pool_timeseries  - dense block grid, forward-filled prices
    5. filter_by_start_time         - cut off the warm-up period

Note on the two time anchors used by the study:
    * the *collection* window (start_ts / end_ts) defines what was pulled from
      Dune;
    * the *study* start time is later than the collection start, leaving a
      warm-up margin so every pool already has a known price to forward-fill
      from before the study window begins. Step 5 applies that cut.
"""

import numpy as np
import pandas as pd

# Columns that stay constant within a pool and should be carried over when we
# build the dense block grid.
_METADATA_COLS = ["pool", "dex", "token0", "token1", "token0_decimals", "token1_decimals"]


def keep_latest_swap_per_block(df, label=""):
    """Keep, per (pool, block), the swap with the highest ``evt_index``.

    The highest ``evt_index`` is the last swap in the block, so its
    ``mid_price`` is the price at the end of that block.
    """
    if df is None or df.empty:
        print(f"[INFO] {label}: empty dataframe")
        return pd.DataFrame()

    df = df.copy()
    df = df.loc[df.groupby(["pool", "evt_block_number"])["evt_index"].idxmax()]
    return df.reset_index(drop=True)


def clean_all(dfs):
    """Apply :func:`keep_latest_swap_per_block` to every DEX DataFrame."""
    filtered = {}
    for name, df in dfs.items():
        filtered[name] = keep_latest_swap_per_block(df, name)
        print(f"Processed {name}: {len(df)} rows -> {len(filtered[name])} rows")
    return filtered


def split_by_pool(filtered_dfs):
    """Split each DEX DataFrame into one DataFrame per pool address.

    Pools are numbered per DEX: ``uniswap_1``, ``uniswap_2``, ``pancake_1`` ...
    """
    result = {}

    for dex_key, df in filtered_dfs.items():
        if df is None or df.empty or "pool" not in df.columns:
            continue

        dex_name = dex_key.replace("df_", "")
        for idx, pool in enumerate(df["pool"].unique(), start=1):
            pool_name = f"{dex_name}_{idx}"
            result[pool_name] = df[df["pool"] == pool].copy()
            print(f"Created {pool_name}: {len(result[pool_name])} rows")

    print(f"\nTotal: {len(result)} dataframes\n{list(result.keys())}")
    return result


def _global_block_range(pool_dfs):
    """Return (min, max) block number across every non-empty pool."""
    all_blocks = [
        b
        for df in pool_dfs.values()
        if not df.empty
        for b in df["evt_block_number"].values
    ]
    return int(min(all_blocks)), int(max(all_blocks))


def filter_pools_by_swap_gap(pool_dfs, max_gap_blocks):
    """Drop pools whose largest gap between swaps exceeds ``max_gap_blocks``.

    A pool that goes too long without a swap can't be reliably forward-filled,
    so it is excluded. The "gap" is the largest of:
        * the biggest gap between two consecutive swaps, and
        * the tail gap (last swap -> global last block).

    Returns
    -------
    (kept, dropped) : (dict, list[(name, reason)])
    """
    _, global_max_block = _global_block_range(pool_dfs)

    kept = {}
    dropped = []

    for pool_name, df in pool_dfs.items():
        if df.empty:
            dropped.append((pool_name, "empty"))
            continue

        blocks = df["evt_block_number"].sort_values().values
        if len(blocks) < 2:
            kept[pool_name] = df
            continue

        max_gap = int(np.diff(blocks).max())
        tail_gap = int(global_max_block - blocks[-1])

        if max(max_gap, tail_gap) <= max_gap_blocks:
            kept[pool_name] = df
        else:
            dropped.append(
                (pool_name, f"max gap = {max_gap} blocks, tail gap = {tail_gap} blocks")
            )

    print(f"k = {max_gap_blocks} blocks")
    print(f"Kept {len(kept)} pools\n")
    print("Kept pools:")
    for pool_name, df in kept.items():
        blocks = df["evt_block_number"].sort_values().values
        max_gap = int(np.diff(blocks).max()) if len(blocks) >= 2 else 0
        print(f"  {pool_name}: {len(df)} swaps, max consecutive gap = {max_gap} blocks")

    if dropped:
        print(f"\nDropped {len(dropped)} pools:")
        for pool_name, reason in dropped:
            print(f"  {pool_name}: {reason}")

    return kept, dropped


def reconstruct_pool_timeseries(pool_dfs):
    """Build a dense, block-by-block, forward-filled price series per pool.

    Every pool is reindexed onto the *same* global block grid (global min ->
    global max block) using a shared, canonical block -> time mapping, so the
    resulting series are directly comparable. Missing prices are forward-filled
    from the last known swap.

    Returns
    -------
    (reconstructed, global_min_block, global_max_block, block_time_map)
    """
    global_min_block, global_max_block = _global_block_range(pool_dfs)
    print(f"Global block range: {global_min_block} to {global_max_block}")
    print(f"Total blocks: {global_max_block - global_min_block + 1}\n")

    # Canonical block -> time mapping shared by all pools.
    block_time_map = {}
    for df in pool_dfs.values():
        if df.empty:
            continue
        for block_num, block_time in zip(df["evt_block_number"], df["evt_block_time"]):
            block_time_map.setdefault(int(block_num), block_time)

    # Complete block grid with canonical times.
    grid = pd.DataFrame(
        {"evt_block_number": range(global_min_block, global_max_block + 1)}
    )
    grid["evt_block_time"] = grid["evt_block_number"].map(block_time_map)

    reconstructed = {}
    for pool_name, df in pool_dfs.items():
        if df.empty:
            print(f"[SKIP] {pool_name}: empty")
            continue

        # One row per block (keep the earliest), then align to the grid.
        one_per_block = (
            df.sort_values("evt_block_number")
            .drop_duplicates("evt_block_number", keep="first")
            .drop(columns="evt_block_time")
        )

        result = grid.merge(one_per_block, on="evt_block_number", how="left")

        # Forward-fill the price and the constant metadata (but never the time).
        result["mid_price"] = result["mid_price"].ffill()
        for col in _METADATA_COLS:
            if col in result.columns:
                result[col] = result[col].ffill()

        reconstructed[pool_name] = result.reset_index(drop=True)

        filled = result["mid_price"].notna().sum()
        print(
            f"{pool_name}: {len(df)} trades -> {filled} blocks "
            f"({100 * filled / len(result):.1f}%)"
        )

    print(f"\nCreated {len(reconstructed)} reconstructed time series")
    return reconstructed, global_min_block, global_max_block, block_time_map


def filter_by_start_time(reconstructed_pools, study_start_time):
    """Trim each reconstructed series to ``evt_block_time >= study_start_time``.

    This drops the warm-up margin between the collection start and the start of
    the actual study window.
    """
    filtered = {}
    for pool_name, ts_df in reconstructed_pools.items():
        trimmed = ts_df[ts_df["evt_block_time"] >= study_start_time].reset_index(drop=True)
        filtered[pool_name] = trimmed
        print(f"{pool_name}: {len(ts_df)} -> {len(trimmed)} rows")

    print(f"\nFiltered all pools by time >= {study_start_time}")
    return filtered
