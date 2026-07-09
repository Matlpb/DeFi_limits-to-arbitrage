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


def verify_token_decimals(swaps, token0, token1, d0, d1):
    """Assert token identity <-> decimals consistency across all pooled swaps.

    Every token address must map to exactly one ``decimals`` value (a conflict
    would be a units bug), and ``token0`` / ``token1`` must be the same expected
    assets in every pool (Uniswap / Pancake order tokens by address, so this is
    not guaranteed a priori). ``token0`` / ``token1`` are the expected addresses
    with expected decimals ``d0`` / ``d1``. Prints a summary; raises on any mismatch.
    """
    expected = {token0: d0, token1: d1}

    pairs = pd.concat([
        swaps[["token0", "token0_decimals"]].rename(
            columns={"token0": "addr", "token0_decimals": "dec"}
        ),
        swaps[["token1", "token1_decimals"]].rename(
            columns={"token1": "addr", "token1_decimals": "dec"}
        ),
    ])
    for addr, decs in pairs.groupby("addr")["dec"].unique().items():
        assert len(decs) == 1, f"{addr} has conflicting decimals {decs}"
        print(f"{addr} -> {decs[0]} decimals")
        if addr in expected:
            assert decs[0] == expected[addr], (
                f"{addr}: expected {expected[addr]} decimals, got {decs[0]}"
            )

    assert set(swaps["token0"].unique()) == {token0}, swaps["token0"].unique()
    assert set(swaps["token1"].unique()) == {token1}, swaps["token1"].unique()
    print(f"\nOK: token0 ({d0} dp), token1 ({d1} dp) consistent across all pools.")


def count_swaps(dfs):
    """Add a ``nb_swaps`` column counting swaps per (pool, block).

    Run this *before* :func:`clean_all` / :func:`keep_latest_swap_per_block`::

        dfs = pp.count_swaps(dfs)      # tag each row with its block's swap count
        filtered_dfs = pp.clean_all(dfs)

    The number is written on every raw swap row, so when ``clean_all`` collapses
    each block down to its last swap, the surviving row keeps the true count of
    swaps that occurred in that block. Each DataFrame is modified in place and
    returned.
    """
    for name, df in dfs.items():
        if df is None or df.empty:
            print(f"[INFO] {name}: empty dataframe")
            continue
        df["nb_swaps"] = df.groupby(["pool", "evt_block_number"])[
            "evt_index"
        ].transform("size")
        print(f"Counted swaps {name}: {len(df)} rows")
    return dfs


def keep_latest_swap_per_block(df, label=""):
    """Keep, per (pool, block), the swap with the highest ``evt_index``.

    The highest ``evt_index`` is the last swap in the block, so its
    ``sqrtPriceX96`` is the price at the end of that block. Before collapsing the
    partition, ``gas_price`` is replaced by two aggregates over *this pool's*
    swaps in that block: ``gas_price_max`` (worst-case) and ``gas_price_med``
    (median), so the kept row carries both rather than just the last swap's gas
    cost. Likewise ``gas_used`` is replaced by its per-(pool, block) median
    ``gas_used_med``. These match the ``(pool, block)`` grouping of ``nb_swaps``
    and the surviving row, so a single-swap partition has ``gas_price_med ==
    gas_price_max`` and ``gas_used_med`` equal to that swap's ``gas_used``.

    If a ``nb_swaps`` column is present (see :func:`count_swaps`), the surviving
    row simply carries it, so the per-block swap count is preserved through the
    collapse.
    """
    if df is None or df.empty:
        print(f"[INFO] {label}: empty dataframe")
        return pd.DataFrame()

    df = df.copy()

    if "gas_price" in df.columns:
        pool_block_gas = df.groupby(["pool", "evt_block_number"])["gas_price"]
        df["gas_price_max"] = pool_block_gas.transform("max")
        df["gas_price_med"] = pool_block_gas.transform("median")
        df = df.drop(columns="gas_price")

    if "gas_used" in df.columns:
        df["gas_used_med"] = df.groupby(["pool", "evt_block_number"])["gas_used"].transform(
            "median"
        )
        df = df.drop(columns="gas_used")

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


def _complete_block_times(grid):
    """Fill ``evt_block_time`` for blocks no pool traded in.

    The grid is a complete, consecutive block range, so a missing block's time
    is linearly interpolated from its neighbours (block times are monotonic in
    block number). The original ``"YYYY-MM-DD HH:MM:SS.fff UTC"`` string format
    is preserved so downstream comparisons keep working.
    """
    naive = grid["evt_block_time"].str.replace(" UTC", "", regex=False)
    ts = pd.to_datetime(naive).interpolate(method="linear")
    grid["evt_block_time"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + " UTC"
    return grid


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

    # Complete block grid with canonical times. Blocks that no pool traded in
    # have no recorded time; interpolate one so every consecutive block keeps a
    # timestamp and none get dropped downstream (see _complete_block_times).
    grid = pd.DataFrame(
        {"evt_block_number": range(global_min_block, global_max_block + 1)}
    )
    grid["evt_block_time"] = grid["evt_block_number"].map(block_time_map)
    grid = _complete_block_times(grid)

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
        result["sqrtPriceX96"] = result["sqrtPriceX96"].ffill()
        for col in _METADATA_COLS:
            if col in result.columns:
                result[col] = result[col].ffill()

        # Reconstructed (forward-filled) blocks had no swap of their own, so
        # their swap count is 0 - never forward-filled.
        if "nb_swaps" in result.columns:
            result["nb_swaps"] = result["nb_swaps"].fillna(0).astype(int)

        reconstructed[pool_name] = result.reset_index(drop=True)

        filled = result["sqrtPriceX96"].notna().sum()
        print(
            f"{pool_name}: {len(df)} trades -> {filled} blocks "
            f"({100 * filled / len(result):.1f}%)"
        )

    print(f"\nCreated {len(reconstructed)} reconstructed time series")
    return reconstructed, global_min_block, global_max_block, block_time_map


def _liquidity_events_by_pool(liq_dfs):
    """Combine the per-DEX liquidity extracts into ``{pool_address -> events}``.

    ``liquidity_delta`` is parsed as a Python ``int`` (the raw values overflow
    int64, so pandas loads them as strings). Events are sorted by
    ``(evt_block_number, evt_index)`` so they can be applied in execution order.
    """
    frames = [df for df in liq_dfs.values() if df is not None and not df.empty]
    if not frames:
        return {}

    events = pd.concat(frames, ignore_index=True)
    events["liquidity_delta"] = events["liquidity_delta"].apply(int)

    by_pool = {}
    for addr, group in events.groupby("pool"):
        by_pool[addr] = group.sort_values(
            ["evt_block_number", "evt_index"]
        ).reset_index(drop=True)
    return by_pool


def _apply_liquidity_events(grid, events):
    """Turn a reconstructed pool grid's ``liquidity`` column into the running
    active-liquidity state ``L`` per block.

    ``L`` is anchored to each swap's authoritative ``liquidity`` field, then
    updated by in-range mint/burn deltas (``tick_lower <= current_tick <
    tick_upper``) between swaps, and carried forward across quiet blocks.

    Returns ``(grid, n_applied)`` where ``n_applied`` is how many deltas landed
    in range.
    """
    grid = grid.sort_values("evt_block_number").reset_index(drop=True).copy()

    has_events = events is not None and not events.empty
    events_by_block = (
        {b: g for b, g in events.groupby("evt_block_number")} if has_events else {}
    )

    # Current price tick, known between swaps via forward-fill, drives the
    # in-range indicator.
    tick_ffill = grid["tick"].ffill()

    blocks = grid["evt_block_number"].to_numpy()
    anchors = grid["liquidity"].to_numpy(dtype=object)

    L = None  # running state (exact int); None until the first swap anchor
    n_applied = 0
    out = []

    for i in range(len(grid)):
        anchor = anchors[i]
        has_swap = not pd.isna(anchor)
        if has_swap:
            L = int(anchor)  # swap liquidity is the per-block ground truth

        block = int(blocks[i])
        if has_events and L is not None and block in events_by_block:
            ic = tick_ffill.iloc[i]
            if not pd.isna(ic):
                ic = int(ic)
                for e in events_by_block[block].itertuples(index=False):
                    # Add every in-range mint/burn delta on top of the swap
                    # liquidity (or the carried value), whenever the current tick
                    # is inside the event's [tick_lower, tick_upper) range.
                    if e.tick_lower <= ic < e.tick_upper:
                        L += int(e.liquidity_delta)
                        n_applied += 1

        # Unchanged L => liquidity is held constant across the hole.
        out.append(L)

    # Keep exact integers: the raw values overflow int64/float64, and assigning a
    # list mixing ints with None would coerce the column to lossy float64. None
    # remains only for the leading warm-up blocks before the first swap.
    grid["liquidity"] = pd.Series(out, index=grid.index, dtype=object)
    return grid, n_applied


def reconstruct_liquidity_states(reconstructed_pools, liq_dfs):
    """Reconstruct active liquidity ``L`` per block for every reconstructed pool.

    For each pool already in ``reconstructed_pools``, the ``liquidity`` column is
    rebuilt as the running liquidity state: swaps act as authoritative anchors,
    in-range mint/burn deltas (from ``liq_dfs``) are applied between swaps using
    the current price tick, and the state is held constant across blocks with no
    update. Pools that appear only in the liquidity extracts are ignored.

    Parameters
    ----------
    reconstructed_pools : dict[str, pd.DataFrame]
        Output of :func:`reconstruct_pool_timeseries` (one dense grid per pool).
    liq_dfs : dict[str, pd.DataFrame]
        Per-DEX mint/burn extracts, e.g. loaded from ``<chain>/liquidity/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same keys as ``reconstructed_pools`` with the ``liquidity`` column
        replaced by the reconstructed state.
    """
    events_by_pool = _liquidity_events_by_pool(liq_dfs)

    result = {}
    n_with_events = 0
    for pool_name, grid in reconstructed_pools.items():
        addr_series = grid["pool"].dropna()
        addr = addr_series.iloc[0] if not addr_series.empty else None
        events = events_by_pool.get(addr)

        result[pool_name], n_applied = _apply_liquidity_events(grid, events)

        if events is not None and not events.empty:
            n_with_events += 1
            print(
                f"{pool_name} ({addr[:10]}...): {len(events)} mint/burn events, "
                f"{n_applied} applied in range"
            )
        else:
            print(f"{pool_name}: no mint/burn events, liquidity forward-filled")

    print(
        f"\nReconstructed liquidity for {len(result)} pools "
        f"({n_with_events} had mint/burn events)"
    )
    return result


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
