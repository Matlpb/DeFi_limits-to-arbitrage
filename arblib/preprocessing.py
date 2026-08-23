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

Two time anchors matter: the collection window (start_ts / end_ts) defines what was
pulled from Dune; the study start is later, leaving a warm-up margin so every pool
already has a known price to forward-fill from before the study begins (step 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Token

_METADATA_COLS = ["pool", "dex", "token0", "token1", "token0_decimals", "token1_decimals"]


def verify_token_decimals(swaps: pd.DataFrame, token0: Token, token1: Token) -> None:
    """Assert token identity <-> decimals consistency across all pooled swaps.

    Every token address must map to exactly one ``decimals`` value (a conflict would
    be a units bug), and the ``token0`` / ``token1`` columns must be the two expected
    tokens in every pool - Uniswap / Pancake order tokens by address, so consistent
    ordering is not guaranteed a priori. Prints a summary; raises on any mismatch.
    """
    expected = {token0.address: token0.decimals, token1.address: token1.decimals}

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

    assert set(swaps["token0"].unique()) == {token0.address}, swaps["token0"].unique()
    assert set(swaps["token1"].unique()) == {token1.address}, swaps["token1"].unique()
    print(f"\nOK: token0 ({token0.decimals} dp), token1 ({token1.decimals} dp) consistent across all pools.")


def count_swaps(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add a ``nb_swaps`` column counting swaps per (pool, block), in place.

    Run this before :func:`clean_all`: the count is written on every raw swap row, so
    when ``clean_all`` collapses each block to its last swap the surviving row keeps
    the true per-(pool, block) swap count.
    """
    for name, df in dfs.items():
        if df is None or df.empty:
            print(f"[INFO] {name}: empty dataframe")
            continue
        df["nb_swaps"] = df.groupby(["pool", "evt_block_number"])["evt_index"].transform("size")
        print(f"Counted swaps {name}: {len(df)} rows")
    return dfs


def keep_latest_swap_per_block(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Keep, per (pool, block), the swap with the highest ``evt_index``.

    The highest ``evt_index`` is the last swap in the block, so its ``sqrtPriceX96`` is
    the end-of-block price. Before collapsing, ``gas_price`` is replaced by two
    aggregates over this pool's swaps in that block - ``gas_price_max`` (worst-case)
    and ``gas_price_med`` (median) - and ``gas_used`` by its ``gas_used_med``. Grouping
    by (pool, block) matches ``nb_swaps`` and the surviving row, so a single-swap
    partition has ``gas_price_med == gas_price_max`` and ``gas_used_med`` equal to that
    swap's ``gas_used``. Any existing ``nb_swaps`` column is carried through.
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
        df["gas_used_med"] = df.groupby(["pool", "evt_block_number"])["gas_used"].transform("median")
        df = df.drop(columns="gas_used")

    df = df.loc[df.groupby(["pool", "evt_block_number"])["evt_index"].idxmax()]
    return df.reset_index(drop=True)


def clean_all(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply :func:`keep_latest_swap_per_block` to every DEX DataFrame."""
    filtered = {}
    for name, df in dfs.items():
        filtered[name] = keep_latest_swap_per_block(df, name)
        print(f"Processed {name}: {len(df)} rows -> {len(filtered[name])} rows")
    return filtered


def split_by_pool(filtered_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
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


def _global_block_range(pool_dfs: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """Return (min, max) block number across every non-empty pool."""
    all_blocks = [
        b
        for df in pool_dfs.values()
        if not df.empty
        for b in df["evt_block_number"].values
    ]
    return int(min(all_blocks)), int(max(all_blocks))


def filter_pools_by_swap_gap(pool_dfs: dict[str, pd.DataFrame],
                             max_gap_blocks: int) -> tuple[dict[str, pd.DataFrame], list]:
    """Drop pools whose largest gap between swaps exceeds ``max_gap_blocks``.

    A pool that goes too long without a swap can't be reliably forward-filled, so it is
    excluded. The gap is the largest of the biggest between-swaps gap and the tail gap
    (last swap -> global last block). Returns ``(kept, dropped)``.
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


def filter_pools_by_constant_fee(pool_dfs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list]:
    """Drop pools whose ``fee`` is not constant over the window (dynamic-fee pools).

    The execution-price / round-trip math assumes a single fixed fee per pool, so a pool
    whose ``fee`` field takes more than one value (e.g. a Uniswap v4 dynamic-fee hook) is
    excluded. Returns ``(kept, dropped)``.
    """
    kept = {}
    dropped = []
    for pool_name, df in pool_dfs.items():
        fees = sorted(df["fee"].dropna().unique().tolist())
        if len(fees) == 1:
            kept[pool_name] = df
        else:
            dropped.append((pool_name, f"fees = {fees}"))

    print(f"Kept {len(kept)} constant-fee pools")
    if dropped:
        print(f"Dropped {len(dropped)} dynamic-fee pool(s):")
        for pool_name, reason in dropped:
            print(f"  {pool_name}: {reason}")
    return kept, dropped


def filter_pools_by_liquidity(pool_dfs: dict[str, pd.DataFrame],
                              max_invalid_frac: float = 0.10) -> tuple[dict[str, pd.DataFrame], list]:
    """Drop pools whose reconstructed active liquidity is invalid on too many blocks.

    A block with no in-range liquidity (``L == 0``) or undefined ``L`` (null, before the pool's first
    swap) has no executable price: the round-trip math divides by ``L``, and ``log(L)`` is ``-inf`` at
    ``L == 0``, so a thin/dead pool poisons the liquidity-growth covariate (a single ``-inf`` makes a
    whole column's mean non-finite downstream). Pools with more than ``max_invalid_frac`` of blocks
    null or ``<= 0`` are dropped. Run this AFTER :func:`reconstruct_liquidity_states` (it reads the
    rebuilt ``liquidity`` column). Returns ``(kept, dropped)``.
    """
    kept = {}
    dropped = []
    for pool_name, df in pool_dfs.items():
        L = pd.to_numeric(df["liquidity"], errors="coerce")
        invalid = float((L.isna() | (L <= 0)).mean())
        if invalid <= max_invalid_frac:
            kept[pool_name] = df
        else:
            dropped.append((pool_name, f"{invalid:.1%} of blocks null or <= 0 liquidity"))

    print(f"Kept {len(kept)} pools with <= {max_invalid_frac:.0%} invalid (null / zero) liquidity")
    if dropped:
        print(f"Dropped {len(dropped)} pool(s):")
        for pool_name, reason in dropped:
            print(f"  {pool_name}: {reason}")
    return kept, dropped


def _complete_block_times(grid: pd.DataFrame) -> pd.DataFrame:
    """Fill ``evt_block_time`` for blocks no pool traded in.

    The grid is a complete, consecutive block range, so a missing block's time is
    linearly interpolated from its neighbours (block times are monotonic in block
    number). The ``"YYYY-MM-DD HH:MM:SS.fff UTC"`` string format is preserved so
    downstream comparisons keep working.
    """
    naive = grid["evt_block_time"].str.replace(" UTC", "", regex=False)
    ts = pd.to_datetime(naive).interpolate(method="linear")
    grid["evt_block_time"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + " UTC"
    return grid


def reconstruct_pool_timeseries(pool_dfs: dict[str, pd.DataFrame]) -> tuple:
    """Build a dense, block-by-block, forward-filled price series per pool.

    Every pool is reindexed onto the same global block grid (global min -> max block)
    using a shared canonical block -> time mapping, so the series are directly
    comparable, and missing prices are forward-filled from the last known swap. Blocks
    no pool traded in get an interpolated time so none are dropped downstream.
    Reconstructed (filled) blocks keep ``nb_swaps == 0`` rather than forward-filling it.

    Returns ``(reconstructed, global_min_block, global_max_block, block_time_map)``.
    """
    global_min_block, global_max_block = _global_block_range(pool_dfs)
    print(f"Global block range: {global_min_block} to {global_max_block}")
    print(f"Total blocks: {global_max_block - global_min_block + 1}\n")

    block_time_map = {}
    for df in pool_dfs.values():
        if df.empty:
            continue
        for block_num, block_time in zip(df["evt_block_number"], df["evt_block_time"]):
            block_time_map.setdefault(int(block_num), block_time)

    grid = pd.DataFrame({"evt_block_number": range(global_min_block, global_max_block + 1)})
    grid["evt_block_time"] = grid["evt_block_number"].map(block_time_map)
    grid = _complete_block_times(grid)

    reconstructed = {}
    for pool_name, df in pool_dfs.items():
        if df.empty:
            print(f"[SKIP] {pool_name}: empty")
            continue

        one_per_block = (
            df.sort_values("evt_block_number")
            .drop_duplicates("evt_block_number", keep="first")
            .drop(columns="evt_block_time")
        )

        result = grid.merge(one_per_block, on="evt_block_number", how="left")

        result["sqrtPriceX96"] = result["sqrtPriceX96"].ffill()
        for col in _METADATA_COLS:
            if col in result.columns:
                result[col] = result[col].ffill()

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


def _liquidity_events_by_pool(liq_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Combine the per-DEX liquidity extracts into ``{pool_address -> events}``.

    ``liquidity_delta`` is parsed as a Python ``int`` because the raw values overflow
    int64 (pandas loads them as strings). Events are sorted by
    ``(evt_block_number, evt_index)`` so they can be applied in execution order.
    """
    frames = [df for df in liq_dfs.values() if df is not None and not df.empty]
    if not frames:
        return {}

    events = pd.concat(frames, ignore_index=True)
    events["liquidity_delta"] = events["liquidity_delta"].apply(int)

    by_pool = {}
    for addr, group in events.groupby("pool"):
        by_pool[addr] = group.sort_values(["evt_block_number", "evt_index"]).reset_index(drop=True)
    return by_pool


def _apply_liquidity_events(grid: pd.DataFrame, events: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Rebuild a pool grid's ``liquidity`` column into the running active-liquidity ``L``.

    ``L`` is anchored to each swap's authoritative ``liquidity`` field, updated by
    in-range mint/burn deltas (``tick_lower <= current_tick < tick_upper``) between
    swaps, and carried forward across quiet blocks. The column is kept as exact-int
    ``object`` dtype - the raw values overflow int64/float64, and ``None`` remains only
    for the leading warm-up blocks before the first swap. Returns ``(grid, n_applied)``.
    """
    grid = grid.sort_values("evt_block_number").reset_index(drop=True).copy()

    has_events = events is not None and not events.empty
    events_by_block = {b: g for b, g in events.groupby("evt_block_number")} if has_events else {}

    tick_ffill = grid["tick"].ffill()

    blocks = grid["evt_block_number"].to_numpy()
    anchors = grid["liquidity"].to_numpy(dtype=object)

    L = None
    n_applied = 0
    out = []

    for i in range(len(grid)):
        anchor = anchors[i]
        if not pd.isna(anchor):
            L = int(anchor)

        block = int(blocks[i])
        if has_events and L is not None and block in events_by_block:
            ic = tick_ffill.iloc[i]
            if not pd.isna(ic):
                ic = int(ic)
                for e in events_by_block[block].itertuples(index=False):
                    if e.tick_lower <= ic < e.tick_upper:
                        L += int(e.liquidity_delta)
                        n_applied += 1

        out.append(L)

    grid["liquidity"] = pd.Series(out, index=grid.index, dtype=object)
    return grid, n_applied


def reconstruct_liquidity_states(reconstructed_pools: dict[str, pd.DataFrame],
                                 liq_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reconstruct active liquidity ``L`` per block for every reconstructed pool.

    For each pool in ``reconstructed_pools`` the ``liquidity`` column is rebuilt via
    :func:`_apply_liquidity_events` using the mint/burn deltas in ``liq_dfs``. Pools
    that appear only in the liquidity extracts are ignored. Same keys as the input.
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


def filter_by_start_time(reconstructed_pools: dict[str, pd.DataFrame],
                         study_start_time: str) -> dict[str, pd.DataFrame]:
    """Trim each reconstructed series to ``evt_block_time >= study_start_time``.

    Drops the warm-up margin between the collection start and the study window start.
    """
    filtered = {}
    for pool_name, ts_df in reconstructed_pools.items():
        trimmed = ts_df[ts_df["evt_block_time"] >= study_start_time].reset_index(drop=True)
        filtered[pool_name] = trimmed
        print(f"{pool_name}: {len(ts_df)} -> {len(trimmed)} rows")

    print(f"\nFiltered all pools by time >= {study_start_time}")
    return filtered