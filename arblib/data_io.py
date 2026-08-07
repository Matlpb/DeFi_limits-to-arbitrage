"""
Disk I/O for the raw extracts, processed pools, and reference quantiles.

Paths come from ``arblib.config.Settings`` (e.g. ``S.swaps_dir``, ``S.processed_dir``),
so these helpers stay parameter-agnostic and keep the load / save logic out of the
notebooks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import SWAP_FILES


def save_dataframes(dfs: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    """Write each non-empty DataFrame in ``{filename: df}`` to ``save_dir`` as CSV."""
    os.makedirs(save_dir, exist_ok=True)

    for filename, df in dfs.items():
        if df is not None and not df.empty:
            path = os.path.join(save_dir, filename)
            df.to_csv(path, index=False)
            print(f"Saved: {path}")
        else:
            print(f"Skipped empty or missing dataframe: {filename}")

    print("Done.")


def save_processed_pools(pool_dfs: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    """Write each processed per-pool series to ``save_dir`` as ``<pool_name>.csv``."""
    os.makedirs(save_dir, exist_ok=True)

    for name, df in pool_dfs.items():
        if df is not None and not df.empty:
            path = os.path.join(save_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved: {path}")
        else:
            print(f"Skipped empty or missing pool: {name}")

    print("Done.")


def save_frames(frames: dict[str, pd.DataFrame], out_dir: str | Path,
                suffix: str = "parquet", label: str | None = None) -> None:
    """Write each frame in ``{name: df}`` to ``out_dir/<name>.<suffix>`` (parquet or csv).

    The one place the "one file per key" save loop lives, reused by the modeling feature build.
    ``label`` only tunes the summary message (e.g. ``"dependent-variable"``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        path = out_dir / f"{name}.{suffix}"
        if suffix == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    print(f"Saved {len(frames)} {label + ' ' if label else ''}{suffix} files to {out_dir}")


def load_pool_csvs(data_dir: str | Path, files: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the per-DEX CSVs found in ``data_dir`` into ``{key: df}``.

    ``files`` maps DataFrame key -> file name (defaults to :data:`arblib.config.SWAP_FILES`);
    only files present on disk are returned. Dune sometimes exports space-padded headers
    and string cells, so ``skipinitialspace`` and header/cell stripping keep merges on
    keys like ``pool`` / ``evt_tx_hash`` matching.
    """
    files = files or SWAP_FILES
    dfs = {}

    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            dfs[name] = df
            print(f"Loaded: {filename}")
            print(df.head())
        else:
            print(f"File not found: {filename}")

    return dfs


def load_processed_pools(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every processed per-pool CSV into ``{pool_name: df}``.

    Reads only ``uniswap_*`` / ``pancake_*`` files (uniswap first, so cross-DEX pairwise
    labels stay stable) from ``processed_dir`` - the output of :func:`save_processed_pools`.
    """
    processed_dir = Path(processed_dir)
    paths = sorted(p for p in processed_dir.glob("*.csv")
                   if p.stem.startswith(("uniswap", "pancake")))
    paths = [p for p in paths if p.stem.startswith("uniswap")] + \
            [p for p in paths if not p.stem.startswith("uniswap")]

    pools = {}
    for path in paths:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        pools[path.stem] = df

    print(f"Loaded {len(pools)} processed pools: {list(pools)}")
    return pools


def load_chain_gas(path: str | Path) -> pd.DataFrame:
    """Load the chain-wide per-block gas table (``block_number`` + ``base_fee_per_gas``)."""
    return pd.read_csv(path)


def load_price_series(path: str | Path) -> pd.DataFrame:
    """Load a 1-minute ``[time, price]`` USD price series (``time`` parsed to UTC datetime)."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def save_quantiles(quantiles: pd.Series, path: str | Path) -> None:
    """Save the trade-size reference quantiles (a Series) to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantiles.rename("usd").rename_axis("quantile").to_csv(path)
    print(f"Saved: {path}")


def load_quantiles(path: str | Path) -> pd.Series:
    """Load the trade-size reference quantiles saved by :func:`save_quantiles`."""
    return pd.read_csv(path).set_index("quantile")["usd"]
