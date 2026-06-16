"""
Disk I/O for the raw DEX extracts.

Each chain gets its own sub-folder (e.g. ``base/``, ``ethereum/``) holding one
CSV per DEX. These helpers keep the save / load logic out of the notebooks.
"""

import os

import pandas as pd

from .config import SWAP_FILES


def save_dataframes(dfs, save_dir):
    """Write each non-empty DataFrame to ``save_dir`` as CSV.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping of file name (e.g. ``"df_uniswap.csv"``) to DataFrame.
    save_dir : str
        Destination directory (created if missing).
    """
    os.makedirs(save_dir, exist_ok=True)

    for filename, df in dfs.items():
        if df is not None and not df.empty:
            path = os.path.join(save_dir, filename)
            df.to_csv(path, index=False)
            print(f"Saved: {path}")
        else:
            print(f"Skipped empty or missing dataframe: {filename}")

    print("Done.")


def save_processed_pools(pool_dfs, save_dir):
    """Write each processed per-pool series to ``save_dir`` as CSV.

    One file per pool, named after its key (e.g. ``"uniswap_1.csv"``,
    ``"pancake_2.csv"``).

    Parameters
    ----------
    pool_dfs : dict[str, pd.DataFrame]
        Mapping of pool key (e.g. ``"uniswap_1"``) to its reconstructed,
        study-window DataFrame.
    save_dir : str
        Destination directory (created if missing), e.g. ``".../ethereum/processed"``.
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, df in pool_dfs.items():
        if df is not None and not df.empty:
            path = os.path.join(save_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved: {path}")
        else:
            print(f"Skipped empty or missing pool: {name}")

    print("Done.")


def load_pool_csvs(data_dir, files=None):
    """Load the per-DEX CSVs found in ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSVs (e.g. ``".../base"``).
    files : dict[str, str], optional
        Mapping of DataFrame key -> file name. Defaults to
        :data:`arblib.config.SWAP_FILES`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Only the files that actually exist on disk.
    """
    files = files or SWAP_FILES
    dfs = {}

    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            # Be robust to CSVs exported pretty-printed (spaces after commas and
            # values padded to fixed width). ``skipinitialspace`` drops leading
            # spaces; stripping headers and string cells drops the trailing
            # padding that would otherwise break e.g. pool-address matching.
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
