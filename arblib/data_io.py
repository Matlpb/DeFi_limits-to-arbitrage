"""
Disk I/O for the raw DEX extracts.

Each chain gets its own sub-folder (e.g. ``base/``, ``ethereum/``) holding one
CSV per DEX. These helpers keep the save / load logic out of the notebooks.
"""

import os

import pandas as pd

from .config import DEX_FILES


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


def load_pool_csvs(data_dir, files=None):
    """Load the per-DEX CSVs found in ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSVs (e.g. ``".../base"``).
    files : dict[str, str], optional
        Mapping of DataFrame key -> file name. Defaults to
        :data:`arblib.config.DEX_FILES`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Only the files that actually exist on disk.
    """
    files = files or DEX_FILES
    dfs = {}

    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"Loaded: {filename}")
            print(dfs[name].head())
        else:
            print(f"File not found: {filename}")

    return dfs
