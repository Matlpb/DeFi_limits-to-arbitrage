"""
Plotting helpers for the reconstructed price series.

Two views:
    * plot_dex_pools          - every pool of one DEX (per-pool + combined)
    * plot_pairwise_prices    - both price series for each cross-DEX pair
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _style_time_axis(ax, hour_interval=2, fmt="%H:%M:%S"):
    """Apply the shared time-axis formatting used by every plot."""
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def plot_dex_pools(filtered_pools, dex="uniswap"):
    """Plot every pool of ``dex``: one subplot per pool plus a combined view."""
    pools = {k: v for k, v in filtered_pools.items() if k.startswith(dex)}
    if not pools:
        print(f"No pools found for DEX '{dex}'")
        return

    n = len(pools)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 5 * (n + 1)))
    axes = [axes] if n + 1 == 1 else axes

    for ax, (pool_name, df) in zip(axes, sorted(pools.items())):
        times = pd.to_datetime(df["evt_block_time"])
        ax.plot(times, df["mid_price"], linewidth=1.5, label=pool_name, alpha=0.8)
        ax.set_title(f"{pool_name.upper()} - Price Over Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Block Time", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)
        _style_time_axis(ax)

    ax_combined = axes[-1]
    for pool_name, df in sorted(pools.items()):
        times = pd.to_datetime(df["evt_block_time"])
        ax_combined.plot(times, df["mid_price"], linewidth=2, label=pool_name, alpha=0.7)
    ax_combined.set_title(f"{dex.upper()} - All Pools Combined", fontsize=14, fontweight="bold")
    ax_combined.set_xlabel("Block Time", fontsize=10)
    ax_combined.set_ylabel("Price", fontsize=10)
    ax_combined.legend(loc="best", fontsize=10)
    _style_time_axis(ax_combined)

    plt.tight_layout()
    plt.show()

    print(f"\nPlotted {n} {dex} pools + 1 combined view")
    for pool_name in sorted(pools):
        print(f"  {pool_name}: {len(pools[pool_name])} blocks")


def plot_pairwise_prices(price_differences, ncols=2):
    """Plot both pool price series for every cross-DEX pair."""
    n_pairs = len(price_differences)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    for ax, (pair_name, df) in zip(axes, sorted(price_differences.items())):
        pool1, pool2 = pair_name.split("_vs_")
        times = pd.to_datetime(df["evt_block_time"])

        ax.plot(times, df[f"{pool1}_price"], linewidth=1.2, label=pool1, alpha=0.85)
        ax.plot(times, df[f"{pool2}_price"], linewidth=1.2, label=pool2, alpha=0.85, linestyle="--")
        ax.set_title(pair_name.replace("_vs_", "  vs  "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Block Time", fontsize=9)
        ax.set_ylabel("Price (USDC/ETH)", fontsize=9)
        ax.legend(fontsize=9)
        _style_time_axis(ax, fmt="%H:%M")

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Price Time Series - All Pairwise DEX Comparisons",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()
