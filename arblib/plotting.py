"""
Plotting helpers for the study.

Views:
    * plot_dex_pools                - every pool of one DEX (per-pool + combined)
    * plot_pairwise_prices          - both price series for each cross-DEX pair
    * plot_trade_size_distributions - per-leg USD trade-size histograms
    * plot_global_trade_size        - pooled trade-size distribution + quantiles
    * plot_directional_spreads      - per-block cross-pool round-trip spreads
    * plot_custom_size              - the same, at a user-chosen trade size

The round-trip plotters take an ``arblib.arbitrage.PoolPair`` and delegate the
P&L to ``arblib.arbitrage``.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import arbitrage


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


def plot_trade_size_distributions(x_in, y_in):
    """Per-leg USD trade-size histograms (USDC-in and WETH-in), log-scaled x-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, color in [
        (axes[0], x_in["amount0_usd"], "amount0 - token0 (USDC) in", "steelblue"),
        (axes[1], y_in["amount1_usd"], "amount1 - token1 (WETH) in", "indianred"),
    ]:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 40)
        ax.hist(data, bins=bins, color=color, edgecolor="white")
        ax.set_xscale("log")
        ax.set_title(f"{title}  ({len(data)} swaps)")
        ax.set_xlabel("trade size [USD, log scale]")
        ax.set_ylabel("number of swaps")

    fig.suptitle("Trade-size distributions in USD - all pools & DEXes", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_global_trade_size(trade_sizes, quantiles):
    """Pooled (both-legs) USD trade-size distribution with the quantile refs marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(trade_sizes.min()), np.log10(trade_sizes.max()), 50)
    ax.hist(trade_sizes, bins=bins, color="seagreen", edgecolor="white")
    ax.set_xscale("log")
    for q, v in quantiles.items():
        ax.axvline(v, color="black", linestyle="--", linewidth=1)
        ax.text(v, ax.get_ylim()[1] * 0.95, f"{int(q * 100)}%\n${v:,.0f}",
                ha="center", va="top", fontsize=9)
    ax.set_title(f"Global trade-size distribution - both tokens, all pools ({len(trade_sizes)} swaps)")
    ax.set_xlabel("trade size [USD, log scale]")
    ax.set_ylabel("number of swaps")
    fig.tight_layout()
    plt.show()


def plot_directional_spreads(pair, Q, title_prefix):
    """Two-panel per-block cross-pool round-trip spread (bps) for USD notional ``Q``.

    Left: direction 1->2 (start ``pair.name1``, close ``pair.name2``); right: 2->1.
    Each panel overlays the two anchors (Y / X). Break-even at 0; a series **below**
    it is a profitable round trip (sign convention of
    :func:`arblib.arbitrage.directional_spreads`, which produces the P&L).
    """
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = arbitrage.directional_spreads(pair, Q)

    fig, (ax12, ax21) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"{title_prefix} - ref ${Q:,.0f}  "
        f"(x_in={x_in:,.2f} USDC, y_in={y_in:.4f} WETH) - chained real amounts",
        fontsize=12,
    )

    ax12.plot(pair.block, S_12_Y * 1e4, lw=1.0, label="S_12_Y (Y-anchor)")
    ax12.plot(pair.block, S_21_X * 1e4, lw=1.0, label="S_21_X (X-anchor)")
    ax12.axhline(0, color="grey", lw=0.8)
    ax12.set_title(f"Direction 1->2 (start {pair.name1}, close {pair.name2})")
    ax12.set_xlabel("block number")
    ax12.set_ylabel("log spread [bps]  (<0 = profitable)")
    ax12.legend(fontsize=8)

    ax21.plot(pair.block, S_21_Y * 1e4, lw=1.0, label="S_21_Y (Y-anchor)")
    ax21.plot(pair.block, S_12_X * 1e4, lw=1.0, label="S_12_X (X-anchor)")
    ax21.axhline(0, color="grey", lw=0.8)
    ax21.set_title(f"Direction 2->1 (start {pair.name2}, close {pair.name1})")
    ax21.set_xlabel("block number")
    ax21.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_custom_size(pair, usd_amount=None, x_amount=None, y_amount=None,
                     title_prefix="Custom trade size"):
    """Directional-spread plot at a user-specified reference size (not a quantile).

    Provide exactly one of ``usd_amount`` (USD notional), ``x_amount`` (USDC, human),
    or ``y_amount`` (WETH, human). ``x_amount`` / ``y_amount`` are converted to an
    equivalent USD ``Q`` via the pair's fixed USD rates.
    """
    n_given = sum(v is not None for v in (usd_amount, x_amount, y_amount))
    if n_given != 1:
        raise ValueError("Provide exactly one of usd_amount, x_amount, y_amount")

    if usd_amount is not None:
        Q = usd_amount
    elif x_amount is not None:
        Q = x_amount * pair.usdc_usd
    else:
        Q = y_amount * pair.weth_usd

    plot_directional_spreads(pair, Q, title_prefix)
