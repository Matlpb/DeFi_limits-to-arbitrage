"""
Plotting helpers for the study.

Views:
    * plot_dex_pools                - every pool of one DEX (per-pool + combined)
    * plot_pairwise_prices          - both price series for each cross-DEX pair
    * plot_trade_size_distributions - per-leg USD trade-size histograms
    * plot_global_trade_size        - pooled trade-size distribution + quantiles
    * plot_directional_spreads      - per-block cross-pool round-trip spreads
    * plot_custom_size              - the same, at a user-chosen trade size
    * plot_arb_index                - heatmap of the round-trip gap over all pairs
    * plot_arb_durations            - arbitrage-hold durations per pool pair

The round-trip plotters take an ``arblib.arbitrage.PoolPair`` and delegate the P&L
to ``arblib.arbitrage``.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from . import arbitrage
from .arbitrage import PoolPair


def _style_time_axis(ax: Axes, hour_interval: int = 2, fmt: str = "%H:%M:%S") -> None:
    """Apply the shared time-axis formatting used by every price plot."""
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def plot_dex_pools(filtered_pools: dict[str, pd.DataFrame], dex: str = "uniswap") -> None:
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


def plot_pairwise_prices(price_differences: dict[str, pd.DataFrame], ncols: int = 2) -> None:
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
        ax.set_ylabel("Price", fontsize=9)
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


def plot_trade_size_distributions(x_in: pd.DataFrame, y_in: pd.DataFrame,
                                  sym0: str = "token0", sym1: str = "token1") -> None:
    """Per-leg USD trade-size histograms (token0-in and token1-in), log-scaled x-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        (axes[0], x_in["amount0_usd"], f"amount0 - token0 ({sym0}) in", "steelblue"),
        (axes[1], y_in["amount1_usd"], f"amount1 - token1 ({sym1}) in", "indianred"),
    ]
    for ax, data, title, color in panels:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 40)
        ax.hist(data, bins=bins, color=color, edgecolor="white")
        ax.set_xscale("log")
        ax.set_title(f"{title}  ({len(data)} swaps)")
        ax.set_xlabel("trade size [USD, log scale]")
        ax.set_ylabel("number of swaps")

    fig.suptitle("Trade-size distributions in USD - all pools & DEXes", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_global_trade_size(trade_sizes: pd.Series, quantiles: pd.Series) -> None:
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


def plot_directional_spreads(pair: PoolPair, Q: float, title_prefix: str) -> None:
    """Two-panel per-block cross-pool round-trip spread (bps) for USD notional ``Q``.

    Left: direction 1->2 (start ``pair.name1``, close ``pair.name2``); right: 2->1.
    Each panel overlays the two anchors. Break-even at 0; a series below it is a
    profitable round trip (sign convention of :func:`arblib.arbitrage.directional_spreads`).
    """
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = arbitrage.directional_spreads(pair, Q)

    fig, (ax12, ax21) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"{title_prefix} - ref ${Q:,.0f}  "
        f"(x_in={x_in:,.2f} {pair.sym0}, y_in={y_in:.4f} {pair.sym1}) - chained real amounts",
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


def plot_custom_size(pair: PoolPair, usd_amount: float | None = None,
                     x_amount: float | None = None, y_amount: float | None = None,
                     title_prefix: str = "Custom trade size") -> None:
    """Directional-spread plot at a user-specified reference size (not a quantile).

    Provide exactly one of ``usd_amount`` (USD notional), ``x_amount`` (token0, human),
    or ``y_amount`` (token1, human). ``x_amount`` / ``y_amount`` are converted to an
    equivalent USD ``Q`` via the pair's fixed USD rates.
    """
    n_given = sum(v is not None for v in (usd_amount, x_amount, y_amount))
    if n_given != 1:
        raise ValueError("Provide exactly one of usd_amount, x_amount, y_amount")

    if usd_amount is not None:
        Q = usd_amount
    elif x_amount is not None:
        Q = x_amount * pair.price0
    else:
        Q = y_amount * pair.price1

    plot_directional_spreads(pair, Q, title_prefix)


def plot_arb_index(arb_index: dict[str, pd.DataFrame], q: float) -> None:
    """Heatmap of the round-trip gap ``Gap_t`` (bps) across pairs and blocks at size ``q``.

    ``arb_index`` is the ``{pair_name: frame}`` mapping from
    :func:`arblib.arbitrage.all_pairs_gap_series`; rows are pool pairs, columns are
    blocks, and ``q`` selects the reference-size column. A coloured cell is a profitable
    round trip after both fees + slippage, so a near-blank map means little arbitrage in
    the sample. The count of live (pair, block) cells is printed alongside.
    """
    pairs = list(arb_index)
    mat = np.vstack([arb_index[p][q].to_numpy() for p in pairs])
    blocks = next(iter(arb_index.values())).index.to_numpy()

    n_live = int((mat > 0).sum())
    print(f"q{int(q * 100)}: {n_live} live (pair, block) cells of {mat.size} "
          f"| max gap = {mat.max():.2f} bps")

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(pairs) + 2))
    im = ax.imshow(
        mat, aspect="auto", cmap="Reds", vmin=0,
        extent=[blocks[0], blocks[-1], len(pairs) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    ax.set_xlabel("block number")
    ax.set_title(f"Round-trip gap Gap_t [bps] by pool pair - reference size q{int(q * 100)}")
    fig.colorbar(im, ax=ax, label="Gap_t [bps]  (blank = no arb)")
    fig.tight_layout()
    plt.show()


def plot_arb_durations(durations: pd.DataFrame, q: float) -> None:
    """One graph for trade size ``q``: the arb-hold duration *sequence* of each pool pair.

    ``durations`` is the long frame from :func:`arblib.arbitrage.arb_hold_durations`. For
    each pool pair its spells (runs of consecutive blocks with ``Gap_t > 0``) are plotted in
    order of occurrence - x = spell index (1st, 2nd, ...), y = duration in blocks - as one
    line per pair. This is the per-pair duration sequence (e.g. 2, 6, 5, 4, ...); a line's
    length is that pair's number of spells. Only pairs with >= 1 spell are shown.
    """
    sub = durations[durations["quantile"] == q]
    seqs = {pair: g["duration"].to_numpy() for pair, g in sub.groupby("pair", sort=False)}
    if not seqs:
        print(f"q{int(q * 100)}: no live arbitrage spells at this trade size")
        return

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for i, (pair, seq) in enumerate(seqs.items()):
        ax.plot(range(1, len(seq) + 1), seq, marker="o", ms=4, lw=1.0, alpha=0.8,
                color=cmap(i % 20), label=pair)

    ax.set_xlabel("spell index (order of occurrence)")
    ax.set_ylabel("arb-hold duration [consecutive blocks with Gap_t > 0]")
    ax.set_title(f"Arbitrage-hold duration sequence per pool pair - trade size q{int(q * 100)}")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.show()
