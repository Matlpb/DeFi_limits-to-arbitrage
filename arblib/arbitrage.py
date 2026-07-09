"""
Cross-pool executable prices and the per-pool-pair arbitrage index.

A naive cross-pool *price spread* compares the same swap side across two pools at
two independent notionals - that is not an executable arbitrage. A real round trip
closes the position: the second leg is the OPPOSITE side (you sell back what you
bought) and is fed the REAL output of the first leg (via ``formulas.swap_out``), so
both fees and both legs' slippage are charged.

Everything is expressed for one ordered pool pair, bundled in :class:`PoolPair`:

    pair = build_pool_pair(p1_df, p2_df, usdc_usd, weth_usd, "pancake_1", "uniswap_1")
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = directional_spreads(pair, Q)
    gaps = gap_series(pair, Q)
    index = arbitrage_index(pair, quantiles)   # tidy per-trade-size summary

``arbitrage_index`` is the per-pair signal a downstream model consumes over every
pool pair.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import formulas

# Columns aligned per block when building a pair.
_KEEP1 = ["evt_block_number", "evt_block_time", "sqrtPriceX96", "liquidity"]
_KEEP2 = ["evt_block_number", "sqrtPriceX96", "liquidity"]


@dataclass
class PoolPair:
    """Per-block state of two aligned pools plus the shared market context.

    ``sqrtP*`` / ``L*`` are per-block arrays; ``fee*`` are the raw pool fee field
    (e.g. ``100`` = 0.01%); ``d0`` / ``d1`` are token0 / token1 decimals; the USD
    prices are held constant over the window. ``name1`` / ``name2`` label the pools.
    """

    block: np.ndarray
    sqrtP1: np.ndarray
    L1: np.ndarray
    fee1: float
    sqrtP2: np.ndarray
    L2: np.ndarray
    fee2: float
    d0: int
    d1: int
    usdc_usd: float
    weth_usd: float
    name1: str = "pool1"
    name2: str = "pool2"


def build_pool_pair(p1_df, p2_df, usdc_usd, weth_usd, name1="pool1", name2="pool2"):
    """Align two processed pool frames on the shared block grid into a ``PoolPair``.

    ``p1_df`` / ``p2_df`` are ``ethereum/processed/*.csv`` frames (per-block
    ``sqrtPriceX96`` / ``liquidity`` / ``fee`` / decimals). ``usdc_usd`` / ``weth_usd``
    are the constant USD prices per token (see ``sizing.constant_usd_prices``).
    """
    fee1 = float(p1_df["fee"].dropna().unique()[0])
    fee2 = float(p2_df["fee"].dropna().unique()[0])
    d0 = int(p1_df["token0_decimals"].dropna().iloc[0])   # USDC = 6
    d1 = int(p1_df["token1_decimals"].dropna().iloc[0])   # WETH = 18

    m = p1_df[_KEEP1].merge(p2_df[_KEEP2], on="evt_block_number", suffixes=("_1", "_2"))

    return PoolPair(
        block=m["evt_block_number"].to_numpy(),
        sqrtP1=m["sqrtPriceX96_1"].astype(float).to_numpy(),
        L1=m["liquidity_1"].astype(float).to_numpy(),
        fee1=fee1,
        sqrtP2=m["sqrtPriceX96_2"].astype(float).to_numpy(),
        L2=m["liquidity_2"].astype(float).to_numpy(),
        fee2=fee2,
        d0=d0,
        d1=d1,
        usdc_usd=usdc_usd,
        weth_usd=weth_usd,
        name1=name1,
        name2=name2,
    )


def directional_spreads(pair, Q):
    """Four chained round-trip spreads for USD notional ``Q``, per block.

    Sign convention: ``S = log(amount_sent_in) - log(amount_received_back)``.
    ``S < 0`` => profitable round trip (you end with MORE than you started);
    ``S > 0`` => loss.

    Direction "1->2": start on pool 1, close out on pool 2. Direction "2->1": the
    reverse. Each direction is estimated two ways (Y-anchored / X-anchored), which
    should agree in sign when a real, size-robust gap exists. Each second leg is fed
    the REAL output amount of the first leg.

    Returns ``(x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X)``.
    """
    sqrtP1, L1, fee1 = pair.sqrtP1, pair.L1, pair.fee1
    sqrtP2, L2, fee2 = pair.sqrtP2, pair.L2, pair.fee2
    d0, d1 = pair.d0, pair.d1

    swap_out = formulas.swap_out
    x_in = formulas.to_usd(Q, pair.usdc_usd, sense="to_token")   # USDC reference size
    y_in = formulas.to_usd(Q, pair.weth_usd, sense="to_token")   # WETH reference size

    # --- Direction 1->2 ---
    # Y-anchor: sell Y on 1, rebuy Y on 2
    x_1out = swap_out(sqrtP1, L1, y_in, fee1 / 100, d0, d1, side="buy_x")
    y_2out = swap_out(sqrtP2, L2, x_1out, fee2 / 100, d0, d1, side="buy_y")
    S_12_Y = np.log(y_in) - np.log(y_2out)

    # X-anchor: sell X on 1, rebuy X on 2
    y_1out = swap_out(sqrtP1, L1, x_in, fee1 / 100, d0, d1, side="buy_y")
    x_2out = swap_out(sqrtP2, L2, y_1out, fee2 / 100, d0, d1, side="buy_x")
    S_12_X = np.log(x_2out) - np.log(x_in)

    # --- Direction 2->1 (venues swapped) ---
    x_2out_first = swap_out(sqrtP2, L2, y_in, fee2 / 100, d0, d1, side="buy_x")
    y_1out_second = swap_out(sqrtP1, L1, x_2out_first, fee1 / 100, d0, d1, side="buy_y")
    S_21_Y = np.log(y_in) - np.log(y_1out_second)

    y_2out_first = swap_out(sqrtP2, L2, x_in, fee2 / 100, d0, d1, side="buy_y")
    x_1out_second = swap_out(sqrtP1, L1, y_2out_first, fee1 / 100, d0, d1, side="buy_x")
    S_21_X = np.log(x_1out_second) - np.log(x_in)

    return x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X


def gap_series(pair, Q):
    """Per-block profit "gap" series derived from :func:`directional_spreads`.

    Turns the four signed spreads into non-negative achievable profits per
    direction / anchor, the headline best-per-block ``Gap_t``, the per-anchor
    direction signal, and whether the two anchors agree on direction. Returns a
    dict of arrays.
    """
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = directional_spreads(pair, Q)

    # --- "X cheap on pool 1" direction ---
    gap_Y_1cheap = np.maximum(0, -S_12_Y)   # Y-anchor: S_12_Y < 0 is profitable
    gap_X_1cheap = np.maximum(0, S_21_X)    # X-anchor: S_21_X > 0 is profitable

    # --- "X cheap on pool 2" direction ---
    gap_Y_2cheap = np.maximum(0, -S_21_Y)   # Y-anchor: S_21_Y < 0 is profitable
    gap_X_2cheap = np.maximum(0, S_12_X)    # X-anchor: S_12_X > 0 is profitable

    # Headline: best achievable profit, whichever direction/anchor, per block.
    Gap_t = np.maximum.reduce([gap_Y_1cheap, gap_X_1cheap, gap_Y_2cheap, gap_X_2cheap])

    # Which direction is winning, per anchor.
    Y_signals_1cheap = gap_Y_1cheap > gap_Y_2cheap
    X_signals_1cheap = gap_X_1cheap > gap_X_2cheap

    # Diagnostic: do the two anchors agree on direction, per block?
    anchor_agrees = Y_signals_1cheap == X_signals_1cheap

    return dict(
        x_in=x_in, y_in=y_in,
        gap_Y_1cheap=gap_Y_1cheap, gap_X_1cheap=gap_X_1cheap,
        gap_Y_2cheap=gap_Y_2cheap, gap_X_2cheap=gap_X_2cheap,
        Gap_t=Gap_t,
        Y_signals_1cheap=Y_signals_1cheap,
        X_signals_1cheap=X_signals_1cheap,
        anchor_agrees=anchor_agrees,
    )


def arbitrage_index(pair, quantiles):
    """Per-trade-size summary of the round-trip gap for one pool pair.

    ``quantiles`` maps a label -> USD reference size (e.g. the pandas Series from
    ``sizing.trade_size_quantiles``). For each size, run :func:`gap_series` and
    summarize the best-per-block round trip ``Gap_t`` (in basis points): how often
    a gap is live, and its mean / median / max, plus the anchor-agreement rate.

    Returns a DataFrame indexed by the quantile label - the per-pair "arbitrage
    index" a downstream model can stack across every pool pair.
    """
    rows = []
    for label, Q in quantiles.items():
        r = gap_series(pair, Q)
        gap = r["Gap_t"]
        rows.append({
            "usd_ref": float(Q),
            "n_blocks": int(gap.size),
            "live_gap_share": float((gap > 0).mean()),
            "mean_gap_bps": float(gap.mean() * 1e4),
            "median_gap_bps": float(np.median(gap) * 1e4),
            "max_gap_bps": float(gap.max() * 1e4),
            "anchor_agreement": float(r["anchor_agrees"].mean()),
        })
    return pd.DataFrame(rows, index=pd.Index(quantiles.index, name="quantile"))
