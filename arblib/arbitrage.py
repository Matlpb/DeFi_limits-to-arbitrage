"""
Cross-pool executable prices and the per-pool-pair arbitrage index.

A naive cross-pool price spread compares the same swap side across two pools at two
independent notionals - that is not an executable arbitrage. A real round trip closes
the position: the second leg is the opposite side (you sell back what you bought) fed
the real output of the first leg (via ``formulas.swap_out``), so both fees and both
legs' slippage are charged.

Everything is expressed for one ordered pool pair, bundled in :class:`PoolPair`:

    pair = build_pool_pair(p1_df, p2_df, S.token0, S.token1, price0, price1, "pancake_1", "uniswap_1")
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = directional_spreads(pair, Q)
    gaps = gap_series(pair, Q)
    index = arbitrage_index(pair, quantiles)

``arbitrage_index`` is the per-pair signal a downstream model consumes over every pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from . import formulas
from .config import Token

_KEEP1 = ["evt_block_number", "evt_block_time", "sqrtPriceX96", "liquidity"]
_KEEP2 = ["evt_block_number", "sqrtPriceX96", "liquidity"]


@dataclass
class PoolPair:
    """Per-block state of two aligned pools plus the shared market context.

    ``sqrtP*`` / ``L*`` are per-block arrays; ``fee*`` are the raw pool fee field
    (e.g. ``100`` = 0.01%). ``d0`` / ``d1`` are token0 / token1 decimals; ``price0`` /
    ``price1`` their constant USD prices; ``sym0`` / ``sym1`` their symbols (labels).
    ``name1`` / ``name2`` label the two pools.
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
    price0: float
    price1: float
    sym0: str
    sym1: str
    name1: str = "pool1"
    name2: str = "pool2"


def build_pool_pair(p1_df: pd.DataFrame, p2_df: pd.DataFrame, token0: Token, token1: Token,
                    price0: float, price1: float, name1: str = "pool1",
                    name2: str = "pool2") -> PoolPair:
    """Align two processed pool frames on the shared block grid into a ``PoolPair``.

    ``p1_df`` / ``p2_df`` are processed per-pool frames (per-block ``sqrtPriceX96`` /
    ``liquidity`` / ``fee``). ``token0`` / ``token1`` supply decimals and symbols;
    ``price0`` / ``price1`` are the constant USD prices of token0 / token1 (see
    ``sizing.constant_usd_prices``).
    """
    fee1 = float(p1_df["fee"].dropna().unique()[0])
    fee2 = float(p2_df["fee"].dropna().unique()[0])

    m = p1_df[_KEEP1].merge(p2_df[_KEEP2], on="evt_block_number", suffixes=("_1", "_2"))

    return PoolPair(
        block=m["evt_block_number"].to_numpy(),
        sqrtP1=m["sqrtPriceX96_1"].astype(float).to_numpy(),
        L1=m["liquidity_1"].astype(float).to_numpy(),
        fee1=fee1,
        sqrtP2=m["sqrtPriceX96_2"].astype(float).to_numpy(),
        L2=m["liquidity_2"].astype(float).to_numpy(),
        fee2=fee2,
        d0=token0.decimals,
        d1=token1.decimals,
        price0=price0,
        price1=price1,
        sym0=token0.symbol,
        sym1=token1.symbol,
        name1=name1,
        name2=name2,
    )


def directional_spreads(pair: PoolPair, Q: float) -> tuple:
    """Four chained round-trip spreads for USD notional ``Q``, per block.

    Sign convention: ``S = log(amount_sent_in) - log(amount_received_back)``, so
    ``S < 0`` is a profitable round trip and ``S > 0`` a loss. Direction "1->2" starts
    on pool 1 and closes on pool 2; "2->1" is the reverse. Each direction is estimated
    two ways (Y-anchored / X-anchored), which should agree in sign when a real,
    size-robust gap exists. Each second leg is fed the real output of the first leg.

    Returns ``(x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X)``.
    """
    sqrtP1, L1, fee1 = pair.sqrtP1, pair.L1, pair.fee1
    sqrtP2, L2, fee2 = pair.sqrtP2, pair.L2, pair.fee2
    d0, d1 = pair.d0, pair.d1

    swap_out = formulas.swap_out
    x_in = formulas.to_usd(Q, pair.price0, sense="to_token")
    y_in = formulas.to_usd(Q, pair.price1, sense="to_token")

    x_1out = swap_out(sqrtP1, L1, y_in, fee1 / 100, d0, d1, side="buy_x")
    y_2out = swap_out(sqrtP2, L2, x_1out, fee2 / 100, d0, d1, side="buy_y")
    S_12_Y = np.log(y_in) - np.log(y_2out)

    y_1out = swap_out(sqrtP1, L1, x_in, fee1 / 100, d0, d1, side="buy_y")
    x_2out = swap_out(sqrtP2, L2, y_1out, fee2 / 100, d0, d1, side="buy_x")
    S_12_X = np.log(x_2out) - np.log(x_in)

    x_2out_first = swap_out(sqrtP2, L2, y_in, fee2 / 100, d0, d1, side="buy_x")
    y_1out_second = swap_out(sqrtP1, L1, x_2out_first, fee1 / 100, d0, d1, side="buy_y")
    S_21_Y = np.log(y_in) - np.log(y_1out_second)

    y_2out_first = swap_out(sqrtP2, L2, x_in, fee2 / 100, d0, d1, side="buy_y")
    x_1out_second = swap_out(sqrtP1, L1, y_2out_first, fee1 / 100, d0, d1, side="buy_x")
    S_21_X = np.log(x_1out_second) - np.log(x_in)

    return x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X


def gap_series(pair: PoolPair, Q: float) -> dict:
    """Per-block profit "gap" series derived from :func:`directional_spreads`.

    The four signed spreads become non-negative achievable profits per direction and
    anchor: "token0 cheap on pool 1" is captured by ``gap_Y_1cheap = max(0, -S_12_Y)``
    (Y-anchor, ``S_12_Y < 0`` profitable) and ``gap_X_1cheap = max(0, S_21_X)``
    (X-anchor, ``S_21_X > 0`` profitable); "token0 cheap on pool 2" symmetrically from
    ``S_21_Y`` / ``S_12_X``. ``Gap_t`` is the best of the four per block; the anchors'
    direction signals and their agreement are returned for diagnostics.
    """
    x_in, y_in, S_12_Y, S_12_X, S_21_Y, S_21_X = directional_spreads(pair, Q)

    gap_Y_1cheap = np.maximum(0, -S_12_Y)
    gap_X_1cheap = np.maximum(0, S_21_X)
    gap_Y_2cheap = np.maximum(0, -S_21_Y)
    gap_X_2cheap = np.maximum(0, S_12_X)

    Gap_t = np.maximum.reduce([gap_Y_1cheap, gap_X_1cheap, gap_Y_2cheap, gap_X_2cheap])

    Y_signals_1cheap = gap_Y_1cheap > gap_Y_2cheap
    X_signals_1cheap = gap_X_1cheap > gap_X_2cheap
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


def arbitrage_index(pair: PoolPair, quantiles: pd.Series) -> pd.DataFrame:
    """Per-trade-size summary of the round-trip gap for one pool pair.

    ``quantiles`` maps a label -> USD reference size. For each size, :func:`gap_series`
    is summarised into how often the best-per-block round trip ``Gap_t`` is live and how
    big it is (basis points), plus the anchor-agreement rate. Returns a DataFrame indexed
    by the quantile label - the per-pair "arbitrage index" a model stacks across pairs.
    """
    rows = []
    for _, Q in quantiles.items():
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


def gap_index_frame(pair: PoolPair, quantiles: pd.Series) -> pd.DataFrame:
    """Per-block best round-trip gap ``Gap_t`` (bps) for one pool pair.

    One column per reference trade size (the ``quantiles`` labels), indexed by block -
    the arbitrage-index *time series* for the pair, of which :func:`arbitrage_index` is
    the cross-block summary.
    """
    data = {label: gap_series(pair, Q)["Gap_t"] * 1e4 for label, Q in quantiles.items()}
    return pd.DataFrame(data, index=pd.Index(pair.block, name="block"))


def all_pairs_gap_series(pools: dict[str, pd.DataFrame], quantiles: pd.Series,
                         token0: Token, token1: Token, price0: float,
                         price1: float) -> dict[str, pd.DataFrame]:
    """Arbitrage-index time series for every pool pair, keyed by pair name.

    Every unordered pair of ``pools`` is built - intra-DEX (e.g. ``uniswap_1_vs_uniswap_2``)
    as well as cross-DEX - and turned into its :func:`gap_index_frame`. ``pools`` are assumed
    constant-fee (the transform step drops dynamic-fee pools). Returns
    ``{"<poolA>_vs_<poolB>": frame}``, the per-pair signal a downstream model stacks.
    """
    out = {}
    for n1, n2 in combinations(pools, 2):
        pair = build_pool_pair(pools[n1], pools[n2], token0, token1, price0, price1, n1, n2)
        out[f"{n1}_vs_{n2}"] = gap_index_frame(pair, quantiles)
    return out
