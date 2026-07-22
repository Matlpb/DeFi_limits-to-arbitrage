"""
Pure pricing / AMM primitives (no state, no plotting).

Everything here is a small, vectorized, side-effect-free function:
    * unit conversions        - to_human_units, to_usd
    * spot price              - mid_price
    * single-swap AMM output  - swap_out

Higher-level analysis (trade-size distributions, cross-pool arbitrage) builds on
these in ``arblib.sizing`` and ``arblib.arbitrage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_Q96 = 2.0**96

##########---------CONVERT-----------###########

def to_human_units(raw: pd.Series | np.ndarray, decimals: int | pd.Series) -> np.ndarray | pd.Series:
    """Convert a raw on-chain integer amount to human units (``raw / 10**decimals``).

    ``raw`` is coerced to float first because large amounts (e.g. 18-decimal WETH)
    overflow int64 and load from CSV as text. Works on scalars, Series, or arrays.
    """
    raw = raw.astype(float) if isinstance(raw, pd.Series) else np.asarray(raw, dtype=float)
    return raw / 10.0**decimals


def to_usd(amount: float | np.ndarray | pd.Series, usd_price: float | np.ndarray,
           sense: str = "to_usd") -> float | np.ndarray | pd.Series:
    """Convert between a human-unit token amount and USD.

    ``sense="to_usd"`` gives ``amount * usd_price``; ``sense="to_token"`` gives
    ``amount / usd_price``. Stays in human units either way.
    """
    if sense == "to_usd":
        return amount * usd_price
    if sense == "to_token":
        return amount / usd_price
    raise ValueError(f"sense must be 'to_usd' or 'to_token', got {sense!r}")

##########---------PRICES-----------###########


def mid_price(sqrtPriceX96: float | np.ndarray | pd.Series,
              token0_decimals: int | pd.Series,
              token1_decimals: int | pd.Series) -> float | np.ndarray:
    """Spot mid price as token0 per token1 in human units (e.g. USDC per WETH).

    Uniswap v3 ``sqrtPriceX96`` encodes the raw token1-per-token0 price as
    ``(sqrtPriceX96 / 2**96)**2``; converting to human units and inverting gives the
    token0-per-token1 quote::

        price = (2**96 / sqrtPriceX96)**2 * 10**(token1_decimals - token0_decimals)
    """
    sqrtP = np.asarray(sqrtPriceX96, dtype=float) / _Q96
    raw_y_per_x = sqrtP**2
    human_y_per_x = raw_y_per_x * 10.0 ** (token0_decimals - token1_decimals)
    return 1.0 / human_y_per_x


def swap_out(sqrtPriceX96: float | np.ndarray, liquidity: float | np.ndarray,
             amount_in_human: float | np.ndarray, fee_bps: float,
             d0: int, d1: int, side: str) -> np.ndarray:
    """Realized OUTPUT amount (human units) of one within-tick swap leg.

    Single-tick, constant-``L`` Uniswap-v3 math with the fee taken on the input.
    ``sqrtPriceX96`` and ``liquidity`` are raw on-chain values; the human input is
    converted to raw units internally. Vectorized over ``amount_in_human`` (and over
    ``sqrtPriceX96`` / ``liquidity``), so one leg's output can feed the next leg's
    input to chain a round trip. ``side="buy_y"`` sends X (token0) in for Y (token1)
    out; ``side="buy_x"`` sends Y in for X out.
    """
    gamma = fee_bps / 10000
    sqrt_P = np.asarray(sqrtPriceX96, dtype=float) / _Q96
    L = np.asarray(liquidity, dtype=float)
    a_in = np.asarray(amount_in_human, dtype=float)

    if side == "buy_y":
        amount_in_eff = (a_in * 10**d0) * (1 - gamma)
        sqrt_P_exit = 1.0 / (1.0 / sqrt_P + amount_in_eff / L)
        return L * (sqrt_P - sqrt_P_exit) / 10**d1
    if side == "buy_x":
        amount_in_eff = (a_in * 10**d1) * (1 - gamma)
        sqrt_P_exit = sqrt_P + amount_in_eff / L
        return L * (1.0 / sqrt_P - 1.0 / sqrt_P_exit) / 10**d0
    raise ValueError(f"side must be 'buy_y' or 'buy_x', got {side!r}")



##########---------PROXY-----------###########$



def _decay_ewma(values: np.ndarray, retain: float) -> np.ndarray:
    """Causal per-block exponential moving average that fades toward 0 on quiet blocks.

    Over the dense block grid: ``out_t = retain * out_{t-1} + (1 - retain) * values_t``.
    A block contributing 0 decays the level by ``retain``; an active block pulls it toward
    ``values_t``. Assumes one row per consecutive block, contemporaneous (includes block t
    itself) - lag downstream when using as a regressor, not here.
    """
    out = np.empty(len(values), dtype=float)
    acc = 0.0
    for i in range(len(values)):
        acc = retain * acc + (1.0 - retain) * values[i]
        out[i] = acc
    return out


def add_mev_intensity(pool_dfs: dict[str, pd.DataFrame], chain_gas: pd.DataFrame,
                       horizon_blocks: int) -> dict[str, pd.DataFrame]:
    """Add the tip-magnitude MEV proxy to each pool series (in place).

    On a block with a swap, the signal is the top priority tip paid:
    ``gas_price_max - base_fee`` (base fee joined from ``chain_gas``). Quiet blocks
    contribute exactly 0 - not imputed, but the correct economic value: no trade means
    no tip was bid, so there is nothing to feed into the EWMA that block. Passed through
    ``_decay_ewma`` with ``retain = exp(-1 / horizon_blocks)`` so the level fades toward 0
    during quiet runs and rises on active blocks.

    Adds column ``mev_intensity`` (wei, contemporaneous - includes block t). Returns
    ``pool_dfs``.
    """
    base_fee = chain_gas.set_index("block_number")["base_fee_per_gas"]
    retain = np.exp(-1.0 / horizon_blocks)

    for name, df in pool_dfs.items():
        base = base_fee.reindex(df["evt_block_number"]).to_numpy(dtype=float)
        nb_swaps = df["nb_swaps"].to_numpy()
        tip = np.nan_to_num(df["gas_price_max"].to_numpy(dtype=float) - base)
        tip_block = np.where(nb_swaps >= 1, tip, 0.0)

        df["mev_intensity"] = _decay_ewma(tip_block, retain)
        print(f"{name}: mev_intensity max {df['mev_intensity'].max():.3e} wei")

    return pool_dfs


def add_contest_freq(pool_dfs: dict[str, pd.DataFrame], horizon_blocks: int) -> dict[str, pd.DataFrame]:
    """Add the swap-count (order-flow) MEV proxy to each pool series (in place).

    Signal is the raw ``nb_swaps`` per block - already dense (0 on quiet blocks, not NaN),
    so no masking step is needed here, unlike the tip signal. Passed through the same
    ``_decay_ewma`` recursion with its own ``horizon_blocks``.

    Adds column ``nb_swaps_ewma`` (contemporaneous - includes block t) to each single-venue
    pool series. This is per-venue by construction; combine the two venues' columns into
    freq_{p,t} = 0.5 * (ewma_venue1 + ewma_venue2) at the pool-pair assembly stage,
    downstream of this function. Returns ``pool_dfs``.
    """
    retain = np.exp(-1.0 / horizon_blocks)

    for name, df in pool_dfs.items():
        nb_swaps = df["nb_swaps"].to_numpy(dtype=float)
        df["nb_swaps_ewma"] = _decay_ewma(nb_swaps, retain)
        print(f"{name}: nb_swaps_ewma max {df['nb_swaps_ewma'].max():.2f}")

    return pool_dfs


def vol_ewma(x_usd: pd.DataFrame, y_usd: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """RiskMetrics EWMA volatility of the token0/token1 exchange rate, from two USD series.

    Aligns the X (token0/USD) and Y (token1/USD) 1-minute price series on ``time``, forms the
    exchange rate ``R = P_X/USD / P_Y/USD``, takes log returns differenced *over time*
    ``r_t = log R_t - log R_{t-1}`` (no demeaning - at 1-minute frequency the mean return is
    negligible), and runs the RiskMetrics recursion via :func:`_decay_ewma`::

        var_t = retain * var_{t-1} + (1 - retain) * r_t ** 2

    with ``retain = exp(-1 / horizon_min)`` - the same e-folding parameterization as the MEV
    proxies, so ``horizon_min`` is the memory span in minutes (mirroring ``mev_horizon_blocks``
    in blocks) rather than an opaque decay factor. Returns ``[time, R, ret, ewma_var,
    ewma_vol]`` (``ewma_vol = sqrt(var)``, per-minute). Feeding the level ``log R_t`` instead of
    the return would give decayed mean-square level dominated by the price level, not
    volatility; hence the differencing.
    """
    retain = np.exp(-1.0 / horizon_min)
    m = x_usd.merge(y_usd, on="time", suffixes=("_x", "_y")).sort_values("time")
    R = (m["price_x"] / m["price_y"]).to_numpy()
    ret = np.diff(np.log(R), prepend=np.nan)
    var = _decay_ewma(np.nan_to_num(ret ** 2), retain)
    return pd.DataFrame({
        "time": m["time"].to_numpy(),
        "R": R,
        "ret": ret,
        "ewma_var": var,
        "ewma_vol": np.sqrt(var),
    })
