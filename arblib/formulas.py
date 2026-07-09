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
