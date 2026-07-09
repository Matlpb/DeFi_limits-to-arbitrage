"""
Pure pricing / AMM primitives (no state, no plotting).

Everything here is a small, vectorized, side-effect-free function:
    * unit conversions        - to_human_units, to_usd
    * spot price              - mid_price
    * single-swap AMM output  - swap_out

Higher-level analysis (trade-size distributions, cross-pool arbitrage) builds on
these in ``arblib.sizing`` and ``arblib.arbitrage``.
"""

import numpy as np
import pandas as pd

# 2**96, the fixed-point scale of Uniswap v3/v4 ``sqrtPriceX96``.
_Q96 = 2.0**96


def to_human_units(raw, decimals):
    """Convert a raw on-chain integer amount to human units.

    ``human = raw / 10**decimals``. Works on scalars or pandas Series
    (elementwise); ``raw`` is coerced to float first because large amounts
    (e.g. 18-decimal WETH) overflow int64 and load from CSV as text.
    """
    raw = raw.astype(float) if isinstance(raw, pd.Series) else np.asarray(raw, dtype=float)
    return raw / 10.0**decimals


def to_usd(amount, usd_price, sense="to_usd"):
    """Convert between a human-unit token amount and USD, both directions.

    ``sense``:
        * ``"to_usd"``   - human token amount -> USD  (``amount * usd_price``);
        * ``"to_token"`` - USD -> human token amount  (``amount / usd_price``).

    Stays in human units either way (no raw pool-format conversion).
    """
    if sense == "to_usd":
        return amount * usd_price
    if sense == "to_token":
        return amount / usd_price
    raise ValueError(f"sense must be 'to_usd' or 'to_token', got {sense!r}")


def mid_price(sqrtPriceX96, token0_decimals, token1_decimals):
    """Spot mid price as token0 per token1 in human units (e.g. USDC per WETH).

    Uniswap v3 ``sqrtPriceX96`` encodes the raw token1-per-token0 price as
    ``(sqrtPriceX96 / 2**96)**2``. Converting to human units and inverting gives
    the token0-per-token1 quote (the ETH price in USDC for a USDC/WETH pool):

        price = (2**96 / sqrtPriceX96)**2 * 10**(token1_decimals - token0_decimals)

    Vectorized over ``sqrtPriceX96`` (scalar or pandas Series / ndarray).
    """
    sqrtP = np.asarray(sqrtPriceX96, dtype=float) / _Q96
    raw_y_per_x = sqrtP**2                                      # token1 per token0, raw
    human_y_per_x = raw_y_per_x * 10.0 ** (token0_decimals - token1_decimals)
    return 1.0 / human_y_per_x                                  # token0 per token1, human


def swap_out(sqrtPriceX96, liquidity, amount_in_human, fee_bps, d0, d1, side):
    """Realized OUTPUT amount (human units) of one within-tick swap leg.

    Single-tick, constant-``L`` Uniswap-v3 math with the fee taken on the input.
    ``sqrtPriceX96`` and ``liquidity`` are raw on-chain values; the human input is
    converted to raw units internally. Vectorized over ``amount_in_human`` (as well
    as ``sqrtPriceX96`` / ``liquidity``), so one leg's real output can be fed as the
    next leg's input to chain a round trip:
        * ``side="buy_y"`` - send X (token0) in  -> Y (token1) out.
        * ``side="buy_x"`` - send Y (token1) in  -> X (token0) out.
    """
    gamma = fee_bps / 10000
    sqrt_P = np.asarray(sqrtPriceX96, dtype=float) / _Q96
    L = np.asarray(liquidity, dtype=float)
    a_in = np.asarray(amount_in_human, dtype=float)

    if side == "buy_y":
        amount_in_eff = (a_in * 10**d0) * (1 - gamma)               # raw X in, after fee
        sqrt_P_exit = 1.0 / (1.0 / sqrt_P + amount_in_eff / L)
        return L * (sqrt_P - sqrt_P_exit) / 10**d1                  # Y out, human
    if side == "buy_x":
        amount_in_eff = (a_in * 10**d1) * (1 - gamma)               # raw Y in, after fee
        sqrt_P_exit = sqrt_P + amount_in_eff / L
        return L * (1.0 / sqrt_P - 1.0 / sqrt_P_exit) / 10**d0      # X out, human
    raise ValueError(f"side must be 'buy_y' or 'buy_x', got {side!r}")
