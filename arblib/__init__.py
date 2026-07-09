"""
arblib - reusable building blocks for the DeFi cross-DEX arbitrage study.

Modules
-------
config        : query ids, token registry, parameter helpers
dune_api      : Dune REST API client (execute / wait / fetch)
data_io       : save / load the raw extracts, processed pools, and quantile refs
formulas      : pure pricing / AMM primitives (to_usd, mid_price, swap_out, ...)
preprocessing : clean swaps -> split by pool -> filter -> reconstruct state series
analysis      : cross-DEX mid-price differences
sizing        : USD trade-size distribution and reference quantiles
arbitrage     : executable prices + per-pool-pair arbitrage index (PoolPair)
plotting      : price, trade-size, and round-trip spread plots
"""

from . import (
    analysis,
    arbitrage,
    config,
    data_io,
    dune_api,
    formulas,
    plotting,
    preprocessing,
    sizing,
)

__all__ = [
    "analysis",
    "arbitrage",
    "config",
    "data_io",
    "dune_api",
    "formulas",
    "plotting",
    "preprocessing",
    "sizing",
]
