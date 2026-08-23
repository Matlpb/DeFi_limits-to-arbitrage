"""
arblib - reusable building blocks for the DeFi cross-DEX arbitrage study.

Modules
-------
config        : query ids, token registry, parameter helpers
dune_api      : Dune REST API client (execute / wait / fetch)
kraken_api    : Kraken REST client (historical 1-min USD prices via Trades)
data_io       : save / load the raw extracts, processed pools, and quantile refs
formulas      : pure pricing / AMM primitives (to_usd, mid_price, swap_out, ...)
preprocessing : clean swaps -> split by pool -> filter -> reconstruct state series
regime        : daily CEX volatility-regime detection (low / mid / high-vol weeks)
analysis      : cross-DEX mid-price differences
sizing        : USD trade-size distribution and reference quantiles
arbitrage     : executable prices + per-pool-pair arbitrage index (PoolPair)
modeling      : feature build - dependent variable + covariates -> parquet
estimation    : panel assembly + the block-level / spell-level models
naming        : shared quantile-suffix / pool-pair naming + iteration conventions
plotting      : price, trade-size, and round-trip spread plots
summary       : cross-regime descriptive summary-statistics tables
reporting     : cross-regime onset / closure / magnitude coefficient tables
"""

from . import (
    analysis,
    arbitrage,
    config,
    data_io,
    dune_api,
    estimation,
    formulas,
    kraken_api,
    modeling,
    naming,
    plotting,
    preprocessing,
    regime,
    reporting,
    sizing,
    summary,
)

__all__ = [
    "analysis",
    "arbitrage",
    "config",
    "data_io",
    "dune_api",
    "estimation",
    "formulas",
    "kraken_api",
    "modeling",
    "naming",
    "plotting",
    "preprocessing",
    "regime",
    "reporting",
    "sizing",
    "summary",
]
