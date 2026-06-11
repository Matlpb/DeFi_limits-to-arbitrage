"""
arblib - reusable building blocks for the DeFi cross-DEX arbitrage study.

Modules
-------
config        : query ids, token registry, parameter helpers
dune_api      : Dune REST API client (execute / wait / fetch)
data_io       : save / load the raw per-DEX CSV extracts
preprocessing : clean swaps -> split by pool -> filter -> reconstruct series
analysis      : cross-DEX pairwise price differences
plotting      : price time-series plots
"""

from . import analysis, config, data_io, dune_api, plotting, preprocessing

__all__ = [
    "analysis",
    "config",
    "data_io",
    "dune_api",
    "plotting",
    "preprocessing",
]
