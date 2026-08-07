"""
Shared naming and pairing conventions.

The quantile suffix (``0.2 -> "q20"``), the pool-pair key (``"uniswap_1_vs_pancake_1"``) and the
enumeration of unordered pool pairs are used across the feature build (:mod:`arblib.arbitrage`,
:mod:`arblib.modeling`) and the estimation layer (:mod:`arblib.estimation`). Defining them once here
keeps the parquet file names, the ``gap_q`` / ``D_q`` column suffixes and the pair iteration exactly
consistent between the module that writes a table and the one that reads it back.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations


def qlabel(quantile: float) -> str:
    """Canonical quantile suffix, e.g. ``0.2 -> "q20"`` (matches the ``gap_q`` / ``D_q`` columns)."""
    return f"q{int(round(quantile * 100))}"


def pair_name(name1: str, name2: str) -> str:
    """Canonical unordered pool-pair key, e.g. ``("uniswap_1", "pancake_1") -> "uniswap_1_vs_pancake_1"``."""
    return f"{name1}_vs_{name2}"


def iter_pairs(pools: "dict") -> Iterator[tuple[str, str, str]]:
    """Yield ``(pair_name, name1, name2)`` for every unordered pair of ``pools``, in a stable order.

    The single source of the "every pool against every other pool" loop, so the pair set and its
    naming stay identical wherever pairs are built (arbitrage index, pool-pair covariates)."""
    for name1, name2 in combinations(pools, 2):
        yield pair_name(name1, name2), name1, name2
