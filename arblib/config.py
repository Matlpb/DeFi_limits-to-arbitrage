"""
Central configuration for the DeFi arbitrage study.

Everything the pipeline needs - which chain, which token pair (addresses +
decimals), the collection and study windows, and every derived path - is defined
once here as the :data:`STUDY` settings instance. Notebooks import it
(``from arblib.config import STUDY as S``) and derive the rest, so re-parametrising
the whole study for a different pair or chain is a single edit to :data:`STUDY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SWAP_QUERY_IDS: dict[str, int] = {
    "uniswap": 7727307,
    "pancake": 7727319,
}

LIQUIDITY_QUERY_IDS: dict[str, int] = {
    "uniswap": 7727404,
    "pancake": 7727327,
}

GAS_QUERY_IDS: dict[str, int] = {
    "chain_gas_price": 7748900,
    "pancake_gas_per_swap": 7749289,
    "uniswap_gas_per_swap": 7749258,
}

SWAP_FILES: dict[str, str] = {
    "df_uniswap": "df_uniswap_swap.csv",
    "df_pancake": "df_pancake_swap.csv",
}

LIQUIDITY_FILES: dict[str, str] = {
    "df_uniswap": "df_uniswap_liq.csv",
    "df_pancake": "df_pancake_liq.csv",
}

GAS_FILES: dict[str, str] = {
    "chain_gas_price": "chain_gas_price.csv",
    "pancake_gas_per_swap": "pancake_gas_per_swap.csv",
    "uniswap_gas_per_swap": "uniswap_gas_per_swap.csv",
}

# Kraken spot market used to price each token in USD (for the volatility control).
KRAKEN_PAIRS: dict[str, str] = {
    "WETH": "ETHUSD",
    "USDC": "USDCUSD",
}


@dataclass(frozen=True)
class Token:
    """One ERC-20 token: its symbol, lower-cased address (as Dune returns it), and decimals."""

    symbol: str
    address: str
    decimals: int


TOKENS: dict[str, dict[str, Token]] = {
    "ethereum": {
        "WETH": Token("WETH", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18),
        "USDC": Token("USDC", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6),
    },
    "base": {
        "WETH": Token("WETH", "0x4200000000000000000000000000000000000006", 18),
        "USDC": Token("USDC", "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913", 6),
    },
}


def build_collection_params(chain: str, token0: str, token1: str,
                            start_ts: str, end_ts: str) -> dict[str, str]:
    """Assemble the superset parameter dict expected by the Dune saved queries."""
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "token0": token0,
        "token1": token1,
        "chain": chain,
    }


@dataclass(frozen=True)
class Settings:
    """Single source of truth for the study: tokens, windows, and derived paths.

    ``base`` / ``quote`` are the conceptual pair (base priced in quote, e.g. WETH in
    USDC); they drive the Dune query and the plot labels. ``token0`` / ``token1`` are
    the same two tokens in on-chain ascending-address order, matching the ``token0`` /
    ``token1`` columns of the swap data (so ``token0.decimals`` replaces literal
    decimals). Every path derives from ``base_dir/chain/"<base>_<quote>"``, so a
    different pair or chain writes to its own folder instead of colliding.
    """

    chain: str
    base: Token
    quote: Token
    max_gap_blocks: int = 6000
    mev_horizon_blocks: int =20
    vol_horizon_min: int = 30
    cex_avg_window_min: int = 60
    regime_start: str = "2025-07-01"    # market_regime_detection: first day of the classified year
    regime_end: str = "2026-06-30"      # ... last day (leaves an end-buffer before "now")
    regime_buffer_days: int = 75        # EWMA warm-up lead before the classified year
    regime_window_days: int = 6         # length of each regime study window (days)
    extract_lead_hours: int = 4         # block-data warm-up lead before a chosen study window
    active_regime: str = "high"         # which window extract / transform operate on: low | mid | high
    reference_quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)   # trade-size reference quantiles
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def token0(self) -> Token:
        return min(self.base, self.quote, key=lambda t: t.address.lower())

    @property
    def token1(self) -> Token:
        return max(self.base, self.quote, key=lambda t: t.address.lower())

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.chain / f"{self.base.symbol}_{self.quote.symbol}"

    @property
    def data_analysis_dir(self) -> Path:
        """Main folder holding the raw / intermediate CSV data (swaps, gas, prices, ...)."""
        return self.data_dir / "data_analysis"

    @property
    def modeling_dir(self) -> Path:
        """Main folder holding modeling-ready transformed data (parquet)."""
        return self.data_dir / "modeling"

    @property
    def dependent_var_dir(self) -> Path:
        """Per-pool-pair dependent-variable parquets (Gap_t(Q) and D_t(Q))."""
        return self.modeling_dir / "dependant_variable"

    @property
    def covariates_dir(self) -> Path:
        """Modeling covariates (features), split into common and pool-pair-specific."""
        return self.modeling_dir / "covariates"

    @property
    def common_covariates_dir(self) -> Path:
        """Covariates shared by every pool pair (CEX volatility, chain gas)."""
        return self.covariates_dir / "common_covariates"

    @property
    def pair_covariates_dir(self) -> Path:
        """Per-pool-pair covariates (e.g. MEV, liquidity); filled downstream, empty for now."""
        return self.covariates_dir / "pool_pair_dependant_covariates"

    @property
    def cex_vol_path(self) -> Path:
        return self.common_covariates_dir / "CEX_volatility.parquet"

    @property
    def chain_covariates_path(self) -> Path:
        return self.common_covariates_dir / "chain_covariates.parquet"

    @property
    def swaps_dir(self) -> Path:
        return self.data_analysis_dir / "swaps"

    @property
    def liquidity_dir(self) -> Path:
        return self.data_analysis_dir / "liquidity"

    @property
    def gas_dir(self) -> Path:
        return self.data_analysis_dir / "gas"

    @property
    def prices_dir(self) -> Path:
        return self.data_analysis_dir / "prices"

    @property
    def processed_dir(self) -> Path:
        return self.data_analysis_dir / "processed"

    @property
    def gas_path(self) -> Path:
        return self.gas_dir / GAS_FILES["chain_gas_price"]

    @property
    def x_price_path(self) -> Path:
        return self.prices_dir / "X_USD_prices.csv"

    @property
    def y_price_path(self) -> Path:
        return self.prices_dir / "Y_USD_prices.csv"

    @property
    def quantiles_path(self) -> Path:
        return self.data_analysis_dir / "trade_size_quantiles.csv"

    @property
    def study_dates_path(self) -> Path:
        """The three chosen regime windows (low / mid / high) written by market_regime_detection."""
        return self.data_analysis_dir / "study_dates.csv"


STUDY = Settings(
    chain="ethereum",
    base=TOKENS["ethereum"]["WETH"],
    quote=TOKENS["ethereum"]["USDC"],
    mev_horizon_blocks=15,   # ~10 min at 12s/block
)
