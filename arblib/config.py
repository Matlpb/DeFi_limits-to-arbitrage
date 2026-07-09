"""
Central configuration for the DeFi arbitrage study.

Keeping these values in one place avoids pasting raw token addresses or query
ids into the notebooks by hand. The notebooks still *choose* which chain /
tokens / time window to study, but they pull the constants from here.
"""

# ---------------------------------------------------------------------------
# Dune saved-query ids (one query per DEX, parameterised by chain + tokens)
# ---------------------------------------------------------------------------
# Swap-level mid prices.
SWAP_QUERY_IDS = {
    "uniswap": 7727307,    # 7423632 Uniswap   (base, ethereum, arbitrum)
    "pancake": 7727319,    # 7429756 PancakeSwap (base, ethereum, arbitrum)
}

# Liquidity state (mint / burn events) used to reconstruct liquidity per block.
LIQUIDITY_QUERY_IDS = {
    "uniswap": 7727404,
    "pancake": 7727327,
}

# Gas queries.
#   ``chain_gas_price``  - per-block base fee + utilization for the whole chain,
#                          parameterised by chain + time window only (not per-DEX).
#   ``*_gas_per_swap``   - realised gas used per swap on each DEX, parameterised
#                          like the swap queries (chain + token pair + window).
GAS_QUERY_IDS = {
    "chain_gas_price":      7748900,
    "pancake_gas_per_swap": 7749289,
    "uniswap_gas_per_swap": 7749258,
}

# Hourly USD price per token (from prices.hour), for both tokens of the pair.
# Parameterised by chain + token pair + time window (not per-DEX).
USD_PRICE_QUERY_ID = {
    "USD_price": 7868130
}
# ---------------------------------------------------------------------------
# Token registry (addresses are lower-cased to match the Dune output)
# ---------------------------------------------------------------------------
TOKENS = {
    "ethereum": {
        "WETH": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    },
    "base": {
        "WETH": "0x4200000000000000000000000000000000000006",
        "USDC": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    },
}

# File name per DEX within each chain sub-folder. Keys keep the ``df_<dex>``
# form so the pool-splitting step names pools ``uniswap_1``, ``pancake_1`` ...
SWAP_FILES = {
    "df_uniswap": "df_uniswap_swap.csv",
    "df_pancake": "df_pancake_swap.csv",
}

LIQUIDITY_FILES = {
    "df_uniswap": "df_uniswap_liq.csv",
    "df_pancake": "df_pancake_liq.csv",
}

# Gas files live under ``<chain>/gas/``. Keys match :data:`GAS_QUERY_IDS`.
GAS_FILES = {
    "chain_gas_price":      "chain_gas_price.csv",
    "pancake_gas_per_swap": "pancake_gas_per_swap.csv",
    "uniswap_gas_per_swap": "uniswap_gas_per_swap.csv",
}

def build_collection_params(chain, token0, token1, start_ts, end_ts):
    """Assemble the parameter dict expected by the Dune saved queries.

    Parameters
    ----------
    chain : str
        Blockchain name, e.g. ``"base"`` or ``"ethereum"``.
    token0, token1 : str
        Token addresses (use the :data:`TOKENS` registry).
    start_ts, end_ts : str
        Collection window, ``"YYYY-MM-DD HH:MM:SS"`` (UTC).

    Returns
    -------
    dict
        Ready to pass to :func:`arblib.dune_api.run_dune_saved_query`.
    """
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "token0": token0,
        "token1": token1,
        "chain": chain,
    }
