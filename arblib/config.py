"""
Central configuration for the DeFi arbitrage study.

Keeping these values in one place avoids pasting raw token addresses or query
ids into the notebooks by hand. The notebooks still *choose* which chain /
tokens / time window to study, but they pull the constants from here.
"""

# ---------------------------------------------------------------------------
# Dune saved-query ids (one query per DEX, parameterised by chain + tokens)
# ---------------------------------------------------------------------------
QUERY_IDS = {
    "uniswap": 7423632,    # Uniswap   (base, ethereum, arbitrum)
    "pancake": 7429756,    # PancakeSwap (base, ethereum, arbitrum)
    "aerodrome": 7430568,  # Aerodrome  (base only)
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

# Default file name per DEX used when saving / loading the raw extracts.
DEX_FILES = {
    "df_uniswap": "df_uniswap.csv",
    "df_pancake": "df_pancake.csv",
    "df_aerodrome": "df_aerodrome.csv",
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
