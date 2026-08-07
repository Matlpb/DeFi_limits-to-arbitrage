-- One row per (pool, block): end-of-block state via MAX_BY(..., evt_index) plus the swap
-- count. Server-side equivalent of keep_latest_swap_per_block, so the download is far
-- smaller while feeding pool reconstruction identical inputs. amount0 / amount1 are the
-- last swap's amounts (the same swap whose sqrtPriceX96 / tick / liquidity we keep).
--
-- Only sqrtPriceX96 / tick / liquidity / amount0 / amount1 change within a block, so they
-- use MAX_BY(evt_index) (the last swap). The pool-level columns are constant within each
-- (pool, block) group, so they go in GROUP BY and are selected directly (no aggregate).
--
-- constant_fee_pools keeps only pools whose fee is constant across the window (the SQL
-- equivalent of preprocessing.filter_pools_by_constant_fee): it drops dynamic-fee pools
-- (e.g. Uniswap v4 hooks) up front, so we never pay to extract them.
WITH v3_pools AS (
    SELECT pool, token0, token1, fee
    FROM uniswap_v3_{{chain}}.uniswapv3factory_evt_poolcreated
    WHERE (token0 = {{token0}} AND token1 = {{token1}})
       OR (token0 = {{token1}} AND token1 = {{token0}})
),
token_decimals AS (
    SELECT contract_address, decimals
    FROM tokens.erc20
    WHERE contract_address IN ({{token0}}, {{token1}})
      AND blockchain = '{{chain}}'
),
v3_swaps AS (
    SELECT
        s.evt_block_time,
        s.evt_block_number,
        s.evt_index,
        s.contract_address AS pool,
        s.sqrtPriceX96,
        s.tick,
        s.liquidity,
        s.amount0,
        s.amount1
    FROM uniswap_v3_{{chain}}.uniswapv3pool_evt_swap s
    INNER JOIN v3_pools p ON s.contract_address = p.pool
    WHERE s.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND s.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
),
v4_pools AS (
    SELECT id, currency0, currency1
    FROM uniswap_v4_{{chain}}.poolmanager_evt_initialize
    WHERE (currency0 = {{token0}} AND currency1 = {{token1}})
       OR (currency0 = {{token1}} AND currency1 = {{token0}})
),
v4_swaps AS (
    SELECT
        s.evt_block_time,
        s.evt_block_number,
        s.evt_index,
        s.id,
        s.sqrtPriceX96,
        s.tick,
        s.liquidity,
        s.fee,
        s.amount0,
        s.amount1
    FROM uniswap_v4_{{chain}}.poolmanager_evt_swap s
    INNER JOIN v4_pools p ON s.id = p.id
    WHERE s.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND s.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
),
v3_final AS (
    SELECT
        s.evt_block_time,
        s.evt_block_number,
        s.evt_index,
        s.pool,
        p.token0,
        p.token1,
        p.fee,
        t0.decimals AS token0_decimals,
        t1.decimals AS token1_decimals,
        'uniswap_v3' AS dex,
        s.tick,
        s.liquidity,
        s.sqrtPriceX96,
        s.amount0,
        s.amount1
    FROM v3_swaps s
    JOIN v3_pools p ON s.pool = p.pool
    LEFT JOIN token_decimals t0 ON p.token0 = t0.contract_address
    LEFT JOIN token_decimals t1 ON p.token1 = t1.contract_address
),
v4_final AS (
    SELECT
        s.evt_block_time,
        s.evt_block_number,
        s.evt_index,
        s.id AS pool,
        p.currency0 AS token0,
        p.currency1 AS token1,
        s.fee,
        t0.decimals AS token0_decimals,
        t1.decimals AS token1_decimals,
        'uniswap_v4' AS dex,
        s.tick,
        s.liquidity,
        s.sqrtPriceX96,
        s.amount0,
        s.amount1
    FROM v4_swaps s
    JOIN v4_pools p ON s.id = p.id
    LEFT JOIN token_decimals t0 ON p.currency0 = t0.contract_address
    LEFT JOIN token_decimals t1 ON p.currency1 = t1.contract_address
),
all_swaps AS (
    SELECT * FROM v3_final
    UNION ALL
    SELECT * FROM v4_final
),
constant_fee_pools AS (
    SELECT pool
    FROM all_swaps
    GROUP BY pool
    HAVING COUNT(DISTINCT fee) = 1
)
SELECT
    a.pool,
    a.evt_block_number,
    a.evt_block_time,
    MAX_BY(a.sqrtPriceX96, a.evt_index)  AS sqrtPriceX96,
    MAX_BY(a.tick, a.evt_index)          AS tick,
    MAX_BY(a.liquidity, a.evt_index)     AS liquidity,
    MAX_BY(a.amount0, a.evt_index)       AS amount0,
    MAX_BY(a.amount1, a.evt_index)       AS amount1,
    a.fee,
    a.token0,
    a.token1,
    a.token0_decimals,
    a.token1_decimals,
    a.dex,
    COUNT(*)                             AS nb_swaps
FROM all_swaps a
INNER JOIN constant_fee_pools c ON a.pool = c.pool
GROUP BY
    a.pool, a.evt_block_number, a.evt_block_time,
    a.fee, a.token0, a.token1, a.token0_decimals, a.token1_decimals, a.dex
