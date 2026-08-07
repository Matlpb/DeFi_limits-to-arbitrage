-- One row per (pool, block): end-of-block state via MAX_BY(..., evt_index) plus the swap
-- count. Server-side equivalent of keep_latest_swap_per_block. amount0 / amount1 are the
-- last swap's amounts (the same swap whose sqrtPriceX96 / tick / liquidity we keep).
--
-- Only sqrtPriceX96 / tick / liquidity / amount0 / amount1 change within a block, so they
-- use MAX_BY(evt_index) (the last swap). The pool-level columns are constant within each
-- (pool, block) group, so they go in GROUP BY and are selected directly. PancakeSwap v3
-- fees are fixed per pool, so no constant-fee filter is needed here.
WITH v3_pools AS (
    SELECT
        pool,
        token0,
        token1,
        fee
    FROM pancakeswap_v3_{{chain}}.pancakev3factory_evt_poolcreated
    WHERE
        (token0 = {{token0}} AND token1 = {{token1}})
        OR (token0 = {{token1}} AND token1 = {{token0}})
),
token_decimals AS (
    SELECT
        contract_address,
        decimals
    FROM tokens.erc20
    WHERE
        blockchain = '{{chain}}'
        AND contract_address IN ({{token0}}, {{token1}})
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
    FROM pancakeswap_v3_{{chain}}.pancakev3pool_evt_swap s
    INNER JOIN v3_pools p ON s.contract_address = p.pool
    WHERE
        s.evt_block_time >= CAST('{{start_ts}}' AS TIMESTAMP)
        AND s.evt_block_time < CAST('{{end_ts}}' AS TIMESTAMP)
),
all_swaps AS (
    SELECT
        s.evt_block_time,
        s.evt_block_number,
        s.evt_index,
        s.pool,
        p.token0,
        p.token1,
        p.fee,
        t0.decimals             AS token0_decimals,
        t1.decimals             AS token1_decimals,
        'pancakeswap_v3'        AS dex,
        s.tick,
        s.liquidity,
        s.sqrtPriceX96,
        s.amount0,
        s.amount1
    FROM v3_swaps s
    JOIN v3_pools p
        ON s.pool = p.pool
    LEFT JOIN token_decimals t0
        ON p.token0 = t0.contract_address
    LEFT JOIN token_decimals t1
        ON p.token1 = t1.contract_address
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
GROUP BY
    a.pool, a.evt_block_number, a.evt_block_time,
    a.fee, a.token0, a.token1, a.token0_decimals, a.token1_decimals, a.dex
