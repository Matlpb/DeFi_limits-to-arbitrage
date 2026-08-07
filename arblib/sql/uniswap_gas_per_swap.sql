-- One row per (pool, block): the max and median transaction gas price across the pool's
-- swaps in that block. Server-side equivalent of the gas_price_max / gas_price_med that
-- keep_latest_swap_per_block used to compute, so it merges to the aggregated swaps on
-- (pool, evt_block_number).
WITH target_pools_v3 AS (
    SELECT pool
    FROM uniswap_v3_{{chain}}.uniswapv3factory_evt_poolcreated
    WHERE (token0 = {{token0}} AND token1 = {{token1}})
       OR (token0 = {{token1}} AND token1 = {{token0}})
),
target_pools_v4 AS (
    SELECT id AS pool
    FROM uniswap_v4_{{chain}}.poolmanager_evt_initialize
    WHERE (currency0 = {{token0}} AND currency1 = {{token1}})
       OR (currency0 = {{token1}} AND currency1 = {{token0}})
),
swap_hashes_v3 AS (
    SELECT
        s.evt_tx_hash,
        s.evt_block_number,
        s.contract_address AS pool
    FROM uniswap_v3_{{chain}}.uniswapv3pool_evt_swap s
    INNER JOIN target_pools_v3 p ON s.contract_address = p.pool
    WHERE s.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND s.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
),
swap_hashes_v4 AS (
    SELECT
        s.evt_tx_hash,
        s.evt_block_number,
        s.id AS pool
    FROM uniswap_v4_{{chain}}.poolmanager_evt_swap s
    INNER JOIN target_pools_v4 p ON s.id = p.pool
    WHERE s.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND s.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
),
swap_hashes AS (
    SELECT evt_tx_hash, evt_block_number, pool FROM swap_hashes_v3
    UNION ALL
    SELECT evt_tx_hash, evt_block_number, pool FROM swap_hashes_v4
),
swap_gas AS (
    SELECT
        s.evt_block_number,
        s.pool,
        t.gas_price
    FROM swap_hashes s
    LEFT JOIN {{chain}}.transactions t
        ON t.hash = s.evt_tx_hash
        AND t.block_time >= CAST('{{start_ts}}' AS timestamp)
        AND t.block_time <  CAST('{{end_ts}}' AS timestamp)
)
SELECT
    pool,
    evt_block_number,
    MAX(gas_price)                    AS gas_price_max,
    APPROX_PERCENTILE(gas_price, 0.5) AS gas_price_med
FROM swap_gas
GROUP BY pool, evt_block_number
