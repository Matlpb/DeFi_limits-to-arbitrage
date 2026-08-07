-- One row per (pool, block): the max and median transaction gas price across the pool's
-- swaps in that block. Merges to the aggregated swaps on (pool, evt_block_number).
WITH target_pools AS (
    SELECT pool
    FROM pancakeswap_v3_{{chain}}.pancakev3factory_evt_poolcreated
    WHERE (token0 = {{token0}} AND token1 = {{token1}})
       OR (token0 = {{token1}} AND token1 = {{token0}})
),
swap_hashes AS (
    SELECT
        s.evt_tx_hash,
        s.evt_block_number,
        s.contract_address AS pool
    FROM pancakeswap_v3_{{chain}}.pancakev3pool_evt_swap s
    INNER JOIN target_pools p ON s.contract_address = p.pool
    WHERE s.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND s.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
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
