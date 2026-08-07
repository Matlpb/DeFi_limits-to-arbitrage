WITH v3_pools AS (
    SELECT pool, token0, token1, tickSpacing
    FROM uniswap_v3_{{chain}}.uniswapv3factory_evt_poolcreated
    WHERE (token0 = {{token0}} AND token1 = {{token1}})
       OR (token0 = {{token1}} AND token1 = {{token0}})
),
v4_pools AS (
    SELECT id AS pool, currency0 AS token0, currency1 AS token1, tickSpacing
    FROM uniswap_v4_{{chain}}.poolmanager_evt_initialize
    WHERE (currency0 = {{token0}} AND currency1 = {{token1}})
       OR (currency0 = {{token1}} AND currency1 = {{token0}})
),
v3_mints AS (
    SELECT
        m.evt_block_time,
        m.evt_block_number,
        m.evt_index,
        m.contract_address  AS pool,
        'mint'              AS event_type,
        'uniswap_v3'        AS dex,
        m.tickLower         AS tick_lower,
        m.tickUpper         AS tick_upper,
        m.amount            AS liquidity_delta
    FROM uniswap_v3_{{chain}}.uniswapv3pool_evt_mint m
    INNER JOIN v3_pools p ON m.contract_address = p.pool
    WHERE m.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND m.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
      AND m.amount > 0
),
v3_burns AS (
    SELECT
        b.evt_block_time,
        b.evt_block_number,
        b.evt_index,
        b.contract_address  AS pool,
        'burn'              AS event_type,
        'uniswap_v3'        AS dex,
        b.tickLower         AS tick_lower,
        b.tickUpper         AS tick_upper,
        -b.amount           AS liquidity_delta
    FROM uniswap_v3_{{chain}}.uniswapv3pool_evt_burn b
    INNER JOIN v3_pools p ON b.contract_address = p.pool
    WHERE b.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND b.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
      AND b.amount > 0
),
v4_modify AS (
    SELECT
        ml.evt_block_time,
        ml.evt_block_number,
        ml.evt_index,
        ml.id               AS pool,
        CASE
            WHEN ml.liquidityDelta > 0 THEN 'mint'
            ELSE 'burn'
        END                 AS event_type,
        'uniswap_v4'        AS dex,
        ml.tickLower        AS tick_lower,
        ml.tickUpper        AS tick_upper,
        ml.liquidityDelta   AS liquidity_delta
    FROM uniswap_v4_{{chain}}.poolmanager_evt_modifyliquidity ml
    INNER JOIN v4_pools p ON ml.id = p.pool
    WHERE ml.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND ml.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
      AND ml.liquidityDelta != 0
),
all_events AS (
    SELECT * FROM v3_mints
    UNION ALL
    SELECT * FROM v3_burns
    UNION ALL
    SELECT * FROM v4_modify
)
SELECT
    MIN(evt_block_time)         AS evt_block_time,
    evt_block_number,
    MIN(evt_index)              AS evt_index,
    pool,
    dex,
    tick_lower,
    tick_upper,
    SUM(liquidity_delta)        AS liquidity_delta
FROM all_events
GROUP BY pool, dex, evt_block_number, tick_lower, tick_upper
HAVING SUM(liquidity_delta) != 0
