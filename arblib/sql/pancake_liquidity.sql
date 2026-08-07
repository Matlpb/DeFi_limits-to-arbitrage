WITH target_pools AS (
    SELECT
        pool,
        token0,
        token1,
        tickSpacing
    FROM pancakeswap_v3_{{chain}}.pancakev3factory_evt_poolcreated
    WHERE (token0 = {{token0}} AND token1 = {{token1}})
       OR (token0 = {{token1}} AND token1 = {{token0}})
),
mints AS (
    SELECT
        m.evt_block_time,
        m.evt_block_number,
        m.evt_index,
        m.contract_address  AS pool,
        'mint'              AS event_type,
        m.tickLower         AS tick_lower,
        m.tickUpper         AS tick_upper,
        m.amount            AS liquidity_delta
    FROM pancakeswap_v3_{{chain}}.pancakev3pool_evt_mint m
    INNER JOIN target_pools p ON m.contract_address = p.pool
    WHERE m.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND m.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
      AND m.amount > 0
),
burns AS (
    SELECT
        b.evt_block_time,
        b.evt_block_number,
        b.evt_index,
        b.contract_address  AS pool,
        'burn'              AS event_type,
        b.tickLower         AS tick_lower,
        b.tickUpper         AS tick_upper,
        -b.amount           AS liquidity_delta
    FROM pancakeswap_v3_{{chain}}.pancakev3pool_evt_burn b
    INNER JOIN target_pools p ON b.contract_address = p.pool
    WHERE b.evt_block_time >= CAST('{{start_ts}}' AS timestamp)
      AND b.evt_block_time <  CAST('{{end_ts}}' AS timestamp)
      AND b.amount > 0
)
SELECT * FROM mints
UNION ALL
SELECT * FROM burns
