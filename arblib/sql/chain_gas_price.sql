WITH tx_tips AS (
    SELECT
        t.block_number,
        APPROX_PERCENTILE(t.gas_price - b.base_fee_per_gas, 0.5) AS tip_p50,
        APPROX_PERCENTILE(t.gas_price - b.base_fee_per_gas, 0.9) AS tip_p90
    FROM {{chain}}.transactions t
    INNER JOIN {{chain}}.blocks b ON b.number = t.block_number
    WHERE t.block_time >= CAST('{{start_ts}}' AS timestamp)
      AND t.block_time <  CAST('{{end_ts}}' AS timestamp)
      AND t.gas_price > 0
    GROUP BY t.block_number
)
SELECT
    b.time,
    b.number AS block_number,
    b.base_fee_per_gas,
    b.gas_used,
    b.gas_limit,
    tx.tip_p50,
    tx.tip_p90
FROM {{chain}}.blocks b
LEFT JOIN tx_tips tx ON tx.block_number = b.number
WHERE b.time >= CAST('{{start_ts}}' AS timestamp)
  AND b.time <  CAST('{{end_ts}}' AS timestamp)
