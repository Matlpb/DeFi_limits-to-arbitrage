# Dune SQL queries

The versioned source of truth for every Dune query the pipeline runs. On the **Free** plan the
Query-Management (CRUD) API is unavailable, so these files are *not* pushed automatically — the
pipeline executes the queries **by id** (`config.*_QUERY_IDS`). Keep the two in sync by hand: when you
edit a `.sql` file, paste it into the **same** saved query on Dune (do not create a new one — the id
must not change).

## What each query returns

| file | Dune query id | returns |
|------|---------------|---------|
| `uniswap_swaps.sql`        | 7727307 | Uniswap swaps for the pair (per-block pool price / liquidity state) |
| `pancake_swaps.sql`        | 7727319 | PancakeSwap swaps for the pair (per-block pool price / liquidity state) |
| `uniswap_liquidity.sql`    | 7727404 | Uniswap mint / burn liquidity events for the pair |
| `pancake_liquidity.sql`    | 7727327 | PancakeSwap mint / burn liquidity events for the pair |
| `chain_gas_price.sql`      | 7748900 | Per-block chain gas: base fee and priority-tip percentiles |
| `uniswap_gas_per_swap.sql` | 7749258 | Per-block gas price paid on Uniswap swap transactions |
| `pancake_gas_per_swap.sql` | 7749289 | Per-block gas price paid on PancakeSwap swap transactions |

## Parameters

Each query is templated with Dune parameters substituted at execution: `{{chain}}`, `{{start_ts}}`,
`{{end_ts}}`, `{{token0}}`, `{{token1}}`. The pipeline sends the superset; unused keys are dropped
per query.
