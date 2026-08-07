# Dune SQL queries

The versioned source of truth for every Dune query the pipeline runs. On the **Free**
plan the Query-Management (CRUD) API is unavailable, so these files are *not* pushed
automatically — the pipeline executes the queries **by id** (see `config.QUERIES` /
`config.*_QUERY_IDS`). Keep the two in sync manually: when you edit a `.sql` file, paste
it into the **same** saved query on Dune (do not create a new one — the id must not
change).

| file | logical name | Dune query id |
|------|--------------|---------------|
| `uniswap_swaps.sql`          | uniswap swaps          | 7727307 |
| `pancake_swaps.sql`          | pancake swaps          | 7727319 |
| `uniswap_liquidity.sql`      | uniswap mint/burn      | 7727404 |
| `pancake_liquidity.sql`      | pancake mint/burn      | 7727327 |
| `chain_gas_price.sql`        | per-block base fee/tip | 7748900 |
| `uniswap_gas_per_swap.sql`   | uniswap gas per block  | 7749258 |
| `pancake_gas_per_swap.sql`   | pancake gas per block  | 7749289 |

## Parameters

Each query is templated with Dune parameters substituted at execution:
`{{chain}}`, `{{start_ts}}`, `{{end_ts}}`, `{{token0}}`, `{{token1}}` (the pipeline sends
the superset; unused keys are dropped per query).

## Aggregation (credit saving)

The **swap** and **gas-per-swap** queries return **one row per (pool, block)** — the
end-of-block state via `MAX_BY(col, evt_index)` plus `nb_swaps = COUNT(*)` (swaps) and
`gas_price_max` / `gas_price_med` (gas). This is the server-side equivalent of
`preprocessing.keep_latest_swap_per_block`, so the download is far smaller while feeding
pool reconstruction identical inputs. Per-swap `amount0/1` are dropped, so the trade-size
distribution needs its own query.

> After changing a query's SQL on Dune, delete the local result cache
> (`<pair>/data_analysis/.dune_cache.json`) so re-runs don't reuse a stale execution of
> the old SQL.
