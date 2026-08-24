# DeFi cross-DEX arbitrage — limits to arbitrage

A panel study of **when** a cross-DEX arbitrage exists, **how large** it is, and **how long** it
persists, for one token pair traded across several DEXes, conditioned on CEX **volatility regimes**.

All the logic lives in the **`arblib`** package; the numbered notebooks are thin, call-only drivers
that run it in order.

## What the library does

`arblib` takes the study from raw data to fitted models in three stages:

1. **Extract** — pull the on-chain data (DEX swaps, liquidity events, gas) from **Dune** and the CEX
   USD prices from **Kraken**, for one study window.
2. **Preprocess** — reconstruct a per-block state for every pool, build the covariates, and assemble
   the cross-DEX arbitrage panel (the dependent variable together with its predictors).
3. **Model** — fit the existence, magnitude, persistence, and survival models on that panel, and
   produce the summary and regression tables.

Everything is parameterised from **one place** — `arblib/config.py` (the `STUDY` instance): the token
pair, the chain, the classified year, the Dune query ids, and every derived output path. Re-run the
whole study for a different pair, chain, or volatility regime by editing `STUDY` — nothing else.

## Setup

> Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter ipykernel
echo "DUNE_API_KEY=your_key_here" > .env
```

You also need the **Dune queries** the pipeline runs. They are versioned as SQL in
[`arblib/sql/`](arblib/sql) (see its README for what each one returns). Create each as a saved query
in your Dune account and put its query id in `arblib/config.py` — the pipeline executes them by id.

## How to run

Run the notebooks in order; each reads what the previous one wrote:

1. `01_market_regime_detection.ipynb` — **automatically selects the study windows**: classifies a year
   of CEX volatility into calm / medium / turbulent and picks one representative week per regime.
2. `02_extract.ipynb` — extract the data for the active regime's window (Dune + Kraken).
3. `03_transform.ipynb` — reconstruct pool state and build the covariates.
4. `04_trade_sizes.ipynb` — the reference trade sizes.
5. `05_arbitrage_index.ipynb` — assemble the cross-DEX arbitrage panel.
6. `06`–`08` — the existence, magnitude, and persistence models.
7. `analysis.ipynb` — the cross-regime summary and regression tables. Compiles results of all regressions in compressed tables + generate the latex.

Which volatility regime the pipeline runs on is set once by `STUDY.active_regime` in `config.py`; each
regime writes to its own folder, so switching regime and re-running never overwrites the last.

## `arblib` modules

- **Data access** — `config` (all parameters), `dune_api` / `kraken_api` (REST clients), `data_io`.
- **Features** — `formulas`, `preprocessing`, `analysis`, `sizing`, `arbitrage`, `modeling` (the panel
  build).
- **Models** — `estimation`, `regime`, `summary`, `reporting`.
- **Support** — `naming`, `plotting`, and the Dune query sources in `sql/`.
