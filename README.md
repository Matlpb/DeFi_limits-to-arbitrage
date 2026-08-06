# DeFi cross-DEX arbitrage — limits to arbitrage

A panel study of **when** a cross-DEX arbitrage exists and **how large / how persistent** it is, for
one token pair (default WETH/USDC on Ethereum, Uniswap vs PancakeSwap), conditioned on CEX
**volatility regimes**. All logic lives in the `arblib` package; the notebooks are call-only drivers
run in a fixed order, each reading what the previous one wrote.

Everything is parameterised from **one place** — `arblib/config.py` (the `STUDY` instance): token
pair, chain, the classified year, the study-window length, the `active_regime`, Dune query ids, and
every derived path. Re-parametrise the study by editing `STUDY`, not the notebooks.

## Setup

> Requires **Python 3.13**. macOS `brew install python@3.13` · Ubuntu `sudo apt install python3.13 python3.13-venv`.

```bash
git clone https://github.com/Matlpb/DeFi_limits-to-arbitrage.git
cd DeFi_limits-to-arbitrage

python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt jupyter ipykernel

echo "DUNE_API_KEY=your_key_here" > .env   # needed only by 02_extract.ipynb
```

> Reactivate the venv each session with `source .venv/bin/activate`.

## Run order

Run the notebooks in this order — each consumes the previous step's output:

| # | notebook | does | reads → writes (under `<chain>/<pair>/`) |
|---|---|---|---|
| 1 | `01_market_regime_detection.ipynb` | pick one calm / medium / turbulent week from a year of Kraken prices | *(Kraken)* → `data_analysis/study_dates.csv` |
| 2 | `02_extract.ipynb` *(Dune)* | pull swaps / liquidity / gas / CEX prices for `active_regime`'s window (+ a warm-up lead) | `study_dates.csv` → `data_analysis/{swaps,liquidity,gas,prices}/` |
| 3 | `03_transform.ipynb` | reconstruct per-block pool state, MEV proxies, shared covariates | extracts → `data_analysis/processed/`, `modeling/covariates/common_covariates/` |
| 4 | `04_trade_sizes.ipynb` | pooled USD trade-size distribution → reference quantiles | swaps, prices → `data_analysis/trade_size_quantiles.csv` |
| 5 | `05_arbitrage_index.ipynb` | executable cross-pool round-trip gaps → dependent variable + pair covariates | processed, quantiles → `modeling/dependant_variable/`, `.../pool_pair_dependant_covariates/` |
| 6 | `06_model_existence` · `07_model_magnitude` · `08_model_persistence` · `09_model_survival` *(any order)* | the four panel models on the assembled panel | `modeling/*` → *(results in-notebook)* |

**Regimes.** Set `STUDY.active_regime` (`"low" \| "mid" \| "high"`) once in `config.py`; steps 2–3 use
it. Process one regime end-to-end (its data overwrites `data_analysis/`), then change `active_regime`
and repeat. Each extraction pulls a few hours *before* the window so price / liquidity are live and
the MEV / frequency EWMAs have converged by the first study block.

## `arblib` modules

**Data & IO** — `config` (the `STUDY` settings + all derived paths) · `dune_api` / `kraken_api` (REST
clients: on-chain events via Dune, CEX prices via Kraken) · `data_io` (generic load/save of the CSV
extracts, processed pools, quantiles).

**Features** (build the modeling-ready data) — `formulas` (AMM/pricing primitives, EWMA vol, MEV
proxies) · `preprocessing` (clean swaps → per-pool block series → active liquidity) · `analysis` /
`sizing` (cross-DEX mid gaps; USD trade-size distribution) · `arbitrage` (executable prices +
per-pool-pair arbitrage index) · `modeling` (writes the dependent-variable and covariate parquets).

**Models & selection** — `estimation` (assembles the pooled panel and fits the four models: FE
logit/probit, within-FE OLS, Cox, Weibull AFT, with cluster-robust SEs) · `regime` (CEX
volatility-regime detection and the study-window handoff) · `plotting`.

## Data layout (per pair, e.g. `ethereum/WETH_USDC/`)

```
data_analysis/   study_dates.csv, trade_size_quantiles.csv,
                 {swaps, liquidity, gas, prices, processed}/
modeling/        dependant_variable/*.parquet
                 covariates/{common_covariates, pool_pair_dependant_covariates}/*.parquet
```
