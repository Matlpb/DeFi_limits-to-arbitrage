# DeFi cross-DEX arbitrage study

Measure price gaps for one token pair across DEXes (Uniswap, PancakeSwap,
Aerodrome) on a chosen blockchain.

## Layout

```
extract_arbitrage.ipynb   # 1. pull swap prices from Dune  -> <chain>/*.csv
arbitrage.ipynb           # 2. clean -> reconstruct series -> cross-DEX gaps
arblib/                   # reusable library (all logic lives here)
  config.py               #   query ids, token registry, param helper
  dune_api.py             #   Dune REST client (execute / wait / fetch)
  data_io.py              #   save / load the raw per-DEX CSVs
  preprocessing.py        #   clean -> split -> filter -> reconstruct
  analysis.py             #   cross-DEX pairwise price differences
  plotting.py             #   price time-series plots
<chain>/                  # raw extracts, one CSV per DEX (e.g. base/, ethereum/)
```

The notebooks only set **parameters** and call `arblib` functions; every step
prints a short progress summary.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Matlpb/DeFi_limits-to-arbitrage.git
cd DeFi_limits-to-arbitrage

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file and add your Dune API key
cp .env.example .env             # then open .env and fill in your key

# 5. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=defi_arbitrage

# 6. Launch Jupyter
jupyter notebook
```

> Each new terminal session, reactivate the venv with `source .venv/bin/activate`.
> In the notebook, select kernel → **defi_arbitrage**.

## Workflow

1. Open **`extract_arbitrage.ipynb`** — set `CHAIN`, tokens and the collection
   window, then run all cells to save `<chain>/df_*.csv`.
2. Open **`arbitrage.ipynb`** — set `CHAIN`, `STUDY_START_TIME` and
   `MAX_GAP_BLOCKS`, then run all cells.

## The two time anchors

- **Collection window** (`START_TS`/`END_TS`, in the extract notebook) — what is
  pulled from Dune.
- **Study start** (`STUDY_START_TIME`, in the analysis notebook) — must be
  *later* than the collection start. The warm-up margin in between guarantees
  every pool already has a known price to forward-fill from before the study
  window begins; that margin is then trimmed off.
