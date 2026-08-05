"""
Feature build - the modeling-ready parquets.

Turns the arbitrage index into the per-pool-pair dependent variable (continuous gap ``Gap_t(Q)``
and its existence ``D_t(Q) = 1[Gap_t(Q) > 0]``) and the pool-pair and shared covariates, each
written to parquet under ``S.modeling_dir``. The estimation layer that reads these back into a
pooled panel and fits the models lives in :mod:`arblib.estimation`.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import formulas


def dependent_variables(arb_index: dict[str, pd.DataFrame],
                        pools: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build the per-pool-pair dependent-variable table from the arbitrage index.

    ``arb_index`` maps a pair to its blocks x quantiles ``Gap_t(Q)`` frame (bps, from
    :func:`arblib.arbitrage.all_pairs_gap_series`); ``pools`` supplies the block -> time map
    (all processed pools share the block grid). For each pair returns a frame with
    ``evt_block_time``, ``blocknumber``, ``gap_q20 ... gap_q80`` (continuous Gap, bps) and
    ``D_q20 ... D_q80`` (``1[gap > 0]``). Gap is kept continuous so both the dependent
    ``D_t`` and the lagged regressor ``Gap_{t-1}`` derive from the same column.
    """
    block_time = (next(iter(pools.values()))
                  .drop_duplicates("evt_block_number")
                  .set_index("evt_block_number")["evt_block_time"])

    out = {}
    for pair, frame in arb_index.items():
        gap_cols = {q: f"gap_q{int(round(q * 100))}" for q in frame.columns}
        df = frame.rename(columns=gap_cols).reset_index().rename(columns={"block": "blocknumber"})
        df["evt_block_time"] = df["blocknumber"].map(block_time)
        for name in gap_cols.values():
            df["D_q" + name.removeprefix("gap_q")] = (df[name] > 0).astype(int)
        ordered = (["evt_block_time", "blocknumber"]
                   + list(gap_cols.values())
                   + ["D_q" + n.removeprefix("gap_q") for n in gap_cols.values()])
        out[pair] = df[ordered]
    return out


def save_dependent_variables(arb_index: dict[str, pd.DataFrame],
                             pools: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write one dependent-variable parquet per pool pair to ``out_dir`` (:func:`dependent_variables`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = dependent_variables(arb_index, pools)
    for pair, df in frames.items():
        df.to_parquet(out_dir / f"{pair}.parquet", index=False)
    print(f"Saved {len(frames)} dependent-variable parquets to {out_dir}")


def cex_volatility(x_usd: pd.DataFrame, y_usd: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """CEX-implied EWMA volatility of the token0/token1 rate, as ``[time, ewma_vol]``.

    Thin wrapper over :func:`arblib.formulas.vol_ewma` keeping only the two columns the panel
    needs: the minute ``time`` and the RiskMetrics per-minute ``ewma_vol``. Common to every pool
    pair (joined to blocks by a backward merge on ``time`` at panel-assembly time)."""
    return formulas.vol_ewma(x_usd, y_usd, horizon_min)[["time", "ewma_vol"]]


def chain_covariates(chain_gas: pd.DataFrame) -> pd.DataFrame:
    """Per-block chain covariates shared by every pool pair, in model-ready form.

    From the chain-wide gas table returns ``[block_number, time, log_base_fee_per_gas, gas_util,
    log1p_tip_p50, log1p_tip_p90]``: the base fee in logs, block fullness ``gas_used / gas_limit``
    in ``[0, 1]``, and the per-block priority-tip p50 / p90 as ``log(1 + tip)`` (tips are
    right-skewed and can be 0). ``time`` is parsed to UTC. Joined to the panel by exact
    ``block_number``."""
    return pd.DataFrame({
        "block_number": chain_gas["block_number"],
        "time": pd.to_datetime(chain_gas["time"], utc=True),
        "log_base_fee_per_gas": np.log(chain_gas["base_fee_per_gas"].astype(float)),
        "gas_util": chain_gas["gas_used"].astype(float) / chain_gas["gas_limit"].astype(float),
        "log1p_tip_p50": np.log1p(chain_gas["tip_p50"].astype(float)),
        "log1p_tip_p90": np.log1p(chain_gas["tip_p90"].astype(float)),
    })


def save_common_covariates(x_usd: pd.DataFrame, y_usd: pd.DataFrame, chain_gas: pd.DataFrame,
                           out_dir: str | Path, vol_horizon_min: int) -> None:
    """Write the pool-independent covariate parquets to ``out_dir``.

    ``CEX_volatility.parquet`` (:func:`cex_volatility`) and ``chain_covariates.parquet``
    (:func:`chain_covariates`) - the covariates every pool pair shares, stored once."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cex_volatility(x_usd, y_usd, vol_horizon_min).to_parquet(out_dir / "CEX_volatility.parquet", index=False)
    chain_covariates(chain_gas).to_parquet(out_dir / "chain_covariates.parquet", index=False)
    print(f"Saved common covariates (CEX_volatility, chain_covariates) to {out_dir}")


def _dlog(liquidity: pd.Series) -> np.ndarray:
    """Per-block active-liquidity log-growth ``log(L_t / L_{t-1})``; NaN on the first row.

    The processed series is dense on the block grid (gap-free reconstruction), so consecutive
    rows are consecutive blocks and this is a true one-block growth rate."""
    return np.diff(np.log(liquidity.astype(float).to_numpy()), prepend=np.nan)


def pair_covariates(pools: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build the per-pool-pair covariate table for every unordered pool pair.

    For each pair (venue 1, venue 2), aligned on ``evt_block_number`` (same names / order as
    :func:`arblib.arbitrage.all_pairs_gap_series`, so the files key identically to the dependent
    variable), returns ``[evt_block_number, evt_block_time, mev_intensity, frequency_intensity,
    per_log_liquidity_growth_rate_avg_venue]``. All three are venue-averaged and contemporaneous
    at block t (lag downstream to make them predetermined, as with the dependent variable):

    * ``mev_intensity`` = 1/2 (log(1 + mev_intensity_1) + log(1 + mev_intensity_2)) - the top-tip
      MEV proxy in logs (it is a wei magnitude), averaged over the two venues.
    * ``frequency_intensity`` = 1/2 (nb_swaps_ewma_1 + nb_swaps_ewma_2) - the order-flow /
      contest-frequency EWMA, averaged over the two venues.
    * ``per_log_liquidity_growth_rate_avg_venue`` = 1/2 (dlog L_1 + dlog L_2) with
      dlog L = log(L_t / L_{t-1}) (:func:`_dlog`) - per-venue active-liquidity log-growth,
      averaged; NaN on the first block. Its one-block lag is the model's
      ``mean_dlog_L_{p,t-1}``.
    """
    out = {}
    for n1, n2 in combinations(pools, 2):
        m = pools[n1].merge(pools[n2], on="evt_block_number", suffixes=("_1", "_2"))
        out[f"{n1}_vs_{n2}"] = pd.DataFrame({
            "evt_block_number": m["evt_block_number"],
            "evt_block_time": m["evt_block_time_1"],
            "mev_intensity": 0.5 * (np.log1p(m["mev_intensity_1"].astype(float))
                                    + np.log1p(m["mev_intensity_2"].astype(float))),
            "frequency_intensity": 0.5 * (m["nb_swaps_ewma_1"].astype(float)
                                          + m["nb_swaps_ewma_2"].astype(float)),
            "per_log_liquidity_growth_rate_avg_venue": 0.5 * (_dlog(m["liquidity_1"])
                                                              + _dlog(m["liquidity_2"])),
        })
    return out


def save_pair_covariates(pools: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write one pool-pair covariate parquet per pair to ``out_dir`` (:func:`pair_covariates`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = pair_covariates(pools)
    for pair, df in frames.items():
        df.to_parquet(out_dir / f"{pair}.parquet", index=False)
    print(f"Saved {len(frames)} pool-pair covariate parquets to {out_dir}")
