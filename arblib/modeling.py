"""
Modeling-ready transformed data + the panel existence logit.

Two halves:

* **Feature build** - turns the arbitrage index into the per-pool-pair dependent variable
  (continuous gap ``Gap_t(Q)`` and its existence ``D_t(Q) = 1[Gap_t(Q) > 0]``) and the pool-pair
  and shared covariates, each written to parquet under ``S.modeling_dir``.
* **Estimation** - reloads those parquets into one pooled panel (:func:`build_panel`), lags the
  predetermined regressors (:func:`prepare_model_frame`), and fits the pool-pair fixed-effects
  logit ``Pr(D_{p,t}=1 | X_{p,t-1}) = Lambda(eta_{p,t})`` with cluster-robust SEs
  (:func:`fit_existence_logit`) and average marginal effects (:func:`average_marginal_effects`).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import formulas
import statsmodels.formula.api as smf





EXISTENCE_TERMS = ["gap_lag", "log_base_fee",  "gas_util_lag", "tip_p90_lag",
                   "mev_lag", "freq_lag", "log_vol", "dlogL_lag"]




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


##########---------PANEL / ESTIMATION-----------###########


def load_parquet_dir(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load every ``<name>.parquet`` in ``directory`` into ``{name: frame}`` (sorted by name)."""
    directory = Path(directory)
    return {p.stem: pd.read_parquet(p) for p in sorted(directory.glob("*.parquet"))}


def build_panel(dep_dir: str | Path, pair_cov_dir: str | Path,
                chain_cov_path: str | Path, vol_path: str | Path) -> pd.DataFrame:
    """Assemble the pooled panel: one row per (pool pair, block), contemporaneous covariates.

    Joins, per pool pair, the dependent variable (``dep_dir``) and the pool-pair covariates
    (``pair_cov_dir``) on the block, stacks all pairs, then attaches the two shared tables:
    ``chain_covariates`` by exact ``block_number`` and ``CEX_volatility`` by the *no-look-ahead*
    minute rule - each block takes the EWMA vol of the last fully-closed 1-minute bar strictly
    before the block's own minute, i.e. ``floor(evt_block_time, 'min') - 1 min``. A 1-minute bar
    labelled T only closes at T+1min, so a block at 15:15:11 must use the 15:14:00 bar (complete
    at 15:15:00), a block at 15:16:xx the 15:15:00 bar, and so on. Returns the panel sorted by
    (``pair``, ``evt_block_number``) with a tz-aware datetime ``evt_block_time`` and a ``vol_minute``
    audit column; lags are applied later in :func:`prepare_model_frame`.
    """
    dv = load_parquet_dir(dep_dir)
    pc = load_parquet_dir(pair_cov_dir)

    frames = []
    for pair, d in dv.items():
        d = d.rename(columns={"blocknumber": "evt_block_number"})
        merged = d.merge(pc[pair].drop(columns=["evt_block_time"]), on="evt_block_number")
        merged.insert(0, "pair", pair)
        frames.append(merged)
    panel = pd.concat(frames, ignore_index=True)
    panel["evt_block_time"] = pd.to_datetime(panel["evt_block_time"], utc=True)

    chain = pd.read_parquet(chain_cov_path).drop(columns=["time"])
    panel = panel.merge(chain, left_on="evt_block_number", right_on="block_number",
                        how="left").drop(columns=["block_number"])

    vol = pd.read_parquet(vol_path).rename(columns={"time": "vol_minute"})
    panel["vol_minute"] = panel["evt_block_time"].dt.floor("min") - pd.Timedelta(minutes=1)
    panel = panel.merge(vol, on="vol_minute", how="left")

    return panel.sort_values(["pair", "evt_block_number"]).reset_index(drop=True)


def prepare_model_frame(panel: pd.DataFrame, quantile: float,
                        drop_constant_pairs: bool = True) -> pd.DataFrame:
    """Build the model-ready frame for one trade-size quantile: outcome ``D_t`` and predetermined ``X``.

    Lags the pair-specific and ``t-1`` chain regressors by one block *within* each pool pair (so no
    value leaks across pairs), while keeping ``log(base_fee)`` and ``log(ewma_vol)`` contemporaneous
    at ``t`` exactly as the equation's subscripts specify (the vol is already backward-looking via
    the minute rule in :func:`build_panel`). Derives hour / day-of-week / iso-week from the block
    time for the calendar fixed effects. Rows with any missing lag are dropped: the first block of
    every pair (all lags), and the second block for ``dlogL_lag`` (its growth needs ``L_{t-2}``).
    With ``drop_constant_pairs`` (default) pool pairs whose ``D`` never varies in-sample are removed
    - under pair fixed effects they are uninformative and would cause perfect separation.

    Columns: ``pair, evt_block_number, evt_block_time, D, gap_lag, log_base_fee, gas_util_lag,
    tip_p90_lag, mev_lag, freq_lag, log_vol, dlogL_lag, hour, dow, week``.
    """
    q = f"q{int(round(quantile * 100))}"
    g = panel.groupby("pair", sort=False)
    out = pd.DataFrame({
        "pair": panel["pair"],
        "evt_block_number": panel["evt_block_number"],
        "evt_block_time": panel["evt_block_time"],
        "D": panel[f"D_{q}"].astype(float),
        "gap_lag": g[f"gap_{q}"].shift(1),
        "log_base_fee": panel["log_base_fee_per_gas"],
        "gas_util_lag": g["gas_util"].shift(1),
        "tip_p90_lag": g["log1p_tip_p90"].shift(1),
        "mev_lag": g["mev_intensity"].shift(1),
        "freq_lag": g["frequency_intensity"].shift(1),
        "log_vol": np.log(panel["ewma_vol"]),

        "dlogL_lag": g["per_log_liquidity_growth_rate_avg_venue"].shift(1),
    })
    out["hour"] = out["evt_block_time"].dt.hour
    out["dow"] = out["evt_block_time"].dt.dayofweek
    out["week"] = out["evt_block_time"].dt.isocalendar().week.astype(int)
    out = out.dropna().reset_index(drop=True)

    if drop_constant_pairs:
        varies = out.groupby("pair")["D"].transform("nunique") > 1
        dropped = sorted(set(out.loc[~varies, "pair"]))
        if dropped:
            print(f"q{q[1:]}: dropping {len(dropped)} pool pairs with no D variation "
                  f"(uninformative under pair FE): {dropped}")
        out = out.loc[varies].reset_index(drop=True)

    return out

def build_risk_set(panel: pd.DataFrame, quantile: float, condition: str) -> pd.DataFrame:
    """Build a model-ready frame restricted to one existence-equation risk set.

    ``condition`` selects the hazard: ``"closure"`` keeps ``gap_lag > 0`` (given a gap is
    open, does it survive?), ``"onset"`` keeps ``gap_lag == 0`` (given no gap is open, does
    one appear?). After filtering, re-checks for pool pairs with no ``D`` variation *within
    this specific subsample* (a pair can vary in the full panel yet be constant once
    restricted) and drops them, since they are uninformative and would cause separation
    under pair fixed effects.
    """
    df = prepare_model_frame(panel, quantile=quantile)
    df = df.loc[df.gap_lag > 0] if condition == "closure" else df.loc[df.gap_lag == 0]
    df = df.reset_index(drop=True)

    varies = df.groupby("pair")["D"].transform("nunique") > 1
    dropped = sorted(set(df.loc[~varies, "pair"]))
    if dropped:
        print(f"{condition}: dropping {len(dropped)} pairs with no D variation: {dropped}")
    df = df.loc[varies].reset_index(drop=True)

    print(f"{condition}: {df.shape} | D mean: {round(df.D.mean(), 4)} | pairs: {df.pair.nunique()}")
    return df


def fit_hazard_logit(model_df: pd.DataFrame, terms: Sequence[str],
                      cluster: str | Sequence[str] = "pair", maxiter: int = 300,
                      direction: str = "logit"):
    """Fit a pooled hazard logit (existence-stage, either risk set) with pool-pair (and,
    when identified, calendar) fixed effects.

    Regresses ``D`` on the supplied ``terms`` plus ``C(pair)`` fixed effects, adding
    ``C(hour)`` / ``C(dow)`` / ``C(week)`` only when the sample spans more than one level.
    Used for both the closure hazard (risk set: gap_lag > 0, terms include gap_lag) and the
    onset hazard (risk set: gap_lag == 0, terms exclude gap_lag - it is constant at 0 on
    this subsample and would be collinear with the intercept). Standard errors are
    cluster-robust: cluster="pair" for one-way, cluster=["pair","evt_block_number"] for the
    two-way (pair x block) robustness check.
    """

    all_terms = list(terms) + ["C(pair)"]
    for fe, col in [("C(hour)", "hour"), ("C(dow)", "dow"), ("C(week)", "week")]:
        if model_df[col].nunique() > 1:
            all_terms.append(fe)
    formula = "D ~ " + " + ".join(all_terms)

    cols = [cluster] if isinstance(cluster, str) else list(cluster)
    codes = np.column_stack([pd.factorize(model_df[c])[0] for c in cols])
    groups = codes[:, 0] if len(cols) == 1 else codes

    if direction == "probit":
        return smf.probit(formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, maxiter=maxiter, disp=False)

    if direction == "logit":
        return smf.logit(formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, maxiter=maxiter, disp=False)

    raise ValueError(f"Unknown direction: {direction!r}")



def average_marginal_effects(result, terms: Sequence[str] | None = None) -> pd.DataFrame:
    """Average marginal effects (``AME = mean_i dPr(D=1)/dX_k``) with the fitted (clustered) SEs.

    Raw logit coefficients are not probability changes, so report AME - the sample average of the
    per-observation marginal effect (``at="overall"``). Returns the tidy ``summary_frame`` of
    ``result.get_margeff``; if ``terms`` is given (e.g. :data:`EXISTENCE_TERMS`), restricts to those
    regressors, dropping the fixed-effect dummies."""
    frame = result.get_margeff(at="overall").summary_frame()
    if terms is not None:
        frame = frame.loc[[t for t in terms if t in frame.index]]
    return frame
