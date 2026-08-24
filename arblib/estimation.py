"""
Panel assembly + the block-level and spell-level models.

Reads the modeling-ready parquets written by :mod:`arblib.modeling` into one pooled panel
(:func:`load_panel` / :func:`build_panel`, which also stamp canonical spell columns), lags the
predetermined regressors (:func:`prepare_model_frame`), carves the sample each specification needs
(:func:`build_risk_set` for the onset / closure hazards, :func:`build_magnitude_sample`,
:func:`build_spell_table` / :func:`build_survival_sample` for the spell-level survival models), and
fits them with pool-pair fixed effects and cluster-robust SEs (:func:`fit_hazard_logit`,
:func:`fit_magnitude_panel_ols`), plus post-estimation diagnostics (:func:`variance_inflation`,
:func:`separation_report`). The parquet feature build lives in :mod:`arblib.modeling`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from . import naming


# Regressor sets, shared by the notebooks so every specification reads from one place.
EXISTENCE_TERMS = ["gap_lag", "log_base_fee", "gas_util_lag", "tip_p90_lag",
                   "mev_lag", "freq_lag", "log_vol"]
ONSET_TERMS = [t for t in EXISTENCE_TERMS if t != "gap_lag"]   # gap_lag == 0 on the onset risk set
CLOSURE_TERMS = ["log1p_gap_lag", *ONSET_TERMS]                # closure hazard on gap_lag > 0; open-gap size in logs
CLOSURE_DURATION_TERMS = CLOSURE_TERMS + ["spell_duration"]    # + elapsed spell age (Wooldridge Ch. 20)
MAGNITUDE_TERMS = ["log1p_gap_lag", "log_base_fee", "gas_util_lag", "tip_p90_lag",
                   "mev_lag", "freq_lag", "log_vol"]
# Spell-level survival baseline (Cox / AFT). No spell_duration: it is the outcome / time axis, not a
# regressor (that role is the discrete-time hazard logit's). Measured pre-onset - see build_survival_sample.
SURVIVAL_TERMS = ["log_base_fee", "gas_util", "tip_p90", "mev_intensity", "freq", "log_vol"]


##########---------PANEL ASSEMBLY-----------###########


def load_parquet_dir(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load every ``<name>.parquet`` in ``directory`` into ``{name: frame}`` (sorted by name)."""
    directory = Path(directory)
    return {p.stem: pd.read_parquet(p) for p in sorted(directory.glob("*.parquet"))}


def add_spell_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add contemporaneous ``spell_id`` / ``spell_duration`` for every gap quantile present.

    For each ``gap_q{XX}`` column a spell is a maximal run of consecutive blocks (within a pool
    pair) with ``gap_q{XX} > 0`` - the same definition used everywhere downstream, so existence,
    magnitude and closure all share one canonical spell boundary per quantile instead of each
    recomputing its own. Adds, per quantile suffix:

    * ``spell_id_q{XX}`` - a string unique across the whole panel (e.g. ``"uniswap_1_vs_pancake_1_q20_3"``),
      null when the block is not in a spell;
    * ``spell_duration_q{XX}`` - 1, 2, 3... within the spell, 0 when not in a spell.

    Both are CONTEMPORANEOUS (age at row t includes block t). :func:`prepare_model_frame` lags
    ``spell_duration`` by one block to get the predetermined age used as a regressor, and derives
    the at-risk ``spell_id`` for the closure sample.
    """
    for gap_col in [c for c in panel.columns if c.startswith("gap_q")]:
        suffix = gap_col.removeprefix("gap_")
        is_open = (panel[gap_col] > 0).astype(int)
        changed = is_open != is_open.groupby(panel["pair"]).shift(1).fillna(0)
        spell_num = changed.groupby(panel["pair"]).cumsum()

        panel[f"spell_duration_{suffix}"] = (
            panel.groupby(["pair", spell_num]).cumcount() + 1).where(is_open == 1, 0)
        panel[f"spell_id_{suffix}"] = np.where(
            is_open == 1, panel["pair"].astype(str) + f"_{suffix}_" + spell_num.astype(str), None)
    return panel


def build_panel(dep_dir: str | Path, pair_cov_dir: str | Path,
                chain_cov_path: str | Path, vol_path: str | Path) -> pd.DataFrame:
    """Assemble the pooled panel: one row per (pool pair, block), contemporaneous covariates.

    Joins, per pool pair, the dependent variable (``dep_dir``) and the pool-pair covariates
    (``pair_cov_dir``) on the block, stacks all pairs, then attaches the two shared tables:
    ``chain_covariates`` by exact ``block_number`` and ``CEX_volatility`` by the *no-look-ahead*
    minute rule - each block takes the EWMA vol of the last fully-closed 1-minute bar strictly
    before the block's own minute, i.e. ``floor(evt_block_time, 'min') - 1 min``. A 1-minute bar
    labelled T only closes at T+1min, so a block at 15:15:11 must use the 15:14:00 bar, a block at
    15:16:xx the 15:15:00 bar, and so on. Finally stamps canonical spell columns
    (:func:`add_spell_columns`). Returns the panel sorted by (``pair``, ``evt_block_number``) with a
    tz-aware ``evt_block_time`` and a ``vol_minute`` audit column; lags are applied later in
    :func:`prepare_model_frame`. Notebooks usually call :func:`load_panel` instead.
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

    panel = panel.sort_values(["pair", "evt_block_number"]).reset_index(drop=True)
    return add_spell_columns(panel)


def load_panel(settings) -> pd.DataFrame:
    """Assemble the pooled panel from the parquets a :class:`~arblib.config.Settings` points at.

    Thin convenience over :func:`build_panel` so every model notebook builds the panel identically
    with one call: ``panel = estimation.load_panel(S)``."""
    return build_panel(settings.dependent_var_dir, settings.pair_covariates_dir,
                       settings.chain_covariates_path, settings.cex_vol_path)


def prepare_model_frame(panel: pd.DataFrame, quantile: float,
                        drop_constant_pairs: bool = True) -> pd.DataFrame:
    """Build the per-quantile model frame: outcome ``D_t``, predetermined ``X``, and spell columns.

    Lags the pair-specific and ``t-1`` chain regressors by one block *within* each pool pair (so no
    value leaks across pairs), keeps ``log(base_fee)`` and ``log(ewma_vol)`` contemporaneous at
    ``t`` (the vol is already backward-looking via the minute rule in :func:`build_panel`), and
    derives the hour-of-day for the time-of-day fixed effect (day-of-week and iso-week are omitted as
    near-collinear and low-signal on a sub-week window), and forms ``log1p_gap_lag`` (the open-gap
    size in logs) for the closure and magnitude equations. Also carries, from the
    canonical spell columns:

    * ``spell_duration`` - the predetermined spell age (contemporaneous age lagged one block; 0 when
      no spell was open at ``t-1``), the closure hazard's duration-dependence regressor;
    * ``spell_id`` - contemporaneous spell membership at ``t`` (null when the block is not in a
      spell), the cluster key for the within-spell two-way SE. :func:`build_risk_set` re-lags this
      to the at-risk spell for the closure set.

    Rows with any missing lag are dropped (the first block of every pair). With
    ``drop_constant_pairs`` (default) pool pairs whose ``D`` never
    varies in-sample are removed - uninformative under pair fixed effects and a source of
    separation.
    """
    q = naming.qlabel(quantile)
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
        "spell_duration": g[f"spell_duration_{q}"].shift(1),
    })
    out["log1p_gap_lag"] = np.log1p(out["gap_lag"])   # open-gap size in logs (closure / magnitude regressor)
    out["hour"] = out["evt_block_time"].dt.hour
    out = out.dropna().reset_index(drop=True)

    # contemporaneous spell membership, merged AFTER dropna so its nulls (closed blocks) don't wipe
    # the sample; it is a label, not a regressor requiring completeness.
    spell = panel[["pair", "evt_block_number", f"spell_id_{q}"]].rename(
        columns={f"spell_id_{q}": "spell_id"})
    out = out.merge(spell, on=["pair", "evt_block_number"], how="left")

    if drop_constant_pairs:
        varies = out.groupby("pair")["D"].transform("nunique") > 1
        dropped = sorted(set(out.loc[~varies, "pair"]))
        if dropped:
            print(f"q{q[1:]}: dropping {len(dropped)} pool pairs with no D variation "
                  f"(uninformative under pair FE): {dropped}")
        out = out.loc[varies].reset_index(drop=True)

    return out


def drop_uninformative_pairs(df: pd.DataFrame, outcome_col: str, min_obs: int,
                             require_variance: bool, label: str = "") -> pd.DataFrame:
    """Drop pool pairs that can't support a pair fixed effect on ``outcome_col``.

    Always requires at least ``min_obs`` rows per pair. When ``require_variance``, additionally
    requires the outcome to take more than one distinct value within the pair - mandatory for a
    binary outcome under FE (a zero-variance entity causes separation, not just imprecision);
    redundant but harmless for a continuous outcome once ``min_obs`` screens out near-degenerate
    pairs.
    """
    counts = df.groupby("pair")[outcome_col].transform("size")
    keep = counts >= min_obs
    if require_variance:
        keep &= df.groupby("pair")[outcome_col].transform("nunique") > 1

    dropped = sorted(set(df.loc[~keep, "pair"]))
    if dropped:
        print(f"{label}: dropping {len(dropped)} pairs: {dropped}")
    return df.loc[keep].reset_index(drop=True)


def drop_pairs_by_event_floor(df: pd.DataFrame, floor: int, outcome_col: str = "D",
                              label: str = "") -> pd.DataFrame:
    """Drop pool pairs whose **rarer outcome class** has fewer than ``floor`` occurrences.

    An events-per-variable (EPV) screen for the logistic risk sets: separation and coefficient
    instability are driven by the minority class, not by "events" under a fixed label. So per pair we
    count *both* outcomes and keep the pair only if ``min(n0, n1) >= floor``. This is direction-free:
    for the onset set ``D == 1`` (an arb appears) is usually the rarer class, but for the closure set,
    with short spells, survival (``D == 1``) can be rarer than closure (``D == 0``) - the binding side
    is computed per pair rather than assumed. For the magnitude OLS the analogue is a total-row floor
    (:func:`drop_uninformative_pairs` with ``require_variance=False``). Returns the filtered frame.
    """
    n1 = df.groupby("pair")[outcome_col].sum()
    n0 = df.groupby("pair")[outcome_col].size() - n1
    rarer = pd.concat([n0, n1], axis=1).min(axis=1)
    keep_pairs = rarer.index[rarer >= floor]

    dropped = sorted(set(df["pair"]) - set(keep_pairs))
    if dropped:
        print(f"{label}: dropping {len(dropped)} pairs with < {floor} rarer-class events "
              f"(min(n_D0, n_D1)): {dropped}")
    out = df.loc[df["pair"].isin(keep_pairs)].reset_index(drop=True)
    print(f"{label}: kept {out.pair.nunique()} pairs, {len(out)} rows")
    return out


def build_risk_set(panel: pd.DataFrame, quantile: float, condition: str,
                   min_obs: int = 2) -> pd.DataFrame:
    """Existence-stage risk set for one trade size: the onset or the closure hazard sample.

    From the per-quantile model frame (:func:`prepare_model_frame`), keeps one at-risk interval per
    row:

    * ``condition="onset"``  - rows with ``gap_lag == 0`` (no gap open at ``t-1``): "does one appear
      at ``t``?" (spell birth). No spell is open, so ``spell_duration`` is 0 and ``spell_id`` is null.
    * ``condition="closure"`` - rows with ``gap_lag > 0`` (a gap is open at ``t-1``): "does it survive
      block ``t``?" (spell survival). This is the discrete-time grouped hazard behind the
      persistence model; ``spell_duration`` is the predetermined spell age and ``spell_id`` is
      re-lagged to the *ongoing* spell (the one open at ``t-1``), so its consecutive within-spell
      rows - including the closing block - share a cluster.

    Pairs that can't support a pair fixed effect on ``D`` in the risk set are then dropped.
    """
    q = naming.qlabel(quantile)
    df = prepare_model_frame(panel, quantile=quantile)
    if condition == "closure":
        # at-risk spell = the spell open at t-1; taken from the full panel (block t-1 is always
        # present there, unlike in the warmup-trimmed frame) so no closure row is left without one.
        prev = (panel[["pair", "evt_block_number", f"spell_id_{q}"]]
                .rename(columns={f"spell_id_{q}": "spell_id"})
                .assign(evt_block_number=lambda t: t["evt_block_number"] + 1))
        df = df.drop(columns="spell_id").merge(prev, on=["pair", "evt_block_number"], how="left")
        df = df.loc[df.gap_lag > 0].reset_index(drop=True)
    elif condition == "onset":
        df = df.loc[df.gap_lag == 0].reset_index(drop=True)
    else:
        raise ValueError(f"condition must be 'onset' or 'closure', got {condition!r}")

    df = drop_uninformative_pairs(df, "D", min_obs=min_obs, require_variance=True, label=condition)
    extra = f" | spell_duration max: {int(df.spell_duration.max())}" if condition == "closure" else ""
    print(f"{condition}: {df.shape} | D mean: {round(df.D.mean(), 4)} | pairs: {df.pair.nunique()}{extra}")
    return df


def build_magnitude_sample(panel: pd.DataFrame, quantile: float,
                           min_obs_per_pair=0) -> pd.DataFrame:
    """Magnitude-equation sample: OLS on ``log(Gap_t)`` conditional on existence (``D == 1``).

    Reuses :func:`prepare_model_frame` for every predetermined covariate (same lag discipline, and
    it already carries ``spell_id`` / ``spell_duration``), joins back the raw contemporaneous gap as
    the outcome, and restricts to ``D == 1`` (so ``Gap_t > 0`` and plain ``log_gap`` is defined). The
    lagged gap enters as ``log1p_gap_lag`` since it can be exactly 0 (the first block of a
    freshly-opened spell). Every row belongs to a spell, so its contemporaneous ``spell_id`` is the
    cluster key for the within-spell two-way robustness. Pairs with fewer than ``min_obs_per_pair``
    rows are dropped (a pair FE off a handful of points is not worth keeping).
    """
    q = naming.qlabel(quantile)
    df = prepare_model_frame(panel, quantile=quantile)
    gap_t = panel[["pair", "evt_block_number", f"gap_{q}"]].rename(columns={f"gap_{q}": "gap_t"})
    df = df.merge(gap_t, on=["pair", "evt_block_number"], how="left")

    df = df.loc[df.D == 1].reset_index(drop=True)
    df["log_gap"] = np.log(df["gap_t"])   # log1p_gap_lag comes from prepare_model_frame

    df = drop_uninformative_pairs(df, "log_gap", min_obs=min_obs_per_pair,
                                  require_variance=False, label="magnitude")
    print(f"magnitude: {df.shape} | mean log_gap: {round(df.log_gap.mean(), 4)} | "
          f"pairs: {df.pair.nunique()}")
    return df


def build_spell_table(panel: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Reshape the block-level panel into one row per arbitrage spell, for survival analysis.

    The closure hazard is a *block-level* at-risk table (one row per open block); a Kaplan-Meier /
    lifelines model instead takes the *spell* as the unit. Reusing the canonical spell columns
    stamped by :func:`add_spell_columns` (a spell = a maximal run of consecutive ``Gap_t(Q) > 0``
    blocks within a pool pair), this collapses each ``spell_id`` to one row with:

    * ``duration`` - the spell length in blocks (= its final ``spell_duration``);
    * ``observed`` - lifelines' ``event_observed`` coding: 1 if closure is observed (the run is
      followed by a ``Gap == 0`` block), 0 if the spell is right-censored because its last block is
      the pair's last block in the window (true length unknown);
    * ``pair`` / ``onset_block`` / ``last_block`` - identifiers;
    * ``left_truncated`` - True if the spell was already open at the pair's first observed block (its
      onset predates the window) - a minor caveat on a short window.

    Descriptive (no fixed effects, no pair screening), so it keeps every spell across all pairs -
    unlike the FE closure hazard, which drops pairs with no within-pair ``D`` variation.
    """
    q = naming.qlabel(quantile)
    id_col, dur_col = f"spell_id_{q}", f"spell_duration_{q}"
    pair_first = panel.groupby("pair")["evt_block_number"].min()
    pair_last = panel.groupby("pair")["evt_block_number"].max()

    open_rows = panel.loc[panel[id_col].notna(), ["pair", "evt_block_number", dur_col, id_col]]
    tbl = (open_rows.groupby(id_col, sort=False)
           .agg(pair=("pair", "first"),
                onset_block=("evt_block_number", "min"),
                last_block=("evt_block_number", "max"),
                duration=(dur_col, "max"))
           .reset_index(drop=True))
    tbl["duration"] = tbl["duration"].astype(int)
    tbl["observed"] = (tbl["last_block"] < tbl["pair"].map(pair_last)).astype(int)
    tbl["left_truncated"] = tbl["onset_block"].eq(tbl["pair"].map(pair_first))

    print(f"spells: {len(tbl)} | closed: {int(tbl.observed.sum())} | "
          f"censored: {int((tbl.observed == 0).sum())} | left-truncated: {int(tbl.left_truncated.sum())} "
          f"| median duration: {tbl.duration.median()}")
    return tbl


def build_survival_sample(panel: pd.DataFrame, quantile: float,
                          drop_left_truncated: bool = True) -> pd.DataFrame:
    """One row per spell for a Cox / AFT survival model: ``duration``, ``event``, baseline covariates.

    The unit is the spell. :func:`build_spell_table` supplies ``duration`` (the time axis), the
    right-censoring flag ``event`` and ``left_truncated``; this adds ``spell_id`` and the baseline
    covariates in :data:`SURVIVAL_TERMS`, each anchored so it is predetermined relative to the spell
    it describes:

    * ``log_base_fee, gas_util, tip_p90, mev_intensity, freq`` - measured at the block *before*
      onset (``onset_block - 1``), the spell-level analogue of the block equations' ``_lag``
      convention (all are shaped by trading activity, so the pre-onset block keeps them strictly
      predetermined);
    * ``log_vol`` - measured at the onset block itself: ``ewma_vol`` is already backward-looking by
      construction (the no-look-ahead minute rule in :func:`build_panel`), so no extra shift.

    Built from the FULL ``panel``, never the closure risk set: the ``gap_lag > 0`` filter excludes
    every spell's own onset block, so it structurally cannot anchor onset covariates. ``spell_duration``
    is deliberately absent - feeding the outcome's time axis back in as a regressor would be circular.
    Left-truncated spells (already open at the pair's first observed block, so their pre-onset block
    predates the window) are dropped by default; a handful more may drop if their pre-onset covariates
    are undefined (onset at the pair's first observed block, which has no preceding block).
    """
    q = naming.qlabel(quantile)
    tbl = build_spell_table(panel, quantile).rename(columns={"observed": "event"})

    onset_id = panel[["pair", "evt_block_number", f"spell_id_{q}"]].rename(
        columns={"evt_block_number": "onset_block", f"spell_id_{q}": "spell_id"})
    tbl = tbl.merge(onset_id, on=["pair", "onset_block"], how="left")

    preonset = {"log_base_fee_per_gas": "log_base_fee", "gas_util": "gas_util",
                "log1p_tip_p90": "tip_p90", "mev_intensity": "mev_intensity",
                "frequency_intensity": "freq"}
    pre = panel[["pair", "evt_block_number", *preonset]].rename(columns=preonset)
    pre["onset_block"] = pre.pop("evt_block_number") + 1     # block b feeds the spell starting at b+1
    tbl = tbl.merge(pre, on=["pair", "onset_block"], how="left")

    vol = panel[["pair", "evt_block_number", "ewma_vol"]].rename(columns={"evt_block_number": "onset_block"})
    vol["log_vol"] = np.log(vol.pop("ewma_vol"))
    tbl = tbl.merge(vol, on=["pair", "onset_block"], how="left")

    n_all, n_trunc = len(tbl), int(tbl["left_truncated"].sum())
    if drop_left_truncated:
        tbl = tbl.loc[~tbl["left_truncated"]]
    n_after = len(tbl)
    tbl = tbl.dropna(subset=SURVIVAL_TERMS).reset_index(drop=True)

    trunc_note = f"dropped {n_trunc} left-truncated" if drop_left_truncated else f"kept {n_trunc} left-truncated (flagged)"
    print(f"survival: {len(tbl)} spells from {n_all} | events: {int(tbl.event.sum())} | "
          f"pairs: {tbl.pair.nunique()} | {trunc_note} | dropped {n_after - len(tbl)} with undefined pre-onset covariate")
    return tbl[["spell_id", "pair", "onset_block", "duration", "event", *SURVIVAL_TERMS, "left_truncated"]]


##########---------ESTIMATION-----------###########


def _cluster_codes(values: pd.Series) -> np.ndarray:
    """Integer cluster codes for a grouping column, with every missing value its own singleton.

    ``pd.factorize`` maps NaN to ``-1``, which statsmodels' clustered covariance rejects; a row
    that belongs to no group (e.g. a block in no spell) should cluster with no one, so each such
    row gets a unique non-negative code instead."""
    codes = pd.factorize(values)[0]
    missing = codes < 0
    if missing.any():
        codes = codes.copy()
        codes[missing] = codes.max() + 1 + np.arange(missing.sum())
    return codes


def fit_hazard_logit(model_df: pd.DataFrame, terms: Sequence[str],
                     cluster: str | Sequence[str] = "pair", maxiter: int = 300,
                     direction: str = "logit"):
    """Fit an existence-stage hazard (onset, closure, or full existence) as a binary FE model.

    Regresses ``D`` on ``terms`` plus ``C(pair)`` fixed effects and ``C(hour)`` time-of-day fixed
    effects (added only when the sample spans more than one hour). Day-of-week and iso-week effects
    are deliberately omitted: on a sub-week study window they are near-collinear with each other and
    carry little signal, so they only cost identification. ``direction`` picks the link, ``"logit"``
    or ``"probit"``. Onset uses
    :data:`ONSET_TERMS` (``gap_lag`` is 0 there); closure uses :data:`CLOSURE_TERMS` or
    :data:`CLOSURE_DURATION_TERMS`. Standard errors are cluster-robust: ``cluster="pair"`` one-way,
    ``cluster=["pair", "spell_id"]`` the within-spell two-way (``spell_id`` is nested in ``pair``, so
    that two-way SE coincides with the pair-clustered one). Returns the fitted statsmodels result.
    """
    fitters = {"logit": smf.logit, "probit": smf.probit}
    if direction not in fitters:
        raise ValueError(f"direction must be one of {list(fitters)}, got {direction!r}")

    all_terms = list(terms) + ["C(pair)"]
    for fe, col in [("C(hour)", "hour")]:
        if col in model_df.columns and model_df[col].nunique() > 1:
            all_terms.append(fe)
    formula = "D ~ " + " + ".join(all_terms)

    cols = [cluster] if isinstance(cluster, str) else list(cluster)
    groups = np.column_stack([_cluster_codes(model_df[c]) for c in cols])
    groups = groups[:, 0] if len(cols) == 1 else groups

    return fitters[direction](formula, data=model_df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, maxiter=maxiter, disp=False)


def fit_magnitude_panel_ols(model_df: pd.DataFrame, terms: Sequence[str] = MAGNITUDE_TERMS,
                            calendar_cols: Sequence[str] = (),
                            two_way: bool = False):
    """Within (demeaned) pool-pair fixed-effects estimator of the magnitude equation.

    The pair effects are swept out by the within transformation rather than carried as ``C(pair)``
    dummies, so the parameter vector is just the economic ``terms`` (+ constant) and the
    entity-clustered covariance has full rank even with few pool pairs. ``two_way=True`` adds block
    (time) clustering on top of the entity clustering. ``linearmodels`` is imported lazily so the
    rest of ``arblib`` does not depend on it. No calendar controls by default (``calendar_cols=()``):
    pass e.g. ``("hour",)`` to add them, but note ``PanelOLS``'s exog interface does no ``C(...)``
    expansion, so they enter as raw integer columns rather than dummies."""

    df = model_df.set_index(["pair", "evt_block_number"])
    all_terms = list(terms) + [c for c in calendar_cols if df[c].nunique() > 1]
    mod = PanelOLS(df["log_gap"], df[all_terms], entity_effects=True)
    if two_way:
        return mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def center_continuous(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Mean-center the given continuous columns (subtract each column's own global mean).

    Used before the LSDV-estimated logits (existence / onset / closure) so the intercept is the
    log-odds at an average observation rather than at the (far-from-data) all-zero point. Does not
    affect slope coefficients - only the intercept's value and precision. Returns a copy."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out[c].mean()
    return out


def variance_inflation(model_df: pd.DataFrame, terms: Sequence[str]) -> pd.Series:
    """Variance inflation factors for ``terms`` - a quick multicollinearity read (VIF > ~10 = worry)."""

    X = sm.add_constant(model_df[list(terms)])
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                     index=X.columns, name="VIF")


def separation_report(result, model_df: pd.DataFrame, eps: float = 1e-4, top: int = 8) -> None:
    """Locate where (quasi-)separation is concentrated in a fitted logit / probit.

    Prints three things:

    * the share of observations the model predicts *near-perfectly* (fitted probability ``< eps`` or
      ``> 1 - eps``) - the same "a fraction ... can be perfectly predicted" statsmodels flags;
    * how those perfectly-predicted rows split across pool pairs (and hours, if in the frame), so you
      can see whether it is a handful of sparse ``pair x hour`` interaction cells rather than the
      whole sample;
    * the fixed-effect dummies carrying the **separation signature** - a coefficient driven to an
      extreme value with an unusually tight SE (large ``|z|``), or an unidentified one (NaN SE).

    A benign fit shows a low perfectly-predicted share and no extreme, tight-SE dummies. ``model_df``
    is the frame passed to the fit (so its ``pair`` / ``hour`` columns align with the fitted rows).
    """
    p = np.asarray(result.predict())
    perfect = (p < eps) | (p > 1 - eps)
    print(f"perfectly predicted (fitted p < {eps:g} or > {1 - eps:g}): "
          f"{int(perfect.sum())} / {len(p)} obs ({perfect.mean():.1%})")

    df = model_df.reset_index(drop=True)
    for col in ["pair", "hour"]:
        if col in df.columns and perfect.any():
            counts = df.loc[perfect, col].value_counts().head(top)
            print(f"\nperfectly-predicted rows by {col} (top {top} of {df[col].nunique()}):")
            print(counts.to_string())

    fe = result.params.index[result.params.index.str.startswith("C(")]
    if len(fe):
        sig = pd.DataFrame({"coef": result.params[fe], "std_err": result.bse[fe]})
        sig["abs_z"] = sig["coef"].abs() / sig["std_err"]
        unidentified = list(sig.index[sig["std_err"].isna()])
        if unidentified:
            print(f"\n{len(unidentified)} fixed-effect dummies UNIDENTIFIED (NaN SE) - complete "
                  f"separation: {unidentified}")
        print(f"\nfixed-effect dummies by |z| (separation signature = big |coef|, tiny SE), top {top}:")
        print(sig.sort_values("abs_z", ascending=False).head(top).round(3).to_string())
