"""
Cross-regime regression tables for the paper.

Refits the three block-level models - the onset hazard (logit), the closure hazard (logit) and the
magnitude equation (within pool-pair FE OLS) - for **every** volatility regime and every reference
quantile, and collapses each into one tidy coefficient table (coefficient with significance stars per
covariate, plus the fit's pseudo / within R^2, N and pool-pair count). The recipes are exactly those
of notebooks 06 / 07 / 08; here they are looped over the regimes so the whole results section is
assembled in one place without touching ``STUDY.active_regime`` - each regime's panel is read from its
own ``<regime>_vol`` folder via :func:`arblib.summary.load_regime_panels`.

Every model reads the pooled panels already in memory; nothing here touches Dune. Notebooks stay
call-only: load the panels once, then call :func:`onset_table` / :func:`closure_table` /
:func:`magnitude_table`.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from . import estimation as est, naming
from .summary import REGIME_LABELS, REGIME_ORDER

# Covariate -> paper column header (shared across the three tables; only the subset each model uses
# is emitted, in the order the LaTeX skeleton lists them).
_DISPLAY = {
    "log_base_fee": "log base fee_t",
    "gas_util_lag": "gas util_{t-1}",
    "tip_p90_lag": "tip p90_{t-1}",
    "mev_lag": "MEV_{t-1}",
    "freq_lag": "freq_{t-1}",
    "log_vol": "log σ_t",
    "spell_duration": "spell dur.",
    "log1p_gap_lag": "log(1+gap_{t-1})",
}

_ONSET_ORDER = ["log_base_fee", "gas_util_lag", "tip_p90_lag", "mev_lag", "freq_lag", "log_vol"]
_CLOSURE_ORDER = _ONSET_ORDER + ["spell_duration", "log1p_gap_lag"]
_MAGNITUDE_ORDER = _ONSET_ORDER + ["log1p_gap_lag"]
_CORR_ORDER = _CLOSURE_ORDER   # the full union of block-level regressors appearing in the tables

# Leading descriptive column of each hazard table: the unconditional base rate of D == 1 *in that
# model's own estimation sample*. Both are the same statistic - mean(D) - read under the risk set the
# equation conditions on: on the onset risk set (gap_lag == 0) D == 1 is a gap appearing, so the rate
# is the existence rate; on the closure risk set (gap_lag > 0) D == 1 is the spell surviving block t,
# so it is the survival rate (1 - closure rate). Survival, not closure, is reported so the descriptive
# number measures literally the outcome the coefficients are signed with respect to.
_RATE_LABEL = {"onset": "existence rate (%)", "closure": "survival rate (%)"}

_DEFAULT_QUANTILES = (0.2, 0.4)


def _stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _coef_star(coef: float, pvalue: float) -> str:
    return f"{coef:.3f}{_stars(pvalue)}"


def _coef_row(result, order, params, pvalues) -> dict[str, str]:
    """One row of coefficient-with-stars cells for the covariates in ``order``."""
    return {_DISPLAY[t]: _coef_star(float(params[t]), float(pvalues[t])) for t in order}


@contextlib.contextmanager
def _quiet(verbose: bool):
    """Swallow the sample-building diagnostics (dropped pairs, shapes) unless ``verbose``."""
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


##########---------ONSET / CLOSURE HAZARD (LOGIT)-----------###########


def _hazard_row(panel: pd.DataFrame, quantile: float, condition: str, order, terms,
                floor: int) -> dict:
    """Fit one onset / closure logit (06 / 08 recipe) and return its coefficient + fit-stat row."""
    rate_label = _RATE_LABEL[condition]
    df = est.build_risk_set(panel, quantile=quantile, condition=condition)
    df = est.drop_pairs_by_event_floor(df, floor, label=condition)
    if df["pair"].nunique() < 2 or df["D"].nunique() < 2:
        rate = f"{100 * float(df['D'].mean()):.2f}" if len(df) else "—"
        return {rate_label: rate, **{_DISPLAY[t]: "—" for t in order}, "pseudo R²": "—",
                "N": len(df), "N_pairs": df["pair"].nunique()}
    df_c = est.center_continuous(df, terms)
    res = est.fit_hazard_logit(df_c, terms, cluster="pair", direction="logit")
    row = _coef_row(res, order, res.params, res.pvalues)
    # Base rate over the rows the logit actually fit (``res.model.endog`` is D after any listwise
    # drop), so #{D==1} / N uses exactly the N reported two columns to the right. Formatted, like the
    # coefficient cells, so ``to_latex()`` emits "2.11" rather than a 6-decimal float.
    row[rate_label] = f"{100 * float(np.mean(res.model.endog)):.2f}"
    row["pseudo R²"] = round(float(res.prsquared), 4)
    row["N"] = int(res.nobs)
    row["N_pairs"] = int(df_c["pair"].nunique())
    return row


def _hazard_table(panels, settings, condition: str, order, terms,
                  quantiles=_DEFAULT_QUANTILES, verbose: bool = False) -> pd.DataFrame:
    rows, index = [], []
    for regime in REGIME_ORDER:
        if regime not in panels:
            continue
        for q in quantiles:
            with _quiet(verbose):
                rows.append(_hazard_row(panels[regime], q, condition, order, terms,
                                        settings.min_events_per_pair))
            index.append((REGIME_LABELS[regime], naming.qlabel(q)))
    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["Regime", "q"]))
    return table[[_RATE_LABEL[condition]] + [_DISPLAY[t] for t in order]
                 + ["pseudo R²", "N", "N_pairs"]]


def onset_table(panels, settings, quantiles=_DEFAULT_QUANTILES, verbose: bool = False) -> pd.DataFrame:
    """Onset-hazard (logit) coefficients by regime x quantile (06 recipe, all regimes at once).

    Risk set ``gap_lag == 0``; ``D`` on :data:`arblib.estimation.ONSET_TERMS` with pool-pair and
    hour-of-day fixed effects and pair-clustered SEs. Cells are ``coef`` with significance stars
    (``*** ``p<0.01, ``**`` p<0.05, ``*`` p<0.10); trailing columns are McFadden pseudo R^2, the
    risk-set N, and the pool-pair count kept after the constant-D and EPV-floor screens.

    The leading ``existence rate (%)`` column is the descriptive base rate the model explains:
    ``100 x #{D == 1} / N`` over that cell's **own** estimation sample - after the EPV floor and
    restricted to ``gap_lag == 0`` - so it is computed on exactly the N rows reported in the same row,
    not on the unconditioned panel of Table 1."""
    return _hazard_table(panels, settings, "onset", _ONSET_ORDER, est.ONSET_TERMS, quantiles, verbose)


def closure_table(panels, settings, quantiles=_DEFAULT_QUANTILES, verbose: bool = False) -> pd.DataFrame:
    """Closure-hazard (logit) coefficients by regime x quantile (08 recipe, all regimes at once).

    Risk set ``gap_lag > 0`` (a gap is open at ``t-1``); ``D`` = "the spell survives block ``t``" on
    :data:`arblib.estimation.CLOSURE_DURATION_TERMS` (adds spell age and the open-gap size) with the
    same fixed effects / clustering as :func:`onset_table`. Same cell and trailing-column format.

    The leading ``survival rate (%)`` column is the persistence analogue of the onset table's
    existence rate: ``100 x #{D == 1} / N`` over that cell's own estimation sample - after the EPV
    floor and restricted to ``gap_lag > 0`` - i.e. the share of at-risk blocks on which the open gap
    survives. The closure rate is its complement, ``100 - survival rate``; survival is the one
    reported because the dependent variable is survival, so the descriptive rate and the coefficient
    signs point the same way."""
    return _hazard_table(panels, settings, "closure", _CLOSURE_ORDER, est.CLOSURE_DURATION_TERMS,
                         quantiles, verbose)


##########---------MAGNITUDE (WITHIN-FE OLS)-----------###########


def _magnitude_row(panel: pd.DataFrame, quantile: float, floor: int) -> dict:
    """Fit one within-FE magnitude OLS (07 recipe) and return its coefficient + fit-stat row."""
    mag = est.build_magnitude_sample(panel, quantile=quantile, min_obs_per_pair=floor)
    if mag["pair"].nunique() < 2 or len(mag) <= len(est.MAGNITUDE_TERMS) + 1:
        return {**{_DISPLAY[t]: "—" for t in _MAGNITUDE_ORDER}, "within R²": "—",
                "N": len(mag), "N_pairs": mag["pair"].nunique()}
    res = est.fit_magnitude_panel_ols(mag, two_way=True)
    row = _coef_row(res, _MAGNITUDE_ORDER, res.params, res.pvalues)
    row["within R²"] = round(float(res.rsquared_within), 4)
    row["N"] = int(res.nobs)
    row["N_pairs"] = int(mag["pair"].nunique())
    return row


def magnitude_table(panels, settings, quantiles=_DEFAULT_QUANTILES, verbose: bool = False) -> pd.DataFrame:
    """Magnitude-equation (within pool-pair FE OLS) coefficients by regime x quantile (07 recipe).

    Conditional on existence (``D == 1``); ``log Gap`` on :data:`arblib.estimation.MAGNITUDE_TERMS`
    with pool-pair fixed effects swept out by the within transformation (no hour effects) and two-way
    (pool-pair x block) clustered SEs. Cells are ``coef`` with stars; trailing columns are the within
    R^2, N (rows with ``D == 1`` after the min-rows-per-pair floor) and the pool-pair count."""
    rows, index = [], []
    for regime in REGIME_ORDER:
        if regime not in panels:
            continue
        for q in quantiles:
            with _quiet(verbose):
                rows.append(_magnitude_row(panels[regime], q, settings.min_events_per_pair))
            index.append((REGIME_LABELS[regime], naming.qlabel(q)))
    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["Regime", "q"]))
    return table[[_DISPLAY[t] for t in _MAGNITUDE_ORDER] + ["within R²", "N", "N_pairs"]]


##########---------REGRESSOR CORRELATIONS-----------###########


def covariate_correlations(panels, quantile: float = 0.2,
                           verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Pearson correlation among the block-level model regressors, one square matrix per regime.

    Computed over the full ``q{quantile}`` model frame (:func:`arblib.estimation.prepare_model_frame`
    with ``drop_constant_pairs=False``) - every pool-pair x block row whose lags are defined, i.e. the
    pooled sample the onset / closure / magnitude risk sets are all carved from. Columns are the union
    of regressors appearing in the three coefficient tables, renamed to the paper headers, so the
    matrix doubles as the multicollinearity read for those regressions. ``log(1+gap_{t-1})`` and
    ``spell dur.`` are lagged-state terms that are 0 off-spell (the large majority of rows), so their
    mutual correlation is mechanical (both non-zero only on open-spell blocks) rather than economic;
    the environment covariates (``log base fee`` ... ``log σ``) carry the read that matters. Returns
    ``{regime label: correlation DataFrame}`` (Calm / Medium / Turbulent)."""
    out = {}
    for regime in REGIME_ORDER:
        if regime not in panels:
            continue
        with _quiet(verbose):
            frame = est.prepare_model_frame(panels[regime], quantile=quantile,
                                            drop_constant_pairs=False)
        corr = frame[_CORR_ORDER].corr()
        labels = [_DISPLAY[t] for t in _CORR_ORDER]
        corr.index, corr.columns = labels, labels
        out[REGIME_LABELS[regime]] = corr
    return out
