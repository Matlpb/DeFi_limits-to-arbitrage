"""
Descriptive summary statistics across volatility regimes.

Assembles the per-regime pooled panels (:func:`arblib.estimation.load_panel`) and reduces them to
the summary-statistics tables the paper reports:

* :func:`summary_table` - one column per regime (Calm / Medium / Turbulent), sectioned into panel
  dimensions, the reference-quantile dependent-variable distribution conditional on existence, and
  covariate means / sds. This is the table that sits next to the main results and describes the raw
  extracted panel feeding them.
* :func:`existence_cross_quantile_table` - the live-gap share at each reference quantile, by regime:
  the compact cross-quantile robustness glance (existence should collapse as the reference trade
  size grows, consistent with larger trades crossing more ticks).

Every number reads from the modeling parquets each ``<regime>_vol`` folder already holds; nothing
here touches Dune. Notebooks stay call-only: load the panels once with :func:`load_regime_panels`,
then hand them to the two table builders.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import estimation, naming

# Paper-facing regime names, ordered calm -> turbulent so a reader sees volatility rise left to right.
REGIME_ORDER = ["low", "mid", "high"]
REGIME_LABELS = {"low": "Calm", "mid": "Medium", "high": "Turbulent"}

_GWEI = 1e9


##########---------PANEL LOADING-----------###########


def _regime_settings(settings, regime: str):
    """A copy of ``settings`` pointing at ``regime``'s ``<regime>_vol`` folder (all paths re-derive)."""
    return replace(settings, active_regime=regime, test_mode=False)


def load_regime_panel(settings, regime: str) -> pd.DataFrame:
    """The pooled panel for one regime, read from that regime's parquets (:func:`arblib.estimation.load_panel`)."""
    return estimation.load_panel(_regime_settings(settings, regime))


def load_regime_panels(settings, regimes: list[str] = REGIME_ORDER) -> dict[str, pd.DataFrame]:
    """Load every regime's panel once, keyed by regime code (``low`` / ``mid`` / ``high``)."""
    return {r: load_regime_panel(settings, r) for r in regimes}


def _window_dates(settings, regime: str) -> str:
    """The inclusive study window ``"win_start - win_end"`` for ``regime`` from ``study_dates.csv``."""
    rows = pd.read_csv(settings.study_dates_path)
    hit = rows.loc[rows["regime"] == regime]
    if hit.empty:
        return "n/a"
    return f"{hit['win_start'].iloc[0]} – {hit['win_end'].iloc[0]}"


##########---------COVARIATE LEVELS-----------###########


def covariate_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """The model covariates in interpretable levels, one row per panel observation.

    Base fee and the priority-tip p90 are stored in logs (:func:`arblib.modeling.chain_covariates`),
    so they are inverted back to Gwei; gas utilisation, the venue-averaged MEV and swap-frequency
    proxies, and the CEX EWMA volatility are already in levels. The volatility is scaled by 1e3 so
    the three regime means stay legible on one decimal scale (its raw level is ~5e-4 to ~1e-3, where
    significant-figure rounding otherwise gives each regime a different number of leading zeros and
    the column reads as if the ordering were reversed). Reported contemporaneously (the regressions
    lag them one block, which leaves the distribution unchanged)."""
    return pd.DataFrame({
        "Base fee (Gwei)": np.exp(panel["log_base_fee_per_gas"]) / _GWEI,
        "Gas utilization": panel["gas_util"].astype(float),
        "Priority fee p90 (Gwei)": np.expm1(panel["log1p_tip_p90"]) / _GWEI,
        "MEV intensity (venue-avg log1p)": panel["mev_intensity"].astype(float),
        "Swap frequency (EWMA/block)": panel["frequency_intensity"].astype(float),
        "CEX EWMA volatility (×10⁻³)": panel["ewma_vol"].astype(float) * 1e3,
    })


##########---------FORMATTING-----------###########


def _fmt_int(n: float) -> str:
    return f"{int(n):,}"


def _fmt_num(x: float, sig: str = ".4g") -> str:
    return format(float(x), sig)


def _fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{_fmt_num(mean)} ({_fmt_num(sd, '.3g')})"


##########---------TABLES-----------###########


def _regime_column(settings, regime: str, panel: pd.DataFrame, quantile: float) -> dict[tuple[str, str], str]:
    """Build one regime's column of the main summary table (see :func:`summary_table`)."""
    q = naming.qlabel(quantile)
    gap = panel[f"gap_{q}"].astype(float)
    d = panel[f"D_{q}"].astype(float)
    open_gap = gap[d == 1]

    dv_section = f"Dependent variable: Gap (bps), {q} | D=1"
    col: dict[tuple[str, str], str] = {}

    dims = "Panel dimensions"
    col[(dims, "Window (UTC, inclusive)")] = _window_dates(settings, regime)
    col[(dims, "N obs (pool-pair × block)")] = _fmt_int(len(panel))
    col[(dims, "N pool-pairs")] = _fmt_int(panel["pair"].nunique())
    col[(dims, "N blocks spanned")] = _fmt_int(panel["evt_block_number"].nunique())

    col[(dv_section, "Existence rate, D=1 (%)")] = _fmt_num(d.mean() * 100, ".3g")
    col[(dv_section, "Mean Gap | D=1")] = _fmt_num(open_gap.mean(), ".4g")
    col[(dv_section, "Median Gap | D=1")] = _fmt_num(open_gap.median(), ".4g")
    col[(dv_section, "p10 Gap | D=1")] = _fmt_num(open_gap.quantile(0.10), ".4g")
    col[(dv_section, "p90 Gap | D=1")] = _fmt_num(open_gap.quantile(0.90), ".4g")

    cov = "Covariates: mean (sd)"
    levels = covariate_levels(panel)
    for name in levels.columns:
        col[(cov, name)] = _fmt_mean_sd(levels[name].mean(), levels[name].std())

    return col


def summary_table(panels: dict[str, pd.DataFrame], settings, quantile: float = 0.2) -> pd.DataFrame:
    """The main summary-statistics table: metrics down the rows, one regime per column.

    ``panels`` maps a regime code to its panel (from :func:`load_regime_panels`). ``quantile`` is the
    lead / primary reference trade size (default q20 - the smallest, least tick-crossing size, so the
    round-trip spread stays cleanly interpretable). Rows are a two-level index
    ``(section, metric)`` grouping panel dimensions, the ``q`` dependent-variable distribution given a
    gap is open, and covariate means / sds; every cell is a pre-formatted string so the table copies
    straight into a LaTeX-writing prompt. Regime columns are ordered Calm -> Medium -> Turbulent."""
    cols = {}
    for regime in REGIME_ORDER:
        if regime in panels:
            cols[REGIME_LABELS[regime]] = _regime_column(settings, regime, panels[regime], quantile)
    table = pd.DataFrame(cols)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Section", "Metric"])
    return table


def existence_cross_quantile_table(panels: dict[str, pd.DataFrame],
                                   quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> pd.DataFrame:
    """Existence rate (share of pool-pair x block cells with a live gap, %) at each reference quantile.

    Rows are the reference quantiles, columns the regimes (Calm -> Medium -> Turbulent). Existence
    should fall as the quantile grows - a larger reference trade crosses more ticks, so fewer blocks
    host an executable round-trip gap - the cheap cross-quantile sanity check on the trade-size
    calibration. Values are rounded percentages."""
    rows = {}
    for q in quantiles:
        label = naming.qlabel(q)
        rows[label] = {REGIME_LABELS[r]: round(panels[r][f"D_{label}"].mean() * 100, 3)
                       for r in REGIME_ORDER if r in panels}
    return pd.DataFrame(rows).T.rename_axis("Reference quantile")
