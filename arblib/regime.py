"""
CEX volatility-regime detection - pick representative low / medium / high-vol weeks.

From a daily EWMA-volatility series (built by :func:`arblib.modeling.cex_volatility` on daily USD
prices), this smooths first and classifies the smoothed value: a centered rolling median gives each
day's "typical volatility over the surrounding week" (robust to a one-day spike or lull inside an
otherwise normal week), which is then split into terciles. Consecutive-day runs that stay in one
tercile are the candidate weeks for a regime-conditioned re-run of the study. Classifying the
smoothed series targets the week-level property directly, rather than the stricter "every single day
must sit in the tercile" that a genuine regime need not satisfy.

Helpers: :func:`calibrate_horizon` picks the EWMA memory by variance-forecast accuracy;
:func:`classify_regimes` / :func:`regime_runs` / :func:`select_windows` build and rank the
candidates; :func:`pick_windows` selects one confound-dodging window per regime; :func:`plot_regimes`
draws it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import formulas
from .config import build_collection_params

_COLORS = {"low": "tab:green", "mid": "tab:orange", "high": "tab:red"}


##########---------HORIZON CALIBRATION-----------###########


def calibrate_horizon(x_usd: pd.DataFrame, y_usd: pd.DataFrame,
                      horizons=range(3, 31), burn: int = 30) -> pd.DataFrame:
    """Score each daily-EWMA horizon by how well it forecasts next-day variance.

    The EWMA variance ``var_t`` (info up to day ``t``) is the one-step forecast of the next day's
    squared log-return ``r_{t+1}^2``. For each candidate ``horizon`` (in days) this builds
    ``var_t`` on the token0/token1 rate and scores that forecast with **QLIKE**
    (``r^2/var - log(r^2/var) - 1``, the standard variance-forecast loss, robust to the noisy
    ``r^2`` proxy) and MSE, discarding the first ``burn`` days of EWMA cold-start. Returns
    ``[horizon, qlike, mse]``; the QLIKE-minimizing horizon is the data-driven memory. Reference:
    RiskMetrics' daily ``lambda = 0.94`` is a ``-1/ln(0.94) ~ 16``-day horizon - crypto vol often
    mean-reverts faster, so expect something in the low-to-mid-teens or shorter.
    """
    m = x_usd.merge(y_usd, on="time", suffixes=("_x", "_y")).sort_values("time")
    ret = np.diff(np.log((m["price_x"] / m["price_y"]).to_numpy()), prepend=np.nan)
    realized = ret[1:] ** 2                       # r_{t+1}^2, aligned to a forecast made at day t

    rows = []
    for h in horizons:
        var = formulas.vol_ewma(x_usd, y_usd, int(h))["ewma_var"].to_numpy()
        forecast = var[:-1]                        # var_t predicts realized[t]
        keep = (np.arange(len(forecast)) >= burn) & (forecast > 0) & (realized > 0)
        f, r = forecast[keep], realized[keep]
        rows.append({"horizon": int(h),
                     "qlike": float(np.mean(r / f - np.log(r / f) - 1)),
                     "mse": float(np.mean((r - f) ** 2))})
    return pd.DataFrame(rows)


def best_horizon(scores: pd.DataFrame, criterion: str = "qlike") -> int:
    """The horizon minimizing ``criterion`` (``"qlike"`` or ``"mse"``) in a :func:`calibrate_horizon` table."""
    return int(scores.loc[scores[criterion].idxmin(), "horizon"])


##########---------REGIME CLASSIFICATION-----------###########


def classify_regimes(vol: pd.DataFrame, window: int = 7, lo_q: float = 1 / 3, hi_q: float = 2 / 3,
                     vol_col: str = "ewma_vol") -> pd.DataFrame:
    """Label each day low / mid / high by the tercile of the *centered rolling-median* vol.

    Smooths ``vol_col`` with a centered ``window``-day rolling median - "was this week calm / medium
    / turbulent", robust to single-day outliers and centred so it asks whether the surrounding week
    (not the trailing one) is representative - then classifies days by the ``lo_q`` / ``hi_q``
    quantile cutoffs of that smoothed series (not of the raw daily vol). Drops the EWMA warm-up
    (``vol_col == 0``). Returns ``vol`` with ``roll_med`` and a ``regime`` categorical in
    {low, mid, high}; the ``window//2`` edge days have ``roll_med`` / ``regime`` NaN.
    """
    out = vol.loc[vol[vol_col] > 0].sort_values("time").reset_index(drop=True)
    out["roll_med"] = out[vol_col].rolling(window, center=True, min_periods=window).median()
    lo, hi = out["roll_med"].quantile([lo_q, hi_q])
    out["regime"] = pd.cut(out["roll_med"], [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
    return out


def regime_runs(daily: pd.DataFrame, min_len: int = 7) -> pd.DataFrame:
    """Maximal runs of consecutive days with a constant ``regime``, one row per run (length >= ``min_len``).

    Returns ``[regime, start, end, length, depth]``, where ``depth`` is the mean rolling-median vol
    over the run - how deep into the tercile it sits, used to prefer unambiguous representatives
    (low ``depth`` for a "low" week, high for a "high" week)."""
    d = daily.dropna(subset=["regime"]).copy()
    d["regime"] = d["regime"].astype(str)
    grp = (d["regime"] != d["regime"].shift()).cumsum()
    runs = (d.groupby(grp)
            .agg(regime=("regime", "first"), start=("time", "min"), end=("time", "max"),
                 length=("time", "size"), depth=("roll_med", "mean"))
            .reset_index(drop=True))
    return runs.loc[runs["length"] >= min_len].reset_index(drop=True)


def select_windows(daily: pd.DataFrame, window: int = 7, min_len: int = 7) -> pd.DataFrame:
    """Rank the qualifying runs per regime and centre a ``window``-day sub-window in each.

    For every regime, ranks its runs (>= ``min_len`` days) by how deep into the tercile they sit -
    the most unambiguous "calm" / "turbulent" week first - breaking ties by recency (prefer the more
    recent, consistent with the study's recency preference). Centres a ``window``-day window inside
    each run (slack when a run is longer than ``window`` lets :func:`pick_windows` slide it off a
    known confound date). Returns ``[regime, start, end, length, depth, score, win_start, win_end,
    rank]`` sorted by (regime, rank); ``rank == 1`` is the best candidate for that regime.
    """
    runs = regime_runs(daily, min_len=min_len)
    if runs.empty:
        return runs.assign(score=[], win_start=pd.Series(dtype="datetime64[ns, UTC]"),
                           win_end=pd.Series(dtype="datetime64[ns, UTC]"), rank=[])

    center = daily["roll_med"].median()

    def _score(regime: str, depth: float) -> float:
        if regime == "low":
            return depth                 # lower vol = deeper into the low tercile (rank first)
        if regime == "high":
            return -depth                # higher vol = deeper into the high tercile
        return abs(depth - center)       # closest to the median = most representative "medium"

    runs = runs.copy()
    runs["score"] = [_score(r, d) for r, d in zip(runs["regime"], runs["depth"])]
    offset = ((runs["length"] - window) // 2).clip(lower=0)
    runs["win_start"] = runs["start"] + pd.to_timedelta(offset, unit="D")
    runs["win_end"] = runs["win_start"] + pd.to_timedelta(window - 1, unit="D")

    runs = runs.sort_values(["regime", "score", "end"], ascending=[True, True, False]).reset_index(drop=True)
    runs["rank"] = runs.groupby("regime").cumcount() + 1
    return runs


##########---------CANDIDATE SELECTION (confound-aware)-----------###########


def monthly_expiries(start, end) -> pd.DatetimeIndex:
    """Crypto monthly options-expiry dates (last Friday of each month) in ``[start, end]`` (UTC).

    A cheap, computable confound to keep out of a chosen window; extend with hand-supplied macro
    dates (FOMC / CPI) when calling :func:`pick_windows`."""
    month_ends = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(),
                               freq="ME", tz="UTC")
    last_fridays = month_ends - pd.to_timedelta((month_ends.weekday - 4) % 7, unit="D")
    return last_fridays[(last_fridays >= pd.Timestamp(start)) & (last_fridays <= pd.Timestamp(end))]


def _place_window(start, end, window: int, avoid: pd.DatetimeIndex):
    """Centre-outward search for a ``window``-day slot in ``[start, end]`` hitting no ``avoid`` date.

    Returns ``(win_start, win_end)`` for the offset closest to centred that dodges every confound,
    or ``None`` if the run cannot host a clean window."""
    max_off = (end - start).days + 1 - window
    if max_off < 0:
        return None
    center = max_off // 2
    for off in sorted(range(max_off + 1), key=lambda o: abs(o - center)):
        ws = start + pd.Timedelta(days=off)
        we = ws + pd.Timedelta(days=window - 1)
        if not ((avoid >= ws) & (avoid <= we)).any():
            return ws, we
    return None


def pick_windows(candidates: pd.DataFrame, avoid=None, window: int = 7) -> pd.DataFrame:
    """Select one window per regime: the highest-ranked run whose window can dodge the ``avoid`` dates.

    Walks each regime's runs best-first (:func:`select_windows` order) and slides a ``window``-day
    window inside the parent run to avoid every ``avoid`` date (:func:`_place_window`); takes the
    first run that admits a clean window. If none does, falls back to the rank-1 centred window and
    marks ``avoided = False``. Returns one row per regime ``[regime, chosen_rank, win_start, win_end,
    run_start, run_end, depth, avoided]``.
    """
    avoid = pd.DatetimeIndex(pd.to_datetime(list(avoid), utc=True)) if avoid is not None \
        else pd.DatetimeIndex([], tz="UTC")

    picks = []
    for regime, grp in candidates.sort_values(["regime", "rank"]).groupby("regime"):
        chosen = None
        for _, run in grp.iterrows():
            win = _place_window(run["start"], run["end"], window, avoid)
            if win is not None:
                chosen = dict(regime=regime, chosen_rank=int(run["rank"]), win_start=win[0],
                              win_end=win[1], run_start=run["start"], run_end=run["end"],
                              depth=run["depth"], avoided=True)
                break
        if chosen is None:
            r = grp.iloc[0]
            chosen = dict(regime=regime, chosen_rank=int(r["rank"]), win_start=r["win_start"],
                          win_end=r["win_end"], run_start=r["start"], run_end=r["end"],
                          depth=r["depth"], avoided=False)
        picks.append(chosen)
    return pd.DataFrame(picks)


##########---------PLOT-----------###########


def plot_regimes(daily: pd.DataFrame, candidates: pd.DataFrame | None = None,
                 vol_col: str = "ewma_vol", window: int = 7):
    """Plot daily vol, its centered rolling median, the two tercile cutoffs, and any candidate windows.

    Candidate windows (from :func:`select_windows` or :func:`pick_windows`) are shaded by regime
    colour, the rank-1 pick of each regime more opaque. Returns the matplotlib ``Axes``."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lo = daily.loc[daily["regime"] == "low", "roll_med"].max()
    hi = daily.loc[daily["regime"] == "high", "roll_med"].min()
    rank_col = "rank" if candidates is not None and "rank" in candidates.columns else "chosen_rank"

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily["time"], daily[vol_col], color="0.75", lw=0.8, label="daily EWMA vol (raw)")
    ax.plot(daily["time"], daily["roll_med"], color="black", lw=1.6,
            label=f"{window}-day rolling median (regime signal)")
    ax.axhline(hi, ls="--", color=_COLORS["high"], lw=1.2, label=f"67% cutoff = {hi:.4f}  (> ⇒ high)")
    ax.axhline(lo, ls="--", color=_COLORS["low"], lw=1.2, label=f"33% cutoff = {lo:.4f}  (≤ ⇒ low)")

    patches = []
    if candidates is not None and len(candidates):
        for _, w in candidates.iterrows():
            ax.axvspan(w["win_start"], w["win_end"], color=_COLORS[w["regime"]],
                       alpha=0.45 if w[rank_col] == 1 else 0.12)
        patches = [Patch(facecolor=_COLORS[r], alpha=0.45, label=f"{r}-vol window") for r in _COLORS]

    ax.set_ylabel("volatility of token0/token1 rate"); ax.set_xlabel("date")
    ax.set_title("CEX volatility regimes — daily EWMA vol, rolling-median signal, terciles "
                 "(solid = rank-1, faint = fallback)")
    lines = ax.get_legend_handles_labels()[0]
    ax.legend(handles=lines + patches, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    return ax


##########---------STUDY-WINDOW HANDOFF-----------###########


def save_study_windows(picks: pd.DataFrame, path: str | Path) -> None:
    """Write the one-window-per-regime table (from :func:`pick_windows`) to ``path`` as CSV.

    ``win_start`` / ``win_end`` are stored as plain dates so the extraction step can read a fixed,
    reproducible study window per regime (see :func:`study_window`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["regime", "chosen_rank", "win_start", "win_end", "run_start", "run_end",
                        "depth", "avoided"] if c in picks.columns]
    out = picks[cols].copy()
    for c in ("win_start", "win_end", "run_start", "run_end"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} regime windows to {path}")


def study_window(settings, regime_label: str) -> dict:
    """Block-level extraction spec for one chosen regime window (read from ``settings.study_dates_path``).

    Loads the ``regime_label`` row saved by :func:`save_study_windows` and returns the timestamps the
    extract pipeline needs. The study period is the full window ``[win_start, win_end + 1 day)``;
    extraction starts ``settings.extract_lead_hours`` earlier - a warm-up lead so on-chain price /
    liquidity are already live and the block-level MEV / frequency EWMAs have converged by the first
    study block (``study_start``). ``cex_start_ts`` widens the Kraken 1-minute fetch a further
    ``cex_avg_window_min`` for the trailing USD-price average. Returns
    ``{regime, start_ts, study_start, end_ts, cex_start_ts, collection_params}`` (timestamps as
    ``"%Y-%m-%d %H:%M:%S"`` UTC, matching the Dune saved-query parameters). For a cheap smoke
    test on an explicit window (independent of the regime dates), see :func:`custom_window`.
    """
    rows = pd.read_csv(settings.study_dates_path)
    hit = rows.loc[rows["regime"] == regime_label]
    if hit.empty:
        raise ValueError(f"regime {regime_label!r} not in {settings.study_dates_path} "
                         f"(available: {list(rows['regime'])})")

    win_start = pd.Timestamp(hit["win_start"].iloc[0], tz="UTC")
    win_end = pd.Timestamp(hit["win_end"].iloc[0], tz="UTC")
    study_start = win_start
    end = win_end + pd.Timedelta(days=1)                                   # include the full last day
    extract_start = study_start - pd.Timedelta(hours=settings.extract_lead_hours)
    cex_start = extract_start - pd.Timedelta(minutes=settings.cex_avg_window_min)

    fmt = "%Y-%m-%d %H:%M:%S"
    return {
        "regime": regime_label,
        "start_ts": extract_start.strftime(fmt),
        "study_start": study_start.strftime(fmt),
        "end_ts": end.strftime(fmt),
        "cex_start_ts": cex_start.strftime(fmt),
        "collection_params": build_collection_params(
            settings.chain, settings.base.address, settings.quote.address,
            extract_start.strftime(fmt), end.strftime(fmt)),
    }


def custom_window(settings, start: str | None, end: str | None) -> dict:
    """Extraction spec for an explicit ``[start, end)`` window, independent of the regime dates.

    Used for a cheap smoke test (``settings.test_mode`` with ``settings.test_start`` /
    ``settings.test_end``): it bypasses ``study_dates.csv`` entirely and extracts exactly the given
    window ``[start, end)``. Extraction starts at ``start``; the study (the part kept by
    ``transform``) starts ``settings.test_lead_min`` minutes later, so the first study block has a
    short warm-up - the in-window analogue of the real mode's ``extract_lead_hours`` before the
    regime window. The Kraken fetch is widened by ``cex_avg_window_min`` for the trailing USD-price
    average. ``start`` / ``end`` are ``"YYYY-MM-DD HH:MM:SS"`` UTC strings; the returned dict has the
    same keys as :func:`study_window` so the extract / transform notebooks are unchanged.
    """
    if start is None or end is None:
        raise ValueError("custom_window needs test_start and test_end (set them when test_mode=True)")

    fmt = "%Y-%m-%d %H:%M:%S"
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    study_start = start_ts + pd.Timedelta(minutes=settings.test_lead_min)
    cex_start = start_ts - pd.Timedelta(minutes=settings.cex_avg_window_min)

    return {
        "regime": "test",
        "start_ts": start_ts.strftime(fmt),
        "study_start": study_start.strftime(fmt),
        "end_ts": end_ts.strftime(fmt),
        "cex_start_ts": cex_start.strftime(fmt),
        "collection_params": build_collection_params(
            settings.chain, settings.base.address, settings.quote.address,
            start_ts.strftime(fmt), end_ts.strftime(fmt)),
    }
