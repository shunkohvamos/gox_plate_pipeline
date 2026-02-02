# src/gox_plate_pipeline/fitting/candidates.py
"""
Candidate window generation and per-well fitting.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .core import (
    FitResult,
    _fit_linear,
    _fit_linear_theilsen,
    _robust_sigma,
    _auto_mono_eps,
    _auto_min_delta_y,
    _detect_step_jump,
    _find_start_index,
)


def generate_candidate_windows(
    time_s: np.ndarray,
    min_points: int = 6,
    max_points: int = 12,
    min_span_s: float = 0.0,
    start_idx: int = 0,
) -> list[Tuple[int, int]]:
    """
    Candidate windows defined by index ranges [i0:i1] (inclusive endpoints) in data order.
    Returns list of (start_idx, end_idx) inclusive.

    Controls:
      - min_points: minimum number of points in a window
      - max_points: maximum number of points in a window (scan multiple sizes)
      - min_span_s: minimum time span (t_end - t_start) required for a window
      - start_idx: restrict windows to start at i0 >= start_idx
    """
    n = len(time_s)
    if n < min_points:
        return []

    min_span_s = float(min_span_s)
    start_idx = max(0, int(start_idx))

    windows: list[Tuple[int, int]] = []
    last_start = n - min_points
    for i0 in range(start_idx, last_start + 1):
        i1_min = i0 + min_points - 1
        i1_max = min(n - 1, i0 + max_points - 1)
        for i1 in range(i1_min, i1_max + 1):
            span = float(time_s[i1] - time_s[i0])
            if span < min_span_s:
                continue
            windows.append((i0, i1))
    return windows


def fit_initial_rate_one_well(
    df_well: pd.DataFrame,
    min_points: int = 6,
    max_points: int = 12,
    min_span_s: float = 0.0,
    mono_eps: Optional[float] = None,
    min_delta_y: Optional[float] = None,
    find_start: bool = True,
    start_max_shift: int = 5,
    start_window: int = 3,
    start_allow_down_steps: int = 1,
    min_t_start_s: float = 0.0,
    down_step_min_frac: Optional[float] = None,
    fit_method: str = "ols",
) -> pd.DataFrame:
    """
    Return candidate fits for one well.

    df_well must have columns:
      - time_s (seconds)
      - signal (numeric)

    Optional robustness (Amplex Red / resorufin):
      - min_t_start_s: ignore windows starting before this time (s); e.g. 60 to skip mixing.
      - down_step_min_frac: only count down steps larger than max(mono_eps, frac * signal_range); small dips ignored.
      - fit_method: "ols" (default) or "theil_sen" (outlier-robust slope).

    Adds per-window diagnostics to support robust selection:
      - dy: y_end - y_start
      - mono_frac: fraction of steps with Δy >= -threshold
      - down_steps: count of steps with Δy < -threshold (threshold may include down_step_min_frac)
      - pos_steps: count of steps with Δy > +mono_eps
      - rmse: fit RMSE within the window
      - snr: abs(dy)/rmse (simple SNR-like score)
      - mono_eps, min_delta_y, start_idx_used: constants per well (debug-friendly)
    """
    d = df_well.copy()
    d["time_s"] = pd.to_numeric(d["time_s"], errors="coerce")
    d["signal"] = pd.to_numeric(d["signal"], errors="coerce")
    d = d.dropna(subset=["time_s", "signal"]).sort_values("time_s").reset_index(drop=True)

    # drop inf/-inf as well
    mask_finite = np.isfinite(d["time_s"].to_numpy(dtype=float)) & np.isfinite(d["signal"].to_numpy(dtype=float))
    d = d.loc[mask_finite].reset_index(drop=True)

    # drop inf/-inf as well (repeated for safety)
    mask_finite = np.isfinite(d["time_s"].to_numpy(dtype=float)) & np.isfinite(d["signal"].to_numpy(dtype=float))
    d = d.loc[mask_finite].reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    # Detect step jumps and restrict to pre-jump segment
    jump_idx = _detect_step_jump(y, threshold_frac=0.25)
    if jump_idx is not None and jump_idx >= min_points - 1:
        # Keep only points up to and including jump_idx
        d = d.iloc[: jump_idx + 1].reset_index(drop=True)
        t = t[: jump_idx + 1]
        y = y[: jump_idx + 1]

    cols = [
        "t_start",
        "t_end",
        "n",
        "slope",
        "intercept",
        "r2",
        # robust / diagnostics for "few outliers but globally linear"
        "slope_trim1",
        "intercept_trim1",
        "r2_trim1",
        "slope_trim2",
        "intercept_trim2",
        "r2_trim2",
        "trim1_drop_idx",
        "trim2_drop_idx1",
        "trim2_drop_idx2",
        "outlier_count",
        "slope_half_drop_frac",
        "start_idx",
        "end_idx",
        "dy",
        "mono_frac",
        "down_steps",
        "pos_steps",
        "pos_steps_eps",
        "pos_eps",
        "rmse",
        "snr",
        "slope_se",
        "slope_t",
        "mono_eps",
        "min_delta_y",
        "start_idx_used",
    ]

    if len(t) < int(min_points):
        return pd.DataFrame(columns=cols)

    eps = float(mono_eps) if mono_eps is not None else _auto_mono_eps(y)
    min_dy = float(min_delta_y) if min_delta_y is not None else _auto_min_delta_y(y, eps)

    start_idx_used = 0
    if bool(find_start):
        start_idx_used = _find_start_index(
            t=t,
            y=y,
            mono_eps=eps,
            max_shift=int(start_max_shift),
            window=int(start_window),
            allow_down_steps=int(start_allow_down_steps),
            min_rise=min_dy,
        )

    wins = generate_candidate_windows(
        t,
        min_points=int(min_points),
        max_points=int(max_points),
        min_span_s=float(min_span_s),
        start_idx=int(start_idx_used),
    )
    min_t_start_s = float(min_t_start_s)
    if min_t_start_s > 0.0:
        wins = [(i0, i1) for (i0, i1) in wins if float(t[i0]) >= min_t_start_s]
    if not wins:
        return pd.DataFrame(columns=cols)

    _fit_fn = _fit_linear_theilsen if (str(fit_method).strip().lower() == "theil_sen") else _fit_linear

    cand: list[dict] = []
    for i0, i1 in wins:
        xw = t[i0 : i1 + 1]
        yw = y[i0 : i1 + 1]

        fr = _fit_fn(xw, yw)

        # --- monotonicity diagnostics: use lightly smoothed y to avoid "cut before noise" ---
        # We smooth only for mono_frac/down_steps/pos_steps/dy, NOT for the linear fit itself.
        if len(yw) >= 3:
            yw_mono = (
                pd.Series(yw)
                .rolling(window=3, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=float)
            )
        else:
            yw_mono = yw

        dy = float(yw_mono[-1] - yw_mono[0])
        step = np.diff(yw_mono)

        # default (so we never hit UnboundLocalError even if code changes later)
        mono_frac = 1.0
        down_steps = 0
        pos_steps = 0
        pos_steps_eps = 0
        pos_eps = float(max(1e-12, 0.25 * eps))  # "significant positive step" threshold

        if step.size > 0:
            # Optionally treat small downward dips as noise (only count "significant" drops)
            down_thresh = float(eps)
            if down_step_min_frac is not None and down_step_min_frac > 0.0:
                y_range = float(np.max(yw_mono) - np.min(yw_mono))
                if y_range > 0.0:
                    down_thresh = max(down_thresh, float(down_step_min_frac) * y_range)
            mono_frac = float(np.mean(step >= -down_thresh))
            down_steps = int(np.sum(step < -down_thresh))

            # loose: any increase counts (robust when eps is huge, but can be fooled by noise)
            pos_steps = int(np.sum(step > 0.0))

            # strict: only clearly positive steps count (protects when min_delta_y is disabled)
            pos_steps_eps = int(np.sum(step > pos_eps))

        # --- residual diagnostics (raw y, not smoothed) ---
        yhat = fr.slope * xw + fr.intercept
        res = yw - yhat

        rmse = float(np.sqrt(np.mean(res**2))) if len(yw) else np.nan
        snr = float(abs(dy) / (rmse + 1e-12)) if np.isfinite(rmse) else np.nan

        # slope uncertainty (standard error) and t-like score
        slope_se = np.nan
        slope_t = np.nan
        if len(xw) >= 3:
            dof = int(len(xw) - 2)
            x_mean = float(np.mean(xw))
            sxx = float(np.sum((xw - x_mean) ** 2))
            if dof > 0 and sxx > 0.0:
                sse = float(np.sum(res**2))
                sigma2 = sse / float(dof)
                slope_se = float(np.sqrt(sigma2 / sxx))
                slope_t = float(fr.slope / (slope_se + 1e-12))

        # count "large residual" points (outlier-like), robust scale
        sigma_res = _robust_sigma(res)
        thr = 3.5 * max(float(sigma_res), 0.5 * float(eps), 1e-12)
        outlier_count = int(np.sum(np.abs(res) > thr))

        # --- indices of points that would be dropped by trim (absolute indices in the well series) ---
        trim1_drop_idx = np.nan
        trim2_drop_idx1 = np.nan
        trim2_drop_idx2 = np.nan

        order = np.argsort(np.abs(res))[::-1] if len(res) else np.array([], dtype=int)
        if order.size >= 1:
            trim1_drop_idx = int(i0 + int(order[0]))
            trim2_drop_idx1 = int(i0 + int(order[0]))
        if order.size >= 2:
            trim2_drop_idx2 = int(i0 + int(order[1]))

        # --- trim-1 refit: remove the single worst residual point ---
        slope_trim1 = float(fr.slope)
        intercept_trim1 = float(fr.intercept)
        r2_trim1 = float(fr.r2)

        if len(xw) >= 5 and order.size >= 1:
            j = int(order[0])
            mask = np.ones(len(xw), dtype=bool)
            mask[j] = False
            fr_t1 = _fit_fn(xw[mask], yw[mask])
            slope_trim1 = float(fr_t1.slope)
            intercept_trim1 = float(fr_t1.intercept)
            r2_trim1 = float(fr_t1.r2)

        # --- trim-2 refit: remove the two worst residual points ---
        slope_trim2 = float(fr.slope)
        intercept_trim2 = float(fr.intercept)
        r2_trim2 = float(fr.r2)

        if len(xw) >= 6 and order.size >= 2:
            mask2 = np.ones(len(xw), dtype=bool)
            mask2[order[:2]] = False
            if int(np.sum(mask2)) >= 4:
                fr_t2 = _fit_fn(xw[mask2], yw[mask2])
                slope_trim2 = float(fr_t2.slope)
                intercept_trim2 = float(fr_t2.intercept)
                r2_trim2 = float(fr_t2.r2)

        # --- curvature-like: slope drop from first half -> second half ---
        slope_half_drop_frac = 0.0
        if len(xw) >= 8:
            mid = int(len(xw) // 2)
            fr_first = _fit_linear(xw[:mid], yw[:mid])
            fr_last = _fit_linear(xw[mid:], yw[mid:])
            s1 = float(fr_first.slope)
            s2 = float(fr_last.slope)
            if s1 > 0.0:
                slope_half_drop_frac = float(max(0.0, (s1 - s2) / s1))

        cand.append(
            {
                "t_start": fr.t_start,
                "t_end": fr.t_end,
                "n": fr.n,
                "slope": float(fr.slope),
                "intercept": float(fr.intercept),
                "r2": float(fr.r2),
                "slope_trim1": slope_trim1,
                "intercept_trim1": intercept_trim1,
                "r2_trim1": r2_trim1,
                "slope_trim2": slope_trim2,
                "intercept_trim2": intercept_trim2,
                "r2_trim2": r2_trim2,
                "trim1_drop_idx": trim1_drop_idx,
                "trim2_drop_idx1": trim2_drop_idx1,
                "trim2_drop_idx2": trim2_drop_idx2,
                "outlier_count": outlier_count,
                "slope_half_drop_frac": slope_half_drop_frac,
                "start_idx": int(i0),
                "end_idx": int(i1),
                "dy": dy,
                "mono_frac": mono_frac,
                "down_steps": down_steps,
                "pos_steps": pos_steps,
                "pos_steps_eps": pos_steps_eps,
                "pos_eps": pos_eps,
                "rmse": rmse,
                "snr": snr,
                "slope_se": slope_se,
                "slope_t": slope_t,
                "mono_eps": eps,
                "min_delta_y": min_dy,
                "start_idx_used": int(start_idx_used),
            }
        )

    return pd.DataFrame(cand, columns=cols)
