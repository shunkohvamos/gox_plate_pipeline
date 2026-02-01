# src/gox_plate_pipeline/fitting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import re

@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int
    t_start: float
    t_end: float


class FitSelectionError(RuntimeError):
    """Raised when no acceptable fitting window can be selected for a well."""


def add_well_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'row' (A-H) and 'col' (1-12) extracted from 'well' like 'A1'.
    Keeps original columns.
    """
    out = df.copy()
    row = out["well"].astype(str).str.extract(r"^([A-H])", expand=False)
    col = out["well"].astype(str).str.extract(r"^[A-H](\d{1,2})$", expand=False)

    if "row" not in out.columns:
        out["row"] = row
    out["col"] = pd.to_numeric(col, errors="coerce").astype("Int64")
    return out


def add_heat_time(df: pd.DataFrame, heat_times: list[float]) -> pd.DataFrame:
    """
    Map 'col' -> heat_min using heat_times list.
    col=1 maps to heat_times[0], col=2 -> heat_times[1], ...
    If col is outside the range, heat_min will be NaN.
    """
    out = add_well_coordinates(df)

    def _map(c: object) -> float:
        if pd.isna(c):
            return np.nan
        ci = int(c)
        if 1 <= ci <= len(heat_times):
            return float(heat_times[ci - 1])
        return np.nan

    out["heat_min"] = out["col"].apply(_map)
    return out


def _fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Ordinary least squares linear fit: y = a*x + b
    Returns slope/intercept and R^2.
    """
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FitResult(
        slope=float(a),
        intercept=float(b),
        r2=float(r2),
        n=int(len(x)),
        t_start=float(np.min(x)),
        t_end=float(np.max(x)),
    )


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)  # MAD -> sigma (normal)


def _percentile_range(y: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    lo, hi = np.percentile(y, [p_low, p_high])
    return float(hi - lo)


def _auto_mono_eps(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    rng = _percentile_range(y)
    sigma_dy = _robust_sigma(np.diff(y)) if y.size >= 2 else 0.0
    # eps is the threshold to call a decrease "significant"
    return float(max(1e-12, 0.01 * rng, 3.0 * sigma_dy))


def _auto_min_delta_y(y: np.ndarray, mono_eps: float) -> float:
    y = np.asarray(y, dtype=float)
    rng = _percentile_range(y)
    # NOTE:
    #   sigma(y) is inflated by the reaction trend itself (monotonic increase),
    #   so using it as "noise" makes min_delta_y unrealistically large.
    #   Use scale-based and step-noise-based terms instead.
    return float(max(1e-12, 0.02 * rng, 3.0 * float(mono_eps)))



def _find_start_index(
    t: np.ndarray,
    y: np.ndarray,
    mono_eps: float,
    max_shift: int = 5,
    window: int = 3,
    allow_down_steps: int = 1,
    min_rise: Optional[float] = None,
) -> int:
    """
    Conservative 'rise start' detection:
      - slide a small window from the beginning (up to max_shift)
      - accept earliest i0 where:
          * significant downs within the window are <= allow_down_steps
          * net rise in the window is >= min_rise (if provided)
    If not found, return 0.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    if n < window:
        return 0

    max_i0 = min(int(max_shift), n - window)
    for i0 in range(0, max_i0 + 1):
        yw = y[i0 : i0 + window]
        dy = np.diff(yw)
        down_steps = int(np.sum(dy < -float(mono_eps)))
        net = float(yw[-1] - yw[0])
        if down_steps <= int(allow_down_steps):
            if min_rise is None or net >= float(min_rise):
                return int(i0)

    return 0


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
) -> pd.DataFrame:
    """
    Return candidate fits for one well.

    df_well must have columns:
      - time_s (seconds)
      - signal (numeric)

    Adds per-window diagnostics to support robust selection:
      - dy: y_end - y_start
      - mono_frac: fraction of steps with Δy >= -mono_eps
      - down_steps: count of steps with Δy < -mono_eps
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

    # drop inf/-inf as well
    mask_finite = np.isfinite(d["time_s"].to_numpy(dtype=float)) & np.isfinite(d["signal"].to_numpy(dtype=float))
    d = d.loc[mask_finite].reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

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
    if not wins:
        return pd.DataFrame(columns=cols)

    cand: list[dict] = []
    for i0, i1 in wins:
        xw = t[i0 : i1 + 1]
        yw = y[i0 : i1 + 1]

        fr = _fit_linear(xw, yw)

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
            mono_frac = float(np.mean(step >= -eps))
            down_steps = int(np.sum(step < -eps))

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
            fr_t1 = _fit_linear(xw[mask], yw[mask])
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
                fr_t2 = _fit_linear(xw[mask2], yw[mask2])
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

def select_fit(
    cands: pd.DataFrame,
    method: str = "initial_positive",
    r2_min: float = 0.98,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
    min_delta_y: Optional[float] = None,
    mono_min_frac: float = 0.85,
    mono_max_down_steps: int = 1,
    min_pos_steps: int = 2,
    min_snr: float = 3.0,
    slope_drop_frac: float = 0.18,
    force_whole: bool = False,
    force_whole_n_min: int = 10,
    force_whole_r2_min: float = 0.985,
    force_whole_mono_min_frac: float = 0.70,
    # If the curve is curving/saturating, the longest-window slope will drop.
    # Only force whole-window when the slope drop is small (i.e., "linear enough").
    force_whole_max_slope_drop_frac: float = 0.06,
    # Allow the "whole" window even if 1-2 points hurt r2, as long as trim looks linear.
    force_whole_allow_trim1: bool = True,
    force_whole_allow_trim2: bool = True,
    force_whole_max_outliers: int = 2,   # <=2 points are "noise", >=3 are not
    force_whole_max_half_drop_frac: float = 0.15,
) -> pd.Series:



    """
    Select one fit from candidates.

    Hard rules (always enforced unless force_whole triggers an early return):
      - slope >= slope_min
      - if max_t_end is not None: t_end <= max_t_end
      - r2 >= r2_min (no silent fallback)
      - dy >= min_delta_y (auto-per-well if not provided)
      - mono_frac >= mono_min_frac
      - down_steps <= mono_max_down_steps
      - pos_steps >= min_pos_steps
      - snr >= min_snr

    method:
      - initial_positive:
          1) apply hard rules
          2) curvature-guard: keep windows whose slope is within (1 - slope_drop_frac) of the max slope
          3) choose earliest t_end; tie-breakers: higher r2, larger n, earlier t_start
      - best_r2:
          choose highest r2 among windows passing hard rules; tie-breakers: larger n, earlier t_end

    Note:
      - When force_whole returns early, the returned Series includes:
          select_method_used = "force_whole"
      - Otherwise, returned Series includes:
          select_method_used = method
    """
    if cands.empty:
        raise FitSelectionError("No candidate windows were generated.")

    required = {
        "dy",
        "mono_frac",
        "down_steps",
        "pos_steps",
        "pos_steps_eps",
        "pos_eps",
        "snr",
        "slope_se",
        "n",
        "t_end",
        "t_start",
        "slope",
        "r2",
    }


    missing = [c for c in sorted(required) if c not in cands.columns]
    if missing:
        raise ValueError(f"cands is missing required columns for robust selection: {missing}")

    c = cands.copy()

    # --- hard filter: slope ---
    c = c[c["slope"] >= float(slope_min)].copy()
    if c.empty:
        smin = float(cands["slope"].min())
        smax = float(cands["slope"].max())
        raise FitSelectionError(
            "No candidates left after filtering: all slopes were below slope_min "
            f"(slope_min={float(slope_min):.6g}, slope range=[{smin:.6g}, {smax:.6g}])."
        )

    # --- hard filter: time ---
    if max_t_end is not None:
        c = c[c["t_end"] <= float(max_t_end)].copy()
        if c.empty:
            tmin = float(cands["t_end"].min())
            tmax = float(cands["t_end"].max())
            raise FitSelectionError(
                "No candidates left after filtering: all windows ended after max_t_end "
                f"(max_t_end={float(max_t_end):.6g}, t_end range=[{tmin:.6g}, {tmax:.6g}])."
            )

    # --- optional: force whole/long window if sufficiently linear (evaluate BEFORE r2_min) ---
    if bool(force_whole):

        pool = c.copy()

        # prefer windows starting at detected start (start_idx_used), if available
        if "start_idx_used" in pool.columns and pool["start_idx_used"].notna().any():
            s0 = int(pool["start_idx_used"].dropna().iloc[0])
            whole_c = pool[pool["start_idx"] == s0].copy()
        else:
            whole_c = pool.copy()

        if not whole_c.empty:
            # choose the longest (largest n, then latest t_end)
            whole = whole_c.sort_values(["n", "t_end"], ascending=[False, False]).iloc[0].copy()

            # guard 1: only force whole-window if slope does not drop much vs max slope
            max_slope = float(pool["slope"].max())
            whole_slope = float(whole["slope"])
            if max_slope > 0.0:
                slope_drop = (max_slope - whole_slope) / max_slope
            else:
                slope_drop = 0.0

            # guard 2: saturation/curvature-like (first half slope -> second half slope)
            half_drop = float(whole.get("slope_half_drop_frac", 0.0))

            # guard 3: allow up to 2 outliers via trim-1/trim-2
            r2_raw = float(whole["r2"])
            r2_t1 = float(whole.get("r2_trim1", r2_raw))
            r2_t2 = float(whole.get("r2_trim2", r2_raw))
            outliers = int(whole.get("outlier_count", 0))

            # If >=3 outliers, do NOT treat as "noise"
            if outliers <= int(force_whole_max_outliers):
                use_trim2 = (
                    bool(force_whole_allow_trim2)
                    and (r2_raw < float(force_whole_r2_min))
                    and (r2_t2 >= float(force_whole_r2_min))
                )
                use_trim1 = (
                    (not use_trim2)
                    and bool(force_whole_allow_trim1)
                    and (r2_raw < float(force_whole_r2_min))
                    and (r2_t1 >= float(force_whole_r2_min))
                )
                r2_ok = (r2_raw >= float(force_whole_r2_min)) or use_trim1 or use_trim2
            else:
                use_trim1 = False
                use_trim2 = False
                r2_ok = False

            # also require reaction-likeness (aligns force_whole with the main gate)
            if min_delta_y is None:
                if "min_delta_y" in whole.index and pd.notna(whole.get("min_delta_y", np.nan)):
                    min_dy_force = float(whole["min_delta_y"])
                else:
                    min_dy_force = 0.0
            else:
                min_dy_force = float(min_delta_y)

            use_strict_pos_force = bool(min_dy_force <= 0.0)
            pos_steps_force = int(whole.get("pos_steps_eps", whole.get("pos_steps", 0))) if use_strict_pos_force else int(whole.get("pos_steps", 0))

            dy_force = float(whole.get("dy", np.nan))
            snr_force = float(whole.get("snr", np.nan))
            down_steps_force = int(whole.get("down_steps", 10**9))

            dy_ok = (min_dy_force <= 0.0) or (np.isfinite(dy_force) and dy_force >= min_dy_force)
            snr_ok = np.isfinite(snr_force) and (snr_force >= float(min_snr))
            down_ok = down_steps_force <= int(mono_max_down_steps)
            pos_ok = pos_steps_force >= int(min_pos_steps)

            if (
                int(whole["n"]) >= int(force_whole_n_min)
                and r2_ok
                and float(whole["mono_frac"]) >= float(force_whole_mono_min_frac)
                and float(slope_drop) <= float(force_whole_max_slope_drop_frac)
                and float(half_drop) <= float(force_whole_max_half_drop_frac)
                and dy_ok
                and snr_ok
                and down_ok
                and pos_ok
            ):

                sel = whole.copy()

                if use_trim2:
                    sel["slope"] = float(sel.get("slope_trim2", sel["slope"]))
                    sel["intercept"] = float(sel.get("intercept_trim2", sel["intercept"]))
                    sel["r2"] = float(r2_t2)
                    sel["select_method_used"] = "force_whole_trim2"
                elif use_trim1:
                    sel["slope"] = float(sel.get("slope_trim1", sel["slope"]))
                    sel["intercept"] = float(sel.get("intercept_trim1", sel["intercept"]))
                    sel["r2"] = float(r2_t1)
                    sel["select_method_used"] = "force_whole_trim1"
                else:
                    sel["select_method_used"] = "force_whole"

                return sel

    # --- hard filter: r2 ---
    ok0 = c[c["r2"] >= float(r2_min)].copy()
    if ok0.empty:
        best = c.sort_values(["r2", "n", "t_end"], ascending=[False, False, True]).iloc[0]
        raise FitSelectionError(
            "No candidates met r2_min="
            f"{float(r2_min):.4g}. Best was r2={float(best['r2']):.4f} "
            f"(n={int(best['n'])}, t_end={float(best['t_end']):.3g}s). "
            "If you want to accept lower-quality fits, lower --r2_min."
        )

    # pool for rejection breakdown: r2-pass pool (BEFORE reaction-likeness filters)
    pre = ok0.copy()

    # min_delta_y: auto from per-well constants if available
    if min_delta_y is None:
        if "min_delta_y" in ok0.columns and ok0["min_delta_y"].notna().any():
            min_dy = float(ok0["min_delta_y"].dropna().iloc[0])
        else:
            min_dy = 0.0
    else:
        min_dy = float(min_delta_y)

    # --- reaction-likeness filters (apply ONCE) ---
    ok = ok0.copy()

    if min_dy > 0.0:
        ok = ok[ok["dy"] >= float(min_dy)].copy()

    ok = ok[ok["mono_frac"] >= float(mono_min_frac)].copy()
    ok = ok[ok["down_steps"] <= int(mono_max_down_steps)].copy()

    # If dy is disabled (min_dy<=0), tighten "positive step" definition to avoid noise-driven false positives.
    use_strict_pos = bool(min_dy <= 0.0)
    pos_col = "pos_steps_eps" if use_strict_pos else "pos_steps"
    ok = ok[ok[pos_col] >= int(min_pos_steps)].copy()

    ok = ok[ok["snr"] >= float(min_snr)].copy()

    if ok.empty:
        # rejection breakdown on the pool that already satisfied r2_min
        n0 = int(len(pre))
        pos_eps_val = (
            float(pre["pos_eps"].dropna().iloc[0])
            if ("pos_eps" in pre.columns and pre["pos_eps"].notna().any())
            else np.nan
        )

        fail_dy = int(np.sum(pre["dy"] < float(min_dy))) if min_dy > 0.0 else 0
        fail_mono = int(np.sum(pre["mono_frac"] < float(mono_min_frac)))
        fail_down = int(np.sum(pre["down_steps"] > int(mono_max_down_steps)))
        fail_pos = int(np.sum(pre[pos_col] < int(min_pos_steps)))
        fail_snr = int(np.sum(pre["snr"] < float(min_snr)))

        best = pre.sort_values(["r2", "n", "t_end"], ascending=[False, False, True]).iloc[0]

        raise FitSelectionError(
            "All candidates were rejected by reaction-likeness filters. "
            f"(min_delta_y={min_dy:.4g}, mono_min_frac={float(mono_min_frac):.3g}, "
            f"mono_max_down_steps={int(mono_max_down_steps)}, min_pos_steps={int(min_pos_steps)}, "
            f"min_snr={float(min_snr):.3g}, pos_steps_mode={pos_col}, pos_eps={pos_eps_val:.4g}) "
            f"Rejected counts out of {n0}: "
            + (
                f"dy<{min_dy:.4g}: {fail_dy}, " if min_dy > 0.0 else ""
            )
            + f"mono_frac<{float(mono_min_frac):.3g}: {fail_mono}, "
              f"down_steps>{int(mono_max_down_steps)}: {fail_down}, "
              f"{pos_col}<{int(min_pos_steps)}: {fail_pos}, "
              f"snr<{float(min_snr):.3g}: {fail_snr}. "
            f"Best (by r2 within r2-pass pool): r2={float(best['r2']):.4f}, "
            f"dy={float(best['dy']):.4g}, mono_frac={float(best['mono_frac']):.3g}, "
            f"down_steps={int(best['down_steps'])}, "
            f"{pos_col}={int(best[pos_col])}, snr={float(best['snr']):.3g} "
            f"(n={int(best['n'])}, t_end={float(best['t_end']):.3g}s)."
        )


    if method == "initial_positive":
        s = ok["slope"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        s_sorted = np.sort(s)

        # robust reference slope: median of top-K slopes (K<=5)
        k = int(min(5, s_sorted.size))
        slope_ref = float(np.median(s_sorted[-k:])) if k > 0 else float(ok["slope"].max())

        floor = slope_ref * (1.0 - float(slope_drop_frac))
        keep = ok[ok["slope"] >= float(floor)].copy()
        if keep.empty:
            keep = ok

        keep = keep.sort_values(
            ["t_end", "r2", "slope_se", "n", "t_start"],
            ascending=[True, False, True, False, True],
        )
        sel = keep.iloc[0].copy()
        sel["select_method_used"] = "initial_positive"
        sel["slope_ref_used"] = slope_ref
        sel["slope_floor_used"] = floor
        return sel

    if method == "best_r2":
        keep = ok.sort_values(["r2", "slope_se", "n", "t_end"], ascending=[False, True, False, True])
        sel = keep.iloc[0].copy()
        sel["select_method_used"] = "best_r2"
        return sel

    if method == "best_score":
        tmp = ok.copy()

        # safe defaults
        if "outlier_count" not in tmp.columns:
            tmp["outlier_count"] = 0
        if "slope_half_drop_frac" not in tmp.columns:
            tmp["slope_half_drop_frac"] = 0.0
        if "slope_t" not in tmp.columns:
            tmp["slope_t"] = np.nan
        if "slope_se" not in tmp.columns:
            tmp["slope_se"] = np.nan

        def _rank01(s: pd.Series, high_is_good: bool) -> pd.Series:
            x = pd.to_numeric(s, errors="coerce")
            n_all = int(len(x))
            n_valid = int(x.notna().sum())

            if n_all <= 1 or n_valid <= 1:
                return pd.Series(np.zeros(n_all, dtype=float), index=x.index)

            xf = x.dropna()
            if xf.size > 0 and float(xf.max() - xf.min()) == 0.0:
                return pd.Series(np.zeros(n_all, dtype=float), index=x.index)

            r = x.rank(method="average", ascending=not high_is_good)
            denom = float(max(1, n_valid - 1))
            out = (r - 1.0) / denom
            return out.fillna(0.0)


        r2_s = _rank01(tmp["r2"], True)
        snr_s = _rank01(tmp["snr"], True)
        n_s = _rank01(tmp["n"], True)
        t_s = _rank01(tmp["t_end"], False)  # earlier is better
        se_s = _rank01(tmp["slope_se"], False)  # smaller SE is better
        ht_s = _rank01(tmp["slope_half_drop_frac"], False)  # less curvature is better
        out_s = _rank01(tmp["outlier_count"], False)  # fewer outliers is better
        st_s = _rank01(tmp["slope_t"].abs(), True)  # larger |t| is better (if available)

        # Composite score (rank-based, scale-free)
        tmp["_score"] = (
            2.0 * r2_s
            + 1.0 * snr_s
            + 0.8 * n_s
            + 0.4 * t_s
            + 0.3 * se_s
            + 0.6 * ht_s
            + 0.5 * out_s
            + 0.4 * st_s
        )

        keep = tmp.sort_values(
            ["_score", "t_end", "r2", "n"],
            ascending=[False, True, False, False],
        )
        sel = keep.iloc[0].copy()
        sel["select_method_used"] = "best_score"
        sel["score"] = float(sel["_score"])
        return sel

    raise ValueError(f"Unknown selection method: {method}")



def _enforce_final_safety(
    sel: pd.Series,
    slope_min: float = 0.0,
    r2_min: float = 0.98,
    max_t_end: Optional[float] = 240.0,
    min_delta_y: Optional[float] = None,
    mono_min_frac: float = 0.85,
    mono_max_down_steps: int = 1,
    min_pos_steps: int = 2,
    min_snr: float = 3.0,
) -> None:
    """
    Final safety gate (must not silently pass):
      - slope must be >= slope_min
      - r2 must be >= r2_min
      - if max_t_end is set: t_end must be <= max_t_end
      - reaction-likeness checks must also hold (dy/monotonicity/SNR)
    """
    # --- slope / r2 / time ---
    slope = float(sel["slope"])
    r2 = float(sel["r2"])
    t_end = float(sel["t_end"])

    if slope < float(slope_min):
        raise FitSelectionError(
            f"Final safety triggered: selected slope {slope:.6g} < slope_min {float(slope_min):.6g}"
        )

    if r2 < float(r2_min):
        raise FitSelectionError(
            f"Final safety triggered: selected r2 {r2:.6g} < r2_min {float(r2_min):.6g}"
        )

    if max_t_end is not None and t_end > float(max_t_end):
        raise FitSelectionError(
            f"Final safety triggered: selected t_end {t_end:.6g} > max_t_end {float(max_t_end):.6g}"
        )

    # --- reaction-likeness required fields ---
    for k in ["dy", "mono_frac", "down_steps", "pos_steps", "snr"]:
        if k not in sel.index:
            raise FitSelectionError(f"Final safety triggered: missing '{k}' in selected fit.")

    dy = float(sel["dy"])
    mono_frac = float(sel["mono_frac"])
    down_steps = int(sel["down_steps"])
    pos_steps = int(sel["pos_steps"])
    snr = float(sel["snr"])

    if min_delta_y is None:
        if "min_delta_y" in sel.index and pd.notna(sel["min_delta_y"]):
            min_dy = float(sel["min_delta_y"])
        else:
            min_dy = 0.0
    else:
        min_dy = float(min_delta_y)

    if min_dy > 0.0 and dy < min_dy:
        raise FitSelectionError(
            f"Final safety triggered: selected dy {dy:.6g} < min_delta_y {min_dy:.6g}"
        )

    if mono_frac < float(mono_min_frac):
        raise FitSelectionError(
            f"Final safety triggered: selected mono_frac {mono_frac:.6g} < mono_min_frac {float(mono_min_frac):.6g}"
        )

    if down_steps > int(mono_max_down_steps):
        raise FitSelectionError(
            f"Final safety triggered: selected down_steps {down_steps} > mono_max_down_steps {int(mono_max_down_steps)}"
        )

    # If dy is disabled (min_dy<=0), use stricter positive-step metric when available.
    if min_dy <= 0.0 and ("pos_steps_eps" in sel.index) and pd.notna(sel.get("pos_steps_eps", np.nan)):
        pos_steps_val = int(sel["pos_steps_eps"])
        pos_label = "pos_steps_eps"
    else:
        pos_steps_val = int(sel["pos_steps"])
        pos_label = "pos_steps"

    if pos_steps_val < int(min_pos_steps):
        raise FitSelectionError(
            f"Final safety triggered: selected {pos_label} {pos_steps_val} < min_pos_steps {int(min_pos_steps)}"
        )

    if snr < float(min_snr):
        raise FitSelectionError(
            f"Final safety triggered: selected snr {snr:.6g} < min_snr {float(min_snr):.6g}"
        )

def _format_heat(heat_min: object) -> str:
    if heat_min is None or (isinstance(heat_min, float) and np.isnan(heat_min)):
        return "NA"
    try:
        return f"{float(heat_min):g} min"
    except Exception:
        return str(heat_min)


def plot_fit_diagnostic(
    df_well: pd.DataFrame,
    meta: dict,
    selected: Optional[pd.Series],
    status: str,
    exclude_reason: str,
    out_png: Path,
) -> None:
    """
    Diagnostic plot for one well.

    Requirements:
      - If excluded: do NOT draw fit line (show "cannot fit" style)
      - If ok: draw dotted line across full plot x-range (edge-to-edge)
      - Title is minimal: plate_id + well + polymer_id (and "EXCLUDED" when excluded)
      - Info box contains: heat_min (+ sample_name if present) and (if ok) slope + R² + n
      - Use unicode R² (superscript 2)
      - Line should be fine dotted
    """
    d = df_well.copy()
    d["time_s"] = pd.to_numeric(d["time_s"], errors="coerce")
    d["signal"] = pd.to_numeric(d["signal"], errors="coerce")
    d = d.dropna(subset=["time_s", "signal"]).sort_values("time_s").reset_index(drop=True)

    mask_finite = np.isfinite(d["time_s"].to_numpy(dtype=float)) & np.isfinite(d["signal"].to_numpy(dtype=float))
    d = d.loc[mask_finite].reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    well = str(meta.get("well", "NA"))
    polymer_id = str(meta.get("polymer_id", "") or "")
    heat_txt = _format_heat(meta.get("heat_min", np.nan))

    _raw_sample = meta.get("sample_name", "")
    sample_name = "" if pd.isna(_raw_sample) else str(_raw_sample).strip()
    # also treat string tokens as "missing"
    if sample_name.lower() in {"nan", "none", "na", "n/a"}:
        sample_name = ""


    # Font: prefer Arial if available (WSL/Linux may not have it)
    font_candidates = ["Arial", "Liberation Sans", "DejaVu Sans"]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    chosen_font = next((f for f in font_candidates if f in available_fonts), "DejaVu Sans")

    # Colors
    c_point = "#0072B2"   # blue
    c_fit = "0.25"        # dark gray
    c_drop = "#D55E00"    # vermillion

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen_font, "DejaVu Sans", "sans-serif"],
        }
    ):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))

        # highlight only the points that were explicitly excluded by trim
        drop_mask = np.zeros(len(t), dtype=bool)

        if status == "ok" and selected is not None:
            method_used = str(selected.get("select_method_used", ""))
            if ("trim1" in method_used) or ("trim2" in method_used):
                for key in ["trim1_drop_idx", "trim2_drop_idx1", "trim2_drop_idx2"]:
                    if key in selected.index and pd.notna(selected[key]):
                        j = int(selected[key])
                        if 0 <= j < len(drop_mask):
                            drop_mask[j] = True

        # paper-like axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.5")
        ax.spines["bottom"].set_color("0.5")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", labelsize=9, width=0.8, colors="0.35")

        # thin polyline to connect points (very light)
        # Draw ONLY when a fit was successfully selected (status == "ok").
        if status == "ok" and selected is not None and len(t) > 1:
            t_line = t[~drop_mask]
            y_line = y[~drop_mask]
            if t_line.size >= 2:
                ax.plot(
                    t_line,
                    y_line,
                    linewidth=0.6,
                    alpha=0.25,
                    color="0.5",
                    zorder=2.6,
                )


        # scatter
        if len(t) > 0:
            ax.scatter(t[~drop_mask], y[~drop_mask], s=16, linewidths=0.0, color=c_point, zorder=3)
            if np.any(drop_mask):
                ax.scatter(t[drop_mask], y[drop_mask], s=16, linewidths=0.0, color=c_drop, zorder=4)


        x0 = float(np.min(t)) if len(t) else 0.0
        x1 = float(np.max(t)) if len(t) else 1.0

        slope_txt = "NA"
        r2_txt = "NA"
        n_txt = "NA"

        if status == "ok" and selected is not None and len(t) > 0:
            slope = float(selected["slope"])
            intercept = float(selected["intercept"])
            r2 = float(selected["r2"])
            n_txt = str(int(selected.get("n", np.nan))) if pd.notna(selected.get("n", np.nan)) else "NA"

            slope_txt = f"{slope:.3g}"
            r2_txt = f"{r2:.3f}"

            # selected window shading
            t0 = float(selected["t_start"])
            t1 = float(selected["t_end"])
            ax.axvspan(t0, t1, facecolor="0.85", alpha=0.18, zorder=2)

            # fit line across full plot x-range (edge-to-edge)
            # Use current axis limits so the selected line is always recognizable.
            xa, xb = ax.get_xlim()
            xx = np.array([xa, xb], dtype=float)
            yy = slope * xx + intercept
            ax.plot(
                xx,
                yy,
                linestyle=(0, (1, 2)),
                linewidth=1.3,
                alpha=0.9,
                color=c_fit,
                zorder=2.7,
            )
            ax.set_xlim(xa, xb)



        plate_txt = str(meta.get("plate_id", "NA"))
        poly_txt = polymer_id or "NA"
        base_title = f"{plate_txt} | Well {well} | {poly_txt}"
        ax.set_title(base_title if status == "ok" else f"{base_title} | EXCLUDED", fontsize=11, pad=6)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Signal (a.u.)", fontsize=10)

        # info box
        # EXCLUDED: show ONLY heat (no sample, no reason).
        info_lines = [f"heat: {heat_txt}"]

        if status == "ok":
            if sample_name:
                info_lines.append(f"sample: {sample_name}")
            info_lines.append(f"slope: {slope_txt}")
            info_lines.append(f"R²: {r2_txt}")
            info_lines.append(f"n: {n_txt}")


        ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
            zorder=10,
        )

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

def _normalize_exclude_reason(reason: object) -> str:
    """
    Normalize verbose FitSelectionError messages into coarse buckets.

    Buckets (English):
      - R² < r2_min
      - Δy < min_delta_y
      - Monotonicity / steps
      - SNR < min_snr
      - t_end > max_t_end
      - Slope < slope_min
      - No candidate windows
      - Other
    """
    s = "" if reason is None else str(reason)
    s = s.strip()
    if (not s) or (s.lower() in {"nan", "none"}):
        return "Other"

    low = s.lower()

    # Case: multi-failure message with per-filter rejection counts
    # e.g. "Rejected counts out of N: dy<...: X, mono_frac<...: Y, down_steps>...: Z, pos_steps...: W, snr<...: V"
    if "rejected counts out of" in low:

        def _m(pat: str) -> int:
            m = re.search(pat, s)
            return int(m.group(1)) if m else 0

        dy = _m(r"dy<[^:]*:\s*(\d+)")
        mono = _m(r"mono_frac<[^:]*:\s*(\d+)")
        down = _m(r"down_steps>[^:]*:\s*(\d+)")
        pos = _m(r"(?:pos_steps_eps|pos_steps)<[^:]*:\s*(\d+)")
        snr = _m(r"snr<[^:]*:\s*(\d+)")

        mono_total = int(mono + down + pos)

        # choose the dominant failure (count-based); tie breaks by this order
        candidates = [
            ("Δy < min_delta_y", dy),
            ("Monotonicity / steps", mono_total),
            ("SNR < min_snr", snr),
        ]
        best = max(candidates, key=lambda x: x[1])
        if best[1] > 0:
            return best[0]

        # fallback if counts were not parsable
        if ("dy<" in low) or ("min_delta_y" in low) or ("selected dy" in low):
            return "Δy < min_delta_y"
        if ("mono_frac" in low) or ("down_steps" in low) or ("pos_steps" in low):
            return "Monotonicity / steps"
        if ("snr" in low) or ("min_snr" in low):
            return "SNR < min_snr"
        return "Other"

    # Single-failure / early-return style messages
    if ("max_t_end" in low) or ("ended after max_t_end" in low):
        return "t_end > max_t_end"

    # NOTE: "R² < r2_min" bucket (explicit square)
    if ("r2_min" in low) or ("no candidates met r2_min" in low) or ("selected r2" in low):
        return "R² < r2_min"

    if ("min_delta_y" in low) or ("dy<" in low) or ("selected dy" in low):
        return "Δy < min_delta_y"

    if ("mono_frac" in low) or ("down_steps" in low) or ("pos_steps" in low):
        return "Monotonicity / steps"

    if ("min_snr" in low) or ("snr<" in low) or ("selected snr" in low):
        return "SNR < min_snr"

    if ("slope_min" in low) or ("slope range" in low) or ("selected slope" in low):
        return "Slope < slope_min"

    if ("no candidate windows were generated" in low) or ("no candidates were generated" in low):
        return "No candidate windows"

    return "Other"
    
def write_fit_qc_report(
    selected: pd.DataFrame,
    out_dir: Path,
    max_t_end: Optional[float] = None,
    prefix: str = "fit_qc",
) -> Path:
    """
    Write a lightweight QC report for fit selection quality.

    Outputs (in out_dir):
      - {prefix}_summary_overall.csv
      - {prefix}_summary_by_plate.csv        (if plate_id exists)
      - {prefix}_summary_by_heat.csv         (if heat_min exists)

      - {prefix}_t_end_hist.png
      - {prefix}_slope_vs_t_end.png

      - {prefix}_select_method_counts.csv    (OK only, if select_method_used exists)
      - {prefix}_select_method_bar.png       (OK only)

      - {prefix}_r2_hist.png                 (OK only)
      - {prefix}_mono_frac_hist.png          (OK only)
      - {prefix}_snr_hist_log10.png          (OK only)

      - {prefix}_exclude_reason_norm_counts.csv   (EXCLUDED only)
      - {prefix}_exclude_reason_norm_bar.png      (EXCLUDED only)

      - {prefix}_report.md


    Returns path to the markdown report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = selected.copy()

    # --- overall ok/excluded rate ---
    if "status" not in df.columns:
        raise ValueError("selected is missing required column: 'status'")

    status = df["status"].astype(str)
    n_total = int(len(df))
    n_ok = int((status == "ok").sum())
    n_ex = int(n_total - n_ok)
    ok_rate = float(n_ok / n_total) if n_total > 0 else np.nan

    overall = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_ok": n_ok,
                "n_excluded": n_ex,
                "ok_rate": ok_rate,
            }
        ]
    )
    overall_csv = out_dir / f"{prefix}_summary_overall.csv"
    overall.to_csv(overall_csv, index=False)

    # --- by plate (optional but highly informative) ---
    by_plate_csv = None
    if "plate_id" in df.columns:
        tmp = df.copy()
        tmp["is_ok"] = (tmp["status"].astype(str) == "ok")
        by_plate = (
            tmp.groupby("plate_id", dropna=False)
            .agg(total=("status", "size"), ok=("is_ok", "sum"))
            .reset_index()
        )
        by_plate["excluded"] = by_plate["total"] - by_plate["ok"]
        by_plate["ok_rate"] = by_plate["ok"] / by_plate["total"]
        by_plate_csv = out_dir / f"{prefix}_summary_by_plate.csv"
        by_plate.to_csv(by_plate_csv, index=False)

    # --- by heat (optional but catches “late-time only fails” etc.) ---
    by_heat_csv = None
    if "heat_min" in df.columns:
        tmp = df.copy()
        tmp["is_ok"] = (tmp["status"].astype(str) == "ok")
        tmp["heat_min_num"] = pd.to_numeric(tmp["heat_min"], errors="coerce")
        by_heat = (
            tmp.groupby("heat_min_num", dropna=False)
            .agg(total=("status", "size"), ok=("is_ok", "sum"))
            .reset_index()
            .rename(columns={"heat_min_num": "heat_min"})
            .sort_values("heat_min")
        )
        by_heat["excluded"] = by_heat["total"] - by_heat["ok"]
        by_heat["ok_rate"] = by_heat["ok"] / by_heat["total"]
        by_heat_csv = out_dir / f"{prefix}_summary_by_heat.csv"
        by_heat.to_csv(by_heat_csv, index=False)

    # --- OK-only frame for distribution plots ---
    ok = df[df["status"].astype(str) == "ok"].copy()

    # --- t_end distribution & slope vs t_end correlation (OK only) ---
    if "t_end" not in ok.columns:
        ok["t_end"] = np.nan
    if "slope" not in ok.columns:
        ok["slope"] = np.nan

    ok["t_end"] = pd.to_numeric(ok["t_end"], errors="coerce")
    ok["slope"] = pd.to_numeric(ok["slope"], errors="coerce")

    t_end = ok["t_end"].to_numpy(dtype=float)
    slope = ok["slope"].to_numpy(dtype=float)
    mask = np.isfinite(t_end) & np.isfinite(slope)
    t_end_f = t_end[mask]
    slope_f = slope[mask]
    n_corr = int(t_end_f.size)

    pearson_r = np.nan
    spearman_rho = np.nan
    if n_corr >= 2:
        pearson_r = float(np.corrcoef(t_end_f, slope_f)[0, 1])
        rx = pd.Series(t_end_f).rank(method="average")
        ry = pd.Series(slope_f).rank(method="average")
        spearman_rho = float(rx.corr(ry))

    # --- plot: t_end histogram ---
    png_hist = out_dir / f"{prefix}_t_end_hist.png"
    plt.figure()
    if np.isfinite(t_end).sum() > 0:
        te = t_end[np.isfinite(t_end)]
        plt.hist(te, bins=30)
        if max_t_end is not None and np.isfinite(float(max_t_end)):
            plt.axvline(float(max_t_end), linestyle=(0, (4, 2)))
        plt.title("Selected t_end Distribution")
        plt.xlabel("t_end (s)")
        plt.ylabel("Count")
    else:
        ax = plt.gca()
        ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
        plt.title("Selected t_end Distribution")
        plt.xlabel("t_end (s)")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png_hist, dpi=200)
    plt.close()

    # --- plot: slope vs t_end scatter ---
    png_scatter = out_dir / f"{prefix}_slope_vs_t_end.png"
    plt.figure()
    ax = plt.gca()
    if n_corr > 0:
        ax.scatter(t_end_f, slope_f)
        ax.set_title("Slope vs t_end")
        ax.set_xlabel("t_end (s)")
        ax.set_ylabel("Slope (a.u./s)")
        txt = (
            f"N={n_corr}\n"
            f"Pearson r={pearson_r:.3f}\n"
            f"Spearman ρ={spearman_rho:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
    else:
        ax.text(0.5, 0.5, "No OK fits with finite t_end & slope", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Slope vs t_end")
        ax.set_xlabel("t_end (s)")
        ax.set_ylabel("Slope (a.u./s)")
    plt.tight_layout()
    plt.savefig(png_scatter, dpi=200)
    plt.close()

    # --- NEW: select_method_used breakdown (OK only) ---
    method_counts_csv = None
    png_method_bar = None
    method_col = None
    if "select_method_used" in ok.columns:
        method_col = "select_method_used"
    elif "select_method" in ok.columns:
        method_col = "select_method"

    method_lines: list[str] = []
    force_ok_frac = np.nan
    force_all_frac = np.nan

    if method_col is not None:
        m = ok[method_col].fillna("NA").astype(str)
        if len(m) > 0:
            vc = m.value_counts(dropna=False)
            method_counts = vc.rename_axis("select_method_used").reset_index(name="count")
            method_counts["fraction"] = method_counts["count"] / float(method_counts["count"].sum())

            method_counts_csv = out_dir / f"{prefix}_select_method_counts.csv"
            method_counts.to_csv(method_counts_csv, index=False)

            # bar plot
            png_method_bar = out_dir / f"{prefix}_select_method_bar.png"
            plt.figure()
            plt.bar(method_counts["select_method_used"].astype(str), method_counts["count"].to_numpy(dtype=int))
            plt.xticks(rotation=45, ha="right")
            plt.title("select_method_used (OK only)")
            plt.xlabel("method")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(png_method_bar, dpi=200)
            plt.close()

            # force_whole firing rate
            force_mask_ok = m.str.startswith("force_whole")
            force_ok_frac = float(force_mask_ok.mean())

            if (method_col in df.columns) and (n_total > 0):
                m_all = df[method_col].fillna("NA").astype(str)
                force_all_frac = float(m_all.str.startswith("force_whole").mean())

            for _, r in method_counts.iterrows():
                name = str(r["select_method_used"])
                cnt = int(r["count"])
                frac = float(r["fraction"]) * 100.0
                method_lines.append(f"- {name}: {cnt} ({frac:.1f}%)")

    # r2 hist
    png_r2 = out_dir / f"{prefix}_r2_hist.png"
    r2v = ok["r2"].to_numpy(dtype=float)
    r2v = r2v[np.isfinite(r2v)]

    plt.figure()
    if r2v.size > 0:
        plt.hist(r2v, bins=30)
        plt.title("Selected R² Distribution")
        plt.xlabel("R²")
        plt.ylabel("Count")
    else:
        ax = plt.gca()
        ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
        plt.title("Selected R² Distribution")
        plt.xlabel("R²")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png_r2, dpi=200)
    plt.close()

    # mono_frac hist
    png_mono = out_dir / f"{prefix}_mono_frac_hist.png"
    mfv = ok["mono_frac"].to_numpy(dtype=float)
    mfv = mfv[np.isfinite(mfv)]
    plt.figure()
    if mfv.size > 0:
        plt.hist(mfv, bins=30)
        plt.title("Selected mono_frac Distribution")
        plt.xlabel("mono_frac")
        plt.ylabel("Count")
    else:
        ax = plt.gca()
        ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
        plt.title("Selected mono_frac Distribution")
        plt.xlabel("mono_frac")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png_mono, dpi=200)
    plt.close()

    # snr hist (log10)
    png_snr = out_dir / f"{prefix}_snr_hist_log10.png"
    snv = ok["snr"].to_numpy(dtype=float)
    snv = snv[np.isfinite(snv)]
    snv = snv[snv > 0.0]
    plt.figure()
    if snv.size > 0:
        sn_log = np.log10(snv + 1e-12)
        plt.hist(sn_log, bins=30)
        plt.title("Selected SNR Distribution (log10)")
        plt.xlabel("log10(snr)")
        plt.ylabel("Count")
    else:
        ax = plt.gca()
        ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
        plt.title("Selected SNR Distribution (log10)")
        plt.xlabel("log10(snr)")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png_snr, dpi=200)
    plt.close()

    # --- NEW: exclude_reason normalization & breakdown (EXCLUDED only) ---
    ex_counts_csv = None
    png_ex_bar = None
    ex_lines: list[str] = []

    ex = df[df["status"].astype(str) != "ok"].copy()
    if "exclude_reason" not in ex.columns:
        ex["exclude_reason"] = ""

    if len(ex) > 0:
        ex["exclude_reason_norm"] = ex["exclude_reason"].apply(_normalize_exclude_reason)

        vc = ex["exclude_reason_norm"].fillna("NA").astype(str).value_counts(dropna=False)
        ex_counts = vc.rename_axis("exclude_reason_norm").reset_index(name="count")
        ex_counts["fraction"] = ex_counts["count"] / float(ex_counts["count"].sum())

        ex_counts_csv = out_dir / f"{prefix}_exclude_reason_norm_counts.csv"
        ex_counts.to_csv(ex_counts_csv, index=False)

        png_ex_bar = out_dir / f"{prefix}_exclude_reason_norm_bar.png"
        plt.figure()
        plt.bar(ex_counts["exclude_reason_norm"].astype(str), ex_counts["count"].to_numpy(dtype=int))
        plt.xticks(rotation=45, ha="right")
        plt.title("exclude_reason_norm (EXCLUDED only)")
        plt.xlabel("reason")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(png_ex_bar, dpi=200)
        plt.close()

        for _, r in ex_counts.iterrows():
            name = str(r["exclude_reason_norm"])
            cnt = int(r["count"])
            frac = float(r["fraction"]) * 100.0
            ex_lines.append(f"- {name}: {cnt} ({frac:.1f}%)")

    # --- markdown report ---
    md_path = out_dir / f"{prefix}_report.md"


    te_ok = ok["t_end"].dropna()
    te_stats = {}
    if len(te_ok) > 0:
        qs = te_ok.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        te_stats = {f"q{int(k*100):02d}": float(v) for k, v in qs.items()}
        te_min = float(te_ok.min())
        te_max = float(te_ok.max())
    else:
        te_min = np.nan
        te_max = np.nan

    thresholds = [30.0, 60.0, 120.0, 240.0]
    if max_t_end is not None and np.isfinite(float(max_t_end)):
        mt = float(max_t_end)
        if mt not in thresholds:
            thresholds.append(mt)
    thresholds = sorted(set(thresholds))

    frac_lines = []
    if len(te_ok) > 0:
        for th in thresholds:
            frac = float((te_ok <= th).mean())
            frac_lines.append(f"- t_end ≤ {th:g} s : {frac*100:.1f}%")
    else:
        frac_lines.append("- t_end: no OK fits")

    def _q_lines(series: pd.Series, label: str) -> list[str]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return [f"- {label}: no OK fits"]
        qs = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        out = [f"- {label} min/max: {float(s.min()):.4g} / {float(s.max()):.4g}"]
        for k in [0.1, 0.25, 0.5, 0.75, 0.9]:
            out.append(f"- {label} q{int(k*100):02d}: {float(qs[k]):.4g}")
        return out

    now_txt = str(pd.Timestamp.now())

    lines: list[str] = []
    lines.append("# Fit QC Report")
    lines.append("")
    lines.append(f"- Generated: {now_txt}")
    lines.append("")
    lines.append("## (a) OK / EXCLUDED")
    lines.append(f"- Total wells: {n_total}")
    lines.append(f"- OK: {n_ok}")
    lines.append(f"- EXCLUDED: {n_ex}")
    lines.append(f"- OK rate: {ok_rate*100:.1f}%")
    lines.append("")
    lines.append(f"- CSV: {overall_csv.name}")
    if by_plate_csv is not None:
        lines.append(f"- CSV (by plate): {by_plate_csv.name}")
    if by_heat_csv is not None:
        lines.append(f"- CSV (by heat): {by_heat_csv.name}")
    lines.append("")

    lines.append("## (b) Selected t_end distribution")
    if len(te_ok) > 0:
        lines.append(f"- t_end min/max: {te_min:.3g} / {te_max:.3g} s")
        for k in ["q10", "q25", "q50", "q75", "q90"]:
            if k in te_stats:
                lines.append(f"- {k}: {te_stats[k]:.3g} s")
    lines.append("")
    lines.extend(frac_lines)
    lines.append("")
    lines.append(f"![t_end hist]({png_hist.name})")
    lines.append("")

    lines.append("## (c) Slope vs t_end")
    lines.append(f"- N (finite): {n_corr}")
    lines.append(f"- Pearson r: {pearson_r:.4g}")
    lines.append(f"- Spearman ρ: {spearman_rho:.4g}")
    lines.append("")
    lines.append(f"![slope vs t_end]({png_scatter.name})")
    lines.append("")

    lines.append("## (d) select_method_used breakdown (OK only)")
    if method_col is None or (method_counts_csv is None) or (png_method_bar is None) or (len(method_lines) == 0):
        lines.append("- select_method_used: not available (column missing or no OK fits)")
    else:
        lines.append(f"- method column used: {method_col}")
        if np.isfinite(force_ok_frac):
            lines.append(f"- force_whole* fraction (among OK): {force_ok_frac*100:.1f}%")
        if np.isfinite(force_all_frac):
            lines.append(f"- force_whole* fraction (among ALL wells): {force_all_frac*100:.1f}%")
        lines.append("")
        lines.append(f"- CSV: {method_counts_csv.name}")
        lines.extend(method_lines)
        lines.append("")
        lines.append(f"![select_method_used]({png_method_bar.name})")
    lines.append("")

    lines.append("## (e) Distributions (OK only)")
    lines.append("### R²")
    lines.extend(_q_lines(ok["r2"], "R²"))
    lines.append("")
    lines.append(f"![r2 hist]({png_r2.name})")
    lines.append("")
    lines.append("### mono_frac")
    lines.extend(_q_lines(ok["mono_frac"], "mono_frac"))
    lines.append("")
    lines.append(f"![mono_frac hist]({png_mono.name})")
    lines.append("")
    lines.append("### snr")
    lines.extend(_q_lines(ok["snr"], "snr"))
    lines.append("")
    lines.append(f"![snr hist]({png_snr.name})")
    lines.append("")

    lines.append("## (f) Exclude reasons (EXCLUDED only)")
    if ex_counts_csv is None or png_ex_bar is None or len(ex_lines) == 0:
        lines.append("- excluded wells: 0")
    else:
        lines.append(f"- CSV: {ex_counts_csv.name}")
        lines.extend(ex_lines)
        lines.append("")
        lines.append(f"![exclude_reason_norm]({png_ex_bar.name})")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path



def compute_rates_and_rea(
    tidy: pd.DataFrame,
    heat_times: List[float],
    min_points: int = 6,
    max_points: int = 20,
    min_span_s: float = 0.0,
    select_method: str = "initial_positive",
    r2_min: float = 0.98,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
    mono_eps: Optional[float] = None,
    min_delta_y: Optional[float] = None,
    find_start: bool = True,
    start_max_shift: int = 5,
    start_window: int = 3,
    start_allow_down_steps: int = 1,
    mono_min_frac: float = 0.85,
    mono_max_down_steps: int = 1,
    min_pos_steps: int = 2,
    min_snr: float = 3.0,
    slope_drop_frac: float = 0.18,
    plot_dir: Optional[Path] = None,
    plot_mode: str = "all",
    # -------------------------
    # QC report (run-level)
    # -------------------------
    qc_report_dir: Optional[Path] = None,
    qc_prefix: str = "fit_qc",
    # -------------------------
    # optional: force "whole/long window" when curve is sufficiently linear
    # -------------------------
    force_whole: bool = False,
    force_whole_n_min: int = 10,
    force_whole_r2_min: float = 0.985,
    force_whole_mono_min_frac: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Compute per-well initial rates (absolute activity) and REA (%).

    Absolute activity:
      - slope from selected linear window per well (no row-averaging)

    REA (%):
      - for each (plate_id, polymer_id), normalize each heat_min by baseline at heat_min==0
      - REA_percent = 100 * abs_activity / baseline_abs_activity
    """
    df = add_heat_time(tidy, heat_times=heat_times)

    selected_rows: list[dict] = []

    group_cols = ["plate_id", "well"]
    if not all(c in df.columns for c in group_cols):
        raise ValueError(f"tidy must contain columns {group_cols}, got: {df.columns.tolist()}")

    plot_mode = str(plot_mode).lower().strip()
    if plot_mode not in {"all", "ok", "excluded"}:
        raise ValueError("plot_mode must be one of: all, ok, excluded")

    for (plate_id, well), g in df.groupby(group_cols, sort=False):
        # ---- sanity checks: metadata should be constant within a (plate_id, well) group ----
        def _nunique_nonnull(s: pd.Series) -> int:
            if s is None or s.empty:
                return 0
            x = s.dropna()
            # treat empty strings as missing-ish for id-like fields
            x = x[~(x.astype(str).str.strip() == "")]
            return int(x.nunique())

        meta_fields = ["polymer_id", "heat_min", "col", "row", "sample_name"]
        bad = []
        for k in meta_fields:
            if k in g.columns:
                if _nunique_nonnull(g[k]) > 1:
                    bad.append(k)

        if bad:
            raise ValueError(
                f"Inconsistent metadata within group (plate_id={plate_id}, well={well}): {bad}. "
                "Rowmap/merge may be broken."
            )

        base = {
            "plate_id": plate_id,
            "well": well,
            "row": g["row"].iloc[0] if "row" in g.columns else "",
            "col": int(g["col"].iloc[0]) if "col" in g.columns and pd.notna(g["col"].iloc[0]) else np.nan,
            "heat_min": float(g["heat_min"].iloc[0]) if "heat_min" in g.columns else np.nan,
            "polymer_id": g["polymer_id"].iloc[0] if "polymer_id" in g.columns else "",
            "sample_name": g["sample_name"].iloc[0] if "sample_name" in g.columns else "",
            "source_file": g["source_file"].iloc[0] if "source_file" in g.columns else "",
        }


        status = "excluded"
        exclude_reason = ""
        sel: Optional[pd.Series] = None

        try:
            cands = fit_initial_rate_one_well(
                g,
                min_points=int(min_points),
                max_points=int(max_points),
                min_span_s=float(min_span_s),
                mono_eps=mono_eps,
                min_delta_y=min_delta_y,
                find_start=bool(find_start),
                start_max_shift=int(start_max_shift),
                start_window=int(start_window),
                start_allow_down_steps=int(start_allow_down_steps),
            )

            sel = select_fit(
                cands,
                method=select_method,
                r2_min=r2_min,
                slope_min=slope_min,
                max_t_end=max_t_end,
                min_delta_y=min_delta_y,
                mono_min_frac=mono_min_frac,
                mono_max_down_steps=mono_max_down_steps,
                min_pos_steps=min_pos_steps,
                min_snr=min_snr,
                slope_drop_frac=slope_drop_frac,
                force_whole=force_whole,
                force_whole_n_min=force_whole_n_min,
                force_whole_r2_min=force_whole_r2_min,
                force_whole_mono_min_frac=force_whole_mono_min_frac,
            )

            # select_fit() sets select_method_used in the returned Series
            select_method_used = str(sel.get("select_method_used", select_method))

            _enforce_final_safety(

                sel,
                slope_min=slope_min,
                r2_min=r2_min,
                max_t_end=max_t_end,
                min_delta_y=min_delta_y,
                mono_min_frac=mono_min_frac,
                mono_max_down_steps=mono_max_down_steps,
                min_pos_steps=min_pos_steps,
                min_snr=min_snr,
            )

            status = "ok"


            row = {
                **base,
                "status": status,
                "abs_activity": float(sel["slope"]),
                "slope": float(sel["slope"]),
                "intercept": float(sel["intercept"]),
                "r2": float(sel["r2"]),
                "n": int(sel["n"]),
                "t_start": float(sel["t_start"]),
                "t_end": float(sel["t_end"]),
                "select_method": select_method_used,
                "select_method_used": select_method_used,
                "exclude_reason": "",
                # diagnostics
                "dy": float(sel["dy"]),
                "mono_frac": float(sel["mono_frac"]),
                "down_steps": int(sel["down_steps"]),
                "pos_steps": int(sel["pos_steps"]),
                "pos_steps_eps": int(sel["pos_steps_eps"]) if "pos_steps_eps" in sel.index else np.nan,
                "pos_eps": float(sel["pos_eps"]) if "pos_eps" in sel.index else np.nan,
                "rmse": float(sel["rmse"]) if pd.notna(sel["rmse"]) else np.nan,
                "snr": float(sel["snr"]) if pd.notna(sel["snr"]) else np.nan,
                "start_idx_used": int(sel["start_idx_used"]) if "start_idx_used" in sel.index else np.nan,
            }

        except FitSelectionError as e:
            status = "excluded"
            exclude_reason = str(e)

            row = {
                **base,
                "status": status,
                "abs_activity": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "n": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "select_method": select_method,
                "select_method_used": select_method,
                "exclude_reason": exclude_reason,
                # diagnostics
                "dy": np.nan,
                "mono_frac": np.nan,
                "down_steps": np.nan,
                "pos_steps": np.nan,
                "rmse": np.nan,
                "snr": np.nan,
                "start_idx_used": np.nan,
            }

        except Exception:
            # unexpected bug: fail fast during development
            raise

            row = {
                **base,
                "status": status,
                "abs_activity": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "n": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "select_method": select_method,
                "exclude_reason": exclude_reason,
                # diagnostics
                "dy": np.nan,
                "mono_frac": np.nan,
                "down_steps": np.nan,
                "pos_steps": np.nan,
                "rmse": np.nan,
                "snr": np.nan,
                "start_idx_used": np.nan,
            }

        selected_rows.append(row)

        if plot_dir is not None:
            do_plot = (
                (plot_mode == "all")
                or (plot_mode == "ok" and status == "ok")
                or (plot_mode == "excluded" and status != "ok")
            )
            if do_plot:
                out_png = plot_dir / f"{plate_id}" / f"{well}.png"
                plot_fit_diagnostic(
                    df_well=g,
                    meta=base,
                    selected=sel if status == "ok" else None,
                    status=status,
                    exclude_reason=exclude_reason,
                    out_png=out_png,
                )

    selected = pd.DataFrame(selected_rows)

    # --- run-level QC report ---
    if qc_report_dir is not None:
        qc_dir = Path(qc_report_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
        write_fit_qc_report(
            selected=selected,
            out_dir=qc_dir,
            max_t_end=max_t_end,
            prefix=str(qc_prefix),
        )

    baseline_mask = (selected["status"] == "ok") & np.isfinite(selected["heat_min"].to_numpy(dtype=float))
    baseline_mask = baseline_mask & np.isclose(selected["heat_min"].to_numpy(dtype=float), 0.0, atol=1e-12)


    baseline = (
        selected[baseline_mask]
        .groupby(["plate_id", "polymer_id"], dropna=False)["abs_activity"]
        .median()
        .rename("baseline_abs_activity")
        .reset_index()
    )


    rea = selected.merge(baseline, on=["plate_id", "polymer_id"], how="left")
    rea["REA_percent"] = 100.0 * rea["abs_activity"] / rea["baseline_abs_activity"]

    return selected, rea

