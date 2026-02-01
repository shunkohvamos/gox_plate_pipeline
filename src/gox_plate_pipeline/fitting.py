# src/gox_plate_pipeline/fitting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    d = df_well.sort_values("time_s").reset_index(drop=True)
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
        "rmse",
        "snr",
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

        if step.size > 0:
            mono_frac = float(np.mean(step >= -eps))
            down_steps = int(np.sum(step < -eps))
            # pos_steps: use >0 to avoid over-rejecting smooth increases when eps becomes large
            pos_steps = int(np.sum(step > 0.0))




        # --- residual diagnostics (raw y, not smoothed) ---
        yhat = fr.slope * xw + fr.intercept
        res = yw - yhat

        rmse = float(np.sqrt(np.mean(res**2))) if len(yw) else np.nan
        snr = float(abs(dy) / (rmse + 1e-12)) if np.isfinite(rmse) else np.nan

        # count "large residual" points (outlier-like), robust scale
        sigma_res = _robust_sigma(res)
        thr = 3.5 * max(float(sigma_res), 1e-12)
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
                "rmse": rmse,
                "snr": snr,
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

    required = {"dy", "mono_frac", "down_steps", "pos_steps", "snr", "n", "t_end", "t_start", "slope", "r2"}
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

            if (
                int(whole["n"]) >= int(force_whole_n_min)
                and r2_ok
                and float(whole["mono_frac"]) >= float(force_whole_mono_min_frac)
                and float(slope_drop) <= float(force_whole_max_slope_drop_frac)
                and float(half_drop) <= float(force_whole_max_half_drop_frac)
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
    ok = c[c["r2"] >= float(r2_min)].copy()
    if ok.empty:
        best = c.sort_values(["r2", "n", "t_end"], ascending=[False, False, True]).iloc[0]
        raise FitSelectionError(
            "No candidates met r2_min="
            f"{float(r2_min):.4g}. Best was r2={float(best['r2']):.4f} "
            f"(n={int(best['n'])}, t_end={float(best['t_end']):.3g}s). "
            "If you want to accept lower-quality fits, lower --r2_min."
        )

    # min_delta_y: auto from per-well constants if available
    if min_delta_y is None:
        if "min_delta_y" in ok.columns and ok["min_delta_y"].notna().any():
            min_dy = float(ok["min_delta_y"].dropna().iloc[0])
        else:
            min_dy = 0.0
    else:
        min_dy = float(min_delta_y)


        # prefer windows starting at detected start (start_idx_used), if available
        if "start_idx_used" in ok.columns and ok["start_idx_used"].notna().any():
            s0 = int(ok["start_idx_used"].dropna().iloc[0])
            whole_c = ok[ok["start_idx"] == s0].copy()
        else:
            whole_c = ok.copy()

        if not whole_c.empty:
            # choose the longest (largest n, then latest t_end)
            whole = whole_c.sort_values(["n", "t_end"], ascending=[False, False]).iloc[0].copy()

            # guard 1: only force whole-window if slope does not drop much vs max slope
            max_slope = float(ok["slope"].max())
            whole_slope = float(whole["slope"])
            if max_slope > 0.0:
                slope_drop = (max_slope - whole_slope) / max_slope
            else:
                slope_drop = 0.0

            # guard 2: saturation/curvature-like (first half slope -> second half slope)
            half_drop = float(whole.get("slope_half_drop_frac", 0.0))

            # guard 3: allow 1-point outlier (trim-1) if configured
            r2_raw = float(whole["r2"])
            r2_t1 = float(whole.get("r2_trim1", r2_raw))
            outliers = int(whole.get("outlier_count", 0))

            use_trim1 = (
                bool(force_whole_allow_trim1)
                and (r2_raw < float(force_whole_r2_min))
                and (r2_t1 >= float(force_whole_r2_min))
                and (outliers <= int(force_whole_max_outliers))
            )
            r2_ok = (r2_raw >= float(force_whole_r2_min)) or use_trim1

            if (
                int(whole["n"]) >= int(force_whole_n_min)
                and r2_ok
                and float(whole["mono_frac"]) >= float(force_whole_mono_min_frac)
                and float(slope_drop) <= float(force_whole_max_slope_drop_frac)
                and float(half_drop) <= float(force_whole_max_half_drop_frac)
            ):
                sel = whole.copy()

                # If only 1 point spoiled r2, use trim-1 parameters so slope isn't pulled by that point
                if use_trim1:
                    sel["slope"] = float(sel.get("slope_trim1", sel["slope"]))
                    sel["intercept"] = float(sel.get("intercept_trim1", sel["intercept"]))
                    sel["r2"] = float(r2_t1)
                    sel["select_method_used"] = "force_whole_trim1"
                else:
                    sel["select_method_used"] = "force_whole"

                return sel


    # --- hard filters: reaction-likeness ---
    if min_dy > 0.0:
        ok = ok[ok["dy"] >= float(min_dy)].copy()

    ok = ok[ok["mono_frac"] >= float(mono_min_frac)].copy()
    ok = ok[ok["down_steps"] <= int(mono_max_down_steps)].copy()
    ok = ok[ok["pos_steps"] >= int(min_pos_steps)].copy()
    ok = ok[ok["snr"] >= float(min_snr)].copy()

    if ok.empty:
        best = c.sort_values(["r2", "n", "t_end"], ascending=[False, False, True]).iloc[0]
        raise FitSelectionError(
            "All candidates were rejected by reaction-likeness filters "
            f"(min_delta_y={min_dy:.4g}, mono_min_frac={float(mono_min_frac):.3g}, "
            f"mono_max_down_steps={int(mono_max_down_steps)}, min_pos_steps={int(min_pos_steps)}, "
            f"min_snr={float(min_snr):.3g}). "
            f"Best remaining by r2 was r2={float(best['r2']):.4f}, dy={float(best['dy']):.4g}, "
            f"mono_frac={float(best['mono_frac']):.3g}, down_steps={int(best['down_steps'])}, "
            f"pos_steps={int(best['pos_steps'])}, snr={float(best['snr']):.3g} "
            f"(n={int(best['n'])}, t_end={float(best['t_end']):.3g}s)."
        )

    if method == "initial_positive":
        max_slope = float(ok["slope"].max())
        floor = max_slope * (1.0 - float(slope_drop_frac))
        keep = ok[ok["slope"] >= float(floor)].copy()
        if keep.empty:
            keep = ok

        keep = keep.sort_values(["t_end", "r2", "n", "t_start"], ascending=[True, False, False, True])
        sel = keep.iloc[0].copy()
        sel["select_method_used"] = "initial_positive"
        return sel

    if method == "best_r2":
        keep = ok.sort_values(["r2", "n", "t_end"], ascending=[False, False, True])
        sel = keep.iloc[0].copy()
        sel["select_method_used"] = "best_r2"
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

    if pos_steps < int(min_pos_steps):
        raise FitSelectionError(
            f"Final safety triggered: selected pos_steps {pos_steps} < min_pos_steps {int(min_pos_steps)}"
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
      - Title contains well + polymer_id + heat_min + slope + R²
      - Use unicode R² (superscript 2)
      - Line should be fine dotted
    """
    d = df_well.sort_values("time_s").reset_index(drop=True)
    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    well = str(meta.get("well", "NA"))
    polymer_id = str(meta.get("polymer_id", "") or "")
    heat_txt = _format_heat(meta.get("heat_min", np.nan))

    # ---- base scatter ----
    plt.figure()

    # highlight only the points that were explicitly excluded by trim (noise-as-outlier)
    drop_mask = np.zeros(len(t), dtype=bool)

    if status == "ok" and selected is not None:
        method_used = str(selected.get("select_method_used", ""))
        if ("trim1" in method_used) or ("trim2" in method_used):
            if "trim1_drop_idx" in selected.index and pd.notna(selected["trim1_drop_idx"]):
                j = int(selected["trim1_drop_idx"])
                if 0 <= j < len(drop_mask):
                    drop_mask[j] = True

            if "trim2_drop_idx1" in selected.index and pd.notna(selected["trim2_drop_idx1"]):
                j = int(selected["trim2_drop_idx1"])
                if 0 <= j < len(drop_mask):
                    drop_mask[j] = True

            if "trim2_drop_idx2" in selected.index and pd.notna(selected["trim2_drop_idx2"]):
                j = int(selected["trim2_drop_idx2"])
                if 0 <= j < len(drop_mask):
                    drop_mask[j] = True

    # draw normal points first
    plt.scatter(t[~drop_mask], y[~drop_mask], zorder=3)

    # draw excluded (trimmed) points on top with a different color
    if np.any(drop_mask):
        plt.scatter(t[drop_mask], y[drop_mask], zorder=4, color="tab:red")


    # x-range for full extension
    x0 = float(np.min(t)) if len(t) else 0.0
    x1 = float(np.max(t)) if len(t) else 1.0

    slope_txt = "NA"
    r2_txt = "NA"

    if status == "ok" and selected is not None:
        slope = float(selected["slope"])
        intercept = float(selected["intercept"])
        r2 = float(selected["r2"])

        slope_txt = f"{slope:.5g}"
        r2_txt = f"{r2:.4f}"

        # full-width fit line (fine dotted) - should be behind the window
        xx = np.array([x0, x1], dtype=float)
        yy = slope * xx + intercept
        plt.plot(xx, yy, linestyle=(0, (1, 2)), zorder=1)  # fine dotted (fine dots)

        # selected window: visible (light shading), above the dotted line but behind points
        t0 = float(selected["t_start"])
        t1 = float(selected["t_end"])
        plt.axvspan(
            t0,
            t1,
            facecolor="0.85",  # light gray (readable, not too loud)
            alpha=0.35,
            zorder=2,
        )



# Title: keep minimal (plate + well + polymer_id)
    plate_txt = str(meta.get("plate_id", "NA"))
    poly_txt = polymer_id or "NA"
    base_title = f"{plate_txt} | Well {well} | {poly_txt}"
    plt.title(base_title if status == "ok" else f"{base_title} | EXCLUDED")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (a.u.)")

# Info box: avoid duplicates with title (plate/polymer are in title)
    info_lines = [f"heat: {heat_txt}"]

    # sample_name: show only if meaningful (not empty / not NaN-like)
    sn = meta.get("sample_name", "")
    sn_str = "" if sn is None else str(sn).strip()
    if sn_str and sn_str.lower() not in {"nan", "none"}:
        info_lines.append(f"sample: {sn_str}")

    if status == "ok" and selected is not None:
        info_lines.append(f"slope: {slope_txt}")
        info_lines.append(f"R²: {r2_txt}")
        info_lines.append(f"n: {int(selected['n'])}")
    # NOTE: exclude_reason is intentionally not shown on the plot to reduce clutter.



    ax = plt.gca()

    # Heuristic: if many points are in the top-right area, move the box to top-left
    if len(t) > 0:
        tx0, tx1 = float(np.min(t)), float(np.max(t))
        ty0, ty1 = float(np.min(y)), float(np.max(y))
        # define "top-right" region
        tr = (t > (tx0 + 0.75 * (tx1 - tx0))) & (y > (ty0 + 0.75 * (ty1 - ty0)))
        crowded_tr = int(np.sum(tr)) >= max(5, int(0.05 * len(t)))
    else:
        crowded_tr = False

    xpos = 0.02 if crowded_tr else 0.98
    ha = "left" if crowded_tr else "right"

    ax.text(
        xpos,
        0.98,
        "\n".join(info_lines),
        ha=ha,
        va="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        zorder=10,
    )


    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


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

                "exclude_reason": "",
                # diagnostics
                "dy": float(sel["dy"]),
                "mono_frac": float(sel["mono_frac"]),
                "down_steps": int(sel["down_steps"]),
                "pos_steps": int(sel["pos_steps"]),
                "rmse": float(sel["rmse"]) if pd.notna(sel["rmse"]) else np.nan,
                "snr": float(sel["snr"]) if pd.notna(sel["snr"]) else np.nan,
                "start_idx_used": int(sel["start_idx_used"]) if "start_idx_used" in sel.index else np.nan,
            }

        except Exception as e:
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

    baseline = (
        selected[(selected["status"] == "ok") & (selected["heat_min"] == 0)]
        .groupby(["plate_id", "polymer_id"], dropna=False)["abs_activity"]
        .median()
        .rename("baseline_abs_activity")
        .reset_index()
    )

    rea = selected.merge(baseline, on=["plate_id", "polymer_id"], how="left")
    rea["REA_percent"] = 100.0 * rea["abs_activity"] / rea["baseline_abs_activity"]

    return selected, rea

