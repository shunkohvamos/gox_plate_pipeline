# src/gox_plate_pipeline/fitting/pipeline.py
"""
Main pipeline for computing rates and REA.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import FitSelectionError, _auto_mono_eps, _auto_min_delta_y
from .preprocessing import add_heat_time
from .candidates import fit_initial_rate_one_well
from .selection import (
    select_fit,
    try_skip_extend,
    try_extend_fit,
    _enforce_final_safety,
    find_best_short_window,
    find_fit_with_outlier_removal,
    detect_internal_outliers,
    detect_curvature_and_shorten,
    fit_with_outlier_skip_full_range,
    apply_conservative_long_override,
    _calc_window_stats,
)
from .plotting import (
    plot_fit_diagnostic,
    write_plate_grid,
    write_plate_grid_from_well_contexts,
)
from .qc import write_fit_qc_report


def _promote_longer_if_similar(
    cands: pd.DataFrame,
    sel: pd.Series,
    *,
    r2_min: float,
    min_snr: float,
    slope_tol_frac: float = 0.20,
    min_n_gain: int = 2,
) -> pd.Series:
    """
    Prefer a longer window when it has comparable linearity and similar slope.
    """
    if cands is None or cands.empty:
        return sel
    try:
        sel_n = int(sel["n"])
        sel_slope = float(sel["slope"])
        sel_r2 = float(sel["r2"])
        sel_start = int(sel["start_idx"])
    except Exception:
        return sel

    if sel_n >= 8 or (not np.isfinite(sel_slope)) or sel_slope <= 0:
        return sel

    pool = cands.copy()
    pool = pool[pool["start_idx"] == sel_start]
    pool = pool[pool["n"] >= sel_n + int(min_n_gain)]
    if pool.empty:
        return sel

    ratio = pool["slope"] / sel_slope
    pool = pool[(ratio >= 0.98) & (ratio <= (1.0 + float(slope_tol_frac)))]
    pool = pool[pool["r2"] >= max(float(r2_min), sel_r2 - 0.01)]
    pool = pool[pool["snr"] >= max(float(min_snr), 2.0)]
    pool = pool[pool["mono_frac"] >= 0.75]
    if pool.empty:
        return sel

    max_slope = float(pool["slope"].max())
    near_max = pool[pool["slope"] >= (0.96 * max_slope)]
    if near_max.empty:
        near_max = pool
    near_max = near_max.sort_values(["n", "r2", "slope", "t_end"], ascending=[False, False, False, True])
    best = near_max.iloc[0].copy()
    best["select_method_used"] = f"{str(sel.get('select_method_used', 'initial_positive'))}_promote_long_ext"
    return best


def _protect_initial_rate(
    prev_sel: pd.Series,
    new_sel: pd.Series,
    *,
    max_slope_drop_frac: float = 0.08,
    min_r2_gain_to_allow_drop: float = 0.02,
) -> pd.Series:
    """
    Keep earlier/steeper fit unless slope reduction is justified by clear R² gain.
    """
    try:
        s0 = float(prev_sel["slope"])
        s1 = float(new_sel["slope"])
        r20 = float(prev_sel["r2"])
        r21 = float(new_sel["r2"])
    except Exception:
        return new_sel

    if (not np.isfinite(s0)) or (not np.isfinite(s1)) or s0 <= 0 or s1 <= 0:
        return new_sel

    drop = (s0 - s1) / s0
    if drop <= float(max_slope_drop_frac):
        return new_sel

    if (r21 - r20) < float(min_r2_gain_to_allow_drop):
        return prev_sel
    return new_sel


def _prefer_early_steeper(
    cands: pd.DataFrame,
    sel: pd.Series,
    *,
    r2_floor: float,
    slope_gain_frac: float = 0.12,
) -> pd.Series:
    """
    Rescue stronger early-window slope when quality is comparable.
    """
    if cands is None or cands.empty:
        return sel
    try:
        sel_slope = float(sel["slope"])
        sel_r2 = float(sel["r2"])
    except Exception:
        return sel
    if (not np.isfinite(sel_slope)) or sel_slope <= 0:
        return sel

    pool = cands.copy()
    pool = pool[pool["start_idx"] == 0]
    pool = pool[pool["r2"] >= max(float(r2_floor), sel_r2 - 0.01)]
    pool = pool[pool["slope"] >= sel_slope * (1.0 + float(slope_gain_frac))]
    if pool.empty:
        return sel

    pool = pool.sort_values(["slope", "r2", "t_end"], ascending=[False, False, True])
    best = pool.iloc[0].copy()
    best["select_method_used"] = f"{str(sel.get('select_method_used', 'initial_positive'))}_early_steep"
    return best


def _prefer_delayed_steeper_when_short(
    cands: pd.DataFrame,
    sel: pd.Series,
    *,
    min_start: int = 3,
    min_n: int = 10,
    slope_gain_frac: float = 0.20,
    r2_tolerance: float = 0.02,
) -> pd.Series:
    """
    If a selected short window is likely underestimating initial rate, allow
    a delayed but steeper/longer candidate when quality remains comparable.
    """
    if cands is None or cands.empty:
        return sel
    try:
        sel_n = int(sel["n"])
        sel_slope = float(sel["slope"])
        sel_r2 = float(sel["r2"])
    except Exception:
        return sel
    method = str(sel.get("select_method_used", ""))
    early_mode = "_early_steep" in method
    if (sel_n > 8 and not early_mode) or (not np.isfinite(sel_slope)) or sel_slope <= 0:
        return sel

    gain_req = float(0.04 if early_mode else slope_gain_frac)
    r2_tol = float(0.03 if early_mode else r2_tolerance)
    min_n_req = int(max(min_n, int(0.7 * sel_n))) if early_mode else int(min_n)

    pool = cands.copy()
    pool = pool[pool["start_idx"] >= int(min_start)]
    pool = pool[pool["n"] >= min_n_req]
    pool = pool[pool["slope"] >= sel_slope * (1.0 + gain_req)]
    pool = pool[pool["r2"] >= (sel_r2 - r2_tol)]
    if pool.empty:
        return sel

    pool = pool.sort_values(["slope", "r2", "start_idx", "n"], ascending=[False, False, True, False])
    best = pool.iloc[0].copy()
    best["select_method_used"] = f"{str(sel.get('select_method_used', 'initial_positive'))}_delayed_steep"
    return best


def _protect_from_over_shortening(
    prev_sel: pd.Series,
    new_sel: pd.Series,
    *,
    min_prev_n: int = 12,
    min_prev_r2: float = 0.965,
    max_n_drop_frac: float = 0.45,
    max_slope_drop_frac: float = 0.05,
) -> pd.Series:
    """
    Prevent aggressive tangent-shortening from replacing a stable long fit.
    """
    try:
        prev_n = int(prev_sel["n"])
        new_n = int(new_sel["n"])
        prev_r2 = float(prev_sel["r2"])
        prev_slope = float(prev_sel["slope"])
        new_slope = float(new_sel["slope"])
    except Exception:
        return new_sel

    if prev_n < int(min_prev_n) or prev_r2 < float(min_prev_r2):
        return new_sel
    if prev_n <= 0 or prev_slope <= 0 or (not np.isfinite(new_slope)):
        return new_sel

    n_drop = float(prev_n - new_n) / float(prev_n)
    slope_drop = float(prev_slope - new_slope) / float(prev_slope)
    if n_drop > float(max_n_drop_frac) and slope_drop > float(max_slope_drop_frac):
        return prev_sel
    return new_sel


def _adjust_r2_gate_for_early_steep(sel: pd.Series, default_gate: float) -> float:
    gate = float(default_gate)
    method = str(sel.get("select_method_used", ""))
    if "_early_steep" not in method:
        return gate
    try:
        r2_sel = float(sel["r2"])
    except Exception:
        return gate
    if not np.isfinite(r2_sel):
        return gate
    return float(min(gate, max(0.95, r2_sel)))


def _rescue_broad_overestimate(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int,
    r2_gate: float,
    fit_method: str = "ols",
    max_sel_n: int = 7,
    late_start_frac: float = 0.20,
    inflate_ratio_min: float = 1.20,
    broad_min_r2_floor: float = 0.60,
    broad_n_frac_min: float = 0.55,
    r2_gap_allow: float = 0.22,
) -> pd.Series:
    """
    Rescue short mid-window overestimation by preferring a broader fit.

    This is intentionally local:
      - only triggers for short windows that start late in the trace
      - only when selected slope is substantially steeper than broad fit
      - broad candidate can include 1-2 dropped outliers
    """
    if sel is None:
        return sel
    if ("slope" not in sel.index) or ("r2" not in sel.index):
        return sel
    if ("start_idx" not in sel.index) or ("n" not in sel.index):
        return sel

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    n_total = int(len(t))
    if n_total < max(int(min_points) + 2, 10):
        return sel

    try:
        sel_slope = float(sel["slope"])
        sel_r2 = float(sel["r2"])
        sel_n = int(sel["n"])
        sel_start = int(sel["start_idx"])
    except Exception:
        return sel

    if (not np.isfinite(sel_slope)) or (not np.isfinite(sel_r2)) or sel_slope <= 0:
        return sel
    if sel_n > int(max_sel_n):
        return sel
    if (sel_start / float(max(1, n_total - 1))) < float(late_start_frac):
        return sel

    eps = _auto_mono_eps(y)
    min_dy = _auto_min_delta_y(y, eps)

    candidates: list[pd.Series] = []

    # Broad contiguous candidate: full window.
    try:
        full_stats = _calc_window_stats(
            t,
            y,
            0,
            n_total - 1,
            mono_eps=float(eps),
            min_delta_y=float(min_dy),
            fit_method=str(fit_method),
        )
        full_sel = pd.Series(full_stats)
        full_sel["skip_indices"] = ""
        full_sel["skip_count"] = 0
        full_sel["select_method_used"] = "broad_full"
        candidates.append(full_sel)
    except Exception:
        pass

    # Broad non-contiguous candidate: allow dropping 1-2 highest residual points.
    trimmed = fit_with_outlier_skip_full_range(
        t=t,
        y=y,
        outlier_sigma=0.0,
        r2_min=0.0,
        min_points=max(int(min_points), 10),
        r2_improvement_min=-1.0,
        max_outliers=2,
    )
    if trimmed is not None:
        trimmed = trimmed.copy()
        trimmed["select_method_used"] = "broad_trim"
        candidates.append(trimmed)

    if not candidates:
        return sel

    broad_pool = []
    for c in candidates:
        try:
            s = float(c["slope"])
            r2 = float(c["r2"])
            n = int(c["n"])
        except Exception:
            continue
        if (not np.isfinite(s)) or (not np.isfinite(r2)) or s <= 0 or n < 2:
            continue
        broad_pool.append(c)

    if not broad_pool:
        return sel

    broad_pool.sort(
        key=lambda c: (
            float(c.get("r2", -np.inf)),
            int(c.get("n", 0)),
            float(c.get("t_end", -np.inf)),
        ),
        reverse=True,
    )
    broad = broad_pool[0].copy()

    broad_slope = float(broad["slope"])
    broad_r2 = float(broad["r2"])
    broad_n = int(broad["n"])

    if broad_n < max(sel_n + 4, int(math.ceil(float(broad_n_frac_min) * n_total))):
        return sel
    if sel_slope < float(inflate_ratio_min) * broad_slope:
        return sel
    if broad_r2 < max(float(broad_min_r2_floor), sel_r2 - float(r2_gap_allow)):
        return sel

    new_sel = sel.copy()
    for k, v in broad.items():
        new_sel[k] = v

    orig_method = str(sel.get("select_method_used", ""))
    suffix = "post_broad_overfit_ext"
    new_sel["select_method_used"] = f"{orig_method}_{suffix}" if orig_method else suffix

    # Broad rescue may require a lower r2 gate than the initial short-window gate.
    if broad_r2 < float(r2_gate):
        gate = max(float(broad_min_r2_floor), broad_r2 - 1e-6)
        new_sel["r2_min_override"] = float(gate)
    if "down_steps" in new_sel.index and pd.notna(new_sel.get("down_steps", np.nan)):
        dsteps = int(new_sel["down_steps"])
        if dsteps > 1:
            new_sel["mono_max_down_steps_override"] = int(dsteps)

    return new_sel


def _rescue_tangent_underestimate(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    *,
    fit_method: str = "ols",
    max_sel_n: int = 6,
    max_sel_start: int = 1,
    max_start_idx: int = 1,
    min_len: int = 4,
    max_len: int = 6,
    slope_gain_frac: float = 0.15,
    r2_tolerance: float = 0.02,
    r2_floor: float = 0.90,
) -> pd.Series:
    """
    Rescue underestimated early fits by preferring a steeper tangent-like window.

    Local guardrails:
      - only when the current fit is short and starts early
      - only when tangent candidate is clearly steeper
      - only when tangent quality is comparable
    """
    if sel is None:
        return sel
    if ("slope" not in sel.index) or ("r2" not in sel.index):
        return sel
    if ("start_idx" not in sel.index) or ("n" not in sel.index):
        return sel

    method = str(sel.get("select_method_used", ""))
    if "tangent" in method:
        return sel

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < int(min_len):
        return sel

    try:
        sel_slope = float(sel["slope"])
        sel_r2 = float(sel["r2"])
        sel_n = int(sel["n"])
        sel_start = int(sel["start_idx"])
    except Exception:
        return sel

    if (not np.isfinite(sel_slope)) or sel_slope <= 0 or (not np.isfinite(sel_r2)):
        return sel
    if sel_n > int(max_sel_n):
        return sel
    if sel_start > int(max_sel_start):
        return sel

    eps = _auto_mono_eps(y)
    min_dy = _auto_min_delta_y(y, eps)
    n_total = int(len(t))

    cands: list[dict] = []
    start_max = min(int(max_start_idx), max(0, n_total - int(min_len)))
    for s0 in range(0, start_max + 1):
        for L in range(int(min_len), int(max_len) + 1):
            e0 = s0 + L - 1
            if e0 >= n_total:
                continue
            try:
                st = _calc_window_stats(
                    t,
                    y,
                    s0,
                    e0,
                    mono_eps=float(eps),
                    min_delta_y=float(min_dy),
                    fit_method=str(fit_method),
                )
            except Exception:
                continue

            s = float(st["slope"])
            r2 = float(st["r2"])
            if (not np.isfinite(s)) or s <= 0:
                continue
            if (not np.isfinite(r2)) or (r2 < max(float(r2_floor), sel_r2 - float(r2_tolerance))):
                continue
            if s < sel_slope * (1.0 + float(slope_gain_frac)):
                continue
            if float(st.get("snr", np.nan)) < 2.0:
                continue
            cands.append(st)

    if not cands:
        return sel

    cands.sort(
        key=lambda st: (
            float(st["slope"]),
            float(st["r2"]),
            int(st["n"]),
            -int(st["start_idx"]),
        ),
        reverse=True,
    )
    best = cands[0]

    new_sel = sel.copy()
    for k, v in best.items():
        new_sel[k] = v
    new_sel["skip_indices"] = ""
    new_sel["skip_count"] = 0

    base_method = str(sel.get("select_method_used", ""))
    suffix = "post_tangent_under"
    new_sel["select_method_used"] = f"{base_method}_{suffix}" if base_method else suffix
    return new_sel


def _apply_local_fit_audit(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int,
    r2_gate: float,
    fit_method: str = "ols",
) -> pd.Series:
    """
    Local post-selection audit:
      - underestimation rescue -> tangent
      - overestimation rescue -> broad/long
    """
    out = _rescue_tangent_underestimate(
        sel,
        t=t,
        y=y,
        fit_method=str(fit_method),
    )
    out = _rescue_broad_overestimate(
        out,
        t=t,
        y=y,
        min_points=int(min_points),
        r2_gate=float(r2_gate),
        fit_method=str(fit_method),
    )
    return out


def _selection_differs(a: pd.Series, b: pd.Series) -> bool:
    keys = ["slope", "r2", "n", "t_start", "t_end", "start_idx", "end_idx", "skip_indices"]
    for k in keys:
        av = a.get(k, np.nan)
        bv = b.get(k, np.nan)
        if k == "skip_indices":
            if str(av) != str(bv):
                return True
            continue
        if pd.isna(av) and pd.isna(bv):
            continue
        try:
            if abs(float(av) - float(bv)) > 1e-12:
                return True
        except Exception:
            if str(av) != str(bv):
                return True
    return False


def _neighbor_recheck_trigger(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    *,
    neighbor_slope: float,
    fit_method: str = "ols",
) -> pd.Series:
    """
    Recheck left well when slope_n < slope_{n+1} in the same row.

    This is a trigger, not a hard rule: only upgrade when a local tangent rescue
    resolves the inversion with comparable quality.
    """
    if sel is None:
        return sel
    try:
        s0 = float(sel["slope"])
        r20 = float(sel["r2"])
        n0 = int(sel["n"])
    except Exception:
        return sel
    if (not np.isfinite(s0)) or (not np.isfinite(r20)) or s0 <= 0:
        return sel
    if (not np.isfinite(neighbor_slope)) or neighbor_slope <= 0:
        return sel
    if s0 >= neighbor_slope:
        return sel
    if n0 > 7 and r20 >= 0.95:
        return sel

    cand = _rescue_tangent_underestimate(
        sel,
        t=t,
        y=y,
        fit_method=str(fit_method),
        max_start_idx=2,
        max_sel_start=2,
        max_len=7,
        slope_gain_frac=0.08,
        r2_tolerance=0.03,
        r2_floor=0.85,
    )
    if not _selection_differs(sel, cand):
        return sel
    try:
        sc = float(cand["slope"])
        r2c = float(cand["r2"])
    except Exception:
        return sel
    if sc < max(0.98 * float(neighbor_slope), 1.05 * s0):
        return sel
    if r2c < max(0.85, r20 - 0.03):
        return sel

    cand = cand.copy()
    cand["select_method_used"] = f"{str(cand.get('select_method_used', ''))}_neighbor_recheck".strip("_")
    return cand


def _neighbor_recheck_right_trigger(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    *,
    left_slope: float,
    min_points: int,
    r2_gate: float,
    fit_method: str = "ols",
) -> pd.Series:
    """
    Recheck right well when slope_left < slope_right.

    Tries to lower obvious overestimation on the right by preferring a broader fit,
    while preserving initial-rate guardrails and avoiding over-correction.
    """
    if sel is None:
        return sel
    try:
        s0 = float(sel["slope"])
        r20 = float(sel["r2"])
        n0 = int(sel["n"])
    except Exception:
        return sel
    if (not np.isfinite(s0)) or (not np.isfinite(r20)) or s0 <= 0:
        return sel
    if (not np.isfinite(left_slope)) or left_slope <= 0:
        return sel
    if s0 <= left_slope:
        return sel
    if n0 >= 10 and r20 >= 0.96:
        return sel

    cand = _rescue_broad_overestimate(
        sel,
        t=t,
        y=y,
        min_points=int(min_points),
        r2_gate=float(r2_gate),
        fit_method=str(fit_method),
        max_sel_n=max(8, int(n0) + 2),
        late_start_frac=0.0,
        inflate_ratio_min=1.05,
        broad_min_r2_floor=0.58,
        broad_n_frac_min=0.45,
        r2_gap_allow=0.25,
    )
    if not _selection_differs(sel, cand):
        return sel
    try:
        sc = float(cand["slope"])
        r2c = float(cand["r2"])
    except Exception:
        return sel
    if (not np.isfinite(sc)) or sc <= 0:
        return sel
    if sc > 0.97 * s0:
        return sel
    if r2c < max(0.60, r20 - 0.08):
        return sel

    left_safe = max(float(left_slope), 1e-12)
    old_ratio = float(s0 / left_safe)
    new_ratio = float(sc / left_safe)

    # Accept when inversion is resolved/almost resolved, or gap is materially reduced.
    if (new_ratio > 1.02) and (new_ratio > old_ratio * 0.85):
        return sel
    # Avoid extreme collapse to a physically implausible right-well slope.
    if sc < 0.70 * left_safe:
        return sel

    cand = cand.copy()
    cand["select_method_used"] = f"{str(cand.get('select_method_used', ''))}_neighbor_recheck_right".strip("_")
    return cand


def _collect_col1_peer_slopes(
    well_contexts: list[dict],
    *,
    exclude_key: Optional[tuple[str, str]] = None,
) -> np.ndarray:
    vals: list[float] = []
    for ctx in well_contexts:
        if str(ctx.get("status", "")) != "ok":
            continue
        sel = ctx.get("sel")
        if sel is None:
            continue
        meta = ctx.get("meta", {}) or {}
        col = meta.get("col", np.nan)
        if pd.isna(col):
            continue
        try:
            if int(col) != 1:
                continue
        except Exception:
            continue
        key = (str(ctx.get("plate_id", "")), str(ctx.get("well", "")))
        if exclude_key is not None and key == exclude_key:
            continue
        try:
            slope = float(sel["slope"])
        except Exception:
            continue
        if np.isfinite(slope) and slope > 0.0:
            vals.append(slope)
    return np.asarray(vals, dtype=float)


def _col1_reference_log_stats(peer_slopes: np.ndarray) -> tuple[float, float]:
    s = np.asarray(peer_slopes, dtype=float)
    s = s[np.isfinite(s) & (s > 0.0)]
    if s.size == 0:
        return np.nan, np.nan
    logs = np.log(s)
    center = float(np.median(logs))
    mad = float(np.median(np.abs(logs - center)))
    sigma = float(1.4826 * mad)
    return center, sigma


def _col1_is_outlier_to_peers(
    sel: pd.Series,
    peer_slopes: np.ndarray,
    *,
    min_peers: int = 5,
    z_thresh: float = 2.5,
    min_abs_log_ratio: float = math.log(1.30),
) -> tuple[bool, float, float]:
    try:
        slope = float(sel["slope"])
    except Exception:
        return False, np.nan, np.nan
    if (not np.isfinite(slope)) or slope <= 0.0:
        return False, np.nan, np.nan

    peers = np.asarray(peer_slopes, dtype=float)
    peers = peers[np.isfinite(peers) & (peers > 0.0)]
    if peers.size < int(min_peers):
        return False, np.nan, np.nan

    center_log, sigma_log = _col1_reference_log_stats(peers)
    if (not np.isfinite(center_log)) or (not np.isfinite(float(np.log(slope)))):
        return False, np.nan, np.nan
    dist = float(abs(float(np.log(slope)) - center_log))
    if dist < float(min_abs_log_ratio):
        return False, dist, center_log

    if np.isfinite(sigma_log) and sigma_log > 1e-9:
        z = float(dist / sigma_log)
        return bool(z >= float(z_thresh)), dist, center_log

    # Degenerate distribution (MAD ~ 0): require a stronger multiplicative mismatch.
    return bool(dist >= math.log(1.55)), dist, center_log


def _col1_candidate_tier(
    cand: pd.Series,
    sel: pd.Series,
) -> Optional[int]:
    try:
        prev_n = int(sel["n"])
        prev_r2 = float(sel["r2"])
        prev_start = int(sel["start_idx"])
        prev_mono = float(sel.get("mono_frac", 1.0))
        prev_snr = float(sel.get("snr", 3.0))

        n = int(cand["n"])
        r2 = float(cand["r2"])
        start_idx = int(cand["start_idx"])
        mono_frac = float(cand.get("mono_frac", np.nan))
        snr = float(cand.get("snr", np.nan))
    except Exception:
        return None

    if (not np.isfinite(r2)) or (not np.isfinite(mono_frac)) or (not np.isfinite(snr)):
        return None
    if start_idx < 0:
        return None

    # Keep the replacement in the same early regime; do not jump to a late segment.
    start_cap = max(int(prev_start) + 3, 6)
    if start_idx > start_cap:
        return None

    if (
        n >= max(6, min(prev_n + 1, 10))
        and r2 >= max(0.88, prev_r2 - 0.03)
        and mono_frac >= max(0.70, prev_mono - 0.15)
        and snr >= max(2.3, prev_snr - 1.5)
    ):
        return 0
    if n >= 4 and r2 >= max(0.78, prev_r2 - 0.08) and mono_frac >= 0.60 and snr >= 1.8:
        return 1
    if n >= 3 and r2 >= max(0.65, prev_r2 - 0.15) and mono_frac >= 0.45 and snr >= 1.0:
        return 2
    if n >= 2 and start_idx <= 2 and r2 >= max(0.50, prev_r2 - 0.25) and mono_frac >= 0.35 and snr >= 0.6:
        return 3
    return None


def _col1_relaxed_used_params(used_params: dict, tier: int) -> dict:
    out = dict(used_params)
    if tier <= 1:
        return out
    if tier == 2:
        out["r2_min"] = min(float(out.get("r2_min", 0.98)), 0.70)
        out["min_delta_y"] = 0.0
        out["mono_min_frac"] = min(float(out.get("mono_min_frac", 0.85)), 0.60)
        out["mono_max_down_steps"] = max(int(out.get("mono_max_down_steps", 1)), 2)
        out["min_pos_steps"] = min(int(out.get("min_pos_steps", 2)), 1)
        out["min_snr"] = min(float(out.get("min_snr", 3.0)), 1.8)
        return out
    out["r2_min"] = min(float(out.get("r2_min", 0.98)), 0.55)
    out["min_delta_y"] = 0.0
    out["mono_min_frac"] = min(float(out.get("mono_min_frac", 0.85)), 0.45)
    out["mono_max_down_steps"] = max(int(out.get("mono_max_down_steps", 1)), 3)
    out["min_pos_steps"] = 0
    out["min_snr"] = min(float(out.get("min_snr", 3.0)), 1.0)
    return out


def _build_col1_short_fallback_candidates(
    t: np.ndarray,
    y: np.ndarray,
    *,
    fit_method: str,
    max_start_idx: int = 2,
) -> list[pd.Series]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 2:
        return []

    eps = _auto_mono_eps(y)
    out: list[pd.Series] = []
    for n_try in (3, 2):
        if t.size < n_try:
            continue
        start_max = min(int(max_start_idx), int(t.size - n_try))
        for i0 in range(0, start_max + 1):
            i1 = i0 + n_try - 1
            try:
                st = _calc_window_stats(
                    t,
                    y,
                    i0,
                    i1,
                    mono_eps=float(eps),
                    min_delta_y=0.0,
                    fit_method=str(fit_method),
                )
            except Exception:
                continue
            cand = pd.Series(st).copy()
            try:
                slope = float(cand["slope"])
            except Exception:
                continue
            if (not np.isfinite(slope)) or slope <= 0.0:
                continue
            cand["skip_indices"] = ""
            cand["skip_count"] = 0
            cand["select_method_used"] = f"col1_consensus_short_n{n_try}"
            out.append(cand)
    return out


def _col1_consensus_recheck(
    sel: pd.Series,
    *,
    df_well: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    peer_slopes: np.ndarray,
    used_params: dict,
    min_points: int,
    max_points: int,
    min_span_s: float,
    slope_min: float,
    max_t_end: Optional[float],
    mono_eps: Optional[float],
    min_delta_y: Optional[float],
    find_start: bool,
    start_max_shift: int,
    start_window: int,
    start_allow_down_steps: int,
    min_t_start_s: float,
    down_step_min_frac: Optional[float],
    fit_method: str,
    mono_max_down_steps_default: int,
) -> pd.Series:
    """
    Recheck obvious col=1 outliers against same-run col=1 peers.

    Priority:
      1) keep initial-rate validity (early/linear/robust windows),
      2) then reduce mismatch to the peer slope consensus.
    """
    is_outlier, dist_old, center_log = _col1_is_outlier_to_peers(sel, peer_slopes)
    if (not is_outlier) or (not np.isfinite(dist_old)) or (not np.isfinite(center_log)):
        return sel

    base_cands = fit_initial_rate_one_well(
        df_well,
        min_points=int(min_points),
        max_points=int(max_points),
        min_span_s=float(min_span_s),
        mono_eps=mono_eps,
        min_delta_y=min_delta_y,
        find_start=bool(find_start),
        start_max_shift=int(start_max_shift),
        start_window=int(start_window),
        start_allow_down_steps=int(start_allow_down_steps),
        min_t_start_s=float(min_t_start_s),
        down_step_min_frac=down_step_min_frac,
        fit_method=str(fit_method),
    )

    candidates: list[pd.Series] = []
    if base_cands is not None and not base_cands.empty:
        for _, row in base_cands.iterrows():
            cand = row.copy()
            cand["skip_indices"] = ""
            cand["skip_count"] = 0
            cand["select_method_used"] = "col1_consensus_base"
            candidates.append(cand)
    candidates.extend(_build_col1_short_fallback_candidates(t, y, fit_method=str(fit_method)))
    if not candidates:
        return sel

    improve_req_by_tier = {0: 0.80, 1: 0.70, 2: 0.55, 3: 0.40}
    scored: list[tuple[tuple, pd.Series]] = []
    base_method = str(sel.get("select_method_used", "initial_positive"))

    for cand_raw in candidates:
        cand = cand_raw.copy()
        try:
            slope_c = float(cand["slope"])
            if (not np.isfinite(slope_c)) or slope_c <= 0.0:
                continue
            dist_new = float(abs(float(np.log(slope_c)) - center_log))
            tier = _col1_candidate_tier(cand, sel)
        except Exception:
            continue
        if tier is None:
            continue
        if dist_new >= dist_old:
            continue
        if dist_new > dist_old * float(improve_req_by_tier.get(tier, 1.0)):
            continue

        cand["select_method_used"] = f"{base_method}_col1_consensus_t{tier}".strip("_")

        local_used_params = _col1_relaxed_used_params(used_params, tier)
        try:
            r2_gate, safety_max_t_end, mono_down_gate = _compute_safety_gates(
                cand,
                local_used_params,
                min_points=int(min_points),
                max_t_end=max_t_end,
                mono_max_down_steps_default=int(mono_max_down_steps_default),
            )
            _enforce_final_safety(
                cand,
                slope_min=float(slope_min),
                r2_min=float(r2_gate),
                max_t_end=safety_max_t_end,
                min_delta_y=local_used_params.get("min_delta_y", used_params.get("min_delta_y")),
                mono_min_frac=float(local_used_params.get("mono_min_frac", used_params.get("mono_min_frac", 0.85))),
                mono_max_down_steps=int(local_used_params.get("mono_max_down_steps", used_params.get("mono_max_down_steps", 1))),
                min_pos_steps=int(local_used_params.get("min_pos_steps", used_params.get("min_pos_steps", 2))),
                min_snr=float(local_used_params.get("min_snr", used_params.get("min_snr", 3.0))),
            )
        except FitSelectionError:
            continue

        try:
            score = (
                int(tier),
                float(dist_new),
                int(cand["start_idx"]),
                -float(cand["r2"]),
                -int(cand["n"]),
                float(cand["t_end"]),
            )
        except Exception:
            continue
        scored.append((score, cand))

    if not scored:
        return sel
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _compute_safety_gates(
    sel: pd.Series,
    used_params: dict,
    *,
    min_points: int,
    max_t_end: Optional[float],
    mono_max_down_steps_default: int,
) -> tuple[float, Optional[float], int]:
    select_method_used = str(sel.get("select_method_used", ""))

    extended = ("skip" in select_method_used) or ("_ext" in select_method_used) or ("outlier" in select_method_used)
    if (not extended) and (max_t_end is not None):
        try:
            if (
                float(sel["t_end"]) > float(max_t_end)
                and int(sel["n"]) >= max(12, 2 * int(min_points))
                and float(sel["r2"]) >= 0.90
                and float(sel["mono_frac"]) >= 0.85
                and float(sel["snr"]) >= 3.0
            ):
                extended = True
        except Exception:
            pass
    safety_max_t_end = None if extended else max_t_end

    r2_gate = float(used_params.get("r2_min", 0.98))
    if "r2_min_override" in sel.index and pd.notna(sel.get("r2_min_override", np.nan)):
        r2_gate = float(sel["r2_min_override"])
    r2_gate = _adjust_r2_gate_for_early_steep(sel, r2_gate)

    mono_down_gate = int(used_params.get("mono_max_down_steps", mono_max_down_steps_default))
    if "mono_max_down_steps_override" in sel.index and pd.notna(sel.get("mono_max_down_steps_override", np.nan)):
        mono_down_gate = max(mono_down_gate, int(sel["mono_max_down_steps_override"]))

    return float(r2_gate), safety_max_t_end, int(mono_down_gate)


def _build_ok_row(base: dict, sel: pd.Series, select_method: str) -> dict:
    select_method_used = str(sel.get("select_method_used", select_method))
    return {
        **base,
        "status": "ok",
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
        "dy": float(sel["dy"]),
        "mono_frac": float(sel["mono_frac"]),
        "down_steps": int(sel["down_steps"]),
        "pos_steps": int(sel["pos_steps"]),
        "pos_steps_eps": int(sel["pos_steps_eps"]) if "pos_steps_eps" in sel.index else np.nan,
        "pos_eps": float(sel["pos_eps"]) if "pos_eps" in sel.index else np.nan,
        "rmse": float(sel["rmse"]) if pd.notna(sel["rmse"]) else np.nan,
        "snr": float(sel["snr"]) if pd.notna(sel["snr"]) else np.nan,
        "start_idx_used": int(sel["start_idx_used"]) if "start_idx_used" in sel.index else np.nan,
        "skip_indices": str(sel.get("skip_indices", "")) if "skip_indices" in sel.index else "",
        "skip_count": int(sel.get("skip_count", 0)) if "skip_count" in sel.index else 0,
    }


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
    qc_report_dir: Optional[Path] = None,
    qc_prefix: str = "fit_qc",
    force_whole: bool = False,
    force_whole_n_min: int = 10,
    force_whole_r2_min: float = 0.985,
    force_whole_mono_min_frac: float = 0.70,
    min_t_start_s: float = 0.0,
    down_step_min_frac: Optional[float] = None,
    fit_method: str = "ols",
    plate_grid_dir: Optional[Path] = None,
    plate_grid_run_id: Optional[str] = None,
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

    well_contexts: list[dict] = []

    group_cols = ["plate_id", "well"]
    if not all(c in df.columns for c in group_cols):
        raise ValueError(f"tidy must contain columns {group_cols}, got: {df.columns.tolist()}")

    plot_mode = str(plot_mode).lower().strip()
    if plot_mode not in {"all", "ok", "excluded"}:
        raise ValueError("plot_mode must be one of: all, ok, excluded")

    for (plate_id, well), g in df.groupby(group_cols, sort=False):
        def _nunique_nonnull(s: pd.Series) -> int:
            if s is None or s.empty:
                return 0
            x = s.dropna()
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
        # Preserve metadata flags from row_map TSV
        # Helper function to parse boolean values (handles both bool and string "True"/"False")
        def _parse_bool_from_series(series_val, default: bool) -> bool:
            if pd.isna(series_val):
                return default
            if isinstance(series_val, bool):
                return series_val
            s = str(series_val).strip().upper()
            if s in ("TRUE", "1", "YES"):
                return True
            if s in ("FALSE", "0", "NO"):
                return False
            return default
        
        if "use_for_bo" in g.columns:
            base["use_for_bo"] = _parse_bool_from_series(g["use_for_bo"].iloc[0], default=True)
        if "include_in_all_polymers" in g.columns:
            base["include_in_all_polymers"] = _parse_bool_from_series(g["include_in_all_polymers"].iloc[0], default=True)
        if "all_polymers_pair" in g.columns:
            base["all_polymers_pair"] = _parse_bool_from_series(g["all_polymers_pair"].iloc[0], default=False)
        t_arr = g["time_s"].to_numpy(dtype=float)
        y_arr = g["signal"].to_numpy(dtype=float)

        status = "excluded"
        exclude_reason = ""
        sel: Optional[pd.Series] = None
        used_params = {
            "r2_min": float(r2_min),
            "min_delta_y": min_delta_y,
            "mono_min_frac": float(mono_min_frac),
            "min_pos_steps": int(min_pos_steps),
            "mono_max_down_steps": int(mono_max_down_steps),
            "min_snr": float(min_snr),
        }

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
                min_t_start_s=float(min_t_start_s),
                down_step_min_frac=down_step_min_frac,
                fit_method=str(fit_method),
            )

            # Fallback attempts: progressively relax parameters for difficult data.
            # Final fallback uses r2_min=0.80 to avoid fitting pure noise.
            fallback_attempts = [
                {"r2_min": r2_min, "min_delta_y": min_delta_y, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.92, "min_delta_y": min_delta_y, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": r2_min, "min_delta_y": 0.0, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.88, "min_delta_y": 0.0, "mono_min_frac": 0.75, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.80, "min_delta_y": 0.0, "mono_min_frac": 0.65, "min_pos_steps": 0},
            ]
            sel = None
            last_err = None
            used_params = fallback_attempts[0]
            for attempt_params in fallback_attempts:
                try:
                    sel = select_fit(
                        cands,
                        method=select_method,
                        r2_min=attempt_params["r2_min"],
                        slope_min=slope_min,
                        max_t_end=max_t_end,
                        min_delta_y=attempt_params["min_delta_y"],
                        mono_min_frac=attempt_params["mono_min_frac"],
                        mono_max_down_steps=mono_max_down_steps,
                        min_pos_steps=attempt_params["min_pos_steps"],
                        min_snr=min_snr,
                        slope_drop_frac=slope_drop_frac,
                        force_whole=force_whole,
                        force_whole_n_min=force_whole_n_min,
                        force_whole_r2_min=force_whole_r2_min,
                        force_whole_mono_min_frac=force_whole_mono_min_frac,
                    )
                    used_params = attempt_params
                    sel = _promote_longer_if_similar(
                        cands,
                        sel,
                        r2_min=float(attempt_params["r2_min"]),
                        min_snr=float(min_snr),
                    )
                    sel = _prefer_early_steeper(
                        cands,
                        sel,
                        r2_floor=float(attempt_params["r2_min"]) - 0.01,
                    )
                    sel = _prefer_delayed_steeper_when_short(
                        cands,
                        sel,
                        min_start=3,
                        min_n=10,
                        slope_gain_frac=0.20,
                        r2_tolerance=0.02,
                    )
                    break
                except FitSelectionError as e:
                    last_err = e
                    continue
            if sel is None:
                # Progressive rescue: Gradually relax both R² and max_points
                # Try: 6+ points with relaxed R² → 5 points → 4 points → 3 points
                # At each point count, gradually relax R² from 0.80 → 0.70 → 0.60 → 0.50
                n_total = len(t_arr)
                
                # Progressive rescue attempts: (max_points, r2_min) pairs
                # Start with slightly reduced max_points, then gradually reduce
                rescue_attempts = [
                    # Try 5 points with relaxed R²
                    (5, 0.80),
                    (5, 0.70),
                    (5, 0.60),
                    # Try 4 points with relaxed R²
                    (4, 0.80),
                    (4, 0.70),
                    (4, 0.60),
                    (4, 0.50),
                    # Try 3 points (minimum for linear fit) with very relaxed R²
                    (3, 0.70),
                    (3, 0.60),
                    (3, 0.50),
                ]
                
                # Before short-window rescue, try robust outlier-removal first.
                # This avoids preferring a tiny short window when a larger
                # outlier-cleaned fit is available.
                sel = find_fit_with_outlier_removal(
                    t=t_arr,
                    y=y_arr,
                    min_points=6,
                    r2_min=0.80,
                    slope_min=slope_min,
                    min_snr=2.0,
                    outlier_sigma=3.0,
                )
                if sel is not None:
                    used_params = {
                        "r2_min": 0.80,
                        "min_delta_y": 0.0,
                        "mono_min_frac": 0.0,
                        "min_pos_steps": 0,
                        "mono_max_down_steps": 99,
                        "min_snr": 2.0,
                    }
                else:
                    sel = None
                for rescue_max_points, rescue_r2_min in rescue_attempts:
                    if sel is not None:
                        break
                    if n_total < rescue_max_points:
                        continue  # Skip if not enough data points
                    
                    # Generate candidates with reduced max_points
                    # For rescue, prioritize initial points (start_idx=0) to capture initial rate
                    rescue_cands = fit_initial_rate_one_well(
                        g,
                        min_points=rescue_max_points,  # Use same as max for focused rescue
                        max_points=rescue_max_points,  # Fixed point count for this attempt
                        min_span_s=float(min_span_s),
                        mono_eps=mono_eps,
                        min_delta_y=min_delta_y,
                        find_start=False,  # Disable start detection for rescue - use start_idx=0 to prioritize initial points
                        start_max_shift=int(start_max_shift),
                        start_window=int(start_window),
                        start_allow_down_steps=int(start_allow_down_steps),
                        min_t_start_s=float(min_t_start_s),
                        down_step_min_frac=down_step_min_frac,
                        fit_method=str(fit_method),
                    )
                    
                    if rescue_cands.empty:
                        continue
                    
                    # Try selecting with relaxed parameters
                    try:
                        sel = select_fit(
                            rescue_cands,
                            method=select_method,
                            r2_min=rescue_r2_min,
                            slope_min=slope_min,
                            max_t_end=max_t_end,
                            min_delta_y=0.0,  # Relaxed
                            mono_min_frac=0.60,  # Relaxed
                            mono_max_down_steps=mono_max_down_steps,
                            min_pos_steps=1,  # Relaxed
                            min_snr=2.0,  # Relaxed
                            slope_drop_frac=slope_drop_frac,
                            force_whole=False,  # Disable force_whole for rescue
                            force_whole_n_min=force_whole_n_min,
                            force_whole_r2_min=force_whole_r2_min,
                            force_whole_mono_min_frac=force_whole_mono_min_frac,
                        )
                        used_params = {
                            "r2_min": rescue_r2_min,
                            "min_delta_y": 0.0,
                            "mono_min_frac": 0.60,
                            "min_pos_steps": 1,
                            "mono_max_down_steps": mono_max_down_steps,
                            "min_snr": 2.0,
                        }
                        sel = _promote_longer_if_similar(
                            rescue_cands,
                            sel,
                            r2_min=float(rescue_r2_min),
                            min_snr=2.0,
                        )
                        sel = _prefer_early_steeper(
                            rescue_cands,
                            sel,
                            r2_floor=float(rescue_r2_min) - 0.01,
                        )
                        sel = _prefer_delayed_steeper_when_short(
                            rescue_cands,
                            sel,
                            min_start=3,
                            min_n=8,
                            slope_gain_frac=0.18,
                            r2_tolerance=0.03,
                        )
                        break  # Success, exit rescue loop
                    except FitSelectionError:
                        continue  # Try next rescue attempt
                
                # Last resort: try to find best short window for very noisy data
                if sel is None:
                    sel = find_best_short_window(
                        t=t_arr,
                        y=y_arr,
                        min_points=4,
                        max_points=8,
                        r2_min=0.55,  # Relaxed to allow noisy initial sections
                        slope_min=slope_min,
                        min_snr=1.5,  # Slightly relaxed SNR for initial rate priority
                    )
                    if sel is not None:
                        # Found a decent short window - use relaxed thresholds
                        used_params = {
                            "r2_min": 0.55,  # Match find_best_short_window threshold
                            "min_delta_y": 0.0,
                            "mono_min_frac": 0.0,
                            "min_pos_steps": 0,
                            "mono_max_down_steps": 99,  # Relaxed for last_resort
                            "min_snr": 1.5,  # Match find_best_short_window threshold
                        }
                        sel = _promote_longer_if_similar(
                            cands,
                            sel,
                            r2_min=0.55,
                            min_snr=1.5,
                        )
                        sel = _prefer_early_steeper(
                            cands,
                            sel,
                            r2_floor=0.54,
                        )
                        sel = _prefer_delayed_steeper_when_short(
                            cands,
                            sel,
                            min_start=3,
                            min_n=8,
                            slope_gain_frac=0.18,
                            r2_tolerance=0.04,
                        )
                    else:
                        if last_err is not None:
                            raise last_err
                        raise FitSelectionError("No acceptable fit found.")

            # Step 1: Try full-range outlier skip first (C1, C3 case)
            # This handles single outliers where we want to include points
            # BEFORE and AFTER the outlier
            full_range_fit = fit_with_outlier_skip_full_range(
                t=t_arr,
                y=y_arr,
                outlier_sigma=3.5,
                r2_min=0.98,
                min_points=10,
                r2_improvement_min=0.005,
            )
            # Prefer full-range one-outlier rescue when quality is comparable.
            # This keeps the intended contiguous domain instead of truncating
            # before an isolated outlier.
            full_range_r2_tolerance = 0.005
            if (
                full_range_fit is not None
                and float(full_range_fit["r2"]) + float(full_range_r2_tolerance) >= float(sel["r2"])
            ):
                slope_drop = 0.0
                if float(sel["slope"]) > 0:
                    slope_drop = (float(sel["slope"]) - float(full_range_fit["slope"])) / float(sel["slope"])
                r2_gain = float(full_range_fit["r2"]) - float(sel["r2"])
                if (slope_drop <= 0.15) or (r2_gain >= 0.02):
                    sel = full_range_fit
                    used_params = {
                        "r2_min": 0.98,
                        "min_delta_y": 0.0,
                        "mono_min_frac": 0.0,
                        "min_pos_steps": 0,
                        "mono_max_down_steps": 99,
                        "min_snr": 2.0,
                    }
            
            # Step 2: Try to extend
            prev_sel = sel.copy()
            sel = try_extend_fit(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                r2_drop_tolerance=0.02,
            )
            sel = _protect_initial_rate(prev_sel, sel)
            
            # Step 3: Try to skip outliers at the end and extend
            prev_sel = sel.copy()
            sel = try_skip_extend(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                max_skip=2,
                min_extend_points=3,
                residual_threshold_sigma=4.5,
                max_t_end=None,
            )
            sel = _protect_initial_rate(prev_sel, sel)
            
            # Step 4: Try extending again after skipping outliers
            prev_sel = sel.copy()
            sel = try_extend_fit(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                r2_drop_tolerance=0.02,
            )
            sel = _protect_initial_rate(prev_sel, sel)
            
            # Step 5: Detect and remove internal outliers (C2 case)
            # Only remove if R² improves by at least 0.005 (keep marginal points for initial velocity)
            prev_sel = sel.copy()
            sel = detect_internal_outliers(
                sel,
                t=t_arr,
                y=y_arr,
                outlier_sigma=3.0,
                r2_min=0.98,
                r2_improvement_min=0.005,
                max_internal_outliers=2,
            )
            sel = _protect_initial_rate(prev_sel, sel)
            
            # Step 6: Detect curvature and shorten for tangent fit (D1-D5 case)
            # If we already selected a full-range one-outlier rescue,
            # keep that full range and do not shorten to a tangent window.
            method_before_curvature = str(sel.get("select_method_used", ""))
            if "full_range_outlier_skip" not in method_before_curvature:
                prev_sel = sel.copy()
                sel = detect_curvature_and_shorten(
                    sel,
                    t=t_arr,
                    y=y_arr,
                    r2_min=0.97,
                    curvature_threshold=0.15,
                )
                sel = _protect_initial_rate(prev_sel, sel)
                sel = _protect_from_over_shortening(prev_sel, sel)

            # Step 7: Conservative post-selection override
            # Rescue clear mid-window overestimates by preferring long/full windows,
            # without changing the normal case.
            prev_sel = sel.copy()
            sel = apply_conservative_long_override(
                sel,
                t=t_arr,
                y=y_arr,
                min_points=int(min_points),
                min_frac=0.60,
                max_trim=3,
                min_delta_y=used_params["min_delta_y"],
                slope_min=slope_min,
                r2_min=used_params["r2_min"],
                mono_min_frac=used_params["mono_min_frac"],
                mono_max_down_steps=used_params.get("mono_max_down_steps", mono_max_down_steps),
                min_pos_steps=used_params["min_pos_steps"],
                min_snr=used_params.get("min_snr", min_snr),
                fit_method=str(fit_method),
            )
            sel = _protect_initial_rate(prev_sel, sel)

            # Step 8: Local audit
            # overestimate -> broad/long, underestimate -> tangent.
            sel = _apply_local_fit_audit(
                sel,
                t=t_arr,
                y=y_arr,
                min_points=int(min_points),
                r2_gate=float(used_params["r2_min"]),
                fit_method=str(fit_method),
            )

            r2_gate, safety_max_t_end, mono_down_gate = _compute_safety_gates(
                sel,
                used_params,
                min_points=int(min_points),
                max_t_end=max_t_end,
                mono_max_down_steps_default=int(mono_max_down_steps),
            )

            _enforce_final_safety(
                sel,
                slope_min=slope_min,
                r2_min=r2_gate,
                max_t_end=safety_max_t_end,
                min_delta_y=used_params["min_delta_y"],
                mono_min_frac=used_params["mono_min_frac"],
                mono_max_down_steps=mono_down_gate,
                min_pos_steps=used_params["min_pos_steps"],
                min_snr=used_params.get("min_snr", min_snr),
            )

            status = "ok"
            exclude_reason = ""

            row = _build_ok_row(base, sel, select_method)

        except FitSelectionError as e:
            status = "excluded"
            exclude_reason = str(e)

            # Column-1 rescue: only if excluded (do not relax normal cases).
            sel = None
            if int(base.get("col", -1)) == 1:
                if len(t_arr) >= 2:
                    rescue_n = 3 if len(t_arr) >= 3 else 2
                    mono_eps_local = _auto_mono_eps(y_arr)
                    stats = _calc_window_stats(
                        t_arr,
                        y_arr,
                        0,
                        rescue_n - 1,
                        mono_eps=mono_eps_local,
                        min_delta_y=0.0,
                        fit_method=str(fit_method),
                    )
                    if float(stats["slope"]) >= float(slope_min):
                        sel = pd.Series(stats)
                        sel["select_method_used"] = f"{select_method}_col1_rescue"
                        sel["skip_indices"] = ""
                        sel["skip_count"] = 0
                        status = "ok"
                        exclude_reason = ""

            if status == "ok" and sel is not None:
                row = _build_ok_row(base, sel, select_method)
            else:
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
                    "dy": np.nan,
                    "mono_frac": np.nan,
                    "down_steps": np.nan,
                    "pos_steps": np.nan,
                    "pos_steps_eps": np.nan,
                    "pos_eps": np.nan,
                    "rmse": np.nan,
                    "snr": np.nan,
                    "start_idx_used": np.nan,
                    "skip_indices": "",
                    "skip_count": 0,
                }

        except Exception:
            raise

        well_contexts.append(
            {
                "plate_id": plate_id,
                "well": well,
                "meta": base,
                "df_well": g,
                "t_arr": t_arr,
                "y_arr": y_arr,
                "used_params": dict(used_params),
                "status": status,
                "exclude_reason": exclude_reason,
                "sel": sel.copy() if status == "ok" and sel is not None else None,
                "row_record": row,
            }
        )

    if well_contexts:
        key_to_ctx = {(ctx["plate_id"], ctx["well"]): ctx for ctx in well_contexts}
        selected_pre = pd.DataFrame([ctx["row_record"] for ctx in well_contexts])

        if (not selected_pre.empty) and {"plate_id", "row", "col", "well", "slope", "status"}.issubset(selected_pre.columns):
            ok_cmp = selected_pre[selected_pre["status"] == "ok"].copy()
            if not ok_cmp.empty:
                ok_cmp = ok_cmp[np.isfinite(ok_cmp["col"].to_numpy(dtype=float))]
                ok_cmp = ok_cmp[np.isfinite(ok_cmp["slope"].to_numpy(dtype=float))]
                if not ok_cmp.empty:
                    ok_cmp["col_int"] = ok_cmp["col"].astype(int)
                    ok_cmp = ok_cmp.sort_values(
                        ["plate_id", "row", "col_int", "well"],
                        kind="mergesort",
                    )

                    for (plate_id, row_id), grp in ok_cmp.groupby(["plate_id", "row"], sort=False):
                        grp = grp.sort_values(["col_int", "well"], kind="mergesort")
                        recs = grp[["well", "col_int", "slope"]].to_dict("records")
                        for i in range(len(recs) - 1):
                            left = recs[i]
                            right = recs[i + 1]
                            if int(right["col_int"]) != int(left["col_int"]) + 1:
                                continue

                            slope_left = float(left["slope"])
                            slope_right = float(right["slope"])
                            if (not np.isfinite(slope_left)) or (not np.isfinite(slope_right)):
                                continue
                            if slope_left >= slope_right:
                                continue

                            ctx_left = key_to_ctx.get((plate_id, left["well"]))
                            ctx_right = key_to_ctx.get((plate_id, right["well"]))
                            if (ctx_left is None) or (ctx_right is None):
                                continue
                            if ctx_left["status"] != "ok" or ctx_left["sel"] is None:
                                continue
                            if ctx_right["status"] != "ok" or ctx_right["sel"] is None:
                                continue

                            cand_left = _neighbor_recheck_trigger(
                                ctx_left["sel"],
                                t=ctx_left["t_arr"],
                                y=ctx_left["y_arr"],
                                neighbor_slope=slope_right,
                                fit_method=str(fit_method),
                            )
                            if _selection_differs(ctx_left["sel"], cand_left):
                                r2_gate, safety_max_t_end, mono_down_gate = _compute_safety_gates(
                                    cand_left,
                                    ctx_left["used_params"],
                                    min_points=int(min_points),
                                    max_t_end=max_t_end,
                                    mono_max_down_steps_default=int(mono_max_down_steps),
                                )

                                try:
                                    _enforce_final_safety(
                                        cand_left,
                                        slope_min=slope_min,
                                        r2_min=r2_gate,
                                        max_t_end=safety_max_t_end,
                                        min_delta_y=ctx_left["used_params"]["min_delta_y"],
                                        mono_min_frac=ctx_left["used_params"]["mono_min_frac"],
                                        mono_max_down_steps=mono_down_gate,
                                        min_pos_steps=ctx_left["used_params"]["min_pos_steps"],
                                        min_snr=ctx_left["used_params"].get("min_snr", min_snr),
                                    )
                                except FitSelectionError:
                                    cand_left = ctx_left["sel"]

                                if _selection_differs(ctx_left["sel"], cand_left):
                                    ctx_left["sel"] = cand_left
                                    ctx_left["status"] = "ok"
                                    ctx_left["exclude_reason"] = ""
                                    ctx_left["row_record"] = _build_ok_row(ctx_left["meta"], cand_left, select_method)

                            try:
                                slope_left_now = float(ctx_left["sel"]["slope"])
                            except Exception:
                                slope_left_now = float(slope_left)
                            try:
                                slope_right_now = float(ctx_right["sel"]["slope"])
                            except Exception:
                                slope_right_now = float(slope_right)
                            if (not np.isfinite(slope_left_now)) or (not np.isfinite(slope_right_now)):
                                continue
                            if slope_left_now >= slope_right_now:
                                continue

                            cand_right = _neighbor_recheck_right_trigger(
                                ctx_right["sel"],
                                t=ctx_right["t_arr"],
                                y=ctx_right["y_arr"],
                                left_slope=float(slope_left_now),
                                min_points=int(min_points),
                                r2_gate=float(ctx_right["used_params"].get("r2_min", r2_min)),
                                fit_method=str(fit_method),
                            )
                            if not _selection_differs(ctx_right["sel"], cand_right):
                                continue

                            r2_gate_r, safety_max_t_end_r, mono_down_gate_r = _compute_safety_gates(
                                cand_right,
                                ctx_right["used_params"],
                                min_points=int(min_points),
                                max_t_end=max_t_end,
                                mono_max_down_steps_default=int(mono_max_down_steps),
                            )

                            try:
                                _enforce_final_safety(
                                    cand_right,
                                    slope_min=slope_min,
                                    r2_min=r2_gate_r,
                                    max_t_end=safety_max_t_end_r,
                                    min_delta_y=ctx_right["used_params"]["min_delta_y"],
                                    mono_min_frac=ctx_right["used_params"]["mono_min_frac"],
                                    mono_max_down_steps=mono_down_gate_r,
                                    min_pos_steps=ctx_right["used_params"]["min_pos_steps"],
                                    min_snr=ctx_right["used_params"].get("min_snr", min_snr),
                                )
                            except FitSelectionError:
                                continue

                            ctx_right["sel"] = cand_right
                            ctx_right["status"] = "ok"
                            ctx_right["exclude_reason"] = ""
                            ctx_right["row_record"] = _build_ok_row(ctx_right["meta"], cand_right, select_method)

            # Column-1 consensus recheck:
            # If one heat=0 well is an obvious outlier versus other polymers in the same run,
            # locally retry fitting to reduce mismatch while keeping initial-rate validity.
            for ctx in well_contexts:
                if str(ctx.get("status", "")) != "ok" or ctx.get("sel") is None:
                    continue
                meta = ctx.get("meta", {}) or {}
                col = meta.get("col", np.nan)
                if pd.isna(col):
                    continue
                try:
                    if int(col) != 1:
                        continue
                except Exception:
                    continue

                peer_slopes = _collect_col1_peer_slopes(
                    well_contexts,
                    exclude_key=(str(ctx.get("plate_id", "")), str(ctx.get("well", ""))),
                )
                cand = _col1_consensus_recheck(
                    ctx["sel"],
                    df_well=ctx["df_well"],
                    t=ctx["t_arr"],
                    y=ctx["y_arr"],
                    peer_slopes=peer_slopes,
                    used_params=ctx["used_params"],
                    min_points=int(min_points),
                    max_points=int(max_points),
                    min_span_s=float(min_span_s),
                    slope_min=float(slope_min),
                    max_t_end=max_t_end,
                    mono_eps=mono_eps,
                    min_delta_y=min_delta_y,
                    find_start=bool(find_start),
                    start_max_shift=int(start_max_shift),
                    start_window=int(start_window),
                    start_allow_down_steps=int(start_allow_down_steps),
                    min_t_start_s=float(min_t_start_s),
                    down_step_min_frac=down_step_min_frac,
                    fit_method=str(fit_method),
                    mono_max_down_steps_default=int(mono_max_down_steps),
                )
                if not _selection_differs(ctx["sel"], cand):
                    continue

                ctx["sel"] = cand
                ctx["status"] = "ok"
                ctx["exclude_reason"] = ""
                ctx["row_record"] = _build_ok_row(ctx["meta"], cand, select_method)

        selected = pd.DataFrame([ctx["row_record"] for ctx in well_contexts])
    else:
        selected = pd.DataFrame(
            columns=[
                "plate_id", "well", "row", "col", "heat_min", "polymer_id",
                "sample_name", "source_file", "status", "abs_activity", "slope",
                "intercept", "r2", "n", "t_start", "t_end", "select_method",
                "select_method_used", "exclude_reason", "dy", "mono_frac",
                "down_steps", "pos_steps", "pos_steps_eps", "pos_eps", "rmse",
                "snr", "start_idx_used", "skip_indices", "skip_count",
            ]
        )

    if plot_dir is not None and well_contexts:
        import gc

        for i, ctx in enumerate(well_contexts, start=1):
            status = str(ctx["status"])
            do_plot = (
                (plot_mode == "all")
                or (plot_mode == "ok" and status == "ok")
                or (plot_mode == "excluded" and status != "ok")
            )
            if do_plot:
                out_png = plot_dir / f"{ctx['plate_id']}" / f"{ctx['well']}.png"
                plot_fit_diagnostic(
                    df_well=ctx["df_well"],
                    meta=ctx["meta"],
                    selected=ctx["sel"] if status == "ok" else None,
                    status=status,
                    exclude_reason=str(ctx["exclude_reason"]),
                    out_png=out_png,
                )
            if i % 20 == 0:
                gc.collect()

    if plate_grid_dir is not None and plate_grid_run_id is not None and well_contexts:
        grid_dir = Path(plate_grid_dir)
        grid_dir.mkdir(parents=True, exist_ok=True)
        # When per-well PNGs are present, assemble existing files to avoid redraw.
        # Otherwise draw plate grid directly from in-memory fit contexts.
        if plot_dir is not None and Path(plot_dir).resolve() == grid_dir.resolve():
            write_plate_grid(grid_dir, str(plate_grid_run_id))
        else:
            write_plate_grid_from_well_contexts(
                well_contexts=well_contexts,
                run_plot_dir=grid_dir,
                run_id=str(plate_grid_run_id),
            )

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

    # Memory optimization: clear large intermediate dataframes before returning
    import gc
    del baseline
    gc.collect()

    return selected, rea
