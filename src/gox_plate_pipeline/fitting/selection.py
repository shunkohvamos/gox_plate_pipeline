# src/gox_plate_pipeline/fitting/selection.py
"""
Fit selection algorithms including fallback strategies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import FitResult, FitSelectionError, _fit_linear, _robust_sigma


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
    force_whole_max_slope_drop_frac: float = 0.06,
    force_whole_allow_trim1: bool = True,
    force_whole_allow_trim2: bool = True,
    force_whole_max_outliers: int = 2,
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
        t_start_min = float(c["t_start"].min()) if "t_start" in c.columns else 0.0
        effective_max_t_end = t_start_min + float(max_t_end)
        c = c[c["t_end"] <= effective_max_t_end].copy()
        if c.empty:
            tmin = float(cands["t_end"].min())
            tmax = float(cands["t_end"].max())
            raise FitSelectionError(
                "No candidates left after filtering: all windows ended after max_t_end "
                f"(max_t_end={float(max_t_end):.6g} relative, effective={effective_max_t_end:.6g}, "
                f"t_start_min={t_start_min:.6g}, t_end range=[{tmin:.6g}, {tmax:.6g}])."
            )

    # --- optional: force whole/long window if sufficiently linear ---
    if bool(force_whole):
        pool = c.copy()

        if "start_idx_used" in pool.columns and pool["start_idx_used"].notna().any():
            s0 = int(pool["start_idx_used"].dropna().iloc[0])
            whole_c = pool[pool["start_idx"] == s0].copy()
        else:
            whole_c = pool.copy()

        if not whole_c.empty:
            whole = whole_c.sort_values(["n", "t_end"], ascending=[False, False]).iloc[0].copy()

            max_slope = float(pool["slope"].max())
            whole_slope = float(whole["slope"])
            if max_slope > 0.0:
                slope_drop = (max_slope - whole_slope) / max_slope
            else:
                slope_drop = 0.0

            half_drop = float(whole.get("slope_half_drop_frac", 0.0))

            r2_raw = float(whole["r2"])
            r2_t1 = float(whole.get("r2_trim1", r2_raw))
            r2_t2 = float(whole.get("r2_trim2", r2_raw))
            outliers = int(whole.get("outlier_count", 0))

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

    pre = ok0.copy()

    if min_delta_y is None:
        if "min_delta_y" in ok0.columns and ok0["min_delta_y"].notna().any():
            min_dy = float(ok0["min_delta_y"].dropna().iloc[0])
        else:
            min_dy = 0.0
    else:
        min_dy = float(min_delta_y)

    ok = ok0.copy()

    if min_dy > 0.0:
        ok = ok[ok["dy"] >= float(min_dy)].copy()

    ok = ok[ok["mono_frac"] >= float(mono_min_frac)].copy()
    ok = ok[ok["down_steps"] <= int(mono_max_down_steps)].copy()

    use_strict_pos = bool(min_dy <= 0.0)
    pos_col = "pos_steps_eps" if use_strict_pos else "pos_steps"
    ok = ok[ok[pos_col] >= int(min_pos_steps)].copy()

    ok = ok[ok["snr"] >= float(min_snr)].copy()

    if ok.empty:
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
        t_s = _rank01(tmp["t_end"], False)
        se_s = _rank01(tmp["slope_se"], False)
        ht_s = _rank01(tmp["slope_half_drop_frac"], False)
        out_s = _rank01(tmp["outlier_count"], False)
        st_s = _rank01(tmp["slope_t"].abs(), True)

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


def try_skip_extend(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    r2_min: float = 0.97,
    max_skip: int = 1,
    min_extend_points: int = 3,
    residual_threshold_sigma: float = 3.0,
    max_t_end: Optional[float] = None,
) -> pd.Series:
    """
    Try to extend a selected fit by skipping intermediate outlier points.
    """
    if "end_idx" not in sel.index or "start_idx" not in sel.index:
        return sel

    start_idx = int(sel["start_idx"])
    end_idx = int(sel["end_idx"])
    n_total = len(t)

    if end_idx + max_skip + min_extend_points >= n_total:
        return sel

    slope = float(sel["slope"])
    intercept = float(sel["intercept"])

    skip_indices = []
    for skip_offset in range(1, max_skip + 2):
        check_idx = end_idx + skip_offset
        if check_idx >= n_total:
            break

        predicted = slope * t[check_idx] + intercept
        residual = abs(y[check_idx] - predicted)

        t_win = t[start_idx : end_idx + 1]
        y_win = y[start_idx : end_idx + 1]
        y_pred_win = slope * t_win + intercept
        res_win = y_win - y_pred_win
        sigma = _robust_sigma(res_win)
        threshold = residual_threshold_sigma * max(sigma, 1.0)

        if residual > threshold:
            if skip_offset <= max_skip:
                skip_indices.append(check_idx)
            else:
                return sel
        else:
            break

    if not skip_indices:
        return sel

    first_new_idx = skip_indices[-1] + 1
    if first_new_idx >= n_total:
        return sel

    skip_set = set(skip_indices)
    base_indices = [i for i in range(start_idx, end_idx + 1)]
    new_indices = []
    for i in range(first_new_idx, n_total):
        if i in skip_set:
            continue
        if max_t_end is not None and t[i] > float(max_t_end):
            break
        new_indices.append(i)

    if not new_indices:
        return sel

    for n_try in range(len(new_indices), 0, -1):
        extended_indices = base_indices + new_indices[:n_try]
        mask = np.array(extended_indices)
        t_ext = t[mask]
        y_ext = y[mask]

        fr = _fit_linear(t_ext, y_ext)
        if fr.r2 >= r2_min:
            new_sel = sel.copy()
            new_sel["slope"] = float(fr.slope)
            new_sel["intercept"] = float(fr.intercept)
            new_sel["r2"] = float(fr.r2)
            new_sel["t_end"] = float(t_ext[-1])
            new_sel["n"] = int(len(t_ext))
            new_sel["end_idx"] = int(max(extended_indices))
            new_sel["skip_indices"] = ",".join(str(i) for i in skip_indices)
            new_sel["skip_count"] = len(skip_indices)
            orig_method = str(sel.get("select_method_used", ""))
            new_sel["select_method_used"] = f"{orig_method}_skip{len(skip_indices)}"
            return new_sel

    return sel


def try_extend_fit(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    r2_min: float = 0.96,
    r2_drop_tolerance: float = 0.005,
) -> pd.Series:
    """
    Try to extend fit to include more points while maintaining R² quality.
    """
    if "end_idx" not in sel.index or "start_idx" not in sel.index:
        return sel

    start_idx = int(sel["start_idx"])
    end_idx = int(sel["end_idx"])
    n_total = len(t)
    orig_r2 = float(sel["r2"])

    if end_idx >= n_total - 1:
        return sel

    skip_set = set()
    if "skip_indices" in sel.index and pd.notna(sel.get("skip_indices", "")):
        skip_str = str(sel["skip_indices"])
        if skip_str:
            for idx_str in skip_str.split(","):
                idx_str = idx_str.strip()
                if idx_str.isdigit():
                    skip_set.add(int(idx_str))

    base_indices = [i for i in range(start_idx, end_idx + 1) if i not in skip_set]

    best_sel = sel
    best_end = end_idx
    best_r2 = orig_r2

    for new_end in range(end_idx + 1, n_total):
        if new_end in skip_set:
            continue

        extended_indices = base_indices + [i for i in range(end_idx + 1, new_end + 1) if i not in skip_set]
        if len(extended_indices) < 4:
            continue

        mask = np.array(extended_indices)
        t_ext = t[mask]
        y_ext = y[mask]

        fr = _fit_linear(t_ext, y_ext)

        r2_drop_from_orig = orig_r2 - fr.r2
        if fr.r2 >= r2_min and r2_drop_from_orig <= r2_drop_tolerance:
            if fr.r2 >= best_r2 - 0.001 or new_end > best_end:
                best_end = new_end
                best_sel = sel.copy()
                best_sel["slope"] = float(fr.slope)
                best_sel["intercept"] = float(fr.intercept)
                best_sel["r2"] = float(fr.r2)
                best_sel["t_end"] = float(t_ext[-1])
                best_sel["n"] = int(len(t_ext))
                best_sel["end_idx"] = int(new_end)
                best_r2 = fr.r2

    if best_end > end_idx:
        orig_method = str(best_sel.get("select_method_used", ""))
        if "_ext" not in orig_method:
            best_sel["select_method_used"] = f"{orig_method}_ext"

    return best_sel


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
    Final safety gate (must not silently pass).
    """
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

    if max_t_end is not None:
        t_start = float(sel.get("t_start", 0.0))
        effective_max_t_end = t_start + float(max_t_end)
        if t_end > effective_max_t_end:
            raise FitSelectionError(
                f"Final safety triggered: selected t_end {t_end:.6g} > effective max_t_end {effective_max_t_end:.6g} "
                f"(t_start={t_start:.6g}, max_t_end={float(max_t_end):.6g} relative)"
            )

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


def find_best_short_window(
    t: np.ndarray,
    y: np.ndarray,
    min_points: int = 4,
    max_points: int = 8,
    r2_min: float = 0.70,
    slope_min: float = 0.0,
    min_snr: float = 2.0,
    max_start_frac: float = 0.5,
    max_noise_cv: float = 0.25,
) -> Optional[pd.Series]:
    """
    Last resort: find the best short window with highest R² for noisy data.

    This function scans short windows in the FIRST HALF of the data and selects
    the one with highest R². Also rejects data that is too noisy overall.

    Parameters
    ----------
    t, y : np.ndarray
        Time and signal arrays
    min_points : int
        Minimum points in window (default 4)
    max_points : int
        Maximum points in window (default 8)
    r2_min : float
        Minimum R² threshold (default 0.70)
    slope_min : float
        Minimum slope threshold (default 0.0)
    min_snr : float
        Minimum SNR threshold (default 2.0)
    max_start_frac : float
        Maximum fraction of data length for window start (default 0.5 = first half)
        This ensures we fit "initial rate", not arbitrary late segments.
    max_noise_cv : float
        Maximum coefficient of variation (std/mean) of residuals from global trend.
        If data is too noisy (CV > max_noise_cv), reject entirely.

    Returns
    -------
    pd.Series or None
        Best fit as Series, or None if no acceptable window found
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)

    if n < min_points:
        return None

    # --- Global noise check: evaluate on FIRST HALF only ---
    # This allows "drift" patterns (rise then fall) to pass if initial portion is clean
    try:
        half_n = max(min_points, n // 2)
        t_half = t[:half_n]
        y_half = y[:half_n]
        
        # Check if first half has positive trend (rising)
        half_fit = _fit_linear(t_half, y_half)
        
        # If slope is negative in first half, this is not initial-rate-like data
        if half_fit.slope <= 0:
            return None
        
        # Compute noise CV on first half only
        half_pred = half_fit.slope * t_half + half_fit.intercept
        half_res = y_half - half_pred
        half_rmse = float(np.sqrt(np.mean(half_res**2)))
        y_half_range = float(np.max(y_half) - np.min(y_half))
        noise_cv = half_rmse / max(y_half_range, 1e-12)
        
        if noise_cv > max_noise_cv:
            # First half is too noisy - reject
            return None
    except Exception:
        pass  # If fit fails, continue with window search

    # Maximum start index (restrict to first half of data)
    max_start_idx = int(n * max_start_frac)
    if max_start_idx < min_points:
        max_start_idx = min_points

    # --- Strategy: "Initial Rate First" ---
    # For enzyme kinetics, we want the EARLIEST window with sufficient quality,
    # not the highest R². Early windows capture the initial rate (steepest slope).
    # Later windows may have higher R² but represent a decaying rate.

    # First pass: Find the best window starting at index 0 (true initial rate)
    # This prioritizes true "initial velocity" measurement
    best_fit_start0 = None
    best_length_start0 = 0

    for length in range(min_points, min(max_points + 1, n + 1)):
        end = length - 1
        if end >= n:
            break

        t_win = t[0 : end + 1]
        y_win = y[0 : end + 1]

        fr = _fit_linear(t_win, y_win)

        if fr.slope < slope_min:
            continue
        if fr.r2 < r2_min:
            continue

        dy = float(y_win[-1] - y_win[0])
        yhat = fr.slope * t_win + fr.intercept
        res = y_win - yhat
        rmse = float(np.sqrt(np.mean(res**2))) if len(y_win) > 0 else 1e12
        snr = abs(dy) / (rmse + 1e-12)

        if snr < min_snr:
            continue

        if len(y_win) >= 3:
            y_smooth = (
                pd.Series(y_win)
                .rolling(window=3, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=float)
            )
        else:
            y_smooth = y_win

        steps = np.diff(y_smooth)
        mono_frac = float(np.mean(steps >= 0)) if steps.size > 0 else 1.0
        down_steps = int(np.sum(steps < 0)) if steps.size > 0 else 0
        pos_steps = int(np.sum(steps > 0)) if steps.size > 0 else 0

        # Prefer longer windows among those starting at 0
        if length > best_length_start0:
            best_length_start0 = length
            best_fit_start0 = pd.Series({
                "t_start": float(t_win[0]),
                "t_end": float(t_win[-1]),
                "n": int(length),
                "slope": float(fr.slope),
                "intercept": float(fr.intercept),
                "r2": float(fr.r2),
                "start_idx": 0,
                "end_idx": int(end),
                "dy": float(dy),
                "mono_frac": float(mono_frac),
                "down_steps": int(down_steps),
                "pos_steps": int(pos_steps),
                "pos_steps_eps": int(pos_steps),
                "pos_eps": 0.0,
                "rmse": float(rmse),
                "snr": float(snr),
                "start_idx_used": 0,
                "select_method_used": "last_resort",
            })

    # If we found a good window starting at 0, use it (true initial rate)
    if best_fit_start0 is not None:
        return best_fit_start0

    # Second pass: If no window starting at 0 works, scan early windows
    # Still prioritize EARLIEST start, not highest R²
    for start in range(1, min(max_start_idx, n - min_points + 1)):
        for length in range(min_points, min(max_points + 1, n - start + 1)):
            end = start + length - 1
            if end >= n:
                break

            t_win = t[start : end + 1]
            y_win = y[start : end + 1]

            fr = _fit_linear(t_win, y_win)

            if fr.slope < slope_min:
                continue
            if fr.r2 < r2_min:
                continue

            dy = float(y_win[-1] - y_win[0])
            yhat = fr.slope * t_win + fr.intercept
            res = y_win - yhat
            rmse = float(np.sqrt(np.mean(res**2))) if len(y_win) > 0 else 1e12
            snr = abs(dy) / (rmse + 1e-12)

            if snr < min_snr:
                continue

            if len(y_win) >= 3:
                y_smooth = (
                    pd.Series(y_win)
                    .rolling(window=3, center=True, min_periods=1)
                    .median()
                    .to_numpy(dtype=float)
                )
            else:
                y_smooth = y_win

            steps = np.diff(y_smooth)
            mono_frac = float(np.mean(steps >= 0)) if steps.size > 0 else 1.0
            down_steps = int(np.sum(steps < 0)) if steps.size > 0 else 0
            pos_steps = int(np.sum(steps > 0)) if steps.size > 0 else 0

            # Return the FIRST acceptable window (earliest start)
            # This captures initial rate, not some arbitrary later segment
            return pd.Series({
                "t_start": float(t_win[0]),
                "t_end": float(t_win[-1]),
                "n": int(length),
                "slope": float(fr.slope),
                "intercept": float(fr.intercept),
                "r2": float(fr.r2),
                "start_idx": int(start),
                "end_idx": int(end),
                "dy": float(dy),
                "mono_frac": float(mono_frac),
                "down_steps": int(down_steps),
                "pos_steps": int(pos_steps),
                "pos_steps_eps": int(pos_steps),
                "pos_eps": 0.0,
                "rmse": float(rmse),
                "snr": float(snr),
                "start_idx_used": int(start),
                "select_method_used": "last_resort",
            })

    return None


def detect_internal_outliers(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    outlier_sigma: float = 3.0,
    r2_min: float = 0.97,
) -> pd.Series:
    """
    Detect and remove internal outliers within the fitting window.
    
    This handles cases like C2 where an outlier exists in the middle of
    the fitting interval, not just at the edges.
    
    Parameters
    ----------
    sel : pd.Series
        Current selected fit
    t, y : np.ndarray
        Full time and signal arrays
    outlier_sigma : float
        Threshold for outlier detection (default 3.0)
    r2_min : float
        Minimum R² after outlier removal (default 0.97)
    
    Returns
    -------
    pd.Series
        Updated fit with internal outlier removed, or original if no improvement
    """
    if "start_idx" not in sel.index or "end_idx" not in sel.index:
        return sel
    
    start_idx = int(sel["start_idx"])
    end_idx = int(sel["end_idx"])
    
    t_win = t[start_idx:end_idx + 1]
    y_win = y[start_idx:end_idx + 1]
    n = len(t_win)
    
    if n < 5:  # Need enough points
        return sel
    
    slope = float(sel["slope"])
    intercept = float(sel["intercept"])
    
    # Calculate residuals
    y_pred = slope * t_win + intercept
    residuals = y_win - y_pred
    
    # Use robust sigma
    sigma = _robust_sigma(residuals)
    if sigma < 1e-12:
        return sel
    
    # Find outliers
    outlier_mask = np.abs(residuals) > outlier_sigma * sigma
    outlier_local_indices = np.where(outlier_mask)[0]
    
    # Only proceed if there's exactly 1 outlier
    if len(outlier_local_indices) != 1:
        return sel
    
    outlier_local_idx = int(outlier_local_indices[0])
    outlier_global_idx = start_idx + outlier_local_idx
    
    # Remove outlier and refit
    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_local_idx] = False
    
    t_clean = t_win[clean_mask]
    y_clean = y_win[clean_mask]
    
    if len(t_clean) < 4:
        return sel
    
    try:
        clean_fit = _fit_linear(t_clean, y_clean)
    except Exception:
        return sel
    
    # Only accept if R² improves or stays high
    if clean_fit.r2 < r2_min:
        return sel
    
    # Create new selection with outlier removed
    new_sel = sel.copy()
    new_sel["slope"] = float(clean_fit.slope)
    new_sel["intercept"] = float(clean_fit.intercept)
    new_sel["r2"] = float(clean_fit.r2)
    new_sel["n"] = int(len(t_clean))
    
    # Compute new dy and stats
    dy = float(y_clean[-1] - y_clean[0])
    new_sel["dy"] = dy
    
    y_pred_clean = clean_fit.slope * t_clean + clean_fit.intercept
    res_clean = y_clean - y_pred_clean
    rmse = float(np.sqrt(np.mean(res_clean**2)))
    new_sel["rmse"] = rmse
    new_sel["snr"] = abs(dy) / (rmse + 1e-12)
    
    # Mark the skipped outlier
    existing_skip = str(sel.get("skip_indices", ""))
    if existing_skip and existing_skip != "nan":
        new_skip = f"{existing_skip},{outlier_global_idx}"
    else:
        new_skip = str(outlier_global_idx)
    new_sel["skip_indices"] = new_skip
    
    orig_method = str(sel.get("select_method_used", ""))
    new_sel["select_method_used"] = f"{orig_method}_intskip1"
    
    return new_sel


def detect_curvature_and_shorten(
    sel: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    r2_min: float = 0.97,
    curvature_threshold: float = 0.15,
) -> pd.Series:
    """
    Detect if fit line crosses the data curve and shorten to tangent fit.
    
    For saturation curves (like D1-D5), the fit line should be tangent to
    the curve, not crossing through it. This function shortens the fit
    to just the initial linear portion.
    
    Parameters
    ----------
    sel : pd.Series
        Current selected fit
    t, y : np.ndarray
        Full time and signal arrays
    r2_min : float
        Minimum R² for shortened fit
    curvature_threshold : float
        Threshold for detecting curvature (ratio of later slope to initial slope)
    
    Returns
    -------
    pd.Series
        Shortened fit if curvature detected, otherwise original
    """
    if "start_idx" not in sel.index or "end_idx" not in sel.index:
        return sel
    
    start_idx = int(sel["start_idx"])
    end_idx = int(sel["end_idx"])
    n_fit = end_idx - start_idx + 1
    
    if n_fit < 6:  # Need enough points to detect curvature
        return sel
    
    t_win = t[start_idx:end_idx + 1]
    y_win = y[start_idx:end_idx + 1]
    
    slope = float(sel["slope"])
    intercept = float(sel["intercept"])
    
    # Check if the fit line crosses the data
    y_pred = slope * t_win + intercept
    residuals = y_win - y_pred
    
    # For a tangent fit, residuals should be mostly negative (line above curve)
    # or mostly positive (line below curve) with a pattern
    # For a crossing fit, residuals change sign through the data
    
    # Simple curvature detection: compare slope of first half vs second half
    half = n_fit // 2
    if half < 3:
        return sel
    
    try:
        first_half_fit = _fit_linear(t_win[:half], y_win[:half])
        second_half_fit = _fit_linear(t_win[half:], y_win[half:])
    except Exception:
        return sel
    
    # If second half slope is significantly lower than first half, we have curvature
    if first_half_fit.slope <= 0:
        return sel
    
    slope_ratio = second_half_fit.slope / first_half_fit.slope
    
    if slope_ratio > (1 - curvature_threshold):
        # Not enough curvature, keep original
        return sel
    
    # Curvature detected! Find optimal shorter window
    # Start from minimum 4 points and find where slope drop starts
    best_fit = None
    best_n = 4
    
    for n_try in range(4, min(half + 2, n_fit)):
        t_short = t_win[:n_try]
        y_short = y_win[:n_try]
        
        try:
            short_fit = _fit_linear(t_short, y_short)
        except Exception:
            continue
        
        if short_fit.r2 >= r2_min and short_fit.slope > 0:
            best_fit = short_fit
            best_n = n_try
    
    if best_fit is None:
        return sel
    
    # Create shortened selection
    new_sel = sel.copy()
    new_sel["slope"] = float(best_fit.slope)
    new_sel["intercept"] = float(best_fit.intercept)
    new_sel["r2"] = float(best_fit.r2)
    new_sel["n"] = int(best_n)
    new_sel["t_end"] = float(t_win[best_n - 1])
    new_sel["end_idx"] = start_idx + best_n - 1
    
    # Compute new stats
    t_short = t_win[:best_n]
    y_short = y_win[:best_n]
    dy = float(y_short[-1] - y_short[0])
    new_sel["dy"] = dy
    
    y_pred_short = best_fit.slope * t_short + best_fit.intercept
    res_short = y_short - y_pred_short
    rmse = float(np.sqrt(np.mean(res_short**2)))
    new_sel["rmse"] = rmse
    new_sel["snr"] = abs(dy) / (rmse + 1e-12)
    
    orig_method = str(sel.get("select_method_used", ""))
    new_sel["select_method_used"] = f"{orig_method}_tangent"
    
    return new_sel


def fit_with_outlier_skip_full_range(
    t: np.ndarray,
    y: np.ndarray,
    outlier_sigma: float = 3.5,
    r2_min: float = 0.97,
    min_points: int = 6,
) -> Optional[pd.Series]:
    """
    Fit full data range with single outlier removal.
    
    This is for cases like C1, C3 where there's an outlier in the middle,
    and we want to fit all points EXCEPT the outlier (not just the points
    after the outlier).
    
    Parameters
    ----------
    t, y : np.ndarray
        Full time and signal arrays
    outlier_sigma : float
        Threshold for outlier detection
    r2_min : float
        Minimum R² threshold
    min_points : int
        Minimum points after outlier removal
    
    Returns
    -------
    pd.Series or None
        Fit with outlier removed, spanning full range
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    
    if n < min_points + 1:
        return None
    
    # First fit all data
    try:
        full_fit = _fit_linear(t, y)
    except Exception:
        return None
    
    # Calculate residuals
    y_pred = full_fit.slope * t + full_fit.intercept
    residuals = y - y_pred
    
    # Use robust sigma
    sigma = _robust_sigma(residuals)
    if sigma < 1e-12:
        return None
    
    # Find outliers
    outlier_mask = np.abs(residuals) > outlier_sigma * sigma
    outlier_indices = np.where(outlier_mask)[0]
    
    # Only proceed if there's exactly 1 outlier
    if len(outlier_indices) != 1:
        return None
    
    outlier_idx = int(outlier_indices[0])
    
    # Remove outlier and refit
    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_idx] = False
    
    t_clean = t[clean_mask]
    y_clean = y[clean_mask]
    
    if len(t_clean) < min_points:
        return None
    
    try:
        clean_fit = _fit_linear(t_clean, y_clean)
    except Exception:
        return None
    
    if clean_fit.r2 < r2_min or clean_fit.slope <= 0:
        return None
    
    # Compute stats
    dy = float(y_clean[-1] - y_clean[0])
    y_pred_clean = clean_fit.slope * t_clean + clean_fit.intercept
    res_clean = y_clean - y_pred_clean
    rmse = float(np.sqrt(np.mean(res_clean**2)))
    snr = abs(dy) / (rmse + 1e-12)
    
    # Mono frac on clean data
    if len(y_clean) >= 3:
        y_smooth = (
            pd.Series(y_clean)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
    else:
        y_smooth = y_clean
    
    steps = np.diff(y_smooth)
    mono_frac = float(np.mean(steps >= 0)) if steps.size > 0 else 1.0
    down_steps = int(np.sum(steps < 0)) if steps.size > 0 else 0
    pos_steps = int(np.sum(steps > 0)) if steps.size > 0 else 0
    
    # Find indices
    clean_indices = np.where(clean_mask)[0]
    start_idx = int(clean_indices[0])
    end_idx = int(clean_indices[-1])
    
    return pd.Series({
        "t_start": float(t_clean[0]),
        "t_end": float(t_clean[-1]),
        "n": int(len(t_clean)),
        "slope": float(clean_fit.slope),
        "intercept": float(clean_fit.intercept),
        "r2": float(clean_fit.r2),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "dy": float(dy),
        "mono_frac": float(mono_frac),
        "down_steps": int(down_steps),
        "pos_steps": int(pos_steps),
        "pos_steps_eps": int(pos_steps),
        "pos_eps": 0.0,
        "rmse": float(rmse),
        "snr": float(snr),
        "start_idx_used": start_idx,
        "skip_indices": str(outlier_idx),
        "select_method_used": "full_range_outlier_skip",
    })


def find_fit_with_outlier_removal(
    t: np.ndarray,
    y: np.ndarray,
    min_points: int = 6,
    r2_min: float = 0.80,
    slope_min: float = 0.0,
    min_snr: float = 2.0,
    outlier_sigma: float = 3.0,
) -> Optional[pd.Series]:
    """
    Try to fit by removing a single outlier point.
    
    This handles cases where data is clean except for one extreme outlier
    (e.g., a spike or measurement error).
    
    Parameters
    ----------
    t, y : np.ndarray
        Time and signal arrays
    min_points : int
        Minimum points required after outlier removal (default 6)
    r2_min : float
        Minimum R² threshold (default 0.80)
    slope_min : float
        Minimum slope threshold (default 0.0)
    min_snr : float
        Minimum SNR threshold (default 2.0)
    outlier_sigma : float
        Threshold for outlier detection in standard deviations (default 3.0)
    
    Returns
    -------
    pd.Series or None
        Best fit with outlier removed, or None if no acceptable fit found
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    
    if n < min_points + 1:  # Need at least min_points after removing 1
        return None
    
    # Step 1: Fit all data to identify potential outlier
    try:
        full_fit = _fit_linear(t, y)
    except Exception:
        return None
    
    # Calculate residuals
    y_pred = full_fit.slope * t + full_fit.intercept
    residuals = y - y_pred
    
    # Use robust sigma estimate (MAD-based)
    sigma = _robust_sigma(residuals)
    if sigma < 1e-12:
        return None
    
    # Find outliers (points with |residual| > outlier_sigma * sigma)
    outlier_mask = np.abs(residuals) > outlier_sigma * sigma
    outlier_indices = np.where(outlier_mask)[0]
    
    # Only proceed if there's exactly 1 outlier
    if len(outlier_indices) != 1:
        return None
    
    outlier_idx = int(outlier_indices[0])
    
    # Step 2: Remove outlier and refit
    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    
    t_clean = t[mask]
    y_clean = y[mask]
    
    if len(t_clean) < min_points:
        return None
    
    try:
        clean_fit = _fit_linear(t_clean, y_clean)
    except Exception:
        return None
    
    # Check quality thresholds
    if clean_fit.slope < slope_min:
        return None
    if clean_fit.r2 < r2_min:
        return None
    
    # Calculate SNR
    dy = float(y_clean[-1] - y_clean[0])
    y_pred_clean = clean_fit.slope * t_clean + clean_fit.intercept
    res_clean = y_clean - y_pred_clean
    rmse = float(np.sqrt(np.mean(res_clean**2)))
    snr = abs(dy) / (rmse + 1e-12)
    
    if snr < min_snr:
        return None
    
    # Calculate mono_frac on clean data
    if len(y_clean) >= 3:
        y_smooth = (
            pd.Series(y_clean)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
    else:
        y_smooth = y_clean
    
    steps = np.diff(y_smooth)
    mono_frac = float(np.mean(steps >= 0)) if steps.size > 0 else 1.0
    down_steps = int(np.sum(steps < 0)) if steps.size > 0 else 0
    pos_steps = int(np.sum(steps > 0)) if steps.size > 0 else 0
    
    # Find indices for clean data
    clean_indices = np.where(mask)[0]
    start_idx = int(clean_indices[0])
    end_idx = int(clean_indices[-1])
    
    return pd.Series({
        "t_start": float(t_clean[0]),
        "t_end": float(t_clean[-1]),
        "n": int(len(t_clean)),
        "slope": float(clean_fit.slope),
        "intercept": float(clean_fit.intercept),
        "r2": float(clean_fit.r2),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "dy": float(dy),
        "mono_frac": float(mono_frac),
        "down_steps": int(down_steps),
        "pos_steps": int(pos_steps),
        "pos_steps_eps": int(pos_steps),
        "pos_eps": 0.0,
        "rmse": float(rmse),
        "snr": float(snr),
        "start_idx_used": start_idx,
        "skip_indices": str(outlier_idx),
        "select_method_used": "outlier_removed",
    })
