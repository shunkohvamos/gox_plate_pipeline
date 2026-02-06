#!/usr/bin/env python3
"""
Analyze problem wells: current fit vs desired fit (from user description).
No code changes to fitting pipeline; data-only analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"


def load_rates(run_id: str) -> pd.DataFrame:
    p = DATA / run_id / "fit" / "rates_with_rea.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_tidy(run_id: str) -> pd.DataFrame:
    p = DATA / run_id / "extract" / "tidy.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def fit_slope_r2(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> tuple[float, float]:
    """Slope and R2 for window [i0, i1] inclusive."""
    if i1 < i0 or i1 - i0 + 1 < 2:
        return np.nan, np.nan
    tt = t[i0 : i1 + 1]
    yy = y[i0 : i1 + 1]
    r = linregress(tt, yy)
    return float(r.slope), float(r.rvalue ** 2)


def analyze_well(run_id: str, plate_id: str, well: str) -> dict:
    tidy = load_tidy(run_id)
    rates = load_rates(run_id)
    if tidy.empty or rates.empty:
        return {}
    sub = tidy[(tidy["plate_id"] == plate_id) & (tidy["well"] == well)].sort_values("time_s")
    if sub.empty:
        return {}
    t = sub["time_s"].values.astype(float)
    y = sub["signal"].values.astype(float)
    n = len(t)
    row = rates[(rates["plate_id"] == plate_id) & (rates["well"] == well)]
    if row.empty:
        cur_slope, cur_r2, cur_n, cur_t_start, cur_t_end, cur_method = np.nan, np.nan, np.nan, np.nan, np.nan, ""
    else:
        row = row.iloc[0]
        cur_slope = float(row["slope"]) if pd.notna(row.get("slope")) else np.nan
        cur_r2 = float(row["r2"]) if pd.notna(row.get("r2")) else np.nan
        cur_n = int(row["n"]) if pd.notna(row.get("n")) else np.nan
        cur_t_start = float(row["t_start"]) if pd.notna(row.get("t_start")) else np.nan
        cur_t_end = float(row["t_end"]) if pd.notna(row.get("t_end")) else np.nan
        cur_method = str(row.get("select_method_used", ""))

    out = {
        "run_id": run_id,
        "plate_id": plate_id,
        "well": well,
        "n_points": n,
        "t_range": (float(t[0]), float(t[-1])),
        "y_range": (float(np.nanmin(y)), float(np.nanmax(y))),
        "cur_slope": cur_slope,
        "cur_r2": cur_r2,
        "cur_n": cur_n,
        "cur_t_start": cur_t_start,
        "cur_t_end": cur_t_end,
        "cur_method": cur_method,
        "target_slope": np.nan,
        "target_descr": "",
        "full_slope": np.nan,
        "full_r2": np.nan,
    }
    if n >= 2:
        full_slope, full_r2 = fit_slope_r2(t, y, 0, n - 1)
        out["full_slope"] = full_slope
        out["full_r2"] = full_r2
    return out, t, y


# Problem wells with desired fitting (user description)
PROBLEM_WELLS = [
    # R2 plate2
    ("260205-R2", "plate2", "A6", "full or (drop first 2 then full) or (drop last 3); avoid mid-window overestimate"),
    ("260205-R2", "plate2", "A7", "full range preferred over 4-point R2-max"),
    ("260205-R2", "plate2", "C6", "first 9 points or similar; avoid late-window overestimate"),
    ("260205-R2", "plate2", "C7", "from 3rd point to end (drop first 2); avoid 4-point overestimate"),
    # R2 plate1
    ("260205-R2", "plate1", "F6", "first 7 points or full; avoid mid-window"),
    ("260205-R2", "plate1", "G7", "mid-to-end or full (lag); avoid first-8 underestimate"),
    ("260205-R2", "plate1", "H7", "full; mid 6 points overestimate (accidental rise)"),
    ("260205-R2", "plate1", "H6", "full range; current early window underestimate"),
    # R1 plate1
    ("260205-R1", "plate1", "H7", "points 4–20 (drop first 3 and last 1) or full; avoid 4–7 overestimate"),
    # R3 plate1
    ("260205-R3", "plate1", "H7", "first 12 points; avoid mid 7–12 window overestimate"),
]


def main() -> None:
    print("=== Problem wells: current vs target (from user description) ===\n")
    for run_id, plate_id, well, descr in PROBLEM_WELLS:
        res = analyze_well(run_id, plate_id, well)
        if not res:
            print(f"{run_id} {plate_id} {well}: no data")
            continue
        info, t, y = res[0], res[1], res[2]
        n = len(t)
        # Compute target slopes for each case
        target_slope = np.nan
        target_descr = descr
        if "260205-R2" == run_id and "plate2" == plate_id and "A6" == well:
            # full or drop first 2 full or drop last 3
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            s_d2, r_d2 = fit_slope_r2(t, y, 2, n - 1) if n > 2 else (np.nan, np.nan)
            s_d3, r_d3 = fit_slope_r2(t, y, 0, n - 4) if n > 3 else (np.nan, np.nan)
            target_slope = s_full
            info["target_slope"] = target_slope
            info["_full"] = s_full
            info["_drop2"] = s_d2
            info["_droplast3"] = s_d3
        elif "260205-R2" == run_id and "plate2" == plate_id and "A7" == well:
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            target_slope = s_full
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate2" == plate_id and "C6" == well:
            # first 9 points
            if n >= 9:
                s9, r9 = fit_slope_r2(t, y, 0, 8)
                target_slope = s9
            else:
                target_slope = fit_slope_r2(t, y, 0, n - 1)[0]
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate2" == plate_id and "C7" == well:
            # from index 2 to end (3rd point to last)
            if n > 2:
                s_2end, r_2end = fit_slope_r2(t, y, 2, n - 1)
                target_slope = s_2end
            else:
                target_slope = np.nan
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate1" == plate_id and "F6" == well:
            s_first7, r_first7 = fit_slope_r2(t, y, 0, min(6, n - 1)) if n >= 7 else (np.nan, np.nan)
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            target_slope = s_first7 if not np.isnan(s_first7) else s_full
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate1" == plate_id and "G7" == well:
            # mid-to-end or full
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            s_7end, r_7end = fit_slope_r2(t, y, 7, n - 1) if n > 7 else (np.nan, np.nan)
            target_slope = s_7end if not np.isnan(s_7end) else s_full
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate1" == plate_id and "H7" == well:
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            target_slope = s_full
            info["target_slope"] = target_slope
        elif "260205-R2" == run_id and "plate1" == plate_id and "H6" == well:
            s_full, r_full = fit_slope_r2(t, y, 0, n - 1)
            target_slope = s_full
            info["target_slope"] = target_slope
        elif "260205-R1" == run_id and "plate1" == plate_id and "H7" == well:
            # indices 3..19 (4th to 20th), 0-based: 3 to 19
            if n >= 20:
                s_4_20, r_4_20 = fit_slope_r2(t, y, 3, 19)
                target_slope = s_4_20
            else:
                target_slope = fit_slope_r2(t, y, 0, n - 1)[0]
            info["target_slope"] = target_slope
        elif "260205-R3" == run_id and "plate1" == plate_id and "H7" == well:
            # first 12 points (indices 0..11)
            if n >= 12:
                s12, r12 = fit_slope_r2(t, y, 0, 11)
                target_slope = s12
            else:
                target_slope = fit_slope_r2(t, y, 0, n - 1)[0]
            info["target_slope"] = target_slope
        else:
            info["target_slope"] = info["full_slope"]

        # Report
        print(f"--- {run_id} {plate_id} {well} ---")
        print(f"  Desired: {target_descr}")
        print(f"  n_pts={info['n_points']}, t=[{info['t_range'][0]:.0f}, {info['t_range'][1]:.0f}], y=[{info['y_range'][0]:.1f}, {info['y_range'][1]:.1f}]")
        print(f"  Current: slope={info['cur_slope']:.6f}, r2={info['cur_r2']:.4f}, n={info['cur_n']}, method={info['cur_method']}")
        print(f"  Full:    slope={info['full_slope']:.6f}, r2={info['full_r2']:.4f}")
        print(f"  Target:  slope={info['target_slope']:.6f}")
        if np.isfinite(info["cur_slope"]) and np.isfinite(info["target_slope"]) and info["target_slope"] != 0:
            ratio = info["cur_slope"] / info["target_slope"]
            print(f"  Ratio cur/target: {ratio:.3f}")
        print()
    return None


if __name__ == "__main__":
    main()
