#!/usr/bin/env python3
"""
Simulate apply_conservative_long_override on all wells (problem + non-problem).
Check: (1) problem wells get target-like outcome, (2) non-problem wells unchanged.
No code changes; uses same formulas as selection.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"

# Import from actual module to match runtime
sys_path = list(__import__("sys").path)
sys_path.insert(0, str(REPO_ROOT / "src"))
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from gox_plate_pipeline.fitting.selection import (
    apply_conservative_long_override,
    _calc_window_stats,
    _best_long_window_stats,
)
from gox_plate_pipeline.fitting.core import _fit_linear, _auto_mono_eps, _auto_min_delta_y


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


def build_sel_from_row(row: pd.Series, t: np.ndarray, y: np.ndarray) -> pd.Series | None:
    """Build a minimal sel Series from rates row so we can call apply_conservative_long_override."""
    if pd.isna(row.get("t_start")) or pd.isna(row.get("t_end")):
        return None
    t_start = float(row["t_start"])
    t_end = float(row["t_end"])
    # Find indices
    i0 = int(np.argmin(np.abs(t - t_start)))
    i1 = int(np.argmin(np.abs(t - t_end)))
    if i1 < i0:
        i0, i1 = i1, i0
    mono_eps = _auto_mono_eps(y)
    min_dy = _auto_min_delta_y(y, mono_eps)
    try:
        stats = _calc_window_stats(t, y, i0, i1, mono_eps=mono_eps, min_delta_y=min_dy, fit_method="ols")
    except Exception:
        return None
    sel = pd.Series(stats)
    sel["select_method_used"] = str(row.get("select_method_used", ""))
    return sel


def main() -> None:
    runs = ["260205-R1", "260205-R2", "260205-R3"]
    problem_set = {
        ("260205-R2", "plate2", "A6"), ("260205-R2", "plate2", "A7"), ("260205-R2", "plate2", "C6"), ("260205-R2", "plate2", "C7"),
        ("260205-R2", "plate1", "F6"), ("260205-R2", "plate1", "G7"), ("260205-R2", "plate1", "H7"), ("260205-R2", "plate1", "H6"),
        ("260205-R1", "plate1", "H7"), ("260205-R3", "plate1", "H7"),
    }
    # Target slopes (from analyze_problem_wells.py output)
    target_slopes = {
        ("260205-R2", "plate2", "A6"): 0.308571,
        ("260205-R2", "plate2", "A7"): 0.116104,
        ("260205-R2", "plate2", "C6"): 0.315556,
        ("260205-R2", "plate2", "C7"): 0.132982,
        ("260205-R2", "plate1", "F6"): 0.342857,
        ("260205-R2", "plate1", "G7"): 0.219487,
        ("260205-R2", "plate1", "H7"): 0.105801,
        ("260205-R2", "plate1", "H6"): 0.289957,
        ("260205-R1", "plate1", "H7"): 0.151961,
        ("260205-R3", "plate1", "H7"): 0.214918,
    }
    print("=== Simulating apply_conservative_long_override ===\n")
    changes_non_problem = []
    problem_results = []
    for run_id in runs:
        rates = load_rates(run_id)
        tidy = load_tidy(run_id)
        if rates.empty or tidy.empty:
            continue
        for _, row in rates.iterrows():
            if row["status"] != "ok":
                continue
            plate_id = row["plate_id"]
            well = row["well"]
            sub = tidy[(tidy["plate_id"] == plate_id) & (tidy["well"] == well)].sort_values("time_s")
            if len(sub) < 6:
                continue
            t = sub["time_s"].values.astype(float)
            y = sub["signal"].values.astype(float)
            sel = build_sel_from_row(row, t, y)
            if sel is None:
                continue
            cur_slope = float(sel["slope"])
            cur_r2 = float(sel["r2"])
            cur_n = int(sel["n"])
            key = (run_id, plate_id, well)
            # Pipeline passes used_params["r2_min"]: 0.55 for last_resort rescue, else 0.98
            r2_min_override = 0.55 if cur_r2 < 0.70 else 0.98
            try:
                new_sel = apply_conservative_long_override(
                    sel.copy(),
                    t,
                    y,
                    min_points=6,
                    min_frac=0.60,
                    max_trim=3,
                    min_delta_y=0.0,
                    slope_min=0.0,
                    r2_min=r2_min_override,
                    mono_min_frac=0.85,
                    mono_max_down_steps=1,
                    min_pos_steps=2,
                    min_snr=3.0,
                    fit_method="ols",
                )
            except Exception as e:
                new_sel = sel
                print(f"  {run_id} {plate_id} {well} exception: {e}")
            new_slope = float(new_sel["slope"])
            changed = abs(new_slope - cur_slope) > 1e-6
            if key in problem_set:
                target = target_slopes.get(key, np.nan)
                problem_results.append({
                    "run": run_id, "plate": plate_id, "well": well,
                    "cur_slope": cur_slope, "new_slope": new_slope, "target_slope": target,
                    "cur_r2": cur_r2, "cur_n": cur_n, "changed": changed,
                })
            elif changed:
                changes_non_problem.append({
                    "run": run_id, "plate": plate_id, "well": well,
                    "cur_slope": cur_slope, "new_slope": new_slope,
                    "cur_method": str(row.get("select_method_used", "")),
                })
    print("--- Problem wells: current vs after override vs target ---")
    for r in problem_results:
        ok = "OK" if abs(r["new_slope"] - r["target_slope"]) / max(r["target_slope"], 1e-12) < 0.15 else "OFF"
        print(f"  {r['run']} {r['plate']} {r['well']}: cur={r['cur_slope']:.5f} -> new={r['new_slope']:.5f} (target={r['target_slope']:.5f}) {ok} changed={r['changed']}")
    print("\n--- Non-problem wells that would change (regressions) ---")
    for r in changes_non_problem[:30]:
        print(f"  {r['run']} {r['plate']} {r['well']}: cur={r['cur_slope']:.5f} -> new={r['new_slope']:.5f}  method={r['cur_method']}")
    if len(changes_non_problem) > 30:
        print(f"  ... and {len(changes_non_problem) - 30} more")
    print(f"\nTotal non-problem wells that would change: {len(changes_non_problem)}")


if __name__ == "__main__":
    main()
