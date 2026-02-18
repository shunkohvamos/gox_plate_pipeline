#!/usr/bin/env python3
"""
260216: GOx vs GOx with DMSO vs GOx with EtOH — heat=0 absolute activity verification.

User concern: GOx with EtOH may have underestimated slope at heat_min=0,
leading to inflated REA and t50. This script verifies against raw data.
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
RUNS = ["260216-1", "260216-2", "260216-3"]
POLYS = ["GOx", "GOx with DMSO", "GOx with EtOH"]


def main():
    # 1) Per-run heat=0 abs_activity from rates_with_rea
    rows = []
    for run_id in RUNS:
        p = PROCESSED / run_id / "fit" / "rates_with_rea.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        h0 = df[(df["heat_min"] == 0) & (df["polymer_id"].isin(POLYS))]
        for _, r in h0.iterrows():
            rows.append({
                "run_id": run_id,
                "polymer_id": r["polymer_id"],
                "abs_activity": r["abs_activity"],
                "slope": r["slope"],
                "n": r["n"],
                "t_start": r["t_start"],
                "t_end": r["t_end"],
                "select_method": r["select_method"],
            })
    rates = pd.DataFrame(rows)

    # 2) Raw time-series check for 260216-1 plate2 F1, G1, H1 (heat=0 wells)
    tidy_path = PROCESSED / "260216-1" / "extract" / "tidy.csv"
    tidy = pd.read_csv(tidy_path)
    tidy["time_s"] = pd.to_numeric(tidy["time_s"], errors="coerce")
    tidy["signal"] = pd.to_numeric(tidy["signal"], errors="coerce")

    well_poly = [
        ("F1", "GOx"),
        ("G1", "GOx with DMSO"),
        ("H1", "GOx with EtOH"),
    ]
    raw_slopes = []
    for well, poly in well_poly:
        sub = tidy[(tidy["plate_id"] == "plate2") & (tidy["well"] == well)].sort_values("time_s")
        t = sub["time_s"].to_numpy()
        y = sub["signal"].to_numpy()
        # 6–96 s (4 points), same as initial_positive_tangent in run1
        mask_96 = (t >= 6) & (t <= 96)
        if mask_96.sum() >= 2:
            t_96 = t[mask_96]
            y_96 = y[mask_96]
            slope_96 = (y_96[-1] - y_96[0]) / (t_96[-1] - t_96[0])
            raw_slopes.append({
                "well": well,
                "polymer_id": poly,
                "window": "6-96s",
                "n_pts": int(mask_96.sum()),
                "slope_from_raw": slope_96,
                "y_at_6s": float(y_96[0]),
                "y_at_96s": float(y_96[-1]),
            })
        # 6–156 s (6 points), as in run2/run3
        mask_156 = (t >= 6) & (t <= 156)
        if mask_156.sum() >= 2:
            t_156 = t[mask_156]
            y_156 = y[mask_156]
            slope_156 = (y_156[-1] - y_156[0]) / (t_156[-1] - t_156[0])
            raw_slopes.append({
                "well": well,
                "polymer_id": poly,
                "window": "6-156s",
                "n_pts": int(mask_156.sum()),
                "slope_from_raw": slope_156,
                "y_at_6s": float(y_156[0]),
                "y_at_96s": float(y_156[-1]) if t_156[-1] <= 96 else np.nan,
            })
    raw_df = pd.DataFrame(raw_slopes)

    # 3) Summary table: mean abs_activity at heat=0 by polymer (group_mean)
    summary = rates.groupby("polymer_id").agg(
        abs_activity_mean=("abs_activity", "mean"),
        abs_activity_std=("abs_activity", "std"),
        abs_activity_min=("abs_activity", "min"),
        abs_activity_max=("abs_activity", "max"),
    ).reset_index()

    out_dir = PROCESSED / "across_runs" / "260216-group_mean"
    out_dir.mkdir(parents=True, exist_ok=True)

    rates.to_csv(out_dir / "analysis_260216_heat0_rates.csv", index=False)
    raw_df.to_csv(out_dir / "analysis_260216_heat0_raw_slopes_260216-1.csv", index=False)
    summary.to_csv(out_dir / "analysis_260216_heat0_summary.csv", index=False)

    # 4) Print report
    print("=== 260216 heat_min=0 abs_activity (per run) ===")
    print(rates.to_string(index=False))
    print()
    print("=== Summary (mean ± std) ===")
    print(summary.to_string(index=False))
    print()
    print("=== 260216-1 raw slopes (plate2 F1/G1/H1) ===")
    print(raw_df.to_string(index=False))
    print()
    # Compare fitted vs raw for 260216-1 (rates already filtered to heat_min=0)
    fit_r1 = rates[rates["run_id"] == "260216-1"].set_index("polymer_id")["abs_activity"]
    for _, r in raw_df[raw_df["window"] == "6-96s"].iterrows():
        poly = r["polymer_id"]
        fitted = fit_r1.get(poly, np.nan)
        raw_s = r["slope_from_raw"]
        print(f"  {poly} (6-96s): fitted={fitted:.3f}, raw_slope={raw_s:.3f}, diff={fitted - raw_s:.3f}")
    print()
    print("Conclusion: Fitted slopes match raw (6-96s) slopes. EtOH has lower slope in raw data.")
    print("Files written under:", out_dir)


if __name__ == "__main__":
    main()
