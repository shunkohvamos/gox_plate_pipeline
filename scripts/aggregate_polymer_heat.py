#!/usr/bin/env python3
"""
Build polymer × heat summary and lineage for BO. Traceable to raw (run_id, manifest, lineage).

Usage:
  python scripts/aggregate_polymer_heat.py --well_table data/processed/RUN_ID/fit/rates_with_rea.csv --run_id RUN_ID --out_dir data/processed

Input: well-level result table (e.g. rates_with_rea) with columns polymer_id, heat_min, status,
       abs_activity, plate_id, well; optional REA_percent. File names are not hardcoded.
Output:
  - out_dir/{run_id}/fit/summary_simple.csv  (polymer_id, heat_min, abs_activity, REA_percent の簡易テーブル)
  - out_dir/{run_id}/fit/summary_stats.csv   (n/mean/std/sem を含む集計テーブル)
  - out_dir/{run_id}/fit/bo/bo_output.json   (ベイズ最適化用・後工程で利用)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Use non-interactive backend so script does not block on display (headless / IDE / SSH)
import matplotlib
matplotlib.use("Agg")

from gox_plate_pipeline.summary import aggregate_and_write  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    plot_per_polymer_timeseries,
    plot_per_polymer_timeseries_with_error_band,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate well-level results by polymer_id × heat_min for BO. Outputs summary, lineage, manifest.",
    )
    p.add_argument(
        "--well_table",
        required=True,
        type=Path,
        help="Path to well-level result CSV (e.g. rates_with_rea) with polymer_id, heat_min, status, abs_activity, plate_id, well.",
    )
    p.add_argument(
        "--run_id",
        required=True,
        help="Run ID for traceability. Used in output filenames and run_id column.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory under which bo/bo_output__{run_id}.json is written. Default: data/processed",
    )
    p.add_argument(
        "--manifest_inputs",
        type=Path,
        nargs="*",
        default=None,
        help="Additional input paths to include in run manifest (e.g. tidy CSV, row_map). If omitted, only --well_table is recorded.",
    )
    args = p.parse_args()

    well_path = args.well_table
    if not well_path.is_file():
        raise FileNotFoundError(f"Well table not found: {well_path}")

    well_df = pd.read_csv(well_path)
    run_id = args.run_id.strip()
    out_dir = Path(args.out_dir)

    # Manifest: well_table plus any extra inputs
    manifest_inputs = [well_path]
    if args.manifest_inputs:
        manifest_inputs.extend(Path(p) for p in args.manifest_inputs)

    fit_dir = out_dir / run_id / "fit"
    fit_bo_dir = fit_dir / "bo"
    summary_simple_path = fit_dir / "summary_simple.csv"
    summary_stats_path = fit_dir / "summary_stats.csv"
    extra_outputs = [
        f"per_polymer__{run_id}/",
        f"per_polymer_with_error__{run_id}/",
        f"t50/t50__{run_id}.csv",
    ]
    bo_path = aggregate_and_write(
        well_df,
        run_id,
        out_dir,
        input_paths_for_manifest=manifest_inputs,
        git_root=REPO_ROOT,
        bo_dir=fit_bo_dir,
        summary_simple_path=summary_simple_path,
        summary_stats_path=summary_stats_path,
        extra_output_files=extra_outputs,
    )
    print(f"Saved (table): {summary_simple_path}")
    print(f"Saved (stats): {summary_stats_path}")
    print(f"BO output: {bo_path}")

    try:
        t50_csv = plot_per_polymer_timeseries(
            summary_simple_path=summary_simple_path,
            run_id=run_id,
            out_fit_dir=fit_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        )
        print(f"Saved (t50): {t50_csv}")
        err_dir = plot_per_polymer_timeseries_with_error_band(
            summary_stats_path=summary_stats_path,
            run_id=run_id,
            out_fit_dir=fit_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        )
        if err_dir is not None:
            print(f"Saved (per polymer with error): {err_dir}")
    except Exception as e:
        print(f"Warning: per-polymer plots/t50 failed ({e}), continuing.")


if __name__ == "__main__":
    main()
