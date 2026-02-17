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
    p.add_argument(
        "--reference_polymer_id",
        type=str,
        default="GOX",
        help="Reference polymer ID for functional panel/reference fallback (default: GOX).",
    )
    p.add_argument(
        "--summary_outlier_filter",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Apply robust replicate-outlier filtering before polymer×heat aggregation "
            "(1=on, 0=off; default=1)."
        ),
    )
    p.add_argument(
        "--summary_outlier_min_samples",
        type=int,
        default=3,
        help="Minimum replicate count for robust per-group outlier detection (default: 3).",
    )
    p.add_argument(
        "--summary_outlier_z_threshold",
        type=float,
        default=3.5,
        help="MAD-based robust z threshold for summary outlier filtering (default: 3.5).",
    )
    p.add_argument(
        "--summary_outlier_ratio_low",
        type=float,
        default=0.33,
        help="Lower ratio-to-median bound for summary outlier filtering (default: 0.33).",
    )
    p.add_argument(
        "--summary_outlier_ratio_high",
        type=float,
        default=3.0,
        help="Upper ratio-to-median bound for summary outlier filtering (default: 3.0).",
    )
    p.add_argument(
        "--summary_outlier_pair_ratio_threshold",
        type=float,
        default=3.0,
        help=(
            "For n=2 replicates only: minimum max/min ratio required to exclude one well "
            "using same-heat run context (default: 3.0)."
        ),
    )
    p.add_argument(
        "--summary_outlier_min_keep",
        type=int,
        default=2,
        help=(
            "Minimum wells kept per polymer×heat after summary outlier filtering (default: 2). "
            "With 2, exclusion is skipped when it would leave only one point, so SEM/error bars remain."
        ),
    )
    args = p.parse_args()

    well_path = args.well_table
    if not well_path.is_file():
        raise FileNotFoundError(f"Well table not found: {well_path}")

    well_df = pd.read_csv(well_path)
    run_id = args.run_id.strip()
    reference_polymer_id = str(args.reference_polymer_id).strip() or "GOX"
    out_dir = Path(args.out_dir)

    # Manifest: well_table plus any extra inputs
    manifest_inputs = [well_path]
    if args.manifest_inputs:
        manifest_inputs.extend(Path(p) for p in args.manifest_inputs)

    fit_dir = out_dir / run_id / "fit"
    fit_bo_dir = fit_dir / "bo"
    summary_simple_path = fit_dir / "summary_simple.csv"
    summary_stats_path = fit_dir / "summary_stats.csv"
    summary_stats_all_path = fit_dir / "summary_stats_all.csv"
    summary_outlier_events_path = fit_dir / "summary_outlier_events.csv"
    extra_outputs = [
        f"per_polymer__{run_id}/",
        f"t50/per_polymer_refnorm__{run_id}/",
        f"t50/rea_comparison_fog_panel__{run_id}/",
        f"t50/rea_comparison_fog_grid__{run_id}.png",
        f"per_polymer_with_error__{run_id}/",
        f"all_polymers_with_error__{run_id}.png",
        f"per_polymer_with_error_all__{run_id}/",
        f"all_polymers_with_error_all__{run_id}.png",
        f"t50/csv/t50__{run_id}.csv",
        "summary_outlier_events.csv",
        "summary_stats_all.csv",
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
        summary_stats_all_path=summary_stats_all_path,
        summary_outlier_events_path=summary_outlier_events_path,
        apply_summary_outlier_filter=bool(int(args.summary_outlier_filter)),
        summary_outlier_min_samples=int(args.summary_outlier_min_samples),
        summary_outlier_z_threshold=float(args.summary_outlier_z_threshold),
        summary_outlier_ratio_low=float(args.summary_outlier_ratio_low),
        summary_outlier_ratio_high=float(args.summary_outlier_ratio_high),
        summary_outlier_pair_ratio_threshold=float(args.summary_outlier_pair_ratio_threshold),
        summary_outlier_min_keep=int(args.summary_outlier_min_keep),
        extra_output_files=extra_outputs,
    )
    print(f"Saved (table): {summary_simple_path}")
    print(f"Saved (stats): {summary_stats_path}")
    if summary_outlier_events_path.is_file():
        print(f"Saved (summary outlier audit): {summary_outlier_events_path}")
    print(f"BO output: {bo_path}")

    try:
        t50_csv = plot_per_polymer_timeseries(
            summary_simple_path=summary_simple_path,
            run_id=run_id,
            out_fit_dir=fit_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
            reference_polymer_id=reference_polymer_id,
        )
        print(f"Saved (t50): {t50_csv}")
        print(f"Saved (per polymer ref-normalized): {fit_dir / 't50' / f'per_polymer_refnorm__{run_id}'}")
        print(f"Saved (REA+FoG panels): {fit_dir / 't50' / f'rea_comparison_fog_panel__{run_id}'}")
        print(f"Saved (REA+FoG panel grid): {fit_dir / 't50' / f'rea_comparison_fog_grid__{run_id}.png'}")
        err_dir = plot_per_polymer_timeseries_with_error_band(
            summary_stats_path=summary_stats_path,
            run_id=run_id,
            out_fit_dir=fit_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
            reference_polymer_id=reference_polymer_id,
            t50_definition="y0_half",
            error_band_suffix="",
        )
        if err_dir is not None:
            print(f"Saved (per polymer with error, robust): {err_dir}")
        all_with_error_robust = fit_dir / f"all_polymers_with_error__{run_id}.png"
        if all_with_error_robust.is_file():
            print(f"Saved (all polymers with error, robust): {all_with_error_robust}")
        if summary_stats_all_path.is_file():
            err_dir_all = plot_per_polymer_timeseries_with_error_band(
                summary_stats_path=summary_stats_all_path,
                run_id=run_id,
                out_fit_dir=fit_dir,
                color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
                reference_polymer_id=reference_polymer_id,
                t50_definition="y0_half",
                error_band_suffix="_all",
            )
            if err_dir_all is not None:
                print(f"Saved (per polymer with error, all data): {err_dir_all}")
            all_with_error_all = fit_dir / f"all_polymers_with_error_all__{run_id}.png"
            if all_with_error_all.is_file():
                print(f"Saved (all polymers with error, all data): {all_with_error_all}")
    except Exception as e:
        print(f"Warning: per-polymer plots/t50 failed ({e}), continuing.")


if __name__ == "__main__":
    main()
