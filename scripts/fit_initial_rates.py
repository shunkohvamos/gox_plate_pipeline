from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure local src/ is used (avoid importing an older installed package)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Use non-interactive backend so script does not block on display (headless / IDE / SSH)
import matplotlib
matplotlib.use("Agg")

from gox_plate_pipeline.fitting import compute_rates_and_rea, write_plate_grid  # noqa: E402
from gox_plate_pipeline.bo_data import (  # noqa: E402
    build_bo_learning_data,
    collect_fog_summary_paths,
    load_bo_catalog,
    write_bo_learning_csv,
    write_exclusion_report,
)
from gox_plate_pipeline.fog import build_fog_summary, write_fog_summary_csv  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    plot_per_polymer_timeseries,
    plot_per_polymer_timeseries_with_error_band,
    plot_per_polymer_timeseries_across_runs_with_error_band,
)
from gox_plate_pipeline.summary import aggregate_and_write  # noqa: E402


def _derive_run_id_from_tidy_path(tidy_path: Path) -> str:
    """run_id: .../run_id/extract/tidy.csv なら親の親、.../run_id/tidy.csv なら親、従来 __tidy なら stem から取得。"""
    if tidy_path.name == "tidy.csv":
        parent = tidy_path.parent
        if parent.name == "extract" and parent.parent.name:
            return parent.parent.name
        if parent.name:
            return parent.name
    stem = tidy_path.stem
    if stem.endswith("__tidy"):
        return stem[: -len("__tidy")]
    return stem


def main() -> None:
    p = argparse.ArgumentParser(description="Fit per-well initial rates and compute REA.")
    p.add_argument("--tidy", required=True, help="Path to tidy CSV (from extract_clean_csv.py).")
    p.add_argument("--config", required=True, help="Path to meta/config.yml (contains heat_times).")
    p.add_argument("--out_dir", default="data/processed", help="Directory to write output CSVs.")
    p.add_argument(
        "--plot_dir",
        default=None,
        help="Directory to write per-well diagnostic plots. If omitted, plots are not generated.",
    )
    p.add_argument(
        "--plot_mode",
        default="all",
        choices=["all", "ok", "excluded"],
        help="Which wells to plot when --plot_dir is provided.",
    )

    # -------------------------
    # fitting controls (basic)
    # -------------------------
    p.add_argument("--min_points", type=int, default=6)
    # IMPORTANT: broaden default so "13+ points looks linear" can be considered if within max_t_end
    p.add_argument("--max_points", type=int, default=30)
    p.add_argument(
        "--min_span_s",
        type=float,
        default=0.0,
        help="Minimum time-span (seconds) for candidate windows. Use to reduce arbitrariness.",
    )
    p.add_argument(
        "--select_method",
        default="initial_positive",
        choices=["initial_positive", "best_r2"],
    )
    p.add_argument("--r2_min", type=float, default=0.96)
    p.add_argument("--slope_min", type=float, default=0.0, help="Exclude windows with slope < slope_min")
    p.add_argument(
        "--max_t_end",
        type=float,
        default=240.0,
        help="Only consider windows with t_end <= this (s). Set negative to disable.",
    )

    # -------------------------
    # fitting controls (robust knobs)
    # NOTE:
    #   - mono_eps / min_delta_y: if omitted -> auto per well (recommended)
    #   - to disable min_delta_y filter, explicitly set --min_delta_y 0
    # -------------------------
    p.add_argument(
        "--mono_eps",
        type=float,
        default=None,
        help="Monotonicity epsilon. If omitted, auto-estimated per well.",
    )
    p.add_argument(
        "--min_delta_y",
        type=float,
        default=None,
        help="Minimum required rise (dy) within a candidate window. If omitted, auto-estimated per well. Set 0 to disable.",
    )

    p.add_argument(
        "--find_start",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to detect and skip initial flat/noisy points (1=yes, 0=no).",
    )
    p.add_argument("--start_max_shift", type=int, default=5)
    p.add_argument("--start_window", type=int, default=3)
    p.add_argument("--start_allow_down_steps", type=int, default=1)

    p.add_argument(
        "--mono_min_frac",
        type=float,
        default=0.85,
        help="Minimum fraction of steps that are non-decreasing within a window (after epsilon).",
    )
    p.add_argument(
        "--mono_max_down_steps",
        type=int,
        default=1,
        help="Maximum allowed number of significant down steps within a window.",
    )
    p.add_argument(
        "--min_pos_steps",
        type=int,
        default=2,
        help="Minimum number of significant positive steps within a window.",
    )
    p.add_argument(
        "--min_snr",
        type=float,
        default=3.0,
        help="Minimum SNR-like score |dy|/rmse within a window.",
    )
    p.add_argument(
        "--slope_drop_frac",
        type=float,
        default=0.18,
        help="Curvature guard: keep windows whose slope is within (1 - slope_drop_frac) of max slope.",
    )
        # -------------------------
    # optional: force "whole window" when the curve is sufficiently linear
    # -------------------------
    p.add_argument(
        "--force_whole",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, prefer the longest window (starting at detected start) when it is sufficiently linear.",
    )
    p.add_argument("--force_whole_n_min", type=int, default=10)
    p.add_argument("--force_whole_r2_min", type=float, default=0.985)
    p.add_argument("--force_whole_mono_min_frac", type=float, default=0.70)

    # optional: mixing skip + robust monotonicity + robust fit (Amplex Red / resorufin)
    p.add_argument(
        "--min_t_start_s",
        type=float,
        default=0.0,
        help="Ignore candidate windows starting before this time (s). E.g. 60 to skip mixing region.",
    )
    p.add_argument(
        "--down_step_min_frac",
        type=float,
        default=None,
        metavar="F",
        help="Only count down steps larger than max(mono_eps, F * signal_range). E.g. 0.02 = small dips ignored.",
    )
    p.add_argument(
        "--fit_method",
        default="ols",
        choices=["ols", "theil_sen"],
        help="Slope estimation: ols (default) or theil_sen (robust to outliers).",
    )

    # naming
    p.add_argument(
        "--run_id",
        default=None,
        help="Run ID for output filenames. If omitted, derived from --tidy filename.",
    )
    # BO learning data (BMA ternary): built when FoG summaries exist and BO catalog is present
    p.add_argument(
        "--bo_catalog",
        type=Path,
        default=None,
        help="Path to BO catalog (polymer_id, frac_MPC, frac_BMA, frac_MTAC). If omitted, meta/bo_catalog_bma.csv is used when present; then BO learning CSV is written under out_dir.",
    )

    args = p.parse_args()

    tidy_path = Path(args.tidy)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    if not tidy_path.exists():
        raise FileNotFoundError(f"--tidy not found: {tidy_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    run_id = args.run_id if args.run_id else _derive_run_id_from_tidy_path(tidy_path)

    # 出力は run_id/fit/ に集約（実行段階ごとに見やすく）
    run_fit_dir = Path(out_dir) / run_id / "fit"
    run_fit_dir.mkdir(parents=True, exist_ok=True)
    if plot_dir is not None:
        plot_dir = run_fit_dir / "plots"
    qc_report_dir = run_fit_dir / "qc"

    # load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "heat_times" not in cfg:
        raise ValueError(f"config must contain 'heat_times': {config_path}")

    heat_times = cfg["heat_times"]
    if not isinstance(heat_times, list) or len(heat_times) == 0:
        raise ValueError(f"heat_times must be a non-empty list in config: {config_path}")

    # interpret max_t_end
    max_t_end = None if (args.max_t_end is not None and float(args.max_t_end) < 0) else float(args.max_t_end)

    # interpret robust knobs
    mono_eps = None if args.mono_eps is None else float(args.mono_eps)
    min_delta_y = None if args.min_delta_y is None else float(args.min_delta_y)

    find_start = bool(int(args.find_start))

    # read tidy
    tidy = pd.read_csv(tidy_path)

    # compute
    selected, rea = compute_rates_and_rea(
        tidy=tidy,
        heat_times=[float(x) for x in heat_times],
        min_points=int(args.min_points),
        max_points=int(args.max_points),
        min_span_s=float(args.min_span_s),
        select_method=str(args.select_method),
        r2_min=float(args.r2_min),
        slope_min=float(args.slope_min),
        max_t_end=max_t_end,
        # robust knobs
        mono_eps=mono_eps,
        min_delta_y=min_delta_y,
        find_start=find_start,
        start_max_shift=int(args.start_max_shift),
        start_window=int(args.start_window),
        start_allow_down_steps=int(args.start_allow_down_steps),
        mono_min_frac=float(args.mono_min_frac),
        mono_max_down_steps=int(args.mono_max_down_steps),
        min_pos_steps=int(args.min_pos_steps),
        min_snr=float(args.min_snr),
        slope_drop_frac=float(args.slope_drop_frac),
        # plotting
        plot_dir=plot_dir,
        plot_mode=str(args.plot_mode),
        qc_report_dir=qc_report_dir,
        qc_prefix="fit_qc",
        force_whole=bool(int(args.force_whole)),
        force_whole_n_min=int(args.force_whole_n_min),
        force_whole_r2_min=float(args.force_whole_r2_min),
        force_whole_mono_min_frac=float(args.force_whole_mono_min_frac),
        min_t_start_s=float(args.min_t_start_s),
        down_step_min_frac=float(args.down_step_min_frac) if args.down_step_min_frac is not None else None,
        fit_method=str(args.fit_method),
    )

    out_rates = run_fit_dir / "rates_selected.csv"
    out_rea = run_fit_dir / "rates_with_rea.csv"

    # Add run_id for traceability (core-rules: provenance)
    selected["run_id"] = run_id
    rea["run_id"] = run_id

    selected.to_csv(out_rates, index=False)
    rea.to_csv(out_rea, index=False)

    print(f"Saved: {out_rates}")
    print(f"Saved: {out_rea}")

    # 集計: fit/ に簡易テーブル、fit/bo/ に BO 用 JSON（後工程で利用）
    summary_simple_path = run_fit_dir / "summary_simple.csv"
    summary_stats_path = run_fit_dir / "summary_stats.csv"
    extra_outputs = [
        f"per_polymer__{run_id}/",
        f"per_polymer_with_error__{run_id}/",
        f"t50/t50__{run_id}.csv",
    ]
    try:
        bo_json_path = aggregate_and_write(
            rea,
            run_id,
            out_dir,
            input_paths_for_manifest=[tidy_path, out_rea],
            git_root=REPO_ROOT,
            bo_dir=run_fit_dir / "bo",
            summary_simple_path=summary_simple_path,
            summary_stats_path=summary_stats_path,
            extra_output_files=extra_outputs,
        )
        print(f"Saved (table): {summary_simple_path}")
        print(f"Saved (stats): {summary_stats_path}")
        print(f"Saved (BO): {bo_json_path}")
    except Exception as e:
        print(f"Warning: BO summary failed ({e}), continuing.")

    # Per-polymer time-series plots + t50 table (derived from summary_simple.csv)
    # t50/FoG creation is critical: user needs t50 to decide round assignment, so failure should stop execution.
    # This is outside try/except so that failures raise errors and stop the script.
    # Try to find row_map TSV for this run_id (for all_polymers_pair setting)
    meta_dir = REPO_ROOT / "data" / "meta"
    row_map_path_for_plot = None
    if meta_dir.is_dir():
        candidate1 = meta_dir / f"{run_id}.tsv"
        candidate2 = meta_dir / f"{run_id}_row_map.tsv"
        if candidate1.is_file():
            row_map_path_for_plot = candidate1
        elif candidate2.is_file():
            row_map_path_for_plot = candidate2
    t50_csv = plot_per_polymer_timeseries(
        summary_simple_path=summary_simple_path,
        run_id=run_id,
        out_fit_dir=run_fit_dir,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        row_map_path=row_map_path_for_plot,
    )
    print(f"Saved (t50): {t50_csv}")
    print(f"Saved (per polymer): {run_fit_dir / f'per_polymer__{run_id}'}")
    err_dir = plot_per_polymer_timeseries_with_error_band(
        summary_stats_path=summary_stats_path,
        run_id=run_id,
        out_fit_dir=run_fit_dir,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
    )
    if err_dir is not None:
        print(f"Saved (per polymer with error): {err_dir}")
    
    # Plot per-polymer with error bands across runs on the same measurement date
    across_runs_dir = plot_per_polymer_timeseries_across_runs_with_error_band(
        run_id=run_id,
        processed_dir=out_dir,
        out_fit_dir=run_fit_dir,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
    )
    if across_runs_dir is not None:
        print(f"Saved (per polymer across runs with error): {across_runs_dir}")
    # FoG summary (t50_polymer / t50_bare_GOx, same run only) for BO
    if t50_csv is None or not t50_csv.is_file():
        raise FileNotFoundError(f"t50 CSV was not created: {t50_csv}. Cannot create FoG summary.")
    # Try to find row_map TSV for this run_id (for use_for_bo flag)
    meta_dir = REPO_ROOT / "data" / "meta"
    row_map_path = row_map_path_for_plot  # Reuse the path found above
    fog_df = build_fog_summary(
        t50_csv,
        run_id,
        manifest_path=run_fit_dir / "bo" / "bo_output.json",
        row_map_path=row_map_path,
    )
    fog_path = run_fit_dir / f"fog_summary__{run_id}.csv"
    write_fog_summary_csv(fog_df, fog_path)
    print(f"Saved (FoG): {fog_path}")

    # BO learning data: when BO catalog exists, join with all FoG summaries under out_dir
    if args.bo_catalog is not None:
        bo_catalog_path = Path(args.bo_catalog)
    else:
        meta_dir = REPO_ROOT / "meta"
        bo_catalog_path = meta_dir / "bo_catalog_bma.csv"
        if not bo_catalog_path.is_file():
            bo_catalog_path = meta_dir / "bo_catalog_bma.tsv"
    if bo_catalog_path.is_file():
        try:
            catalog_df = load_bo_catalog(bo_catalog_path, validate_sum=True)
            fog_paths = collect_fog_summary_paths(Path(out_dir), run_ids=None)
            if fog_paths:
                learning_df, excluded_df = build_bo_learning_data(catalog_df, fog_paths)
                bo_learning_dir = Path(out_dir) / "bo_learning"
                bo_learning_dir.mkdir(parents=True, exist_ok=True)
                bo_learning_path = bo_learning_dir / "bo_learning.csv"
                bo_excluded_path = bo_learning_dir / "bo_learning_excluded.csv"
                write_bo_learning_csv(learning_df, bo_learning_path)
                write_exclusion_report(excluded_df, bo_excluded_path)
                print(f"Saved (BO learning): {bo_learning_path} ({len(learning_df)} rows)")
                print(f"Saved (BO excluded): {bo_excluded_path} ({len(excluded_df)} rows)")
            else:
                print("BO catalog present but no fog_summary CSV found under out_dir; skipping BO learning CSV.")
        except Exception as e:
            print(f"Warning: BO learning data build failed ({e}), continuing.")

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saved plots under: {plot_dir}")
        grid_paths = write_plate_grid(plot_dir, run_id)
        for grid_path in grid_paths:
            if grid_path.exists():
                print(f"Saved plate grid: {grid_path}")


if __name__ == "__main__":
    main()
