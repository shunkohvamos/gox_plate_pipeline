from __future__ import annotations

import argparse
import shutil
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

from gox_plate_pipeline.fitting import compute_rates_and_rea  # noqa: E402
from gox_plate_pipeline.bo_data import (  # noqa: E402
    build_bo_learning_data,
    collect_fog_summary_paths,
    load_bo_catalog,
    write_bo_learning_csv,
    write_exclusion_report,
)
from gox_plate_pipeline.fog import build_fog_summary, write_fog_summary_csv  # noqa: E402
from gox_plate_pipeline.fog import write_run_ranking_outputs  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    plot_per_polymer_timeseries,
    plot_per_polymer_timeseries_with_error_band,
)
from gox_plate_pipeline.summary import aggregate_and_write  # noqa: E402
from gox_plate_pipeline.meta_paths import get_meta_paths  # noqa: E402

META = get_meta_paths(REPO_ROOT)


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


def _remove_stale_well_plot_dirs(run_plot_root: Path) -> list[Path]:
    """
    Remove stale per-well plot subdirectories (e.g. fit/plots/plate1/*) so default
    fit outputs only contain plate-grid PNGs.
    """
    removed: list[Path] = []
    if not run_plot_root.is_dir():
        return removed
    for child in sorted(run_plot_root.iterdir()):
        if not child.is_dir():
            continue
        shutil.rmtree(child)
        removed.append(child)
    return removed


def main() -> None:
    p = argparse.ArgumentParser(description="Fit per-well initial rates and compute REA.")
    p.add_argument("--tidy", required=True, help="Path to tidy CSV (from extract_clean_csv.py).")
    p.add_argument("--config", required=True, help="Path to assay config YAML (e.g. meta/config.yml, contains heat_times).")
    p.add_argument("--out_dir", default="data/processed", help="Directory to write output CSVs.")
    p.add_argument(
        "--plot_dir",
        default=None,
        help="Directory to write per-well diagnostic plots when --write_well_plots=1.",
    )
    p.add_argument(
        "--write_well_plots",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to write per-well diagnostic plots. 1=on, 0=off (default).",
    )
    p.add_argument(
        "--plot_mode",
        default="all",
        choices=["all", "ok", "excluded"],
        help="Which wells to plot when --write_well_plots=1.",
    )
    p.add_argument(
        "--write_plate_grid",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to write plate-grid PNG(s) (A1-H7 style). 1=on (default), 0=off.",
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
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition for REA curves: y0_half (default) or rea50.",
    )
    p.add_argument(
        "--native_activity_min_rel",
        type=float,
        default=0.70,
        help=(
            "Native-activity feasibility threshold for constrained FoG objective. "
            "Defined as abs_activity_at_0 / same-run GOx abs_activity_at_0 (default: 0.70)."
        ),
    )
    p.add_argument(
        "--reference_polymer_id",
        type=str,
        default="GOX",
        help=(
            "Reference polymer ID used for FoG denominator and functional objective "
            "(default: GOX; e.g., set BETAGAL for beta-gal runs)."
        ),
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
    # BO learning data (BMA ternary): built when FoG summaries exist and BO catalog is present
    p.add_argument(
        "--bo_catalog",
        type=Path,
        default=None,
        help="Path to BO catalog (polymer_id, frac_MPC, frac_BMA, frac_MTAC). If omitted, meta/bo/catalog_bma.csv is used when present; then BO learning CSV is written under out_dir.",
    )

    args = p.parse_args()

    tidy_path = Path(args.tidy)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    requested_plot_dir = Path(args.plot_dir) if args.plot_dir else None

    if not tidy_path.exists():
        raise FileNotFoundError(f"--tidy not found: {tidy_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    run_id = args.run_id if args.run_id else _derive_run_id_from_tidy_path(tidy_path)
    reference_polymer_id = str(args.reference_polymer_id).strip() or "GOX"
    print(f"t50 definition: {args.t50_definition}")
    print(
        f"native activity threshold (abs0/{reference_polymer_id}0): "
        f"{float(args.native_activity_min_rel):.3f}"
    )

    # 出力は run_id/fit/ に集約（実行段階ごとに見やすく）
    run_fit_dir = Path(out_dir) / run_id / "fit"
    run_fit_dir.mkdir(parents=True, exist_ok=True)
    write_well_plots = bool(int(args.write_well_plots))
    if write_well_plots:
        plot_dir = requested_plot_dir if requested_plot_dir is not None else run_fit_dir / "plots"
    else:
        plot_dir = None
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
    if "polymer_id" in tidy.columns:
        n_before = int(len(tidy))
        tidy = tidy[tidy["polymer_id"].astype(str).str.strip().ne("")].copy()
        n_dropped = n_before - int(len(tidy))
        if n_dropped > 0:
            print(f"Skipped rows with empty polymer_id: {n_dropped}")

    write_plate_grid = bool(int(args.write_plate_grid))
    plate_grid_dir = run_fit_dir / "plots" if write_plate_grid else None

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
        plate_grid_dir=plate_grid_dir,
        plate_grid_run_id=run_id if write_plate_grid else None,
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
    summary_stats_all_path = run_fit_dir / "summary_stats_all.csv"
    summary_outlier_events_path = run_fit_dir / "summary_outlier_events.csv"
    extra_outputs = [
        f"per_polymer__{run_id}/",
        f"per_polymer_v2__{run_id}/",
        f"t50/per_polymer_refnorm__{run_id}/",
        f"t50/rea_comparison_fog_panel__{run_id}/",
        f"t50/rea_comparison_fog_grid__{run_id}.png",
        f"per_polymer_with_error__{run_id}/",
        f"all_polymers_with_error__{run_id}.png",
        f"representative_4__{run_id}.png",
        f"representative_objective_loglinear_main__{run_id}.png",
        f"per_polymer_with_error_all__{run_id}/",
        f"all_polymers_with_error_all__{run_id}.png",
        "summary_stats_all.csv",
        f"t50/csv/t50__{run_id}.csv",
        "summary_outlier_events.csv",
        f"ranking/csv/t50_ranking__{run_id}.csv",
        f"ranking/csv/fog_ranking__{run_id}.csv",
        f"ranking/csv/objective_activity_bonus_penalty_ranking__{run_id}.csv",
        f"ranking/csv/objective_loglinear_main_ranking__{run_id}.csv",
        f"ranking/csv/objective_activity_bonus_penalty_profile_ranks__{run_id}.csv",
        f"ranking/csv/functional_ranking__{run_id}.csv",
        f"ranking/figure_guide__{run_id}.md",
        f"ranking/t50_ranking__{run_id}.png",
        f"ranking/fog_ranking__{run_id}.png",
        f"ranking/objective_activity_bonus_penalty_ranking__{run_id}.png",
        f"ranking/objective_loglinear_main_ranking__{run_id}.png",
        f"ranking/objective_activity_bonus_penalty_tradeoff__{run_id}.png",
        f"ranking/objective_activity_bonus_penalty_proxy_curves__{run_id}.png",
        f"ranking/new/objective_activity_bonus_penalty_proxy_curves_grid__{run_id}.png",
        f"ranking/new/objective_activity_bonus_penalty_profile_tradeoff_grid__{run_id}.png",
        f"ranking/new/objective_activity_bonus_penalty_profile_rank_heatmap__{run_id}.png",
        f"ranking/mainA_native0_vs_fog__{run_id}.png",
        f"ranking/mainA_abs0_vs_fog_solvent__{run_id}.png",
        f"ranking/mainE_u0_vs_fog_loglog_regression__{run_id}.png",
        f"ranking/mainF_u0_vs_t50_loglog_regression__{run_id}.png",
        f"ranking/old/csv/fog_ranking__{run_id}.csv",
        f"ranking/old/csv/fog_native_constrained_ranking__{run_id}.csv",
        f"ranking/old/csv/objective_native_soft_ranking__{run_id}.csv",
        f"ranking/old/fog_ranking__{run_id}.png",
        f"ranking/old/fog_native_constrained_ranking__{run_id}.png",
        f"ranking/old/objective_native_soft_ranking__{run_id}.png",
        f"ranking/old/fog_native_constrained_decision__{run_id}.png",
        f"ranking/old/fog_native_constrained_tradeoff__{run_id}.png",
        f"ranking/old/objective_native_soft_tradeoff__{run_id}.png",
        f"ranking/old/mainB_feasible_fog_ranking__{run_id}.png",
        f"ranking/old/supp_theta_sensitivity__{run_id}.png",
        f"ranking/old/csv/supp_theta_sensitivity__{run_id}.csv",
        f"ranking/csv/primary_objective_table__{run_id}.csv",
        f"ranking/functional_ranking__{run_id}.png",
    ]
    summary_write_error: Exception | None = None
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
        print(f"Saved (BO): {bo_json_path}")
    except Exception as e:
        summary_write_error = e
        print(f"Warning: BO summary failed ({e}), continuing.")

    if not summary_simple_path.is_file() or not summary_stats_path.is_file():
        if summary_write_error is not None:
            raise RuntimeError(
                f"Failed to create summary outputs for run_id={run_id}. "
                "Check aggregate_and_write error above."
            ) from summary_write_error
        missing = []
        if not summary_simple_path.is_file():
            missing.append(str(summary_simple_path))
        if not summary_stats_path.is_file():
            missing.append(str(summary_stats_path))
        raise FileNotFoundError(f"Required summary output is missing: {', '.join(missing)}")

    # Per-polymer time-series plots + t50 table (derived from summary_simple.csv)
    # t50/FoG creation is critical: user needs t50 to decide round assignment, so failure should stop execution.
    # This is outside try/except so that failures raise errors and stop the script.
    # Try to find row_map TSV for this run_id (for all_polymers_pair setting)
    meta_dir = META.row_maps_dir
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
        color_map_path=META.polymer_colors,
        row_map_path=row_map_path_for_plot,
        t50_definition=args.t50_definition,
        reference_polymer_id=reference_polymer_id,
        native_activity_min_rel=float(args.native_activity_min_rel),
        processed_dir=Path(out_dir),
        run_round_map_path=META.run_round_map if META.run_round_map.is_file() else None,
    )
    print(f"Saved (t50): {t50_csv}")
    print(f"Saved (per polymer): {run_fit_dir / f'per_polymer__{run_id}'}")
    print(f"Saved (per polymer v2): {run_fit_dir / 't50' / f'per_polymer_v2__{run_id}'}")
    print(f"Saved (per polymer ref-normalized): {run_fit_dir / 't50' / f'per_polymer_refnorm__{run_id}'}")
    print(f"Saved (REA+FoG panels): {run_fit_dir / 't50' / f'rea_comparison_fog_panel__{run_id}'}")
    print(f"Saved (REA+FoG panel grid): {run_fit_dir / 't50' / f'rea_comparison_fog_grid__{run_id}.png'}")
    rep_t50_png = run_fit_dir / f"representative_4__{run_id}.png"
    rep_obj_png = run_fit_dir / f"representative_objective_loglinear_main__{run_id}.png"
    if rep_t50_png.is_file():
        print(f"Saved (representative, t50): {rep_t50_png}")
    if rep_obj_png.is_file():
        print(f"Saved (representative, objective): {rep_obj_png}")
    err_dir = plot_per_polymer_timeseries_with_error_band(
        summary_stats_path=summary_stats_path,
        run_id=run_id,
        out_fit_dir=run_fit_dir,
        color_map_path=META.polymer_colors,
        reference_polymer_id=reference_polymer_id,
        native_activity_min_rel=float(args.native_activity_min_rel),
        t50_definition=args.t50_definition,
        error_band_suffix="",
    )
    if err_dir is not None:
        print(f"Saved (per polymer with error, robust): {err_dir}")
    all_with_error_robust = run_fit_dir / f"all_polymers_with_error__{run_id}.png"
    if all_with_error_robust.is_file():
        print(f"Saved (all polymers with error, robust): {all_with_error_robust}")
    if summary_stats_all_path.is_file():
        err_dir_all = plot_per_polymer_timeseries_with_error_band(
            summary_stats_path=summary_stats_all_path,
            run_id=run_id,
            out_fit_dir=run_fit_dir,
            color_map_path=META.polymer_colors,
            reference_polymer_id=reference_polymer_id,
            native_activity_min_rel=float(args.native_activity_min_rel),
            t50_definition=args.t50_definition,
            error_band_suffix="_all",
        )
        if err_dir_all is not None:
            print(f"Saved (per polymer with error, all data): {err_dir_all}")
        all_with_error_all = run_fit_dir / f"all_polymers_with_error_all__{run_id}.png"
        if all_with_error_all.is_file():
            print(f"Saved (all polymers with error, all data): {all_with_error_all}")

    # FoG summary (t50_polymer / t50_bare_GOx, same run only) for BO
    if t50_csv is None or not t50_csv.is_file():
        raise FileNotFoundError(f"t50 CSV was not created: {t50_csv}. Cannot create FoG summary.")
    # Try to find row_map TSV for this run_id (for use_for_bo flag)
    meta_dir = META.row_maps_dir
    row_map_path = row_map_path_for_plot  # Reuse the path found above
    fog_df = build_fog_summary(
        t50_csv,
        run_id,
        manifest_path=run_fit_dir / "bo" / "bo_output.json",
        row_map_path=row_map_path,
        polymer_solvent_path=(META.polymer_stock_solvent if META.polymer_stock_solvent.is_file() else None),
        native_activity_min_rel=float(args.native_activity_min_rel),
        reference_polymer_id=reference_polymer_id,
    )
    fog_path = run_fit_dir / f"fog_summary__{run_id}.csv"
    write_fog_summary_csv(fog_df, fog_path)
    print(f"Saved (FoG): {fog_path}")
    ranking_outputs = write_run_ranking_outputs(
        fog_df=fog_df,
        run_id=run_id,
        out_dir=run_fit_dir / "ranking",
        color_map_path=META.polymer_colors,
        polymer_solvent_path=(META.polymer_stock_solvent if META.polymer_stock_solvent.is_file() else None),
        reference_polymer_id=reference_polymer_id,
    )
    if "t50_ranking_csv" in ranking_outputs:
        print(f"Saved (t50 ranking): {ranking_outputs['t50_ranking_csv']}")
    if "fog_ranking_csv" in ranking_outputs:
        print(f"Saved (FoG ranking): {ranking_outputs['fog_ranking_csv']}")
    if "fog_native_constrained_ranking_csv" in ranking_outputs:
        print(
            "Saved (FoG native-constrained ranking): "
            f"{ranking_outputs['fog_native_constrained_ranking_csv']}"
        )
    if "objective_native_soft_ranking_csv" in ranking_outputs:
        print(
            "Saved (legacy soft-objective ranking): "
            f"{ranking_outputs['objective_native_soft_ranking_csv']}"
        )
    if "objective_activity_bonus_penalty_ranking_csv" in ranking_outputs:
        print(
            "Saved (FoG-activity bonus/penalty objective ranking): "
            f"{ranking_outputs['objective_activity_bonus_penalty_ranking_csv']}"
        )
    if "objective_loglinear_main_ranking_csv" in ranking_outputs:
        print(
            "Saved (primary objective ranking): "
            f"{ranking_outputs['objective_loglinear_main_ranking_csv']}"
        )
    if "objective_activity_bonus_penalty_profile_ranks_csv" in ranking_outputs:
        print(
            "Saved (objective profile rank sensitivity table): "
            f"{ranking_outputs['objective_activity_bonus_penalty_profile_ranks_csv']}"
        )
    if "t50_ranking_png" in ranking_outputs:
        print(f"Saved (t50 ranking plot): {ranking_outputs['t50_ranking_png']}")
    if "fog_ranking_png" in ranking_outputs:
        print(f"Saved (FoG ranking plot): {ranking_outputs['fog_ranking_png']}")
    if "fog_native_constrained_ranking_png" in ranking_outputs:
        print(
            "Saved (FoG native-constrained ranking plot): "
            f"{ranking_outputs['fog_native_constrained_ranking_png']}"
        )
    if "objective_native_soft_ranking_png" in ranking_outputs:
        print(
            "Saved (legacy soft-objective ranking plot): "
            f"{ranking_outputs['objective_native_soft_ranking_png']}"
        )
    if "objective_activity_bonus_penalty_ranking_png" in ranking_outputs:
        print(
            "Saved (FoG-activity bonus/penalty objective ranking plot): "
            f"{ranking_outputs['objective_activity_bonus_penalty_ranking_png']}"
        )
    if "objective_loglinear_main_ranking_png" in ranking_outputs:
        print(
            "Saved (primary objective ranking plot): "
            f"{ranking_outputs['objective_loglinear_main_ranking_png']}"
        )
    if "fog_native_constrained_decision_png" in ranking_outputs:
        print(
            "Saved (FoG native-constrained decision plot): "
            f"{ranking_outputs['fog_native_constrained_decision_png']}"
        )
    if "objective_native_soft_tradeoff_png" in ranking_outputs:
        print(
            "Saved (legacy soft-objective tradeoff map): "
            f"{ranking_outputs['objective_native_soft_tradeoff_png']}"
        )
    if "objective_activity_bonus_penalty_tradeoff_png" in ranking_outputs:
        print(
            "Saved (FoG-activity bonus/penalty objective tradeoff map): "
            f"{ranking_outputs['objective_activity_bonus_penalty_tradeoff_png']}"
        )
    if "objective_activity_bonus_penalty_proxy_curves_png" in ranking_outputs:
        print(
            "Saved (FoG-activity bonus/penalty proxy curves): "
            f"{ranking_outputs['objective_activity_bonus_penalty_proxy_curves_png']}"
        )
    if "objective_activity_bonus_penalty_proxy_curves_grid_png" in ranking_outputs:
        print(
            "Saved (FoG-activity bonus/penalty proxy curves, all-polymer grid): "
            f"{ranking_outputs['objective_activity_bonus_penalty_proxy_curves_grid_png']}"
        )
    if "objective_activity_bonus_penalty_profile_tradeoff_grid_png" in ranking_outputs:
        print(
            "Saved (objective profile sensitivity tradeoff grid): "
            f"{ranking_outputs['objective_activity_bonus_penalty_profile_tradeoff_grid_png']}"
        )
    if "objective_activity_bonus_penalty_profile_rank_heatmap_png" in ranking_outputs:
        print(
            "Saved (objective profile rank heatmap): "
            f"{ranking_outputs['objective_activity_bonus_penalty_profile_rank_heatmap_png']}"
        )
    if "mainA_native0_vs_fog_png" in ranking_outputs:
        print(f"Saved (MainA native-vs-FoG): {ranking_outputs['mainA_native0_vs_fog_png']}")
    if "mainA_abs0_vs_fog_solvent_png" in ranking_outputs:
        print(
            "Saved (MainA solvent abs-vs-FoG): "
            f"{ranking_outputs['mainA_abs0_vs_fog_solvent_png']}"
        )
    if "mainE_u0_vs_fog_loglog_regression_png" in ranking_outputs:
        print(
            "Saved (MainE U0*-FoG* log-log regression): "
            f"{ranking_outputs['mainE_u0_vs_fog_loglog_regression_png']}"
        )
    if "mainF_u0_vs_t50_loglog_regression_png" in ranking_outputs:
        print(
            "Saved (MainF U0*-t50 log-log regression): "
            f"{ranking_outputs['mainF_u0_vs_t50_loglog_regression_png']}"
        )
    if "mainB_feasible_fog_ranking_png" in ranking_outputs:
        print(f"Saved (MainB feasible ranking): {ranking_outputs['mainB_feasible_fog_ranking_png']}")
    if "supp_theta_sensitivity_png" in ranking_outputs:
        print(f"Saved (theta sensitivity): {ranking_outputs['supp_theta_sensitivity_png']}")
    if "primary_objective_table_csv" in ranking_outputs:
        print(f"Saved (primary objective table): {ranking_outputs['primary_objective_table_csv']}")
    if "functional_ranking_csv" in ranking_outputs:
        print(f"Saved (functional ranking): {ranking_outputs['functional_ranking_csv']}")
    if "functional_ranking_png" in ranking_outputs:
        print(f"Saved (functional ranking plot): {ranking_outputs['functional_ranking_png']}")
    if "figure_guide_md" in ranking_outputs:
        print(f"Saved (ranking figure guide): {ranking_outputs['figure_guide_md']}")

    # BO learning data: when BO catalog exists, join with all FoG summaries under out_dir
    if args.bo_catalog is not None:
        bo_catalog_path = Path(args.bo_catalog)
    else:
        bo_catalog_path = META.bo_catalog_bma
        if not bo_catalog_path.is_file():
            bo_catalog_path = META.bo_catalog_bma.parent / "catalog_bma.tsv"
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
        print(f"Saved per-well plots under: {plot_dir}")
    else:
        removed_dirs = _remove_stale_well_plot_dirs(run_fit_dir / "plots")
        print("Per-well plots: disabled (default). Use 'Well plots only' to generate them on demand.")
        if removed_dirs:
            print(f"Removed stale per-well plot directories: {len(removed_dirs)}")
            for d in removed_dirs:
                print(f"Removed stale per-well plot dir: {d}")
    if write_plate_grid and plate_grid_dir is not None:
        grid_paths = sorted(plate_grid_dir.glob(f"plate_grid__{run_id}__*.png"))
        legacy = plate_grid_dir / f"plate_grid__{run_id}.png"
        if legacy.exists():
            grid_paths.append(legacy)
        if grid_paths:
            print(f"Saved plate grid under: {plate_grid_dir}")
            for grid_path in sorted(set(grid_paths)):
                print(f"Saved plate grid: {grid_path}")


if __name__ == "__main__":
    main()
