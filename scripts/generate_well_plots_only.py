#!/usr/bin/env python3
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


def _derive_run_id_from_tidy_path(tidy_path: Path) -> str:
    """run_id: .../run_id/extract/tidy.csv -> parent of extract; .../run_id/tidy.csv -> parent."""
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
    p = argparse.ArgumentParser(
        description="Generate per-well fit diagnostic plots only (no rates/REA/t50/FoG CSV outputs).",
    )
    p.add_argument("--tidy", required=True, help="Path to tidy CSV (from extract_clean_csv.py).")
    p.add_argument("--config", required=True, help="Path to meta/config.yml (contains heat_times).")
    p.add_argument("--out_dir", default="data/processed", help="Processed root directory.")
    p.add_argument(
        "--plot_dir",
        default=None,
        help="Directory for per-well plots. Default: data/processed/{run_id}/fit/plots",
    )
    p.add_argument(
        "--plot_mode",
        default="all",
        choices=["all", "ok", "excluded"],
        help="Which wells to plot.",
    )
    p.add_argument(
        "--run_id",
        default=None,
        help="Run ID for plot paths. If omitted, derived from --tidy filename.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print additional counters for troubleshooting.",
    )

    # fitting controls (same defaults as fit_initial_rates.py)
    p.add_argument("--min_points", type=int, default=6)
    p.add_argument("--max_points", type=int, default=30)
    p.add_argument("--min_span_s", type=float, default=0.0)
    p.add_argument("--select_method", default="initial_positive", choices=["initial_positive", "best_r2"])
    p.add_argument("--r2_min", type=float, default=0.96)
    p.add_argument("--slope_min", type=float, default=0.0)
    p.add_argument(
        "--max_t_end",
        type=float,
        default=240.0,
        help="Only consider windows with t_end <= this (s). Set negative to disable.",
    )

    p.add_argument("--mono_eps", type=float, default=None)
    p.add_argument("--min_delta_y", type=float, default=None)
    p.add_argument("--find_start", type=int, default=1, choices=[0, 1])
    p.add_argument("--start_max_shift", type=int, default=5)
    p.add_argument("--start_window", type=int, default=3)
    p.add_argument("--start_allow_down_steps", type=int, default=1)
    p.add_argument("--mono_min_frac", type=float, default=0.85)
    p.add_argument("--mono_max_down_steps", type=int, default=1)
    p.add_argument("--min_pos_steps", type=int, default=2)
    p.add_argument("--min_snr", type=float, default=3.0)
    p.add_argument("--slope_drop_frac", type=float, default=0.18)

    p.add_argument("--force_whole", type=int, default=0, choices=[0, 1])
    p.add_argument("--force_whole_n_min", type=int, default=10)
    p.add_argument("--force_whole_r2_min", type=float, default=0.985)
    p.add_argument("--force_whole_mono_min_frac", type=float, default=0.70)

    p.add_argument("--min_t_start_s", type=float, default=0.0)
    p.add_argument("--down_step_min_frac", type=float, default=None)
    p.add_argument("--fit_method", default="ols", choices=["ols", "theil_sen"])

    args = p.parse_args()

    tidy_path = Path(args.tidy)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)

    if not tidy_path.exists():
        raise FileNotFoundError(f"--tidy not found: {tidy_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    run_id = args.run_id if args.run_id else _derive_run_id_from_tidy_path(tidy_path)
    run_fit_dir = out_dir / run_id / "fit"
    run_plot_dir = Path(args.plot_dir) if args.plot_dir else run_fit_dir / "plots"
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "heat_times" not in cfg:
        raise ValueError(f"config must contain 'heat_times': {config_path}")
    heat_times = cfg["heat_times"]
    if not isinstance(heat_times, list) or len(heat_times) == 0:
        raise ValueError(f"heat_times must be a non-empty list in config: {config_path}")

    max_t_end = None if (args.max_t_end is not None and float(args.max_t_end) < 0) else float(args.max_t_end)
    mono_eps = None if args.mono_eps is None else float(args.mono_eps)
    min_delta_y = None if args.min_delta_y is None else float(args.min_delta_y)
    find_start = bool(int(args.find_start))

    tidy = pd.read_csv(tidy_path)

    selected, _rea = compute_rates_and_rea(
        tidy=tidy,
        heat_times=[float(x) for x in heat_times],
        min_points=int(args.min_points),
        max_points=int(args.max_points),
        min_span_s=float(args.min_span_s),
        select_method=str(args.select_method),
        r2_min=float(args.r2_min),
        slope_min=float(args.slope_min),
        max_t_end=max_t_end,
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
        plot_dir=run_plot_dir,
        plot_mode=str(args.plot_mode),
        qc_report_dir=None,
        qc_prefix="fit_qc",
        force_whole=bool(int(args.force_whole)),
        force_whole_n_min=int(args.force_whole_n_min),
        force_whole_r2_min=float(args.force_whole_r2_min),
        force_whole_mono_min_frac=float(args.force_whole_mono_min_frac),
        min_t_start_s=float(args.min_t_start_s),
        down_step_min_frac=float(args.down_step_min_frac) if args.down_step_min_frac is not None else None,
        fit_method=str(args.fit_method),
    )

    print(f"Saved per-well plots under: {run_plot_dir}")
    if "status" in selected.columns:
        ok_count = int((selected["status"] == "ok").sum())
        excluded_count = int((selected["status"] != "ok").sum())
        print(f"Wells: total={len(selected)}, ok={ok_count}, excluded={excluded_count}")
        if args.debug:
            reason_counts = (
                selected.loc[selected["status"] != "ok", "exclude_reason"]
                .fillna("")
                .astype(str)
                .value_counts()
            )
            if not reason_counts.empty:
                print("Exclude reasons:")
                for reason, count in reason_counts.items():
                    print(f"  - {reason or '(none)'}: {int(count)}")

    grid_paths = write_plate_grid(run_plot_dir, run_id)
    if grid_paths:
        for grid_path in grid_paths:
            if grid_path.exists():
                print(f"Saved plate grid: {grid_path}")
    else:
        print("No per-well plots found for plate-grid assembly.")


if __name__ == "__main__":
    main()
