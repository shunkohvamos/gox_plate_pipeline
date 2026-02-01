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

from gox_plate_pipeline.fitting import compute_rates_and_rea  # noqa: E402


def _derive_run_id_from_tidy_path(tidy_path: Path) -> str:
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
    p.add_argument("--r2_min", type=float, default=0.98)
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


    # naming
    p.add_argument(
        "--run_id",
        default=None,
        help="Run ID for output filenames. If omitted, derived from --tidy filename.",
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

    # NEW: keep plots separated per run to avoid mixing old/new styles
    if plot_dir is not None:
        plot_dir = plot_dir / run_id

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
        qc_report_dir=Path(out_dir) / "qc",
        qc_prefix="fit_qc",
        force_whole=bool(int(args.force_whole)),
        force_whole_n_min=int(args.force_whole_n_min),
        force_whole_r2_min=float(args.force_whole_r2_min),
        force_whole_mono_min_frac=float(args.force_whole_mono_min_frac),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    out_rates = out_dir / f"{run_id}__rates_selected.csv"
    out_rea = out_dir / f"{run_id}__rates_with_rea.csv"

    selected.to_csv(out_rates, index=False)
    rea.to_csv(out_rea, index=False)

    print(f"Saved: {out_rates}")
    print(f"Saved: {out_rea}")

    if plot_dir is not None:
        print(f"Saved plots under: {plot_dir}")


if __name__ == "__main__":
    main()
