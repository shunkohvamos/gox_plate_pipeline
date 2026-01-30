# scripts/fit_initial_rates.py
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
        help="Directory to write per-well fit plots. If omitted, plots are not generated.",
    )

    # fitting controls
    p.add_argument("--min_points", type=int, default=6)
    p.add_argument("--max_points", type=int, default=12)
    p.add_argument("--select_method", default="initial_positive", choices=["initial_positive", "best_r2"])
    p.add_argument("--r2_min", type=float, default=0.98)
    p.add_argument("--slope_min", type=float, default=0.0, help="Exclude windows with slope < slope_min")
    p.add_argument(
        "--max_t_end",
        type=float,
        default=240.0,
        help="Only consider windows with t_end <= this (s). Set negative to disable.",
    )

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

    # load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "heat_times" not in cfg:
        raise ValueError(f"config must contain 'heat_times': {config_path}")

    heat_times = cfg["heat_times"]
    if not isinstance(heat_times, list) or len(heat_times) == 0:
        raise ValueError(f"heat_times must be a non-empty list in config: {config_path}")

    # interpret max_t_end
    max_t_end = None if (args.max_t_end is not None and args.max_t_end < 0) else float(args.max_t_end)

    # read tidy
    tidy = pd.read_csv(tidy_path)

    # compute
    selected, rea = compute_rates_and_rea(
        tidy=tidy,
        heat_times=[float(x) for x in heat_times],
        min_points=int(args.min_points),
        max_points=int(args.max_points),
        select_method=str(args.select_method),
        r2_min=float(args.r2_min),
        slope_min=float(args.slope_min),
        max_t_end=max_t_end,
        plot_dir=plot_dir,
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
