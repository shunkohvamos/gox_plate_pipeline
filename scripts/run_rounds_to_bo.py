#!/usr/bin/env python3
"""
One-shot pipeline for BO using all runs that have round assignment.

Steps:
  1) run_fit_then_round_fog.py        (extract + fit for round-assigned runs, round-averaged FoG)
  2) build_fog_plate_aware.py         (plate-aware per-row and round-averaged FoG)
  3) run_bayesian_optimization.py     (rebuild bo_learning from plate-aware round averages and run BO)

Usage:
  python scripts/run_rounds_to_bo.py --t50_definition y0_half|rea50 [--dry_run] [--force_fit]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run(cmd: list[str], *, env: dict, debug: bool) -> None:
    if debug:
        print("Run:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run round-assigned extract+fit and execute Bayesian optimization in one shot.",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=REPO_ROOT / "meta" / "bo_run_round_map.tsv",
        help="Path to run_id→round_id map.",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed root directory.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "meta" / "config.yml",
        help="Path to config.yml for extract/fit.",
    )
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition used in fit and plate-aware FoG.",
    )
    p.add_argument(
        "--force_fit",
        action="store_true",
        help="Force re-fit in step 1.",
    )
    p.add_argument(
        "--bo_run_id",
        type=str,
        default=None,
        help="Optional BO run ID.",
    )
    p.add_argument(
        "--out_bo_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_runs",
        help="BO output root directory.",
    )
    p.add_argument("--n_suggestions", type=int, default=8, help="Number of BO suggestions.")
    p.add_argument(
        "--acquisition",
        type=str,
        default="ei",
        choices=["ei", "ucb"],
        help="Acquisition function for BO.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned commands.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging.",
    )
    args = p.parse_args()

    if not args.run_round_map.is_file():
        raise FileNotFoundError(f"Run-round map not found: {args.run_round_map}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    fog_round_averaged = args.processed_dir / "fog_round_averaged" / "fog_round_averaged.csv"
    plate_aware_dir = args.processed_dir / "fog_plate_aware"
    fog_plate_aware = plate_aware_dir / "fog_plate_aware.csv"
    fog_plate_aware_round = plate_aware_dir / "fog_plate_aware_round_averaged.csv"
    bo_learning = args.processed_dir / "bo_learning" / "bo_learning_plate_aware.csv"
    bo_excluded = args.processed_dir / "bo_learning" / "bo_learning_excluded_plate_aware.csv"

    step1 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_fit_then_round_fog.py"),
        "--run_round_map", args.run_round_map.relative_to(REPO_ROOT).as_posix(),
        "--processed_dir", args.processed_dir.relative_to(REPO_ROOT).as_posix(),
        "--config", args.config.relative_to(REPO_ROOT).as_posix(),
        "--out_fog", fog_round_averaged.relative_to(REPO_ROOT).as_posix(),
        "--t50_definition", str(args.t50_definition),
    ]
    if args.force_fit:
        step1.append("--force_fit")
    if args.debug:
        step1.append("--debug")

    step2 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_fog_plate_aware.py"),
        "--run_round_map", args.run_round_map.relative_to(REPO_ROOT).as_posix(),
        "--processed_dir", args.processed_dir.relative_to(REPO_ROOT).as_posix(),
        "--out_dir", plate_aware_dir.relative_to(REPO_ROOT).as_posix(),
        "--t50_definition", str(args.t50_definition),
    ]

    step3 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bayesian_optimization.py"),
        "--rebuild_learning",
        "--catalog", "meta/bo_catalog_bma.csv",
        "--fog_round_averaged", fog_plate_aware_round.relative_to(REPO_ROOT).as_posix(),
        "--bo_learning", bo_learning.relative_to(REPO_ROOT).as_posix(),
        "--exclusion_report", bo_excluded.relative_to(REPO_ROOT).as_posix(),
        "--fog_plate_aware", fog_plate_aware.relative_to(REPO_ROOT).as_posix(),
        "--out_dir", args.out_bo_dir.relative_to(REPO_ROOT).as_posix(),
        "--n_suggestions", str(int(args.n_suggestions)),
        "--acquisition", str(args.acquisition),
        "--min_component", "0.02",
        "--max_component", "0.95",
        "--min_fraction_distance", "0.06",
        "--objective_column", "log_fog_corrected",
    ]
    if args.bo_run_id is not None and str(args.bo_run_id).strip():
        step3.extend(["--bo_run_id", str(args.bo_run_id).strip()])

    if args.dry_run or args.debug:
        print("Step 1: round-assigned extract+fit")
        print(" ", " ".join(step1))
        print("Step 2: plate-aware FoG")
        print(" ", " ".join(step2))
        print("Step 3: Bayesian optimization")
        print(" ", " ".join(step3))
        print("t50 definition:", args.t50_definition)

    if args.dry_run:
        return

    _run(step1, env=env, debug=args.debug)
    _run(step2, env=env, debug=args.debug)
    _run(step3, env=env, debug=args.debug)
    print("Completed: round-assigned runs -> BO one-shot pipeline.")


if __name__ == "__main__":
    main()
