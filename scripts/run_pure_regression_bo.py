#!/usr/bin/env python3
"""
Run pure regression BO (2D x,y): no classifier, no anchor/replicate slots.

Design: x = BMA/(BMA+MTAC), y = BMA+MTAC. GP on (x,y) with z-score; EI or UCB;
diversity in fraction space; composition bounds (min/max per fraction).
Outputs: candidates CSV and 6 figures (observed scatter, mu/sigma heatmaps,
acquisition+proposals, distance comparison, learning curve).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from gox_plate_pipeline.bo_engine import run_pure_regression_bo  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pure regression BO: propose batch in (x,y), save candidates and 6 figures.",
    )
    p.add_argument(
        "--learning",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_learning" / "bo_learning_plate_aware.csv",
        help="BO learning CSV: frac_MPC, frac_BMA, frac_MTAC (or x, y) and log_fog_corrected or log_fog.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_runs" / "pure",
        help="Output directory for candidates CSV and figures.",
    )
    p.add_argument("--q", type=int, default=5, help="Number of batch proposals.")
    p.add_argument(
        "--acquisition",
        type=str,
        choices=["ei", "ucb"],
        default="ei",
        help="Acquisition: ei or ucb.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_id", type=str, default="", help="Run ID for filenames. Default: timestamp.")
    p.add_argument(
        "--objective_column",
        type=str,
        default="log_fog_corrected",
        help="Column name for objective (maximized). Fallback: log_fog.",
    )
    p.add_argument("--ei_xi", type=float, default=0.01, help="EI exploration parameter.")
    p.add_argument("--ucb_kappa", type=float, default=2.0, help="UCB exploration parameter.")
    p.add_argument(
        "--ucb_beta",
        type=float,
        default=None,
        help="Optional UCB beta (kappa = sqrt(beta)). If set, overrides --ucb_kappa.",
    )
    p.add_argument(
        "--min_fraction_distance",
        type=float,
        default=0.05,
        help="Diversity: min L2 distance in (MPC,BMA,MTAC) space to known/selected points.",
    )
    p.add_argument(
        "--min_component",
        type=float,
        default=0.05,
        help="Lower bound for each fraction (MPC, BMA, MTAC).",
    )
    p.add_argument(
        "--max_component",
        type=float,
        default=0.95,
        help="Upper bound for each fraction.",
    )
    p.add_argument(
        "--n_random_candidates",
        type=int,
        default=5000,
        help="Number of random (x,y) candidates evaluated by acquisition.",
    )
    p.add_argument(
        "--disable_sparse_isotropic",
        action="store_true",
        help="Disable sparse-data isotropic kernel fallback (auto-enabled for <=20 points by default).",
    )
    p.add_argument(
        "--sparse_isotropic_max_unique_points",
        type=int,
        default=20,
        help="Auto-enable isotropic kernel when unique design points <= this value (default: 20).",
    )
    p.add_argument(
        "--min_length_scale_sparse_isotropic",
        type=float,
        default=0.5,
        help="Minimum length scale for sparse isotropic kernel (larger = smoother gradient). Default 0.5.",
    )
    p.add_argument("--no_plots", action="store_true", help="Skip writing figures.")
    p.add_argument(
        "--fog_plate_aware",
        type=Path,
        default=None,
        help="Path to fog_plate_aware.csv for generating ranking bar charts. If not provided, bar charts are skipped.",
    )
    p.add_argument(
        "--polymer_colors",
        type=Path,
        default=REPO_ROOT / "meta" / "polymer_colors.yml",
        help="Path to polymer colors YAML file for bar chart coloring.",
    )
    args = p.parse_args()

    learning_path = Path(args.learning)
    if not learning_path.is_file():
        raise FileNotFoundError(f"Learning CSV not found: {learning_path}")

    learning_df = pd.read_csv(learning_path)
    composition_constraints = {
        "min_mpc": args.min_component,
        "max_mpc": args.max_component,
        "min_bma": args.min_component,
        "max_bma": args.max_component,
        "min_mtac": args.min_component,
        "max_mtac": args.max_component,
    }
    diversity_params = {"min_fraction_distance": args.min_fraction_distance}

    outputs = run_pure_regression_bo(
        learning_df,
        args.out_dir,
        args.q,
        args.acquisition,
        diversity_params=diversity_params,
        composition_constraints=composition_constraints,
        seed=args.seed,
        run_id=args.run_id or None,
        objective_column=args.objective_column,
        ei_xi=args.ei_xi,
        ucb_kappa=args.ucb_kappa,
        ucb_beta=args.ucb_beta,
        n_random_candidates=args.n_random_candidates,
        sparse_force_isotropic=not args.disable_sparse_isotropic,
        sparse_isotropic_max_unique_points=args.sparse_isotropic_max_unique_points,
        min_length_scale_sparse_isotropic=args.min_length_scale_sparse_isotropic,
        write_plots=not args.no_plots,
        learning_input_path=learning_path,
        fog_plate_aware_path=args.fog_plate_aware,
        polymer_colors_path=args.polymer_colors,
    )

    print("Pure regression BO finished. Outputs:")
    for key, path in sorted(outputs.items(), key=lambda x: x[0]):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
