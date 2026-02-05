#!/usr/bin/env python3
"""
Run pure-regression Bayesian optimization from prepared (or rebuilt) BO learning data.

Default workflow:
1) Use plate-aware round-averaged FoG to build bo_learning_plate_aware.csv (optional rebuild).
2) Fit GP on (x, y) and output BO proposal maps/logs for FoG maximization.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import (  # noqa: E402
    build_bo_learning_data_from_round_averaged,
    load_bo_catalog,
    write_bo_learning_csv,
    write_exclusion_report,
)
from gox_plate_pipeline.bo_engine import run_pure_regression_bo  # noqa: E402


def _build_learning_if_needed(
    *,
    rebuild_learning: bool,
    bo_learning_path: Path,
    exclusion_report_path: Path,
    catalog_path: Path,
    fog_round_averaged_path: Path,
) -> None:
    if not rebuild_learning and bo_learning_path.is_file():
        return
    if not catalog_path.is_file():
        raise FileNotFoundError(f"BO catalog not found: {catalog_path}")
    if not fog_round_averaged_path.is_file():
        raise FileNotFoundError(f"Round-averaged FoG not found: {fog_round_averaged_path}")

    catalog_df = load_bo_catalog(catalog_path, validate_sum=True)
    learning_df, excluded_df = build_bo_learning_data_from_round_averaged(
        catalog_df,
        fog_round_averaged_path,
    )
    bo_learning_path.parent.mkdir(parents=True, exist_ok=True)
    write_bo_learning_csv(learning_df, bo_learning_path)
    write_exclusion_report(excluded_df, exclusion_report_path)
    print(f"Saved BO learning: {bo_learning_path} ({len(learning_df)} rows)")
    print(f"Saved BO excluded: {exclusion_report_path} ({len(excluded_df)} rows)")


def _default_bo_run_id() -> str:
    now = datetime.now()
    return f"bo_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run pure-regression Bayesian optimization (FoG objective) and export BO figures/tables."
    )
    p.add_argument(
        "--bo_learning",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_learning" / "bo_learning_plate_aware.csv",
        help="Path to BO learning CSV (polymer_id, round_id, frac_*, log_fog).",
    )
    p.add_argument(
        "--fog_plate_aware",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_plate_aware" / "fog_plate_aware.csv",
        help="Legacy option. Kept for compatibility; pure regression BO does not require this input.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_runs",
        help="Output root directory. Results go under out_dir/{bo_run_id}/",
    )
    p.add_argument(
        "--bo_run_id",
        type=str,
        default=None,
        help="Optional BO run ID. If omitted, local timestamp-based ID is used.",
    )

    # Optional rebuild of BO learning data.
    p.add_argument(
        "--rebuild_learning",
        action="store_true",
        help="Rebuild BO learning CSV from plate-aware round-averaged FoG before BO.",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "meta" / "bo_catalog_bma.csv",
        help="BO catalog path used when --rebuild_learning is enabled.",
    )
    p.add_argument(
        "--fog_round_averaged",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_plate_aware" / "fog_plate_aware_round_averaged.csv",
        help="Round-averaged FoG path used when --rebuild_learning is enabled.",
    )
    p.add_argument(
        "--exclusion_report",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_learning" / "bo_learning_excluded_plate_aware.csv",
        help="Exclusion report path used when --rebuild_learning is enabled.",
    )

    # BO policy (pure-regression mode)
    p.add_argument("--n_suggestions", type=int, default=8)
    p.add_argument("--exploration_ratio", type=float, default=0.35)
    p.add_argument("--anchor_fraction", type=float, default=0.12)
    p.add_argument("--replicate_fraction", type=float, default=0.12)
    p.add_argument("--anchor_count", type=int, default=None)
    p.add_argument("--replicate_count", type=int, default=None)
    p.add_argument(
        "--anchor_polymer_ids",
        type=str,
        default="PMPC,PMTAC",
        help="Comma-separated polymer IDs for fixed-composition slots (anchor re-proposal). Not GOx.",
    )
    p.add_argument(
        "--no_exact_anchor",
        action="store_true",
        help="Use nearest grid point for anchor instead of exact composition (default: exact).",
    )
    p.add_argument(
        "--replicate_source",
        type=str,
        default="exploit",
        choices=["exploit", "explore", "all"],
        help="Pool to draw replicates from: exploit, explore, or all (default: exploit).",
    )
    p.add_argument("--candidate_step", type=float, default=0.02)
    p.add_argument("--min_component", type=float, default=0.02)
    p.add_argument("--min_distance_between", type=float, default=0.06)
    p.add_argument("--min_distance_to_train", type=float, default=0.03)
    p.add_argument("--ei_xi", type=float, default=0.01)
    p.add_argument("--ucb_kappa", type=float, default=2.0)
    p.add_argument(
        "--ucb_beta",
        type=float,
        default=None,
        help="Optional UCB beta (kappa = sqrt(beta)). If set, overrides --ucb_kappa.",
    )
    p.add_argument(
        "--acquisition",
        type=str,
        choices=["ei", "ucb"],
        default="ei",
        help="Acquisition function used for proposal selection.",
    )
    p.add_argument(
        "--objective_column",
        type=str,
        default="log_fog_corrected",
        help="Objective column to maximize. Falls back to log_fog when unavailable.",
    )
    p.add_argument(
        "--min_fraction_distance",
        type=float,
        default=None,
        help="Diversity threshold in fraction-space distance. Defaults to --min_distance_between.",
    )
    p.add_argument(
        "--max_component",
        type=float,
        default=0.95,
        help="Upper bound for each component fraction (MPC/BMA/MTAC).",
    )
    p.add_argument(
        "--n_random_candidates",
        type=int,
        default=5000,
        help="Number of random (x,y) candidates evaluated per BO proposal step.",
    )
    p.add_argument(
        "--disable_heteroskedastic_noise",
        action="store_true",
        help="Disable per-observation noise scaling (default: enabled).",
    )
    p.add_argument("--noise_rel_min", type=float, default=0.35)
    p.add_argument("--noise_rel_max", type=float, default=3.0)
    p.add_argument(
        "--priority_weight_fog",
        type=float,
        default=0.45,
        help="Priority score weight for predicted FoG (top5 next-experiment table).",
    )
    p.add_argument(
        "--priority_weight_t50",
        type=float,
        default=0.45,
        help="Priority score weight for predicted t50 (top5 next-experiment table).",
    )
    p.add_argument(
        "--priority_weight_ei",
        type=float,
        default=0.10,
        help="Priority score weight for EI (exploration component).",
    )
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--anchor_correction",
        action="store_true",
        help="Enable round anchor correction of log_fog (default: off; scale handled experimentally).",
    )
    p.add_argument(
        "--min_anchor_polymers",
        type=int,
        default=2,
        help="Minimum number of shared anchor polymers required to apply round correction.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plotting (for quick debug runs).",
    )
    p.add_argument(
        "--use_simplex_gp",
        action="store_true",
        help="Use GP on (frac_MPC, frac_BMA, frac_MTAC) with 3 length scales; no xy_2x2 panels.",
    )
    p.add_argument(
        "--use_bma_mtac_coords",
        action="store_true",
        help="Use GP on (frac_BMA, frac_MTAC) and 2x2 panels in BMA–MTAC plane (default: use x,y).",
    )
    p.add_argument(
        "--use_xy_coords",
        action="store_true",
        help="Use (x,y) coordinates; default, so this flag is optional.",
    )
    p.add_argument(
        "--disable_sparse_isotropic",
        action="store_true",
        help="Disable sparse-data isotropic kernel fallback (default: enabled).",
    )
    p.add_argument(
        "--sparse_isotropic_max_unique_points",
        type=int,
        default=15,
        help="Apply isotropic kernel when unique design points are <= this value (default: 15).",
    )
    p.add_argument(
        "--sparse_isotropic_apply_min_below_n",
        type=int,
        default=10,
        help="When isotropic and unique points <= this, apply min length scale for smooth gradient (default: 10).",
    )
    p.add_argument(
        "--min_length_scale_sparse_isotropic",
        type=float,
        default=0.2,
        help="Min length scale when sparse isotropic is applied (default: 0.2).",
    )
    p.add_argument(
        "--std_color_gamma",
        type=float,
        default=8.0,
        help="PowerNorm gamma for std map (default: 8.0). Use 0 or omit for linear (script uses 8.0). Set via config to None for linear.",
    )
    p.add_argument(
        "--std_color_linear",
        action="store_true",
        help="Use linear color scale for std map (overrides --std_color_gamma).",
    )
    p.add_argument(
        "--polymer_colors",
        type=Path,
        default=REPO_ROOT / "meta" / "polymer_colors.yml",
        help="Path to polymer_id color map YAML for ranking bars and FoG vs t50 scatter (default: meta/polymer_colors.yml).",
    )
    args = p.parse_args()

    _build_learning_if_needed(
        rebuild_learning=bool(args.rebuild_learning),
        bo_learning_path=args.bo_learning,
        exclusion_report_path=args.exclusion_report,
        catalog_path=args.catalog,
        fog_round_averaged_path=args.fog_round_averaged,
    )

    if not args.bo_learning.is_file():
        raise FileNotFoundError(f"BO learning CSV not found: {args.bo_learning}")
    if args.min_component > args.max_component:
        raise ValueError(
            f"min_component must be <= max_component, got min={args.min_component}, max={args.max_component}"
        )

    learning_df = pd.read_csv(args.bo_learning)
    bo_run_id = str(args.bo_run_id).strip() if args.bo_run_id else _default_bo_run_id()
    out_dir = Path(args.out_dir) / bo_run_id
    min_fraction_distance = (
        float(args.min_fraction_distance)
        if args.min_fraction_distance is not None
        else float(args.min_distance_between)
    )
    diversity_params = {"min_fraction_distance": min_fraction_distance}
    composition_constraints = {
        "min_mpc": float(args.min_component),
        "max_mpc": float(args.max_component),
        "min_bma": float(args.min_component),
        "max_bma": float(args.max_component),
        "min_mtac": float(args.min_component),
        "max_mtac": float(args.max_component),
    }

    # Keep legacy knobs accepted in CLI/launch configs, but notify that pure BO ignores them.
    legacy_ignored = []
    if args.use_simplex_gp or args.use_bma_mtac_coords or args.use_xy_coords:
        legacy_ignored.append("coordinate-mode flags (--use_simplex_gp/--use_bma_mtac_coords/--use_xy_coords)")
    if (
        args.anchor_count is not None
        or args.replicate_count is not None
        or float(args.anchor_fraction) != 0.12
        or float(args.replicate_fraction) != 0.12
        or str(args.anchor_polymer_ids).strip() != "PMPC,PMTAC"
        or bool(args.no_exact_anchor)
        or str(args.replicate_source) != "exploit"
    ):
        legacy_ignored.append("anchor/replicate settings")
    if args.candidate_step != 0.02 or args.exploration_ratio != 0.35:
        legacy_ignored.append("candidate_step/exploration_ratio")
    if legacy_ignored:
        print(
            "Note: pure-regression mode is active; ignored legacy options: "
            + ", ".join(legacy_ignored)
        )

    outputs = run_pure_regression_bo(
        learning_df=learning_df,
        out_dir=out_dir,
        q=int(args.n_suggestions),
        acquisition=str(args.acquisition),
        diversity_params=diversity_params,
        composition_constraints=composition_constraints,
        seed=int(args.random_state),
        run_id=bo_run_id,
        objective_column=str(args.objective_column),
        ei_xi=float(args.ei_xi),
        ucb_kappa=float(args.ucb_kappa),
        ucb_beta=args.ucb_beta,
        n_random_candidates=int(args.n_random_candidates),
        sparse_force_isotropic=not bool(args.disable_sparse_isotropic),
        sparse_isotropic_max_unique_points=int(args.sparse_isotropic_max_unique_points),
        write_plots=not bool(args.no_plots),
        learning_input_path=args.bo_learning,
    )

    print("Pure-regression BO finished. Output files:")
    for key, path in sorted(outputs.items(), key=lambda x: x[0]):
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
