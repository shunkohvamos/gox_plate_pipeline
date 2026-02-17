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
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run(cmd: list[str], *, env: dict, debug: bool) -> None:
    if debug:
        print("Run:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _parse_theta_grid(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        v = float(s)
        if v <= 0:
            continue
        vals.append(v)
    if not vals:
        vals = [0.60, 0.70, 0.75]
    return sorted(set(vals))


def _write_theta_sensitivity(
    *,
    fog_plate_aware_csv: Path,
    out_dir: Path,
    theta_grid: List[float],
    reference_polymer_id: str,
) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not fog_plate_aware_csv.is_file():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(fog_plate_aware_csv)
    if df.empty:
        return
    df["polymer_id"] = df.get("polymer_id", "").astype(str).str.strip()
    ref_norm = str(reference_polymer_id).strip().upper()
    df = df[df["polymer_id"].str.upper() != ref_norm].copy()
    if df.empty:
        return
    native_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    df[native_col] = pd.to_numeric(df.get(native_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df = df[np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return

    per_theta_rows = []
    topk_sets: dict[float, set[str]] = {}
    baseline_theta = min(theta_grid, key=lambda x: abs(float(x) - 0.70))
    top_k = 5
    for theta in theta_grid:
        sub = df[np.isfinite(df[native_col]) & (df[native_col] >= float(theta))].copy()
        if sub.empty:
            per_theta_rows.append(
                {
                    "theta": float(theta),
                    "feasible_rows": 0,
                    "feasible_polymers": 0,
                    "top1_polymer_id": "",
                    "top5_polymer_ids": "",
                }
            )
            topk_sets[float(theta)] = set()
            continue
        agg = (
            sub.groupby("polymer_id", as_index=False)
            .agg(
                fog_median=("fog", "median"),
                fog_mean=("fog", "mean"),
                native0_median=(native_col, "median"),
                n_rows=("polymer_id", "size"),
            )
            .sort_values(["fog_median", "fog_mean"], ascending=[False, False], kind="mergesort")
            .reset_index(drop=True)
        )
        top_ids = agg["polymer_id"].head(top_k).astype(str).tolist()
        topk_sets[float(theta)] = set(top_ids)
        per_theta_rows.append(
            {
                "theta": float(theta),
                "feasible_rows": int(len(sub)),
                "feasible_polymers": int(agg["polymer_id"].nunique()),
                "top1_polymer_id": str(agg["polymer_id"].iloc[0]) if not agg.empty else "",
                "top5_polymer_ids": ",".join(top_ids),
            }
        )
    sens_df = pd.DataFrame(per_theta_rows)
    baseline_set = topk_sets.get(float(baseline_theta), set())
    agree_vals = []
    for theta in sens_df["theta"].tolist():
        s = topk_sets.get(float(theta), set())
        denom = max(1, len(baseline_set))
        agree_vals.append(float(len(s & baseline_set)) / float(denom))
    sens_df["top5_overlap_vs_theta70"] = agree_vals
    out_csv = csv_dir / "theta_sensitivity_summary.csv"
    sens_df.to_csv(out_csv, index=False)
    legacy_csv = out_dir / "theta_sensitivity_summary.csv"
    if legacy_csv.is_file():
        legacy_csv.unlink(missing_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.plot(sens_df["theta"], sens_df["feasible_polymers"], marker="o", linewidth=1.0, color="#1f77b4")
    ax1.set_xlabel("theta")
    ax1.set_ylabel("Feasible polymers (n)")
    ax1.set_title("Feasible count vs theta")
    ax1.grid(True, linestyle=":", alpha=0.35)

    ax2.plot(sens_df["theta"], sens_df["top5_overlap_vs_theta70"], marker="o", linewidth=1.0, color="#2ca02c")
    ax2.set_xlabel("theta")
    ax2.set_ylabel("Top5 overlap vs theta=0.70")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Top-k stability")
    ax2.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout(pad=0.3)
    out_png = out_dir / "theta_sensitivity_summary.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
        "--native_activity_min_rel",
        type=float,
        default=0.70,
        help=(
            "Native-activity feasibility threshold used in fit and plate-aware FoG "
            "(abs_activity_at_0 / GOx_abs_activity_at_0 reference)."
        ),
    )
    p.add_argument(
        "--reference_polymer_id",
        type=str,
        default="GOX",
        help="Reference polymer ID used in fit and FoG steps (default: GOX).",
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
        "--objective_column",
        type=str,
        default="log_fog_native_constrained",
        help=(
            "BO objective column for run_bayesian_optimization.py "
            "(default: log_fog_native_constrained)."
        ),
    )
    p.add_argument(
        "--policy_v2_dir",
        type=Path,
        default=None,
        help="Output root for policy_v2 artifacts. Default: <processed_dir>/policy_v2",
    )
    p.add_argument(
        "--theta_grid",
        type=str,
        default="0.60,0.70,0.75",
        help="Comma-separated theta values for sensitivity summary (default: 0.60,0.70,0.75).",
    )
    p.add_argument(
        "--ref_agg_method",
        type=str,
        default="median",
        choices=["median", "trimmed_mean", "mean"],
        help="Reference aggregation method for policy_v2 fallback (default: median).",
    )
    p.add_argument(
        "--ref_trimmed_mean_proportion",
        type=float,
        default=0.1,
        help="Trimming proportion for --ref_agg_method=trimmed_mean.",
    )
    p.add_argument(
        "--reference_qc_mad_rel_threshold",
        type=float,
        default=0.25,
        help="Run-level reference abs0 QC threshold on relative MAD.",
    )
    p.add_argument(
        "--reference_qc_min_abs0",
        type=float,
        default=0.0,
        help="Run-level reference abs0 QC threshold on median abs0 (0.0 disables low-floor filtering).",
    )
    p.add_argument(
        "--reference_qc_exclude",
        action="store_true",
        help="Exclude QC-failed runs from constrained objective in policy_v2 (default: flag-only).",
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
    policy_v2_dir = args.policy_v2_dir if args.policy_v2_dir is not None else (args.processed_dir / "policy_v2")
    plate_aware_dir = policy_v2_dir / "fog_plate_aware"
    fog_plate_aware = plate_aware_dir / "fog_plate_aware.csv"
    fog_plate_aware_round = plate_aware_dir / "fog_plate_aware_round_averaged.csv"
    bo_learning = policy_v2_dir / "bo_learning" / "bo_learning_plate_aware.csv"
    bo_excluded = policy_v2_dir / "bo_learning" / "bo_learning_excluded_plate_aware.csv"
    sensitivity_dir = policy_v2_dir / "sensitivity"

    step1 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_fit_then_round_fog.py"),
        "--run_round_map", args.run_round_map.relative_to(REPO_ROOT).as_posix(),
        "--processed_dir", args.processed_dir.relative_to(REPO_ROOT).as_posix(),
        "--config", args.config.relative_to(REPO_ROOT).as_posix(),
        "--out_fog", fog_round_averaged.relative_to(REPO_ROOT).as_posix(),
        "--t50_definition", str(args.t50_definition),
        "--native_activity_min_rel", str(float(args.native_activity_min_rel)),
        "--reference_polymer_id", str(args.reference_polymer_id).strip() or "GOX",
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
        "--native_activity_min_rel", str(float(args.native_activity_min_rel)),
        "--reference_polymer_id", str(args.reference_polymer_id).strip() or "GOX",
        "--native_reference_mode", "same_run_then_round",
        "--ref_agg_method", str(args.ref_agg_method),
        "--ref_trimmed_mean_proportion", str(float(args.ref_trimmed_mean_proportion)),
        "--reference_qc_mad_rel_threshold", str(float(args.reference_qc_mad_rel_threshold)),
        "--reference_qc_min_abs0", str(float(args.reference_qc_min_abs0)),
    ]
    if args.reference_qc_exclude:
        step2.append("--reference_qc_exclude")

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
        "--objective_column", str(args.objective_column),
        "--theta_grid", str(args.theta_grid),
        "--theta_sensitivity_out", (policy_v2_dir / "sensitivity").as_posix(),
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
        print("native activity threshold:", float(args.native_activity_min_rel))
        print("reference polymer id:", str(args.reference_polymer_id).strip() or "GOX")
        print("policy_v2 dir:", policy_v2_dir)

    if args.dry_run:
        return

    _run(step1, env=env, debug=args.debug)
    _run(step2, env=env, debug=args.debug)
    _write_theta_sensitivity(
        fog_plate_aware_csv=fog_plate_aware,
        out_dir=sensitivity_dir,
        theta_grid=_parse_theta_grid(args.theta_grid),
        reference_polymer_id=str(args.reference_polymer_id).strip() or "GOX",
    )
    _run(step3, env=env, debug=args.debug)
    print("Completed: round-assigned runs -> BO one-shot pipeline.")


if __name__ == "__main__":
    main()
