#!/usr/bin/env python3
"""
Run pure-regression Bayesian optimization from prepared (or rebuilt) BO learning data.

Default workflow:
1) Use plate-aware round-averaged FoG to build bo_learning_plate_aware.csv (optional rebuild).
2) Fit GP on (x, y) and output BO proposal maps/logs for FoG maximization.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import (  # noqa: E402
    build_round_coverage_summary,
    build_bo_learning_data_from_round_averaged,
    file_fingerprint,
    load_bo_catalog,
    load_run_round_map,
    write_bo_learning_csv,
    write_exclusion_report,
)
from gox_plate_pipeline.bo_engine import run_pure_regression_bo  # noqa: E402
from gox_plate_pipeline.meta_paths import get_meta_paths  # noqa: E402
from gox_plate_pipeline.summary import build_run_manifest_dict  # noqa: E402

META = get_meta_paths(REPO_ROOT)


def _build_learning_if_needed(
    *,
    rebuild_learning: bool,
    trace_run_id: str,
    bo_learning_path: Path,
    exclusion_report_path: Path,
    catalog_path: Path,
    fog_round_averaged_path: Path,
    write_manifest: bool = True,
    include_run_id_column: bool = False,
    write_lineage_csv: bool = False,
    lineage_out_path: Path | None = None,
    run_round_map_path: Path | None = None,
    strict_round_coverage: bool = True,
) -> None:
    if not rebuild_learning and bo_learning_path.is_file():
        return
    if not catalog_path.is_file():
        raise FileNotFoundError(f"BO catalog not found: {catalog_path}")
    if not fog_round_averaged_path.is_file():
        raise FileNotFoundError(f"Round-averaged FoG not found: {fog_round_averaged_path}")

    catalog_df = load_bo_catalog(catalog_path, validate_sum=True)
    run_round_map = None
    round_coverage = None
    run_round_map_file_meta = None
    if run_round_map_path is not None:
        if not Path(run_round_map_path).is_file():
            raise FileNotFoundError(f"Run-round map not found: {run_round_map_path}")
        run_round_map = load_run_round_map(Path(run_round_map_path))
        run_round_map_file_meta = file_fingerprint(Path(run_round_map_path))
    learning_df, excluded_df = build_bo_learning_data_from_round_averaged(
        catalog_df,
        fog_round_averaged_path,
        run_round_map=run_round_map,
        strict_round_coverage=bool(strict_round_coverage),
    )
    if run_round_map is not None:
        round_coverage = build_round_coverage_summary(learning_df.get("round_id", pd.Series([], dtype=object)).tolist(), run_round_map)
    learning_to_write = learning_df.copy()
    excluded_to_write = excluded_df.copy()
    if bool(include_run_id_column):
        if "run_id" not in learning_to_write.columns:
            learning_to_write.insert(0, "run_id", trace_run_id)
        if "run_id" not in excluded_to_write.columns:
            excluded_to_write.insert(0, "run_id", trace_run_id)
    bo_learning_path.parent.mkdir(parents=True, exist_ok=True)
    write_bo_learning_csv(learning_to_write, bo_learning_path)
    write_exclusion_report(excluded_to_write, exclusion_report_path)
    print(f"Saved BO learning: {bo_learning_path} ({len(learning_df)} rows)")
    print(f"Saved BO excluded: {exclusion_report_path} ({len(excluded_df)} rows)")

    lineage_path = None
    if bool(write_lineage_csv) or lineage_out_path is not None:
        lineage_cols = [
            "run_id",
            "lineage_row_id",
            "polymer_id",
            "round_id",
            "source_run_id",
            "source_n_observations",
            "source_objective_source",
        ]
        lineage_path = Path(lineage_out_path) if lineage_out_path is not None else bo_learning_path.parent / f"bo_learning_lineage__{trace_run_id}.csv"
        lineage_rows: list[dict] = []
        work = learning_df.reset_index(drop=True).copy()
        for idx, row in work.iterrows():
            source_ids = [s.strip() for s in str(row.get("run_ids", "")).split(",") if s.strip()]
            if not source_ids:
                source_ids = [""]
            for sid in source_ids:
                lineage_rows.append(
                    {
                        "run_id": trace_run_id,
                        "lineage_row_id": int(idx),
                        "polymer_id": str(row.get("polymer_id", "")),
                        "round_id": str(row.get("round_id", "")),
                        "source_run_id": sid,
                        "source_n_observations": row.get("n_observations", ""),
                        "source_objective_source": str(row.get("objective_source", "")),
                    }
                )
        pd.DataFrame(lineage_rows, columns=lineage_cols).to_csv(lineage_path, index=False)
        print(f"Saved BO learning lineage: {lineage_path} ({len(lineage_rows)} rows)")

    run_round_map_meta_path = None
    if run_round_map is not None:
        run_round_map_meta_path = bo_learning_path.parent / f"run_round_map_meta__{trace_run_id}.json"
        meta_payload = {
            "run_id": trace_run_id,
            "operation": "run_bayesian_optimization.rebuild_learning",
            "run_round_map_file": run_round_map_file_meta,
            "round_coverage": round_coverage,
        }
        with open(run_round_map_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved BO run-round map meta: {run_round_map_meta_path}")

    if write_manifest:
        manifest_path = bo_learning_path.parent / f"bo_learning_manifest__{trace_run_id}.json"
        output_files = [bo_learning_path.name, exclusion_report_path.name]
        if lineage_path is not None:
            output_files.append(lineage_path.name)
        if run_round_map_meta_path is not None:
            output_files.append(run_round_map_meta_path.name)
        manifest = build_run_manifest_dict(
            run_id=trace_run_id,
            input_paths=[catalog_path, fog_round_averaged_path]
            + ([Path(run_round_map_path)] if run_round_map_path is not None else []),
            git_root=REPO_ROOT,
            extra={
                "operation": "run_bayesian_optimization.rebuild_learning",
                "n_learning_rows": int(len(learning_df)),
                "n_excluded_rows": int(len(excluded_df)),
                "strict_round_coverage": bool(strict_round_coverage),
                "round_coverage": round_coverage,
                "output_files": output_files,
                "cli_args": sys.argv[1:],
            },
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved BO learning manifest: {manifest_path}")


def _default_bo_run_id() -> str:
    now = datetime.now()
    return f"bo_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def _parse_theta_grid(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        v = float(s)
        if v > 0:
            vals.append(v)
    if not vals:
        vals = [0.60, 0.70, 0.75]
    return sorted(set(vals))


def _sanitize_objective_token(objective_column: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(objective_column).strip())
    token = token.strip("_")
    return token or "objective"


def _write_bo_comparison_csv(
    *,
    bo_run_id: str,
    out_dir: Path,
    primary_objective: str,
    secondary_objective: str,
    primary_candidates_path: Path,
    secondary_candidates_path: Path,
) -> Path | None:
    if (not primary_candidates_path.is_file()) or (not secondary_candidates_path.is_file()):
        return None
    primary = pd.read_csv(primary_candidates_path)
    secondary = pd.read_csv(secondary_candidates_path)
    if primary.empty or secondary.empty:
        return None
    if "selection_order" not in primary.columns:
        primary["selection_order"] = np.arange(1, len(primary) + 1, dtype=int)
    if "selection_order" not in secondary.columns:
        secondary["selection_order"] = np.arange(1, len(secondary) + 1, dtype=int)
    primary = primary.copy()
    secondary = secondary.copy()
    primary["selection_order"] = pd.to_numeric(primary["selection_order"], errors="coerce").astype(int)
    secondary["selection_order"] = pd.to_numeric(secondary["selection_order"], errors="coerce").astype(int)
    primary_cols = {
        c: f"primary_{c}"
        for c in primary.columns
        if c != "selection_order"
    }
    secondary_cols = {
        c: f"secondary_{c}"
        for c in secondary.columns
        if c != "selection_order"
    }
    merged = (
        primary.rename(columns=primary_cols)
        .merge(
            secondary.rename(columns=secondary_cols),
            on="selection_order",
            how="outer",
        )
        .sort_values("selection_order", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )
    merged.insert(0, "bo_run_id", str(bo_run_id))
    merged.insert(1, "primary_objective_column", str(primary_objective))
    merged.insert(2, "secondary_objective_column", str(secondary_objective))

    frac_cols = ["frac_MPC", "frac_BMA", "frac_MTAC"]
    for col in frac_cols:
        p_col = f"primary_{col}"
        s_col = f"secondary_{col}"
        if p_col not in merged.columns:
            merged[p_col] = np.nan
        if s_col not in merged.columns:
            merged[s_col] = np.nan
    primary_frac = merged[[f"primary_{c}" for c in frac_cols]].to_numpy(dtype=float)
    secondary_frac = merged[[f"secondary_{c}" for c in frac_cols]].to_numpy(dtype=float)
    both_valid = np.isfinite(primary_frac).all(axis=1) & np.isfinite(secondary_frac).all(axis=1)
    l2_order = np.full(len(merged), np.nan, dtype=float)
    if np.any(both_valid):
        l2_order[both_valid] = np.linalg.norm(primary_frac[both_valid] - secondary_frac[both_valid], axis=1)
    merged["frac_l2_distance_order_matched"] = l2_order

    sec_full = secondary[[c for c in frac_cols if c in secondary.columns]].copy()
    sec_full = sec_full.apply(pd.to_numeric, errors="coerce")
    if set(frac_cols).issubset(sec_full.columns):
        sec_all = sec_full[frac_cols].to_numpy(dtype=float)
        sec_arr = sec_all[np.isfinite(sec_all).all(axis=1)]
    else:
        sec_arr = np.empty((0, 3))
    min_to_secondary = np.full(len(merged), np.nan, dtype=float)
    if sec_arr.size:
        for i, row in enumerate(primary_frac):
            if np.isfinite(row).all():
                min_to_secondary[i] = float(np.min(np.linalg.norm(sec_arr - row.reshape(1, 3), axis=1)))
    merged["primary_min_frac_l2_to_any_secondary"] = min_to_secondary

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bo_comparison__{bo_run_id}.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def _write_theta_sensitivity(
    *,
    fog_plate_aware_path: Path,
    out_dir: Path,
    theta_grid: List[float],
    reference_polymer_id: str = "GOX",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    path = Path(fog_plate_aware_path)
    if not path.is_file():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
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

    rows = []
    top_sets: dict[float, set[str]] = {}
    for theta in theta_grid:
        sub = df[np.isfinite(df[native_col]) & (df[native_col] >= float(theta))].copy()
        if sub.empty:
            rows.append({"theta": float(theta), "feasible_rows": 0, "feasible_polymers": 0})
            top_sets[float(theta)] = set()
            continue
        agg = (
            sub.groupby("polymer_id", as_index=False)
            .agg(fog_median=("fog", "median"))
            .sort_values("fog_median", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        rows.append(
            {
                "theta": float(theta),
                "feasible_rows": int(len(sub)),
                "feasible_polymers": int(len(agg)),
                "top1_polymer_id": str(agg["polymer_id"].iloc[0]) if len(agg) else "",
                "top5_polymer_ids": ",".join(agg["polymer_id"].head(5).astype(str).tolist()),
            }
        )
        top_sets[float(theta)] = set(agg["polymer_id"].head(5).astype(str).tolist())
    sens = pd.DataFrame(rows)
    base_theta = min(theta_grid, key=lambda x: abs(float(x) - 0.70))
    base_set = top_sets.get(float(base_theta), set())
    sens["top5_overlap_vs_theta70"] = [
        float(len(top_sets.get(float(th), set()) & base_set)) / max(1, len(base_set))
        for th in sens["theta"].tolist()
    ]
    out_csv = csv_dir / "theta_sensitivity_summary.csv"
    sens.to_csv(out_csv, index=False)
    legacy_csv = out_dir / "theta_sensitivity_summary.csv"
    if legacy_csv.is_file():
        legacy_csv.unlink(missing_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.plot(sens["theta"], sens["feasible_polymers"], marker="o", linewidth=1.0, color="#1f77b4")
    ax1.set_xlabel("theta")
    ax1.set_ylabel("Feasible polymers (n)")
    ax1.set_title("Feasible count vs theta")
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax2.plot(sens["theta"], sens["top5_overlap_vs_theta70"], marker="o", linewidth=1.0, color="#2ca02c")
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
        description="Run pure-regression Bayesian optimization (FoG objective) and export BO figures/tables."
    )
    p.add_argument(
        "--bo_mode",
        type=str,
        choices=["pure_regression"],
        default="pure_regression",
        help="BO execution mode. Locked to pure_regression in this script.",
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
        default=META.bo_catalog_bma,
        help="BO catalog path used when --rebuild_learning is enabled.",
    )
    p.add_argument(
        "--fog_round_averaged",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_plate_aware" / "fog_plate_aware_round_averaged.csv",
        help="Round-averaged FoG path used when --rebuild_learning is enabled.",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=META.run_round_map,
        help="run_id→round_id map used for strict round-coverage checks when rebuilding learning data.",
    )
    p.add_argument(
        "--allow_unmapped_round_ids",
        action="store_true",
        help=(
            "Allow round IDs in --fog_round_averaged that are missing in --run_round_map. "
            "Default is strict (raise on mismatch)."
        ),
    )
    p.add_argument(
        "--exclusion_report",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bo_learning" / "bo_learning_excluded_plate_aware.csv",
        help="Exclusion report path used when --rebuild_learning is enabled.",
    )
    p.add_argument(
        "--no_learning_manifest",
        action="store_true",
        help="Skip writing bo_learning_manifest__*.json when --rebuild_learning is used.",
    )
    p.add_argument(
        "--include_run_id_column",
        action="store_true",
        help="Add run_id column to rebuilt bo_learning/excluded CSVs (compat mode off by default).",
    )
    p.add_argument(
        "--write_learning_lineage_csv",
        action="store_true",
        help="Deprecated compatibility flag. bo_learning_lineage CSV is now written by default.",
    )
    p.add_argument(
        "--no_learning_lineage_csv",
        action="store_true",
        help="Disable bo_learning_lineage CSV output when --rebuild_learning is used.",
    )
    p.add_argument(
        "--learning_lineage_out",
        type=Path,
        default=None,
        help="Optional explicit path for bo_learning_lineage CSV.",
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
        default="log_fog_activity_bonus_penalty",
        help=(
            "Objective column to maximize (fail-fast). "
            "Default: log_fog_activity_bonus_penalty."
        ),
    )
    p.add_argument(
        "--compare_objective_column",
        type=str,
        default="objective_loglinear_main",
        help=(
            "Secondary objective column used for one-round parallel comparison. "
            "Default: objective_loglinear_main."
        ),
    )
    p.add_argument(
        "--disable_objective_comparison",
        action="store_true",
        help="Disable secondary objective comparison and bo_comparison CSV output.",
    )
    p.add_argument(
        "--theta_grid",
        type=str,
        default="0.60,0.70,0.75",
        help="Comma-separated theta values for sensitivity summary (default: 0.60,0.70,0.75).",
    )
    p.add_argument(
        "--theta_sensitivity_out",
        type=Path,
        default=None,
        help="Output directory for theta sensitivity summary. Default: <bo_run_dir>/sensitivity.",
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
        default=10,
        help="Apply isotropic kernel when unique design points are <= this value (default: 10).",
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
        default=0.5,
        help="Min length scale when sparse isotropic is applied (default: 0.5).",
    )
    p.add_argument(
        "--enable_sparse_trend",
        dest="sparse_trend",
        action="store_true",
        help="Enable sparse polynomial trend + GP residual fitting (default: disabled).",
    )
    p.add_argument(
        "--disable_sparse_trend",
        dest="sparse_trend",
        action="store_false",
        help="Disable sparse polynomial trend + GP residual fitting.",
    )
    p.set_defaults(sparse_trend=False)
    p.add_argument(
        "--sparse_trend_max_unique_points",
        type=int,
        default=8,
        help="Apply sparse trend when unique design points are <= this value (default: 8).",
    )
    p.add_argument(
        "--trend_ridge",
        type=float,
        default=1e-5,
        help="Ridge regularization for sparse trend fit (default: 1e-5).",
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
        default=META.polymer_colors,
        help="Path to polymer_id color map YAML for ranking bars and FoG vs t50 scatter (default: meta/polymers/colors.yml).",
    )
    args = p.parse_args()

    bo_run_id = str(args.bo_run_id).strip() if args.bo_run_id else _default_bo_run_id()

    _build_learning_if_needed(
        rebuild_learning=bool(args.rebuild_learning),
        trace_run_id=f"{bo_run_id}__learning",
        bo_learning_path=args.bo_learning,
        exclusion_report_path=args.exclusion_report,
        catalog_path=args.catalog,
        fog_round_averaged_path=args.fog_round_averaged,
        write_manifest=not bool(args.no_learning_manifest),
        include_run_id_column=bool(args.include_run_id_column),
        write_lineage_csv=(not bool(args.no_learning_lineage_csv)) or bool(args.write_learning_lineage_csv),
        lineage_out_path=args.learning_lineage_out,
        run_round_map_path=args.run_round_map,
        strict_round_coverage=not bool(args.allow_unmapped_round_ids),
    )

    if not args.bo_learning.is_file():
        raise FileNotFoundError(f"BO learning CSV not found: {args.bo_learning}")
    if args.min_component > args.max_component:
        raise ValueError(
            f"min_component must be <= max_component, got min={args.min_component}, max={args.max_component}"
        )

    learning_df = pd.read_csv(args.bo_learning)
    out_dir = Path(args.out_dir) / bo_run_id
    theta_sensitivity_dir = (
        Path(args.theta_sensitivity_out)
        if args.theta_sensitivity_out is not None
        else (out_dir / "sensitivity")
    )
    _write_theta_sensitivity(
        fog_plate_aware_path=args.fog_plate_aware,
        out_dir=theta_sensitivity_dir,
        theta_grid=_parse_theta_grid(args.theta_grid),
    )
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
    if bool(args.anchor_correction):
        legacy_ignored.append("--anchor_correction (disabled by policy in pure-regression mode)")
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
        min_length_scale_sparse_isotropic=float(args.min_length_scale_sparse_isotropic),
        sparse_use_trend=bool(args.sparse_trend),
        sparse_trend_max_unique_points=int(args.sparse_trend_max_unique_points),
        trend_ridge=float(args.trend_ridge),
        enable_heteroskedastic_noise=not bool(args.disable_heteroskedastic_noise),
        noise_rel_min=float(args.noise_rel_min),
        noise_rel_max=float(args.noise_rel_max),
        write_plots=not bool(args.no_plots),
        learning_input_path=args.bo_learning,
        fog_plate_aware_path=args.fog_plate_aware,
        polymer_colors_path=args.polymer_colors,
    )
    comparison_csv_path: Path | None = None
    secondary_outputs: dict[str, Path] = {}
    primary_objective_col = str(args.objective_column).strip()
    secondary_objective_col = str(args.compare_objective_column).strip()
    if (not bool(args.disable_objective_comparison)) and secondary_objective_col and (secondary_objective_col != primary_objective_col):
        secondary_run_id = f"{bo_run_id}__cmp__{_sanitize_objective_token(secondary_objective_col)}"
        secondary_outputs = run_pure_regression_bo(
            learning_df=learning_df,
            out_dir=out_dir,
            q=int(args.n_suggestions),
            acquisition=str(args.acquisition),
            diversity_params=diversity_params,
            composition_constraints=composition_constraints,
            seed=int(args.random_state),
            run_id=secondary_run_id,
            objective_column=secondary_objective_col,
            ei_xi=float(args.ei_xi),
            ucb_kappa=float(args.ucb_kappa),
            ucb_beta=args.ucb_beta,
            n_random_candidates=int(args.n_random_candidates),
            sparse_force_isotropic=not bool(args.disable_sparse_isotropic),
            sparse_isotropic_max_unique_points=int(args.sparse_isotropic_max_unique_points),
            min_length_scale_sparse_isotropic=float(args.min_length_scale_sparse_isotropic),
            sparse_use_trend=bool(args.sparse_trend),
            sparse_trend_max_unique_points=int(args.sparse_trend_max_unique_points),
            trend_ridge=float(args.trend_ridge),
            enable_heteroskedastic_noise=not bool(args.disable_heteroskedastic_noise),
            noise_rel_min=float(args.noise_rel_min),
            noise_rel_max=float(args.noise_rel_max),
            write_plots=False,
            learning_input_path=args.bo_learning,
            fog_plate_aware_path=args.fog_plate_aware,
            polymer_colors_path=args.polymer_colors,
        )
        primary_candidates = Path(outputs.get("candidates", ""))
        secondary_candidates = Path(secondary_outputs.get("candidates", ""))
        comparison_csv_path = _write_bo_comparison_csv(
            bo_run_id=bo_run_id,
            out_dir=out_dir,
            primary_objective=primary_objective_col,
            secondary_objective=secondary_objective_col,
            primary_candidates_path=primary_candidates,
            secondary_candidates_path=secondary_candidates,
        )
        if comparison_csv_path is not None and comparison_csv_path.is_file():
            outputs["bo_comparison_csv"] = comparison_csv_path
            outputs["secondary_candidates_csv"] = secondary_candidates
    sens_csv = theta_sensitivity_dir / "csv" / "theta_sensitivity_summary.csv"
    sens_png = theta_sensitivity_dir / "theta_sensitivity_summary.png"
    if sens_csv.is_file():
        outputs["theta_sensitivity_csv"] = sens_csv
    if sens_png.is_file():
        outputs["theta_sensitivity_png"] = sens_png

    execution_policy = {
        "run_id": bo_run_id,
        "bo_mode": str(args.bo_mode),
        "strategy_locked": True,
        "ignored_legacy_options": legacy_ignored,
        "primary_objective_column": str(args.objective_column),
        "secondary_objective_column": str(args.compare_objective_column),
        "objective_comparison_enabled": not bool(args.disable_objective_comparison),
        "bo_comparison_csv": (comparison_csv_path.name if comparison_csv_path is not None else None),
        "strict_round_coverage": not bool(args.allow_unmapped_round_ids),
        "round_map_path": str(args.run_round_map),
        "round_map_exists": bool(Path(args.run_round_map).is_file()),
        "cli_args": sys.argv[1:],
    }
    execution_policy_path = out_dir / f"bo_execution_policy__{bo_run_id}.json"
    with open(execution_policy_path, "w", encoding="utf-8") as f:
        json.dump(execution_policy, f, indent=2, ensure_ascii=False)
    outputs["execution_policy"] = execution_policy_path

    summary_path = Path(outputs.get("summary", ""))
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
        summary_payload["bo_mode"] = str(args.bo_mode)
        summary_payload["strategy_locked"] = True
        summary_payload["ignored_legacy_options"] = legacy_ignored
        summary_payload["primary_objective_column"] = str(args.objective_column)
        summary_payload["secondary_objective_column"] = str(args.compare_objective_column)
        summary_payload["bo_comparison_csv"] = (
            comparison_csv_path.name if comparison_csv_path is not None else None
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    manifest_path = Path(outputs.get("manifest", ""))
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_payload = json.load(f)
        manifest_extra = manifest_payload.setdefault("extra", {})
        manifest_extra["bo_mode"] = str(args.bo_mode)
        manifest_extra["strategy_locked"] = True
        manifest_extra["ignored_legacy_options"] = legacy_ignored
        manifest_extra["primary_objective_column"] = str(args.objective_column)
        manifest_extra["secondary_objective_column"] = str(args.compare_objective_column)
        manifest_extra["bo_comparison_csv"] = (
            comparison_csv_path.name if comparison_csv_path is not None else None
        )
        manifest_extra["execution_policy_file"] = execution_policy_path.name
        output_files = manifest_extra.get("output_files")
        if isinstance(output_files, list):
            if execution_policy_path.name not in output_files:
                output_files.append(execution_policy_path.name)
            if comparison_csv_path is not None and comparison_csv_path.name not in output_files:
                output_files.append(comparison_csv_path.name)
            manifest_extra["output_files"] = sorted(output_files)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    print("Pure-regression BO finished. Output files:")
    for key, path in sorted(outputs.items(), key=lambda x: x[0]):
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
