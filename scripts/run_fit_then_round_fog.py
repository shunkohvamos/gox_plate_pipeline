#!/usr/bin/env python3
"""
Run Fit rates+REA on all raw data folders that belong to a round, then build round-averaged FoG
and GOx traceability CSVs.

- Reads run_round_map (run_id → round_id). Only run_ids with a valid round_id are processed.
- For each such run_id: if extract/tidy.csv is missing, runs extract_clean_csv; if fit/fog_summary
  is missing (or --force_fit), runs fit_initial_rates. Then builds fog_round_averaged.csv and
  fog_round_gox_traceability.csv.
- Round-averaged FoG: one row per (round_id, polymer_id) with mean_fog; same round → one FoG per polymer.
- GOx traceability: round_id, run_id, heat_min, plate_id, well, abs_activity, REA_percent (all pre-averaged).

Usage:
  python scripts/run_fit_then_round_fog.py --run_round_map meta/bo_run_round_map.tsv [--dry_run] [--force_fit] [--debug] [--t50_definition y0_half|rea50]
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


def _discover_datasets(repo_root: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (run_id, raw_path, row_map_path) for each valid pair (same logic as generate_launch_json)."""
    raw_dir = repo_root / "data" / "raw"
    meta_dir = repo_root / "data" / "meta"
    if not raw_dir.is_dir():
        return []

    pairs: list[tuple[str, Path, Path]] = []
    seen_run_ids: set[str] = set()

    for raw_path in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        csvs = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            continue
        run_id = raw_path.name
        row_map = meta_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = meta_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))
            seen_run_ids.add(run_id)

    for raw_path in sorted(raw_dir.glob("*.csv")):
        run_id = raw_path.stem
        if run_id in seen_run_ids:
            continue
        row_map = meta_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = meta_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))

    return pairs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Fit rates+REA on all round-associated runs, then build round-averaged FoG + GOx traceability.",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=REPO_ROOT / "meta" / "bo_run_round_map.tsv",
        help="Path to run_id→round_id map (TSV/YAML/CSV).",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed output directory (extract/fit per run_id).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "meta" / "config.yml",
        help="Config YAML for fit (heat_times etc.).",
    )
    p.add_argument(
        "--out_fog",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_round_averaged" / "fog_round_averaged.csv",
        help="Output path for round-averaged FoG CSV.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be run; do not execute extract/fit or write CSVs.",
    )
    p.add_argument(
        "--force_fit",
        action="store_true",
        help="Re-run fit_initial_rates for each run even if fog_summary already exists.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose output (print each step).",
    )
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition passed to fit_initial_rates.py.",
    )
    args = p.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    # Load run_round_map and get run_ids with valid round_id
    from gox_plate_pipeline.bo_data import load_run_round_map  # noqa: E402

    if not args.run_round_map.is_file():
        print(f"Run-round map not found: {args.run_round_map}", file=sys.stderr)
        sys.exit(1)
    run_round_map = load_run_round_map(Path(args.run_round_map))
    run_ids_needed = [
        rid for rid, oid in run_round_map.items()
        if oid and str(oid).strip() and str(oid).strip().upper() not in ("—", "NA", "NAN")
    ]
    if not run_ids_needed:
        print("No run_id with valid round_id in run_round_map.", file=sys.stderr)
        sys.exit(1)

    datasets = _discover_datasets(REPO_ROOT)
    by_run = {run_id: (raw_path, row_map) for run_id, raw_path, row_map in datasets}

    if args.dry_run or args.debug:
        print("Run IDs with round (will process):", run_ids_needed)
        print("Discovered raw datasets:", list(by_run.keys()))

    # Plan: extract then fit for each run_id that needs it
    # Note: If tidy.csv exists, we can fit even if raw is missing (for re-processing past runs).
    to_extract: list[str] = []
    to_fit: list[str] = []
    for run_id in run_ids_needed:
        tidy_path = args.processed_dir / run_id / "extract" / "tidy.csv"
        fog_path = args.processed_dir / run_id / "fit" / f"fog_summary__{run_id}.csv"
        if not tidy_path.is_file():
            if run_id not in by_run:
                print(f"Warning: {run_id} has round but no raw+row_map and no tidy.csv; skip.", file=sys.stderr)
                continue
            to_extract.append(run_id)
        # If tidy.csv exists, we can fit even without raw (for re-processing past runs)
        if not fog_path.is_file() or args.force_fit:
            to_fit.append(run_id)

    if args.dry_run:
        print("Would run extract for:", to_extract)
        print("Would run fit for:", to_fit)
        print("t50 definition for fit:", args.t50_definition)
        print("Would then write:", args.out_fog, "and", args.out_fog.parent / "fog_round_gox_traceability.csv")
        return

    # Run extract for runs missing tidy
    for run_id in to_extract:
        raw_path, row_map_path = by_run[run_id]
        raw_rel = raw_path.relative_to(REPO_ROOT).as_posix()
        row_rel = row_map_path.relative_to(REPO_ROOT).as_posix()
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "extract_clean_csv.py"),
            "--raw", raw_rel,
            "--row_map", row_rel,
            "--config", str(args.config.relative_to(REPO_ROOT)),
            "--out_dir", str(args.processed_dir.relative_to(REPO_ROOT)),
        ]
        if args.debug:
            print("Run extract:", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    # Run fit for runs missing fog_summary or --force_fit
    for run_id in to_fit:
        tidy_path = args.processed_dir / run_id / "extract" / "tidy.csv"
        if not tidy_path.is_file():
            print(f"Skip fit for {run_id}: tidy not found.", file=sys.stderr)
            continue
        tidy_rel = tidy_path.relative_to(REPO_ROOT).as_posix()
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fit_initial_rates.py"),
            "--tidy", tidy_rel,
            "--config", str(args.config.relative_to(REPO_ROOT)),
            "--out_dir", str(args.processed_dir.relative_to(REPO_ROOT)),
            "--write_well_plots", "0",
            "--t50_definition", str(args.t50_definition),
        ]
        if args.debug:
            print("Run fit:", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    # Build round-averaged FoG and GOx traceability
    from gox_plate_pipeline.fog import build_round_averaged_fog, build_round_gox_traceability  # noqa: E402

    fog_df = build_round_averaged_fog(run_round_map, Path(args.processed_dir))
    args.out_fog.parent.mkdir(parents=True, exist_ok=True)
    fog_df.to_csv(args.out_fog, index=False)
    print(f"Saved: {args.out_fog} ({len(fog_df)} rows)")

    gox_out = args.out_fog.parent / "fog_round_gox_traceability.csv"
    gox_df = build_round_gox_traceability(run_round_map, Path(args.processed_dir))
    gox_df.to_csv(gox_out, index=False)
    print(f"Saved (GOx traceability): {gox_out} ({len(gox_df)} rows)")


if __name__ == "__main__":
    main()
