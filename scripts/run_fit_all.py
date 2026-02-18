#!/usr/bin/env python3
"""
Run fit_initial_rates.py for all runs.

By default:
  - Targets all discovered raw/meta pairs plus runs that already have processed/{run_id}/extract/tidy.csv.
  - If tidy.csv is missing and raw/meta pair exists, runs extract_clean_csv.py first.
  - Runs fit_initial_rates.py unless fog_summary__{run_id}.csv exists (unless --force_fit).

Usage:
  python scripts/run_fit_all.py --t50_definition y0_half|rea50 [--dry_run] [--force_fit]
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

from gox_plate_pipeline.meta_paths import get_meta_paths  # noqa: E402

META = get_meta_paths(REPO_ROOT)


def _resolve_from_repo_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _repo_rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _discover_raw_datasets(repo_root: Path) -> list[tuple[str, Path, Path]]:
    raw_dir = repo_root / "data" / "raw"
    row_maps_dir = get_meta_paths(repo_root).row_maps_dir
    if not raw_dir.is_dir():
        return []

    pairs: list[tuple[str, Path, Path]] = []
    seen_run_ids: set[str] = set()

    for raw_path in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        csvs = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            continue
        run_id = raw_path.name
        row_map = row_maps_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = row_maps_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))
            seen_run_ids.add(run_id)

    for raw_path in sorted(raw_dir.glob("*.csv")):
        run_id = raw_path.stem
        if run_id in seen_run_ids:
            continue
        row_map = row_maps_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = row_maps_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))

    return pairs


def _discover_runs_with_tidy(processed_dir: Path) -> list[str]:
    out: list[str] = []
    if not processed_dir.is_dir():
        return out
    for d in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        tidy = d / "extract" / "tidy.csv"
        if tidy.is_file():
            out.append(d.name)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run fit_initial_rates.py for all runs.")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed root directory.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=META.config,
        help="Path to config.yml.",
    )
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition passed to fit_initial_rates.py.",
    )
    p.add_argument(
        "--native_activity_min_rel",
        type=float,
        default=0.70,
        help=(
            "Native-activity feasibility threshold passed to fit_initial_rates.py "
            "(abs_activity_at_0 / same-run GOx abs_activity_at_0)."
        ),
    )
    p.add_argument(
        "--reference_polymer_id",
        type=str,
        default="GOX",
        help="Reference polymer ID passed to fit_initial_rates.py (default: GOX).",
    )
    p.add_argument(
        "--run_ids",
        nargs="*",
        default=None,
        help="Optional explicit run_id list. If omitted, all available runs are targeted.",
    )
    p.add_argument(
        "--force_fit",
        action="store_true",
        help="Re-run fit even if fog_summary__{run_id}.csv exists.",
    )
    p.add_argument(
        "--no_extract_if_missing",
        action="store_true",
        help="Do not run extract when tidy.csv is missing.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned actions.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose output.",
    )
    args = p.parse_args()

    config_path = _resolve_from_repo_root(args.config, REPO_ROOT)
    processed_dir = _resolve_from_repo_root(args.processed_dir, REPO_ROOT)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw_datasets = _discover_raw_datasets(REPO_ROOT)
    by_raw = {rid: (raw_path, row_map_path) for rid, raw_path, row_map_path in raw_datasets}
    runs_with_tidy = set(_discover_runs_with_tidy(processed_dir))
    all_runs = sorted(set(by_raw.keys()) | runs_with_tidy)
    if args.run_ids:
        target_runs = [str(r).strip() for r in args.run_ids if str(r).strip()]
    else:
        target_runs = all_runs

    extract_if_missing = not bool(args.no_extract_if_missing)
    to_extract: list[str] = []
    to_fit: list[str] = []
    skipped_no_input: list[str] = []
    skipped_fitted: list[str] = []

    for run_id in target_runs:
        tidy = processed_dir / run_id / "extract" / "tidy.csv"
        fog = processed_dir / run_id / "fit" / f"fog_summary__{run_id}.csv"

        if not tidy.is_file():
            if extract_if_missing and run_id in by_raw:
                to_extract.append(run_id)
            elif run_id not in by_raw:
                skipped_no_input.append(run_id)

        if fog.is_file() and not args.force_fit:
            skipped_fitted.append(run_id)
            continue
        to_fit.append(run_id)

    if args.dry_run or args.debug:
        print("Target runs:", target_runs)
        print("Will extract (missing tidy):", sorted(set(to_extract)))
        print("Will fit:", sorted(set(to_fit)))
        if skipped_fitted:
            print("Skip fit (already has fog_summary):", sorted(set(skipped_fitted)))
        if skipped_no_input:
            print("Skip (no tidy and no raw/meta pair):", sorted(set(skipped_no_input)))
        print("t50 definition for fit:", args.t50_definition)
        print("native activity threshold for fit:", float(args.native_activity_min_rel))
        print("reference polymer id for fit:", str(args.reference_polymer_id).strip() or "GOX")

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    extracted = 0
    for run_id in sorted(set(to_extract)):
        raw_row = by_raw.get(run_id)
        if raw_row is None:
            continue
        raw_path, row_map_path = raw_row
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "extract_clean_csv.py"),
            "--raw", _repo_rel_or_abs(raw_path, REPO_ROOT),
            "--row_map", _repo_rel_or_abs(row_map_path, REPO_ROOT),
            "--config", _repo_rel_or_abs(config_path, REPO_ROOT),
            "--out_dir", _repo_rel_or_abs(processed_dir, REPO_ROOT),
        ]
        if args.debug:
            print("Run extract:", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        extracted += 1

    fitted = 0
    for run_id in sorted(set(to_fit)):
        tidy = processed_dir / run_id / "extract" / "tidy.csv"
        if not tidy.is_file():
            print(f"Skip fit for {run_id}: tidy not found.", file=sys.stderr)
            continue
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fit_initial_rates.py"),
            "--tidy", _repo_rel_or_abs(tidy, REPO_ROOT),
            "--config", _repo_rel_or_abs(config_path, REPO_ROOT),
            "--out_dir", _repo_rel_or_abs(processed_dir, REPO_ROOT),
            "--write_well_plots", "0",
            "--t50_definition", str(args.t50_definition),
            "--native_activity_min_rel", str(float(args.native_activity_min_rel)),
            "--reference_polymer_id", str(args.reference_polymer_id).strip() or "GOX",
        ]
        if args.debug:
            print("Run fit:", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        fitted += 1

    print(f"Finished: extracted={extracted}, fitted={fitted}")


if __name__ == "__main__":
    main()
