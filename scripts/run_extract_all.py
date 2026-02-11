#!/usr/bin/env python3
"""
Run extract_clean_csv.py for all discovered raw datasets.

Dataset discovery:
  - Preferred: data/raw/{run_id}/*.csv + data/meta/{run_id}.tsv (or {run_id}_row_map.tsv)
  - Legacy:    data/raw/{run_id}.csv   + data/meta/{run_id}.tsv (or {run_id}_row_map.tsv)

Usage:
  python scripts/run_extract_all.py [--dry_run] [--force_extract] [--debug]
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


def _resolve_from_repo_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _repo_rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _discover_datasets(repo_root: Path) -> list[tuple[str, Path, Path]]:
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
    p = argparse.ArgumentParser(description="Run extract_clean_csv.py for all discovered runs.")
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
        help="Path to config.yml (heat_times etc.).",
    )
    p.add_argument(
        "--run_ids",
        nargs="*",
        default=None,
        help="Optional explicit run_id list. If omitted, all discovered runs are targeted.",
    )
    p.add_argument(
        "--force_extract",
        action="store_true",
        help="Re-run extract even if processed/{run_id}/extract/tidy.csv already exists.",
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

    datasets = _discover_datasets(REPO_ROOT)
    if not datasets:
        print("No datasets discovered under data/raw with matching data/meta TSV.")
        return

    by_run = {rid: (raw_path, row_map_path) for rid, raw_path, row_map_path in datasets}
    if args.run_ids:
        target_runs = [str(r).strip() for r in args.run_ids if str(r).strip()]
    else:
        target_runs = sorted(by_run.keys())

    to_extract: list[str] = []
    skipped_existing: list[str] = []
    skipped_missing: list[str] = []
    for run_id in target_runs:
        if run_id not in by_run:
            skipped_missing.append(run_id)
            continue
        tidy_path = processed_dir / run_id / "extract" / "tidy.csv"
        if tidy_path.is_file() and not args.force_extract:
            skipped_existing.append(run_id)
            continue
        to_extract.append(run_id)

    if args.dry_run or args.debug:
        print("Discovered runs:", sorted(by_run.keys()))
        print("Target runs:", target_runs)
        print("Will extract:", to_extract)
        if skipped_existing:
            print("Skip (already extracted):", skipped_existing)
        if skipped_missing:
            print("Skip (missing raw/meta pair):", skipped_missing)

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    done = 0
    for run_id in to_extract:
        raw_path, row_map_path = by_run[run_id]
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
        done += 1

    print(f"Finished extract: {done} run(s).")


if __name__ == "__main__":
    main()
