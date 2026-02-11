#!/usr/bin/env python3
"""
Output a TSV of all raw/processed run folders and their BO round correspondence.

- Scans data/raw (folder names + CSV stems) and data/processed (subdir names).
- Writes meta/bo_run_round_map.tsv with columns run_id, round_id.
- If meta/bo_run_round_map.yml or meta/bo_run_round_map.tsv already exists, fills in
  round_id for those runs; others get round_id = "—" (not used for BO).
- Round "—" or empty or NA means this run is not used for BO; build_bo_learning_data
  skips such runs when reading the TSV as run_round_map.

Usage: run from repo root (e.g. Run and Debug "全フォルダ–Round対応TSVを出力").
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import load_run_round_map  # noqa: E402

# Sentinel written for "not used for BO" so it's visible in the TSV
ROUND_NOT_USED_LABEL = "—"

# Pattern for valid run_id: YYMMDD followed by optional -R<number> or -<number>
# Examples: 260205-R1, 260209-1, 260205-R2
# Must start with 6 digits (date), optionally followed by hyphen and alphanumeric
RUN_ID_PATTERN = re.compile(r"^\d{6}(-[R]?\d+)?$")


def is_valid_run_id(name: str) -> bool:
    """
    Check if a name matches the run_id pattern.
    
    Valid run_id examples:
    - 260205-R1 (YYMMDD-RN)
    - 260209-1 (YYMMDD-N)
    - 260205-R2
    
    Invalid (excluded):
    - bo_learning (contains underscore)
    - bo_runs (contains underscore)
    - fog_plate_aware (contains underscore)
    - sim_proposed_logic_focus_plots_v2 (contains underscore)
    """
    return bool(RUN_ID_PATTERN.match(name))


def discover_run_ids(repo_root: Path) -> list[str]:
    """
    Collect all run_id from data/raw and data/processed, merged and sorted.
    
    Only includes names that match the run_id pattern (YYMMDD-RN or YYMMDD-N format).
    Excludes output folders like bo_learning, bo_runs, fog_plate_aware, etc.
    """
    run_ids: set[str] = set()

    raw_dir = repo_root / "data" / "raw"
    if raw_dir.is_dir():
        for p in raw_dir.iterdir():
            if p.is_dir():
                if any(f.suffix.lower() == ".csv" for f in p.iterdir() if f.is_file()):
                    if is_valid_run_id(p.name):
                        run_ids.add(p.name)
        for p in raw_dir.glob("*.csv"):
            if p.is_file():
                if is_valid_run_id(p.stem):
                    run_ids.add(p.stem)

    processed_dir = repo_root / "data" / "processed"
    if processed_dir.is_dir():
        for p in processed_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                if is_valid_run_id(p.name):
                    run_ids.add(p.name)

    return sorted(run_ids)


def main() -> None:
    repo_root = REPO_ROOT
    meta_dir = repo_root / "meta"
    out_path = meta_dir / "bo_run_round_map.tsv"

    run_ids = discover_run_ids(repo_root)
    if not run_ids:
        print("No run_id folders found under data/raw or data/processed.", file=sys.stderr)
        sys.exit(1)

    # Load existing map: prefer TSV (so re-run preserves your edits), then YAML, then CSV
    # Note: TSV is the primary format; YAML is legacy and may cause confusion if both exist.
    existing: dict[str, str] = {}
    yml_candidates = [meta_dir / "bo_run_round_map.yml", meta_dir / "bo_run_round_map.yaml"]
    yml_exists = any(p.is_file() for p in yml_candidates)
    if yml_exists:
        yml_path = next(p for p in yml_candidates if p.is_file())
        if out_path.is_file():
            print(f"Warning: Both TSV ({out_path.name}) and YAML ({yml_path.name}) exist. TSV will be used; consider removing {yml_path.name} to avoid confusion.", file=sys.stderr)
        else:
            print(f"Info: Found YAML ({yml_path.name}). Will convert to TSV format.", file=sys.stderr)
    for candidate in [out_path, *yml_candidates, meta_dir / "bo_run_round_map.csv"]:
        if candidate.is_file():
            try:
                existing = load_run_round_map(candidate)
            except Exception as e:
                print(f"Warning: could not load {candidate}: {e}", file=sys.stderr)
            break

    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("run_id\tround_id\n")
        for rid in run_ids:
            round_id = existing.get(rid) if existing.get(rid) else ROUND_NOT_USED_LABEL
            f.write(f"{rid}\t{round_id}\n")

    print(f"Saved: {out_path}")
    print(f"  {len(run_ids)} run(s). Round '—' or empty/NA = not used for BO.")
    used = sum(1 for rid in run_ids if existing.get(rid))
    if used:
        print(f"  {used} run(s) currently assigned to a round (from existing map).")
    print("  Edit round_id column (R1, R2, …) for runs you use for BO; leave '—' for others.")


if __name__ == "__main__":
    main()
