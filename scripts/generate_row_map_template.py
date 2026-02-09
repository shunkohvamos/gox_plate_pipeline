"""
Generate a row-map TSV template from raw Synergy H1 export.

- Detects number of plates and rows (A, B, ...) from the raw file(s).
- Writes data/meta/{run_id}.tsv with columns: plate, row, polymer_id, sample_name, use_for_bo.
- polymer_id and sample_name are empty for you to fill in.
- use_for_bo defaults to True (include in Bayesian optimization).
  Set to False for background wells (e.g., "without GOx" entries) that should be fitted
  but not used in BO learning data.

Usage:
  With --raw (file):   generate template for that file (overwrites if exists).
  With --raw (folder): generate one template for all CSVs in that folder (same run_id).
  Without --raw:       generate for raw folders first, then legacy raw files, when TSV is missing.

If a CSV file starts with N- (e.g. 2-sample.csv), inferred plate IDs are remapped
to start from plateN (plateN, plateN+1, ...). This matches extract_clean_csv.py behavior.

Note: Rows with empty polymer_id are excluded from fitting. To include a well in fitting,
enter a polymer_id (e.g., "without GOx" for background measurements).

Run from "Run and Debug" (one-click) or after adding a new raw file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.loader import infer_plate_row_from_synergy  # noqa: E402
from gox_plate_pipeline.raw_bundle import (  # noqa: E402
    remap_plate_row_pairs_for_file,
    sort_plate_row_pairs,
)


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _write_template(pairs: list[tuple[str, str]], out_path: Path, repo_root: Path) -> bool:
    if not pairs:
        return False
    sorted_pairs = sort_plate_row_pairs(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["plate\trow\tpolymer_id\tsample_name\tuse_for_bo"]
    for plate_id, row in sorted_pairs:
        lines.append(f"{plate_id}\t{row}\t\t\tTrue")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        disp = str(out_path.relative_to(repo_root))
    except Exception:
        disp = str(out_path)
    print(f"  Wrote: {disp} ({len(sorted_pairs)} rows)")
    return True


def generate_one(raw_path: Path, out_path: Path, repo_root: Path) -> bool:
    """Generate TSV template for one raw file OR one raw folder. Returns True if written."""
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        csvs = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            print(f"  Skip (no CSV in folder): {raw_path.name}")
            return False

        all_pairs: list[tuple[str, str]] = []
        for csv_path in csvs:
            pairs = infer_plate_row_from_synergy(csv_path)
            if not pairs:
                continue
            all_pairs.extend(remap_plate_row_pairs_for_file(pairs, raw_file=csv_path))

        if not all_pairs:
            print(f"  Skip (no plate/row detected): {raw_path.name}")
            return False
        return _write_template(all_pairs, out_path, repo_root)

    pairs = infer_plate_row_from_synergy(raw_path)
    if not pairs:
        print(f"  Skip (no plate/row detected): {raw_path.name}")
        return False
    return _write_template(pairs, out_path, repo_root)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate row-map TSV template from raw Synergy export.")
    p.add_argument(
        "--raw",
        default=None,
        help="Path to raw CSV or raw folder. If omitted, generate templates for new raw folders (preferred) and legacy raw CSV files.",
    )
    p.add_argument(
        "--out_dir",
        default="data/meta",
        help="Directory for output TSV (default: data/meta).",
    )
    args = p.parse_args()

    out_dir = _resolve(args.out_dir, REPO_ROOT)
    raw_dir = REPO_ROOT / "data" / "raw"

    if args.raw is not None:
        raw_path = _resolve(args.raw, REPO_ROOT)
        if not raw_path.exists():
            raise FileNotFoundError(f"--raw not found: {raw_path}")
        stem = raw_path.stem
        out_path = out_dir / f"{stem}.tsv"
        generate_one(raw_path, out_path, REPO_ROOT)
        return

    # No --raw: preferred raw folder layout first
    if not raw_dir.is_dir():
        print("No data/raw directory.")
        return

    written = 0

    seen_run_ids: set[str] = set()
    for raw_folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        csvs = sorted(p for p in raw_folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            continue
        run_id = raw_folder.name
        tsv_path = out_dir / f"{run_id}.tsv"
        seen_run_ids.add(run_id)
        if tsv_path.exists():
            continue
        if generate_one(raw_folder, tsv_path, REPO_ROOT):
            written += 1

    # Legacy layout: data/raw/*.csv
    for raw_path in sorted(raw_dir.glob("*.csv")):
        run_id = raw_path.stem
        if run_id in seen_run_ids:
            continue
        tsv_path = out_dir / f"{run_id}.tsv"
        if tsv_path.exists():
            continue
        if generate_one(raw_path, tsv_path, REPO_ROOT):
            written += 1

    if written == 0:
        print("No new raw folders/files without a TSV template.")


if __name__ == "__main__":
    main()
