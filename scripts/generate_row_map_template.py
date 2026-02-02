"""
Generate a row-map TSV template from raw Synergy H1 export.

- Detects number of plates and rows (A, B, ...) from the raw file.
- Writes data/meta/{stem}.tsv with columns: plate, row, polymer_id, sample_name.
- polymer_id and sample_name are empty for you to fill in.

Usage:
  With --raw: generate template for that file (overwrites if exists).
  Without --raw: generate for every data/raw/*.csv that does not yet have data/meta/{stem}.tsv.

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


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def generate_one(raw_path: Path, out_path: Path, repo_root: Path) -> bool:
    """Generate TSV template for one raw file. Returns True if written."""
    pairs = infer_plate_row_from_synergy(raw_path)
    if not pairs:
        print(f"  Skip (no plate/row detected): {raw_path.name}")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["plate\trow\tpolymer_id\tsample_name"]
    for plate_id, row in pairs:
        lines.append(f"{plate_id}\t{row}\t\t")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote: {out_path.relative_to(repo_root)} ({len(pairs)} rows)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Generate row-map TSV template from raw Synergy export.")
    p.add_argument(
        "--raw",
        default=None,
        help="Path to raw CSV. If omitted, generate for all data/raw/*.csv that lack data/meta/{stem}.tsv.",
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

    # No --raw: find all raw CSVs that don't have a matching TSV
    if not raw_dir.is_dir():
        print("No data/raw directory.")
        return
    written = 0
    for raw_path in sorted(raw_dir.glob("*.csv")):
        stem = raw_path.stem
        tsv_path = out_dir / f"{stem}.tsv"
        if tsv_path.exists():
            continue
        if generate_one(raw_path, tsv_path, REPO_ROOT):
            written += 1
    if written == 0:
        print("No new raw files without a TSV (all data/raw/*.csv already have data/meta/{stem}.tsv).")


if __name__ == "__main__":
    main()
