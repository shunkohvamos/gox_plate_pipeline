#!/usr/bin/env python3
"""
Generate TSV to control include/exclude for cross-run group aggregation.

Output: meta/run_group_map.tsv
Columns:
  - run_id
  - group_id
  - include_in_group_mean
  - notes

Behavior:
  - Auto-discovers run_id from data/raw and data/processed.
  - Preserves existing group/include/notes values when TSV already exists.
  - New run_id rows are added with:
      group_id = date prefix (for convenience; editable)
      include_in_group_mean = True

This file is independent from:
  - data/meta/{run_id}.tsv (row map)
  - meta/bo_run_round_map.tsv (BO round assignment)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUN_ID_PATTERN = re.compile(r"^\d{6}(-[R]?\d+)?$")


def _is_valid_run_id(name: str) -> bool:
    return bool(RUN_ID_PATTERN.match(str(name)))


def _default_group_id(run_id: str) -> str:
    rid = str(run_id).strip()
    if "-" in rid:
        return rid.split("-", 1)[0]
    return rid


def discover_run_ids(repo_root: Path) -> list[str]:
    run_ids: set[str] = set()

    raw_dir = repo_root / "data" / "raw"
    if raw_dir.is_dir():
        for p in raw_dir.iterdir():
            if p.is_dir():
                csvs = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
                if csvs and _is_valid_run_id(p.name):
                    run_ids.add(p.name)
            elif p.is_file() and p.suffix.lower() == ".csv" and _is_valid_run_id(p.stem):
                run_ids.add(p.stem)

    processed_dir = repo_root / "data" / "processed"
    if processed_dir.is_dir():
        for p in processed_dir.iterdir():
            if p.is_dir() and _is_valid_run_id(p.name):
                run_ids.add(p.name)

    return sorted(run_ids)


def main() -> None:
    out_path = REPO_ROOT / "meta" / "run_group_map.tsv"
    legacy_path = REPO_ROOT / "meta" / "same_date_run_include.tsv"
    run_ids = discover_run_ids(REPO_ROOT)
    if not run_ids:
        print("No run_id found under data/raw or data/processed.", file=sys.stderr)
        sys.exit(1)

    existing_group: dict[str, str] = {}
    existing_include: dict[str, str] = {}
    existing_notes: dict[str, str] = {}
    source_path = out_path if out_path.is_file() else (legacy_path if legacy_path.is_file() else None)
    if source_path is not None:
        try:
            old = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False)
            if "run_id" in old.columns:
                group_col = None
                for c in ["group_id", "analysis_group_id", "date_prefix"]:
                    if c in old.columns:
                        group_col = c
                        break
                include_col = None
                for c in ["include_in_group_mean", "include_in_same_date_mean", "include", "enabled", "use"]:
                    if c in old.columns:
                        include_col = c
                        break
                notes_col = "notes" if "notes" in old.columns else None
                for _, row in old.iterrows():
                    rid = str(row.get("run_id", "")).strip()
                    if not rid:
                        continue
                    if group_col is not None:
                        g = str(row.get(group_col, "")).strip()
                        if g:
                            existing_group[rid] = g
                    if include_col is not None:
                        existing_include[rid] = str(row.get(include_col, "")).strip() or "True"
                    if notes_col is not None:
                        existing_notes[rid] = str(row.get(notes_col, "")).strip()
        except Exception as e:
            print(f"Warning: failed to read existing TSV ({source_path}): {e}", file=sys.stderr)

    all_run_ids = sorted(set(run_ids) | set(existing_group.keys()) | set(existing_include.keys()) | set(existing_notes.keys()))
    rows: list[dict[str, str]] = []
    for rid in all_run_ids:
        rows.append(
            {
                "run_id": rid,
                "group_id": existing_group.get(rid, _default_group_id(rid)),
                "include_in_group_mean": existing_include.get(rid, "True"),
                "notes": existing_notes.get(rid, ""),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["group_id", "run_id"], kind="mergesort").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path}")
    print(f"  {len(out_df)} run(s).")
    print("  Edit group_id to combine runs under the same analysis condition.")
    print("  Edit include_in_group_mean to True/False for each run_id.")
    print("  This TSV is separate from per-run row_map TSV and bo_run_round_map.tsv.")


if __name__ == "__main__":
    main()
