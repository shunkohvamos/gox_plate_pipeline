from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gox_plate_pipeline.loader import (
    attach_row_map,
    extract_tidy_from_synergy_export,
    load_yaml,
    read_row_map_tsv,
)
from gox_plate_pipeline.raw_bundle import (
    derive_run_id_from_raw_input,
    list_raw_csv_files,
    remap_plate_ids_for_file,
)


def _resolve_from_repo_root(p: str | Path, repo_root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        required=True,
        help="Raw CSV file OR raw folder containing multiple CSV files to analyze together.",
    )
    parser.add_argument("--row_map", required=True, help="Row map TSV (may include plate column)")
    parser.add_argument("--config", default="meta/config.yml", help="Config YAML")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--out_prefix",
        default=None,
        help="Output prefix (default: raw folder name or raw file stem).",
    )
    args = parser.parse_args()

    # scripts/ 配下から1つ上がプロジェクトルート
    REPO_ROOT = Path(__file__).resolve().parents[1]

    raw_input = _resolve_from_repo_root(args.raw, REPO_ROOT)
    row_map_path = _resolve_from_repo_root(args.row_map, REPO_ROOT)
    config_path = _resolve_from_repo_root(args.config, REPO_ROOT)
    out_dir = _resolve_from_repo_root(args.out_dir, REPO_ROOT)

    if not raw_input.exists():
        raise FileNotFoundError(f"--raw not found: {raw_input}\nExisting entries in data/raw:\n{list((REPO_ROOT/'data/raw').glob('*'))}")

    prefix = args.out_prefix if args.out_prefix else derive_run_id_from_raw_input(raw_input)

    config = load_yaml(config_path)

    raw_files = list_raw_csv_files(raw_input)
    tidy_parts: list[pd.DataFrame] = []
    used_plate_ids: set[str] = set()
    for raw_file in raw_files:
        part = extract_tidy_from_synergy_export(raw_file, config=config)
        part, _ = remap_plate_ids_for_file(
            part,
            raw_file=raw_file,
            used_plate_ids=used_plate_ids,
        )
        part["source_file"] = raw_file.name
        tidy_parts.append(part)

    tidy = pd.concat(tidy_parts, ignore_index=True)

    row_map = read_row_map_tsv(row_map_path)
    tidy = attach_row_map(tidy, row_map=row_map)

    # 解析対象だけ残す（polymer_id が空の行は除外）
    if "polymer_id" in tidy.columns:
        tidy = tidy[tidy["polymer_id"].astype(str).str.strip().ne("")].copy()

    # Add run_id for traceability (core-rules: provenance)
    tidy["run_id"] = prefix

    # 生データごと・実行段階ごとにフォルダ分け: processed/{prefix}/extract/tidy.csv, wide.csv
    extract_dir = out_dir / prefix / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = extract_dir / "tidy.csv"
    wide_path = extract_dir / "wide.csv"

    tidy.to_csv(tidy_path, index=False)
    wide = tidy.pivot_table(index=["plate_id", "time_s"], columns="well", values="signal", aggfunc="first")
    wide = wide.reset_index()
    # Add run_id to wide format (preserve from tidy before pivot)
    # Since pivot_table loses run_id, we need to add it back
    # run_id is constant per run, so we can add it from prefix
    wide["run_id"] = prefix
    # Reorder columns to put run_id first for consistency
    cols = ["run_id"] + [c for c in wide.columns if c != "run_id"]
    wide = wide[cols]
    wide.to_csv(wide_path, index=False)

    print(f"Saved: {tidy_path}")
    print(f"Saved: {wide_path}")
    print(f"Raw files analyzed together ({len(raw_files)}): {[p.name for p in raw_files]}")


if __name__ == "__main__":
    main()
