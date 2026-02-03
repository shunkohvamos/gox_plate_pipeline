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


def _resolve_from_repo_root(p: str | Path, repo_root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Synergy H1 export (tab-delimited text; *.csv OK)")
    parser.add_argument("--row_map", required=True, help="Row map TSV (may include plate column)")
    parser.add_argument("--config", default="meta/config.yml", help="Config YAML")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--out_prefix", default=None, help="Output prefix (default: raw file stem)")
    args = parser.parse_args()

    # scripts/ 配下から1つ上がプロジェクトルート
    REPO_ROOT = Path(__file__).resolve().parents[1]

    raw_path = _resolve_from_repo_root(args.raw, REPO_ROOT)
    row_map_path = _resolve_from_repo_root(args.row_map, REPO_ROOT)
    config_path = _resolve_from_repo_root(args.config, REPO_ROOT)
    out_dir = _resolve_from_repo_root(args.out_dir, REPO_ROOT)

    if not raw_path.exists():
        raise FileNotFoundError(f"--raw not found: {raw_path}\nExisting files in data/raw:\n{list((REPO_ROOT/'data/raw').glob('*'))}")

    prefix = args.out_prefix if args.out_prefix else raw_path.stem

    config = load_yaml(config_path)
    tidy = extract_tidy_from_synergy_export(raw_path, config=config)

    row_map = read_row_map_tsv(row_map_path)
    tidy = attach_row_map(tidy, row_map=row_map)

    # 解析対象だけ残す（polymer_id が空の行は除外）
    if "polymer_id" in tidy.columns:
        tidy = tidy[tidy["polymer_id"].astype(str).str.strip().ne("")].copy()

    # 解析csvに生データファイル名を載せる（run_idの代わり）
    tidy["source_file"] = raw_path.name

    # 生データごと・実行段階ごとにフォルダ分け: processed/{prefix}/extract/tidy.csv, wide.csv
    extract_dir = out_dir / prefix / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = extract_dir / "tidy.csv"
    wide_path = extract_dir / "wide.csv"

    tidy.to_csv(tidy_path, index=False)
    wide = tidy.pivot_table(index=["plate_id", "time_s"], columns="well", values="signal", aggfunc="first")
    wide.reset_index().to_csv(wide_path, index=False)

    print(f"Saved: {tidy_path}")
    print(f"Saved: {wide_path}")


if __name__ == "__main__":
    main()
