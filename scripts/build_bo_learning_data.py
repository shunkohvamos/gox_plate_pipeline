#!/usr/bin/env python3
"""
Build BO learning CSV from BO catalog (BMA composition) and FoG summaries.

- Joins BO catalog with fog_summary__{run_id}.csv; only catalog polymers (BMA-only).
- X columns: frac_MPC, frac_BMA, frac_MTAC (order fixed). y: log_fog.
- Excluded rows (missing log_fog / not in catalog) are written to an exclusion report.

Usage:
  python scripts/build_bo_learning_data.py --catalog meta/bo_catalog_bma.csv --processed_dir data/processed --out data/processed/bo_learning.csv --exclusion_report data/processed/bo_learning_excluded.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import (
    load_bo_catalog,
    load_run_round_map,
    collect_fog_summary_paths,
    build_bo_learning_data,
    build_bo_learning_data_from_round_averaged,
    write_bo_learning_csv,
    write_exclusion_report,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build BO learning CSV from BO catalog (BMA) and FoG summaries.",
    )
    p.add_argument("--catalog", required=True, type=Path, help="Path to BO catalog CSV (polymer_id, frac_MPC, frac_BMA, frac_MTAC; optional round_id, x, y).")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory containing {run_id}/fit/fog_summary__{run_id}.csv",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=None,
        help="Path to run_id→round_id map (YAML or CSV). When given, only these runs are used and each row gets round_id from the run.",
    )
    p.add_argument(
        "--fog_round_averaged",
        type=Path,
        default=None,
        help="Path to round-averaged FoG CSV (round_id, polymer_id, mean_fog, mean_log_fog, ...). When given, BO learning data is built from this (one row per round_id, polymer_id) instead of per-run fog summaries.",
    )
    p.add_argument("--run_ids", nargs="*", default=None, help="Optional list of run_id to include. If omitted (and no --run_round_map), all runs under processed_dir are used.")
    p.add_argument("--out", type=Path, default=None, help="Output path for BO learning CSV. Default: bo_learning/bo_learning.csv or bo_learning/bo_learning_plate_aware.csv depending on input.")
    p.add_argument("--exclusion_report", type=Path, default=None, help="Output path for exclusion report CSV. If omitted, report is not written.")
    args = p.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"BO catalog not found: {catalog_path}")

    catalog_df = load_bo_catalog(catalog_path, validate_sum=True)

    if args.fog_round_averaged is not None:
        if not args.fog_round_averaged.is_file():
            raise FileNotFoundError(f"Round-averaged FoG not found: {args.fog_round_averaged}")
        learning_df, excluded_df = build_bo_learning_data_from_round_averaged(catalog_df, Path(args.fog_round_averaged))
    else:
        run_round_map = None
        run_ids = args.run_ids
        if args.run_round_map is not None:
            if not args.run_round_map.is_file():
                raise FileNotFoundError(f"Run-round map not found: {args.run_round_map}")
            run_round_map = load_run_round_map(Path(args.run_round_map))
            run_ids = list(run_round_map.keys())

        fog_paths = collect_fog_summary_paths(Path(args.processed_dir), run_ids=run_ids)

        if not fog_paths:
            print("No fog_summary CSV found under processed_dir (for given run_ids/run_round_map).", file=sys.stderr)
            sys.exit(1)

        learning_df, excluded_df = build_bo_learning_data(catalog_df, fog_paths, run_round_map=run_round_map)

    # Determine output paths
    if args.out is None:
        # Default: bo_learning/ folder
        bo_learning_dir = REPO_ROOT / "data" / "processed" / "bo_learning"
        bo_learning_dir.mkdir(parents=True, exist_ok=True)
        if args.fog_round_averaged is not None:
            # Plate-aware or regular round-averaged
            if "plate_aware" in str(args.fog_round_averaged):
                out_path = bo_learning_dir / "bo_learning_plate_aware.csv"
                excluded_path = bo_learning_dir / "bo_learning_excluded_plate_aware.csv"
            else:
                out_path = bo_learning_dir / "bo_learning.csv"
                excluded_path = bo_learning_dir / "bo_learning_excluded.csv"
        else:
            # Per-run fog summaries
            out_path = bo_learning_dir / "bo_learning.csv"
            excluded_path = bo_learning_dir / "bo_learning_excluded.csv"
    else:
        out_path = Path(args.out)
        excluded_path = args.exclusion_report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bo_learning_csv(learning_df, out_path)
    print(f"Saved: {out_path} ({len(learning_df)} rows)")

    if excluded_path is not None:
        excluded_path = Path(excluded_path)
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        write_exclusion_report(excluded_df, excluded_path)
        print(f"Saved (excluded): {excluded_path} ({len(excluded_df)} rows)")

    # Write README.md if output is in bo_learning directory
    if args.out is None or "bo_learning" in str(args.out):
        readme_path = out_path.parent / "README.md"
        readme_content = """# BO学習データ

このフォルダには、Bayesian Optimization用の学習データが保存されます。

## ファイル一覧

- **bo_learning.csv**: BO学習データ（従来版、round-averaged FoGから生成）
  - 列: `polymer_id`, `round_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`, `x`, `y`, `log_fog`, `run_ids`, `n_observations`
  - 入力: `fog_round_averaged/fog_round_averaged.csv`

- **bo_learning_excluded.csv**: 除外されたデータ（従来版）
  - 列: `round_id`, `polymer_id`, `reason`
  - 除外理由: `not_in_bo_catalog`, `log_fog_missing_or_invalid` など

- **bo_learning_plate_aware.csv**: BO学習データ（plate-aware版）
  - 列: `polymer_id`, `round_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`, `x`, `y`, `log_fog`, `run_ids`, `n_observations`
  - 入力: `fog_plate_aware/fog_plate_aware_round_averaged.csv`

- **bo_learning_excluded_plate_aware.csv**: 除外されたデータ（plate-aware版）

## 生成方法

### 従来版（round-averaged FoGから）

**「BO学習データ作成（Round平均FoG）」**を実行

または、コマンドラインから：
```bash
python scripts/build_bo_learning_data.py \\
  --catalog meta/bo_catalog_bma.csv \\
  --run_round_map meta/bo_run_round_map.tsv \\
  --fog_round_averaged data/processed/fog_round_averaged/fog_round_averaged.csv
```

### Plate-aware版

**「BO学習データ作成（Plate-aware Round平均FoG）」**を実行

または、コマンドラインから：
```bash
python scripts/build_bo_learning_data.py \\
  --catalog meta/bo_catalog_bma.csv \\
  --run_round_map meta/bo_run_round_map.tsv \\
  --fog_round_averaged data/processed/fog_plate_aware/fog_plate_aware_round_averaged.csv
```

## データ形式

- **X列**: `frac_MPC`, `frac_BMA`, `frac_MTAC`（組成比、合計=1.0）
- **y列**: `log_fog`（log(FoG)）
- **round_id**: 実験ラウンドID
- **lineage**: `run_ids`, `n_observations`（追跡可能性情報）

## 除外条件

以下の条件で除外されます：
- BO catalogに含まれていないpolymer_id
- `log_fog`が欠損または無効な値
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"Saved (README): {readme_path}")


if __name__ == "__main__":
    main()
