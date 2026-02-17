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
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import (
    build_round_coverage_summary,
    file_fingerprint,
    load_bo_catalog,
    load_run_round_map,
    collect_fog_summary_paths,
    build_bo_learning_data,
    build_bo_learning_data_from_round_averaged,
    write_bo_learning_csv,
    write_exclusion_report,
)
from gox_plate_pipeline.summary import build_run_manifest_dict


def _default_trace_run_id(prefix: str) -> str:
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def _add_run_id_column_if_requested(df, *, trace_run_id: str, include_run_id_column: bool):
    if not include_run_id_column:
        return df
    out = df.copy()
    if "run_id" in out.columns:
        return out
    out.insert(0, "run_id", trace_run_id)
    return out


def _build_bo_learning_lineage(df, *, trace_run_id: str):
    """
    Build a compatibility-safe lineage table without changing the main learning CSV schema.
    One row per source run used for each learning row.
    """
    cols = [
        "run_id",
        "lineage_row_id",
        "polymer_id",
        "round_id",
        "source_run_id",
        "source_n_observations",
        "source_objective_source",
    ]
    if df is None or len(df) == 0:
        return __import__("pandas").DataFrame(columns=cols)

    import pandas as pd

    work = df.reset_index(drop=True).copy()
    rows: list[dict] = []
    for idx, row in work.iterrows():
        polymer_id = str(row.get("polymer_id", ""))
        round_id = str(row.get("round_id", "")) if "round_id" in work.columns else ""
        n_obs = row.get("n_observations", "")
        obj_src = str(row.get("objective_source", "")) if "objective_source" in work.columns else ""

        source_ids: list[str] = []
        if "run_id" in work.columns and str(row.get("run_id", "")).strip():
            source_ids = [str(row.get("run_id", "")).strip()]
        elif "run_ids" in work.columns and str(row.get("run_ids", "")).strip():
            source_ids = [s.strip() for s in str(row.get("run_ids", "")).split(",") if s.strip()]

        if not source_ids:
            source_ids = [""]

        for source_run_id in source_ids:
            rows.append(
                {
                    "run_id": trace_run_id,
                    "lineage_row_id": int(idx),
                    "polymer_id": polymer_id,
                    "round_id": round_id,
                    "source_run_id": source_run_id,
                    "source_n_observations": n_obs,
                    "source_objective_source": obj_src,
                }
            )

    return pd.DataFrame(rows, columns=cols)


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
    p.add_argument(
        "--trace_run_id",
        type=str,
        default=None,
        help="Traceability run_id for manifest filename/content. Default: timestamp-based bo_learning_*.",
    )
    p.add_argument(
        "--no_manifest",
        action="store_true",
        help="Skip writing bo_learning_manifest__*.json (default: write).",
    )
    p.add_argument(
        "--include_run_id_column",
        action="store_true",
        help="Add run_id column to output CSVs (compat mode off by default).",
    )
    p.add_argument(
        "--write_lineage_csv",
        action="store_true",
        help="Deprecated compatibility flag. Lineage CSV is now written by default.",
    )
    p.add_argument(
        "--no_lineage_csv",
        action="store_true",
        help="Disable bo_learning_lineage CSV output.",
    )
    p.add_argument(
        "--allow_unmapped_round_ids",
        action="store_true",
        help=(
            "Allow round IDs in --fog_round_averaged that are not present in --run_round_map. "
            "Default is strict (raise error on mismatch)."
        ),
    )
    p.add_argument(
        "--lineage_out",
        type=Path,
        default=None,
        help="Optional explicit path for bo_learning_lineage CSV.",
    )
    args = p.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"BO catalog not found: {catalog_path}")

    catalog_df = load_bo_catalog(catalog_path, validate_sum=True)

    input_paths: list[Path] = [catalog_path]
    mode = "per_run"
    fog_paths: list[Path] = []
    run_round_map = None
    run_round_map_path: Path | None = None
    if args.run_round_map is not None:
        run_round_map_path = Path(args.run_round_map)
        input_paths.append(run_round_map_path)
        if not run_round_map_path.is_file():
            raise FileNotFoundError(f"Run-round map not found: {run_round_map_path}")
        run_round_map = load_run_round_map(run_round_map_path)
    if args.fog_round_averaged is not None:
        mode = "round_averaged"
        input_paths.append(Path(args.fog_round_averaged))
        if not args.fog_round_averaged.is_file():
            raise FileNotFoundError(f"Round-averaged FoG not found: {args.fog_round_averaged}")
        if run_round_map is None:
            print(
                "Warning: --fog_round_averaged was used without --run_round_map. "
                "Round coverage provenance checks are skipped.",
                file=sys.stderr,
            )
        learning_df, excluded_df = build_bo_learning_data_from_round_averaged(
            catalog_df,
            Path(args.fog_round_averaged),
            run_round_map=run_round_map,
            strict_round_coverage=not bool(args.allow_unmapped_round_ids),
        )
    else:
        run_ids = args.run_ids
        if run_round_map is not None:
            run_ids = list(run_round_map.keys())

        fog_paths = collect_fog_summary_paths(Path(args.processed_dir), run_ids=run_ids)

        if not fog_paths:
            print("No fog_summary CSV found under processed_dir (for given run_ids/run_round_map).", file=sys.stderr)
            sys.exit(1)
        input_paths.extend(fog_paths)

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

    trace_run_id = str(args.trace_run_id).strip() if args.trace_run_id else _default_trace_run_id("bo_learning")
    learning_to_write = _add_run_id_column_if_requested(
        learning_df,
        trace_run_id=trace_run_id,
        include_run_id_column=bool(args.include_run_id_column),
    )
    excluded_to_write = _add_run_id_column_if_requested(
        excluded_df,
        trace_run_id=trace_run_id,
        include_run_id_column=bool(args.include_run_id_column),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bo_learning_csv(learning_to_write, out_path)
    print(f"Saved: {out_path} ({len(learning_df)} rows)")

    if excluded_path is not None:
        excluded_path = Path(excluded_path)
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        write_exclusion_report(excluded_to_write, excluded_path)
        print(f"Saved (excluded): {excluded_path} ({len(excluded_df)} rows)")

    lineage_path = None
    write_lineage_csv = (not bool(args.no_lineage_csv)) or bool(args.write_lineage_csv)
    if write_lineage_csv or args.lineage_out is not None:
        lineage_path = Path(args.lineage_out) if args.lineage_out is not None else out_path.parent / f"bo_learning_lineage__{trace_run_id}.csv"
        lineage_df = _build_bo_learning_lineage(learning_df, trace_run_id=trace_run_id)
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
        lineage_df.to_csv(lineage_path, index=False)
        print(f"Saved (lineage): {lineage_path} ({len(lineage_df)} rows)")

    round_coverage_summary = None
    run_round_map_meta_path = None
    if run_round_map is not None:
        observed_round_ids = (
            learning_df["round_id"].tolist()
            if "round_id" in learning_df.columns
            else pd.Series([], dtype=object).tolist()
        )
        round_coverage_summary = build_round_coverage_summary(observed_round_ids, run_round_map)
        run_round_map_meta_path = out_path.parent / f"run_round_map_meta__{trace_run_id}.json"
        run_round_map_meta_payload = {
            "run_id": trace_run_id,
            "mode": mode,
            "run_round_map_file": file_fingerprint(run_round_map_path) if run_round_map_path is not None else None,
            "round_coverage": round_coverage_summary,
        }
        with open(run_round_map_meta_path, "w", encoding="utf-8") as f:
            json.dump(run_round_map_meta_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved (run-round map meta): {run_round_map_meta_path}")

    if not bool(args.no_manifest):
        manifest_path = out_path.parent / f"bo_learning_manifest__{trace_run_id}.json"
        output_files = [out_path]
        if excluded_path is not None:
            output_files.append(Path(excluded_path))
        if lineage_path is not None:
            output_files.append(Path(lineage_path))
        if run_round_map_meta_path is not None:
            output_files.append(Path(run_round_map_meta_path))
        manifest = build_run_manifest_dict(
            run_id=trace_run_id,
            input_paths=input_paths,
            git_root=REPO_ROOT,
            extra={
                "operation": "build_bo_learning_data",
                "mode": mode,
                "n_learning_rows": int(len(learning_df)),
                "n_excluded_rows": int(len(excluded_df)),
                "strict_round_coverage": not bool(args.allow_unmapped_round_ids),
                "round_coverage": round_coverage_summary,
                "output_files": [p.name for p in output_files],
                "cli_args": sys.argv[1:],
            },
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved (manifest): {manifest_path}")

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
