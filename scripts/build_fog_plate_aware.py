#!/usr/bin/env python3
"""
Build FoG with denominator rule: same plate GOx → same round GOx.

- Reads run_round_map and rates_with_rea per run; computes per-plate t50 from REA vs heat_min.
- For each (run, plate, polymer): gox_t50 = GOx from same (run, plate) if present, else mean GOx in that round.
- Outputs: fog_plate_aware.csv (per row), fog_plate_aware_round_averaged.csv, fog_round_gox_traceability.csv.
- Fit rates+REA must have been run for all round-associated runs beforehand (this script does not run fit).

Usage:
  python scripts/build_fog_plate_aware.py --run_round_map meta/bo_run_round_map.tsv --processed_dir data/processed --out_dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import load_run_round_map  # noqa: E402
from gox_plate_pipeline.fog import build_fog_plate_aware  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build FoG with same_plate → same_round denominator rule.",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=REPO_ROOT / "meta" / "bo_run_round_map.tsv",
        help="Path to run_id→round_id map (TSV/YAML/CSV).",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed directory (each run_id/fit/rates_with_rea.csv).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_plate_aware",
        help="Output directory for fog_plate_aware.csv, fog_plate_aware_round_averaged.csv, fog_round_gox_traceability.csv, warnings.md.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only check inputs and list what would be written; do not write files.",
    )
    p.add_argument(
        "--exclude_outlier_gox",
        action="store_true",
        help="Exclude outlier GOx t50 values from round average calculation. "
        "Outliers are detected using median-based thresholds (default: < median*0.33 or > median*3.0).",
    )
    p.add_argument(
        "--gox_outlier_low_threshold",
        type=float,
        default=0.33,
        help="Lower threshold multiplier for GOx t50 outlier detection (default: 0.33). "
        "Values below median * this threshold are considered outliers.",
    )
    p.add_argument(
        "--gox_outlier_high_threshold",
        type=float,
        default=3.0,
        help="Upper threshold multiplier for GOx t50 outlier detection (default: 3.0). "
        "Values above median * this threshold are considered outliers.",
    )
    p.add_argument(
        "--disable_gox_guard",
        action="store_true",
        help="Disable denominator guard for same_plate GOx (default: enabled).",
    )
    p.add_argument(
        "--gox_guard_low_threshold",
        type=float,
        default=None,
        help="Lower threshold multiplier for same_plate guard. Default: same as --gox_outlier_low_threshold.",
    )
    p.add_argument(
        "--gox_guard_high_threshold",
        type=float,
        default=None,
        help="Upper threshold multiplier for same_plate guard. Default: same as --gox_outlier_high_threshold.",
    )
    p.add_argument(
        "--gox_round_fallback_stat",
        type=str,
        default="median",
        choices=["median", "mean", "trimmed_mean"],
        help="Representative round GOx used for same_round fallback (default: median).",
    )
    p.add_argument(
        "--gox_round_trimmed_mean_proportion",
        type=float,
        default=0.1,
        help="Proportion to trim from each tail when using trimmed_mean (default: 0.1).",
    )
    args = p.parse_args()

    if not args.run_round_map.is_file():
        raise FileNotFoundError(f"Run-round map not found: {args.run_round_map}")

    run_round_map = load_run_round_map(Path(args.run_round_map))
    if not run_round_map:
        print("No run_id with valid round_id in run_round_map.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        from gox_plate_pipeline.fog import build_round_gox_traceability  # noqa: E402
        round_to_runs = {}
        for rid, oid in run_round_map.items():
            oid = str(oid).strip()
            if not oid or oid.upper() in ("—", "NA", "NAN"):
                continue
            round_to_runs.setdefault(oid, []).append(str(rid).strip())
        print("Rounds and runs:", round_to_runs)
        for round_id, run_ids in round_to_runs.items():
            for run_id in run_ids:
                path = args.processed_dir / run_id / "fit" / "rates_with_rea.csv"
                print(f"  {round_id} / {run_id}: rates_with_rea {'OK' if path.is_file() else 'MISSING'}")
        print("Would write:")
        print(f"  {args.out_dir / 'fog_plate_aware.csv'}")
        print(f"  {args.out_dir / 'fog_plate_aware_round_averaged.csv'}")
        print(f"  {args.out_dir / 'fog_round_gox_traceability.csv'}")
        print(f"  {args.out_dir / 'warnings.md'}")
        print(f"  {args.out_dir / 'README.md'}")
        return

    per_row_df, round_averaged_df, gox_trace_df, warning_info = build_fog_plate_aware(
        run_round_map,
        Path(args.processed_dir),
        exclude_outlier_gox=args.exclude_outlier_gox,
        gox_outlier_low_threshold=args.gox_outlier_low_threshold,
        gox_outlier_high_threshold=args.gox_outlier_high_threshold,
        gox_guard_same_plate=not args.disable_gox_guard,
        gox_guard_low_threshold=args.gox_guard_low_threshold,
        gox_guard_high_threshold=args.gox_guard_high_threshold,
        gox_round_fallback_stat=args.gox_round_fallback_stat,
        gox_round_trimmed_mean_proportion=args.gox_round_trimmed_mean_proportion,
    )

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_per = args.out_dir / "fog_plate_aware.csv"
    out_round = args.out_dir / "fog_plate_aware_round_averaged.csv"
    out_gox = args.out_dir / "fog_round_gox_traceability.csv"
    out_warning = args.out_dir / "warnings.md"
    out_readme = args.out_dir / "README.md"

    per_row_df.to_csv(out_per, index=False)
    print(f"Saved: {out_per} ({len(per_row_df)} rows)")

    round_averaged_df.to_csv(out_round, index=False)
    print(f"Saved: {out_round} ({len(round_averaged_df)} rows)")

    gox_trace_df.to_csv(out_gox, index=False)
    print(f"Saved (GOx traceability): {out_gox} ({len(gox_trace_df)} rows)")

    # Write warning file if there are any warnings
    if warning_info.outlier_gox or warning_info.guarded_same_plate or warning_info.missing_rates_files:
        from gox_plate_pipeline.fog import write_fog_warning_file  # noqa: E402
        write_fog_warning_file(warning_info, out_warning, exclude_outlier_gox=args.exclude_outlier_gox)
        print(f"Saved (warnings): {out_warning}")

    # Write README.md
    readme_content = """# Plate-aware FoG 計算結果

このフォルダには、「FoG（同一プレート→同一ラウンド）計算」を実行した結果が保存されます。

## ファイル一覧

- **fog_plate_aware.csv**: 各ポリマーのFoG値（詳細）
  - 列: `round_id`, `run_id`, `plate_id`, `polymer_id`, `t50_min`, `gox_t50_used_min`, `denominator_source`, `fog`, `log_fog`
  - `denominator_source`: `same_plate`（同じplateのGOxを使用）または `same_round`（round平均GOxを使用）

- **fog_plate_aware_round_averaged.csv**: Round平均FoG値
  - 列: `round_id`, `polymer_id`, `mean_fog`, `mean_log_fog`, `robust_fog`, `robust_log_fog`, `log_fog_mad`, `n_observations`, `run_ids`
  - 同じround内の同じpolymer_idのFoG値を平均化

- **fog_round_gox_traceability.csv**: GOxの追跡可能性情報
  - 列: `round_id`, `run_id`, `heat_min`, `plate_id`, `well`, `abs_activity`, `REA_percent`
  - 各roundで使用されたGOxの詳細情報

- **warnings.md**: 警告情報（警告がある場合のみ生成）
  - 異常GOx t50値の検出情報
  - 見つからない`rates_with_rea.csv`の情報
  - 「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」の実行方法

## 計算方法

- **分母の選択ルール**: 同じplateのGOx → 同じroundのGOx
  - 各`(run_id, plate_id, polymer_id)`について：
    - 同じ`(run_id, plate_id)`にGOxがある場合: そのGOx t50を使用（`denominator_source = "same_plate"`）
      - ただし、same_plate GOxが異常値閾値を外れる場合は `same_round` にフォールバック（分母ガード）
    - 同じplateにGOxがない場合: round平均GOx t50を使用（`denominator_source = "same_round"`）

- **Round平均GOx t50**: すべての`(run, plate)`のGOx t50を単純平均（各plateが等しい重み）

- **Round平均FoG**: 各`(round_id, polymer_id)`について、FoG値を平均化

## 関連する実行設定

- **「FoG（同一プレート→同一ラウンド）計算」**: 通常実行（警告のみ、異常値は除外しない）
- **「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」**: 異常GOx t50値を除外して実行

## 次のステップ

Round平均FoGが生成されたら、BO学習データを作成できます：
- **「BO学習データ作成（Plate-aware Round平均FoG）」**を実行
"""
    with open(out_readme, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved (README): {out_readme}")


if __name__ == "__main__":
    main()
