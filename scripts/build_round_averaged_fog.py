#!/usr/bin/env python3
"""
Build round-averaged FoG CSV from per-run fog_summary CSVs.

- Reads run_round_map (run_id → round_id). For each round, loads fog_summary__{run_id}.csv
  for all runs in that round.
- If a round has no GOx in any run, raises an error.
- Output: round_id, polymer_id, mean_fog, mean_log_fog, n_observations, run_ids.
  Same polymer_id in multiple runs within a round is averaged.
- Optionally writes GOx traceability CSV: round_id, run_id, heat_min, plate_id, well, abs_activity,
  REA_percent (all pre-averaged GOx values per round so which GOx was used is auditable).

Usage:
  python scripts/build_round_averaged_fog.py --run_round_map meta/bo_run_round_map.tsv --processed_dir data/processed --out data/processed/fog_round_averaged.csv [--out_gox_traceability data/processed/fog_round_gox_traceability.csv]
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
from gox_plate_pipeline.fog import build_round_averaged_fog, build_round_gox_traceability  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build round-averaged FoG CSV from per-run fog_summary CSVs.",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        required=True,
        help="Path to run_id→round_id map (YAML or TSV/CSV).",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory containing {run_id}/fit/fog_summary__{run_id}.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fog_round_averaged" / "fog_round_averaged.csv",
        help="Output path for round-averaged FoG CSV.",
    )
    p.add_argument(
        "--out_gox_traceability",
        type=Path,
        default=None,
        help="Path for GOx traceability CSV. Default: same dir as --out, name fog_round_gox_traceability.csv.",
    )
    args = p.parse_args()

    if not args.run_round_map.is_file():
        raise FileNotFoundError(f"Run-round map not found: {args.run_round_map}")

    run_round_map = load_run_round_map(Path(args.run_round_map))
    if not run_round_map:
        print("No run_id with valid round_id in run_round_map.", file=sys.stderr)
        sys.exit(1)

    df = build_round_averaged_fog(run_round_map, Path(args.processed_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({len(df)} rows)")

    # GOx traceability: same dir as --out, name fog_round_gox_traceability.csv (or explicit path)
    gox_out = args.out_gox_traceability if args.out_gox_traceability is not None else args.out.parent / "fog_round_gox_traceability.csv"
    gox_df = build_round_gox_traceability(run_round_map, Path(args.processed_dir))
    gox_out.parent.mkdir(parents=True, exist_ok=True)
    gox_df.to_csv(gox_out, index=False)
    print(f"Saved (GOx traceability): {gox_out} ({len(gox_df)} rows)")

    # Write README.md
    readme_path = args.out.parent / "README.md"
    readme_content = """# Round-averaged FoG 計算結果（従来版）

このフォルダには、「Fit+REA 全run → Round平均FoGまとめ」または「build_round_averaged_fog.py」を実行した結果が保存されます。

## ファイル一覧

- **fog_round_averaged.csv**: Round平均FoG値
  - 列: `round_id`, `polymer_id`, `mean_fog`, `mean_log_fog`, `robust_fog`, `robust_log_fog`, `log_fog_mad`, `n_observations`, `run_ids`
  - 同じround内の同じpolymer_idのFoG値を平均化
  - 分母は各runのGOx t50を使用（same-run GOx）

- **fog_round_gox_traceability.csv**: GOxの追跡可能性情報
  - 列: `round_id`, `run_id`, `heat_min`, `plate_id`, `well`, `abs_activity`, `REA_percent`
  - 各roundで使用されたGOxの詳細情報

## 計算方法

- **FoGの計算**: 各runで `t50_polymer / t50_GOx_same_run` を計算
- **Round平均**: 同じ`(round_id, polymer_id)`のFoG値を平均化

## 関連する実行設定

- **「Fit+REA 全run → Round平均FoGまとめ」**: Extract + Fit rates+REAを実行後、round平均FoGを生成

## 注意

このフォルダの結果は、**plate-aware FoG**（`fog_plate_aware/`フォルダ）とは異なります。
- このフォルダ: 各runのGOx t50を使用（same-run GOx）
- `fog_plate_aware/`フォルダ: 同じplateのGOx → 同じroundのGOxを使用

## 次のステップ

Round平均FoGが生成されたら、BO学習データを作成できます：
- **「BO学習データ作成（Round平均FoG）」**を実行
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved (README): {readme_path}")


if __name__ == "__main__":
    main()
