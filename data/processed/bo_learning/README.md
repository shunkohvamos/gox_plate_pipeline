# BO学習データ

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
python scripts/build_bo_learning_data.py \
  --catalog meta/bo_catalog_bma.csv \
  --run_round_map meta/bo_run_round_map.tsv \
  --fog_round_averaged data/processed/fog_round_averaged/fog_round_averaged.csv
```

### Plate-aware版

**「BO学習データ作成（Plate-aware Round平均FoG）」**を実行

または、コマンドラインから：
```bash
python scripts/build_bo_learning_data.py \
  --catalog meta/bo_catalog_bma.csv \
  --run_round_map meta/bo_run_round_map.tsv \
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
