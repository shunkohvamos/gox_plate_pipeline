# data/processed の整理

このディレクトリは**生データフォルダ（run_id）ごと**にフォルダを分け、その中で**実行段階（extract / fit）ごと**にサブフォルダを分けて出力します。

- 推奨運用: `data/raw/{run_id}/` に同時解析する CSV をまとめる
- 互換運用: `data/raw/{run_id}.csv` 単体ファイル
- 同一 run_id 内の CSV は同時に解析され、`data/processed/{run_id}/` に統合出力される

## ディレクトリ構成

```
data/processed/
└── {run_id}/                    ← 生データファイル名（拡張子なし）に対応
    ├── extract/                 ← Extract の生成物
    │   ├── tidy.csv
    │   └── wide.csv
    └── fit/                     ← Fit の生成物
        ├── rates_selected.csv
        ├── rates_with_rea.csv
        ├── summary_simple.csv   ← 簡易テーブル（polymer_id, heat_min, abs_activity, REA_percent）
        ├── summary_stats.csv    ← 集計テーブル（n / mean / std / sem）
        ├── bo/                  ← ベイズ最適化用（後工程で利用）
        │   └── bo_output.json   （summary + lineage + manifest）
        ├── per_polymer__{run_id}/      ← polymer_id ごと・Absolute（左）＋ REA（右）統合 PNG
        ├── per_polymer_with_error__{run_id}/  ← 重複測定ありの場合の mean±SEM 版
        ├── t50/                 ← t50 計算結果
        │   └── t50__{run_id}.csv
        ├── qc/                  ← フィット QC レポート・ヒストグラム等
        └── plots/               ← ウェルごとの診断プロット（--plot_dir 指定時）
            ├── plate1/*.png
            ├── plate2/*.png
            └── plate_grid__{run_id}__plate{N}.png
```

| パス                                      | 説明                                                               | 出力元スクリプト                                                          |
| ----------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `{run_id}/extract/tidy.csv`               | 抽出済み tidy 形式（時系列・ウェル単位）                           | `scripts/extract_clean_csv.py`                                            |
| `{run_id}/extract/wide.csv`               | 抽出済み wide 形式（plate_id, time_s × well）                      | `scripts/extract_clean_csv.py`                                            |
| `{run_id}/fit/rates_selected.csv`         | ウェルごとの初期速度（選択ウィンドウ情報）                         | `scripts/fit_initial_rates.py`                                            |
| `{run_id}/fit/rates_with_rea.csv`         | ウェルごとの初期速度 + REA                                         | `scripts/fit_initial_rates.py`                                            |
| `{run_id}/fit/summary_simple.csv`         | 簡易テーブル（polymer_id, heat_min, abs_activity, REA_percent）    | `scripts/fit_initial_rates.py` または `scripts/aggregate_polymer_heat.py` |
| `{run_id}/fit/summary_stats.csv`          | 集計テーブル（n, mean, std, sem）                                  | 同上                                                                      |
| `{run_id}/fit/bo/bo_output.json`          | ベイズ最適化用（summary + lineage + manifest・後工程で利用）       | 同上                                                                      |
| `{run_id}/fit/per_polymer__{run_id}/`     | polymer_id ごと・Absolute（左）＋ REA（右）統合（paper-grade PNG） | 同上（Fit 実行時に付随）                                                  |
| `{run_id}/fit/per_polymer_with_error__{run_id}/` | polymer_id ごと mean±SEM 版（重複測定がある場合のみ）        | 同上                                                                      |
| `{run_id}/fit/t50/t50__{run_id}.csv`      | t50（linear / exp）・fit パラメータ等                              | 同上                                                                      |
| `{run_id}/fit/qc/`                        | フィット QC レポート・ヒストグラム等                               | `scripts/fit_initial_rates.py`                                            |
| `{run_id}/fit/plots/`                     | ウェルごとの診断プロット・プレート別グリッド                       | `scripts/fit_initial_rates.py`（`--plot_dir` 指定時）                     |

- `{run_id}`: 生データフォルダ名（推奨）または生データファイル名（拡張子なし）

## 実行の流れ

1. **Extract**: `extract_clean_csv.py --raw data/raw/XXX/`（または `data/raw/XXX.csv`）→ `XXX/extract/tidy.csv`, `XXX/extract/wide.csv`
2. **Fit**: `fit_initial_rates.py --tidy data/processed/XXX/extract/tidy.csv` → `XXX/fit/rates_*.csv`, `XXX/fit/summary_simple.csv`, `XXX/fit/bo/bo_output.json`, `XXX/fit/per_polymer__XXX/`, `XXX/fit/t50/t50__XXX.csv`, `XXX/fit/qc/`, （任意）`XXX/fit/plots/`
3. **集計のみ再実行**: `aggregate_polymer_heat.py --well_table data/processed/XXX/fit/rates_with_rea.csv --run_id XXX` → 同上（summary_simple + bo + per_polymer + t50）
