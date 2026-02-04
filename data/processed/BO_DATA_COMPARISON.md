# ベイズ最適化データ比較：以前のコード vs 現在のプロジェクト

## 以前のコードで使用していたデータ

### 入力CSV（GOx_timeseries_RA_REA_251111.csv）
- **基本情報**: `Copolymer`, `Enzyme`, `RoundID`
- **組成情報**: `x`, `y`, `MPC_mol%`, `MTAC_mol%`, `BMA_mol%`
- **時系列データ**:
  - `Abs_0` から `Abs_60`（各サンプルの吸収度）
  - `Control_Abs_0` から `Control_Abs_60`（コントロールの吸収度）
  - `RA_0` から `RA_60`（相対活性、計算済み）
  - `REA_0` から `REA_60`（残存酵素活性、計算済み）

### 以前のコードの処理フロー
1. CSVから時系列データ（REA）を読み込み
2. REA時系列から`t50`を計算（`_compute_t50`関数）
3. 各ラウンドのコントロール`t50`を計算
4. `FoG = t50_polymer / t50_control`を計算
5. `log_fog`を計算
6. ベイズ最適化の入力: `x`, `y`（または`frac_MPC`, `frac_BMA`, `frac_MTAC`）
7. ベイズ最適化の出力: `log_fog`（または`t50_fold_rea`）

### MolLogP
- **以前のコード**: ハードコード（`LOGP_MPC = 0.3135`, `LOGP_MTAC = -2.1841`, `LOGP_BMA = 1.9058`）
- **用途**: Ternary mapでのWeighted MolLogP可視化

---

## 現在のプロジェクトで利用可能なデータ

### 1. BOカタログ（`meta/bo_catalog_bma.csv`）
- **列**: `polymer_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`, `x`, `y`
- **役割**: ポリマーIDと組成のマスター（BMA三元のみ）
- **状態**: ✅ 揃っている

### 2. FoGサマリ（`data/processed/{run_id}/fit/fog_summary__{run_id}.csv`）
- **列**: `polymer_id`, `run_id`, `t50_min`, `fog`, `log_fog`, `lineage`（`input_t50_file`, `input_tidy`など）
- **役割**: 各runごとのFoG計算結果
- **状態**: ✅ 揃っている（フィッティング実行時に自動生成）

### 3. Round平均FoG（`data/processed/fog_plate_aware/fog_plate_aware_round_averaged.csv`）
- **列**: `round_id`, `polymer_id`, `mean_fog`, `mean_log_fog`, `n_observations`, `run_ids`
- **役割**: ラウンドごとのFoG平均値
- **状態**: ✅ 揃っている

### 4. BO学習データ（`data/processed/bo_learning/bo_learning_plate_aware.csv`）
- **列**: `polymer_id`, `round_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`, `log_fog`, `run_ids`, `n_observations`
- **役割**: BO用の学習データ（カタログとFoGサマリのJOIN）
- **状態**: ✅ 揃っている（`build_bo_learning_data.py`で生成）

### 5. Run-Roundマップ（`meta/bo_run_round_map.tsv`）
- **列**: `run_id`, `round_id`
- **役割**: run_idとround_idの対応関係
- **状態**: ✅ 揃っている

---

## 不足している・まとめた方が良いデータ

### 1. MolLogP（重要度: 中）
- **現状**: `bo_catalog_bma.csv`に含まれていない
- **以前のコード**: ハードコード（`LOGP_MPC`, `LOGP_MTAC`, `LOGP_BMA`）
- **推奨**: `bo_catalog_bma.csv`に`MolLogP_MPC`, `MolLogP_MTAC`, `MolLogP_BMA`列を追加するか、別のポリマー名簿ファイル（例: `meta/polymer_properties.csv`）で管理
- **理由**: 
  - ポリマー名簿と同じようにまとめておく方が良い（ユーザー指摘）
  - Ternary mapでのWeighted MolLogP可視化に必要
  - モノマー単位の値なので、カタログに含めるか別ファイルで管理するのが適切

### 2. `x`, `y`列のBO学習データへの追加（重要度: 低）
- **現状**: `bo_learning_plate_aware.csv`には`x`, `y`列が含まれていない
- **以前のコード**: `x`, `y`を直接使用してベイズ最適化
- **推奨**: `build_bo_learning_data.py`で`x`, `y`をカタログからJOINして追加（既にコード内で対応可能）
- **理由**: 
  - 以前のコードでは`x`, `y`を入力として使用
  - 現在のプロジェクトでも`x`, `y`は`bo_catalog_bma.csv`に存在するため、JOINで追加可能

### 3. 時系列データ（重要度: 低）
- **現状**: 時系列データ（RA/REA）は各runの`rates_with_rea.csv`に保存されているが、BOには直接使用しない
- **以前のコード**: CSVから時系列データを読み込んで`t50`を計算
- **推奨**: 現在のプロジェクトでは既に`t50`が計算済みなので、時系列データはBOには不要
- **理由**: 
  - 現在のプロジェクトでは`fit_initial_rates.py`で既に`t50`を計算済み
  - BOには`log_fog`（または`t50`）のみが必要

---

## データの対応関係

| 以前のコード | 現在のプロジェクト | 状態 |
|------------|------------------|------|
| `Copolymer` | `polymer_id`（`bo_catalog_bma.csv`） | ✅ |
| `RoundID` | `round_id`（`bo_run_round_map.tsv`経由） | ✅ |
| `x`, `y` | `x`, `y`（`bo_catalog_bma.csv`） | ✅ |
| `MPC_mol%`, `MTAC_mol%`, `BMA_mol%` | `frac_MPC`, `frac_BMA`, `frac_MTAC`（`bo_catalog_bma.csv`） | ✅ |
| REA時系列から`t50`計算 | `t50_min`（`fog_summary__{run_id}.csv`） | ✅ |
| `FoG = t50_polymer / t50_control` | `fog`（`fog_summary__{run_id}.csv`） | ✅ |
| `log_fog` | `log_fog`（`fog_summary__{run_id}.csv`または`bo_learning_plate_aware.csv`） | ✅ |
| MolLogP（ハードコード） | **未実装** | ❌ 要追加 |

---

## まとめ

### ✅ 揃っているデータ
1. **組成情報**: `bo_catalog_bma.csv`に`frac_MPC`, `frac_BMA`, `frac_MTAC`, `x`, `y`が揃っている
2. **FoGデータ**: `fog_summary__{run_id}.csv`と`fog_plate_aware_round_averaged.csv`が揃っている
3. **BO学習データ**: `bo_learning_plate_aware.csv`が生成可能（`build_bo_learning_data.py`で生成）
4. **Run-Roundマップ**: `bo_run_round_map.tsv`で管理されている

### ⚠️ まとめた方が良いデータ
1. **MolLogP**: `bo_catalog_bma.csv`に追加するか、別のポリマー名簿ファイルで管理
   - 推奨: `bo_catalog_bma.csv`に`MolLogP_MPC`, `MolLogP_MTAC`, `MolLogP_BMA`列を追加
   - または: `meta/polymer_properties.csv`のような別ファイルで管理

### 📝 実装時の注意点
1. **BO学習データの`x`, `y`列**: `build_bo_learning_data.py`でカタログからJOINして追加可能（既にコード内で対応可能）
2. **時系列データ**: BOには不要（既に`t50`が計算済み）
3. **MolLogP**: Ternary map可視化に必要なので、カタログまたは別ファイルで管理する

---

## 結論

**現在のプロジェクトでベイズ最適化を行うために必要な情報は基本的に揃っています。**

唯一の不足は**MolLogP**ですが、これは可視化用途（Ternary map）のみで、BOのコア機能（候補提案）には不要です。ただし、ユーザーの指摘通り、ポリマー名簿と同じようにまとめておく方が良いでしょう。

**推奨アクション**:
1. `bo_catalog_bma.csv`に`MolLogP_MPC`, `MolLogP_MTAC`, `MolLogP_BMA`列を追加（または別ファイルで管理）
2. `build_bo_learning_data.py`で`x`, `y`列をBO学習データに追加（既にコード内で対応可能だが、出力に含める）
