# Plate-aware FoG 計算結果

このフォルダには、「FoG（同一プレート→同一ラウンド）計算」を実行した結果が保存されます。

## ファイル一覧

- **fog_plate_aware.csv**: 各ポリマーのFoG値（詳細）
  - 列: `round_id`, `run_id`, `plate_id`, `polymer_id`, `t50_min`, `t50_definition`, `gox_t50_used_min`, `denominator_source`, `fog`, `log_fog`, `native_activity_rel_at_0`, `native_0`, `native_activity_feasible`, `fog_native_constrained`, `log_fog_native_constrained`, `U_*`, `t_theta`, `t_theta_censor_flag`, `reference_qc_*`
  - `denominator_source`: `same_plate`（同じplateのGOxを使用）または `same_round`（round平均GOxを使用）

- **fog_plate_aware_round_averaged.csv**: Round平均FoG値
  - 列: `round_id`, `polymer_id`, `mean_fog`, `mean_log_fog`, `robust_fog`, `robust_log_fog`, `log_fog_mad`, `mean_fog_native_constrained`, `mean_log_fog_native_constrained`, `robust_fog_native_constrained`, `robust_log_fog_native_constrained`, `native_feasible_fraction`, `n_observations`, `run_ids`
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

- **t50定義**: `--t50_definition` で `y0_half` または `rea50` を選択

## 関連する実行設定

- **「FoG（同一プレート→同一ラウンド）計算」**: 通常実行（警告のみ、異常値は除外しない）
- **「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」**: 異常GOx t50値を除外して実行

## 次のステップ

Round平均FoGが生成されたら、BO学習データを作成できます：
- **「BO学習データ作成（Plate-aware Round平均FoG）」**を実行
