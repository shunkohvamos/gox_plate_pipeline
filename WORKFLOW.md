# 実行とデバッグの使い方ガイド

生データから BO 学習データまで、VS Code の Run and Debug から実行する手順。

---

## 📋 実行順序（生データを入れた後）

### ステップ 1: 新規 run を追加したとき（初回のみ）

**1-1. Row map を用意**
- `data/meta/{run_id}.tsv` が無い場合:
  - **「Generate TSV template from raw」** を実行
  - 生成されたテンプレートを編集して `data/meta/{run_id}.tsv` として保存

**1-2. launch.json を更新**
- **「Generate launch.json from data」** を実行
- 新しい run 用の Extract / Fit 設定が追加される

**1-3. 集計グループ TSV を更新（任意）**
- **「全フォルダ–集計グループTSVを出力」** を実行
- `meta/run_group_map.tsv` が更新される
- `group_id` を編集して同条件runをまとめる
- `include_in_group_mean` を `True/False` で編集（グループ横断集計に含めるrunを制御）

---

### ステップ 2: Extract → Fit rates+REA（各 run ごと）

**方法 A: 個別に実行（推奨）**
- **「Extract clean CSV ({run_id})」** を実行
- 続けて **「Fit rates+REA [t50=y0/2] ({run_id})」** または **「Fit rates+REA [t50=REA50] ({run_id})」** を実行（通常: well単位fit画像なし、plate統合図あり）
  - run単位ランキング（`fit/ranking/t50_ranking__{run_id}.csv/.png`, `fit/ranking/fog_ranking__{run_id}.csv/.png`）を自動出力
- well単位fit画像が必要な場合のみ **「Well plots only ({run_id})」**（必要なら **Debug** 版）を実行
- 複数runをまとめる場合は **「Group mean plots+ranking [t50=...] ({run_id})」** を実行
  - `data/processed/across_runs/{group_id}-group_mean/plots/`: ポリマーごとの平均フィット + SEMエラーバー
  - `data/processed/across_runs/{group_id}-group_mean/ranking/`: 平均t50/FoGランキングCSV/PNG
  - 対象runは `meta/run_group_map.tsv` の同一 `group_id` かつ `include_in_group_mean=True` を使用（`--run_ids` 指定時はそれを優先）
- これを各 run ごとに繰り返す

**方法 B: まとめて実行（round に含まれる run のみ）**
- **「Fit+REA 全run → Round平均FoGまとめ [t50=y0/2]」** または **「Fit+REA 全run → Round平均FoGまとめ [t50=REA50]」** を実行
- Round に round_id が付いている run について、extract → fit を自動実行
- ⚠️ **注意**: 既に fit が完了している run はスキップされる（`fog_summary__{run_id}.csv` が存在する場合）

**方法 C: 全runを一括実行**
- **「Extract clean CSV 全run」** で全rawフォルダを一括extract
- **「Fit rates+REA 全run [t50=y0/2]」** または **「Fit rates+REA 全run [t50=REA50]」** で全runを一括fit
- round指定runだけで BO まで一括実行したい場合は **「Round指定全run → BO一括 [t50=...]」** を実行

---

### ステップ 3: Round 割り当てを決める

**3-1. per_polymer の曲線と t50 を確認**
- ファイルエクスプローラーで以下を開いて確認:
  - `data/processed/{run_id}/fit/t50/per_polymer__{run_id}/` の PNG
  - `data/processed/{run_id}/fit/t50/t50__{run_id}.csv`

**3-2. Round 割り当てを設定**
- **「全フォルダ–Round対応TSVを出力」** を実行
- `meta/bo_run_round_map.tsv` が更新される
- エディタで開き、BO に使う run に `R1`, `R2`, … を、使わない run に `—` を設定

---

### ステップ 4: Round 平均 FoG を計算

- **「FoG（同一プレート→同一ラウンド）計算」** を実行
- 出力:
  - `data/processed/fog_plate_aware.csv`
  - `data/processed/fog_plate_aware_round_averaged.csv`
  - `data/processed/fog_round_gox_traceability.csv`

---

### ステップ 5: BO 学習データを作成

- **「BO学習データ作成（Plate-aware Round平均FoG）」** を実行
- 出力:
  - `data/processed/bo_learning_plate_aware.csv`
  - `data/processed/bo_learning_excluded_plate_aware.csv`

---

### ステップ 6: ベイズ最適化を実行

- **「Bayesian Optimization（Pure Regression / Plate-aware）」** を実行（ワンクリック）
  - `bo_learning_plate_aware.csv` を再生成してから BO を実行
- 既存学習データを使う場合は **「Bayesian Optimization（Pure Regression / 既存学習データ）」**
- 出力先:
  - `data/processed/bo_runs/{bo_run_id}/`
  - 三角図: `ternary_mean_log_fog__{bo_run_id}.png`, `ternary_std_log_fog__{bo_run_id}.png`, `ternary_ei__{bo_run_id}.png`, `ternary_ucb__{bo_run_id}.png`
  - 2x2 パネル（Mean / Std / EI / UCB）:
    - 既定（推奨）: `bma_mtac_2x2_mean_std_ei_ucb__{bo_run_id}.png`
    - 旧xy座標を使う場合: `xy_2x2_mean_std_ei_ucb__{bo_run_id}.png`
  - 次実験向け上位5提案: `next_experiment_top5__{bo_run_id}.csv`
    - `priority_rank` と `recommended_top3` を見れば、実験本数が 1〜3 本でもすぐ選べる
    - 優先度重みの既定: `FoG 0.45 / t50 0.45 / EI 0.10`
  - ランキング表: `t50_ranking_*.csv`, `fog_ranking_*.csv`
  - 提案ログ: `bo_candidate_log__{bo_run_id}.csv`, `bo_suggestions__{bo_run_id}.csv`
  - マニフェスト: `bo_manifest__{bo_run_id}.json`

---

## 🎯 よく使う設定（必須）

| 設定名 | いつ使う | 頻度 |
|--------|----------|------|
| **Extract clean CSV ({run_id})** | 新規 run の extract | 新規 run ごと |
| **Fit rates+REA [t50=y0/2] ({run_id})** | 新規 run の fit（通常: well図なし、plate統合図あり） | 新規 run ごと |
| **Fit rates+REA [t50=REA50] ({run_id})** | 新規 run の fit（通常: well図なし、plate統合図あり） | 新規 run ごと |
| **Extract clean CSV 全run** | 全runを一括extract | 必要時 |
| **Fit rates+REA 全run [t50=y0/2] / [t50=REA50]** | 全runを一括fit | 必要時 |
| **Round指定全run → BO一括 [t50=y0/2] / [t50=REA50]** | round指定runを収集してBOまで一括 | 必要時 |
| **Well plots only ({run_id})** | well単位fit画像のみを生成 | 必要時のみ |
| **Well plots only (Debug) ({run_id})** | 上記 + 除外理由カウントを表示 | 必要時のみ |
| **Group mean plots+ranking [t50=y0/2] ({run_id})** | runグループ横断の平均可視化 | 必要時のみ |
| **Group mean plots+ranking [t50=REA50] ({run_id})** | runグループ横断の平均可視化 | 必要時のみ |
| **全フォルダ–Round対応TSVを出力** | Round 割り当てを設定/更新 | per_polymer 確認後 |
| **全フォルダ–集計グループTSVを出力** | runグループ横断の include/exclude を設定/更新 | 必要時 |
| **FoG（同一プレート→同一ラウンド）計算** | Round 平均 FoG を計算 | Round 割り当て確定後 |
| **BO学習データ作成（Plate-aware Round平均FoG）** | BO 学習データを作成 | FoG 計算後 |
| **Bayesian Optimization（Pure Regression / Plate-aware）** | BO 実行（提案・三角図・ランキング） | 学習データ作成後 |

---

## 🔍 確認・デバッグ用（必要に応じて）

| 設定名 | いつ使う | 説明 |
|--------|----------|------|
| **FoG（同一プレート→同一ラウンド）Dry run** | FoG 計算前に確認 | どの run に `rates_with_rea.csv` があるか確認 |
| **Extract clean CSV 全run (Dry run)** | 一括extract前に確認 | どのrunをextractするか確認 |
| **Fit rates+REA 全run (Dry run)** | 一括fit前に確認 | どのrunをfitするか確認 |
| **Round指定全run → BO一括 (Dry run)** | 一括BO前に確認 | 実行予定の3ステップを確認 |
| **Fit+REA 全run → Round平均FoGまとめ (Dry run)** | 一括実行前に確認 | どの run で extract/fit が実行されるか確認 |
| **Fit+REA 全run → Round平均FoGまとめ (Debug)** | 一括実行時に詳細ログ | 実行中のコマンドを詳しく見たいとき |

---

## ⚙️ 設定・メンテナンス用（たまに使う）

| 設定名 | いつ使う | 説明 |
|--------|----------|------|
| **Generate launch.json from data** | 新規 run を追加したとき | launch.json に新しい run の設定を追加 |
| **Generate TSV template from raw** | row map がないとき | row map のテンプレートを生成 |

---

## ❌ 通常は使わない設定

| 設定名 | 理由 |
|--------|------|
| **Fit+REA 全run → Round平均FoGまとめ [t50=y0/2] / [t50=REA50]** | 既に fit が完了している場合、個別実行の方が柔軟 |
| **BO学習データ作成（Round平均FoG）** | Plate-aware 版を使う方が適切（プレート間の系統誤差を考慮） |

---

## 📝 典型的なワークフロー例

### シナリオ 1: 新しい実験データを追加した

1. `data/raw/{new_run_id}/` に CSV を入れる
2. `data/meta/{new_run_id}.tsv` を用意（なければ「Generate TSV template from raw」）
3. 「Generate launch.json from data」で launch を更新
4. 「Extract clean CSV ({new_run_id})」を実行
5. 「Fit rates+REA [t50=y0/2] ({new_run_id})」または「Fit rates+REA [t50=REA50] ({new_run_id})」を実行（通常）
6. well単位fit画像が必要なら「Well plots only ({new_run_id})」（必要なら Debug 版）を実行
7. per_polymer の曲線と t50 を確認
8. 「全フォルダ–Round対応TSVを出力」で round を設定
9. 「FoG（同一プレート→同一ラウンド）計算」を実行
10. 「BO学習データ作成（Plate-aware Round平均FoG）」を実行
11. 「Bayesian Optimization（Pure Regression / Plate-aware）」を実行

### シナリオ 2: 既存データで round を再設定したい

1. per_polymer の曲線と t50 を確認
2. 「全フォルダ–Round対応TSVを出力」で round を再設定
3. 「FoG（同一プレート→同一ラウンド）計算」を実行
4. 「BO学習データ作成（Plate-aware Round平均FoG）」を実行
5. 「Bayesian Optimization（Pure Regression / Plate-aware）」を実行

---

## 💡 ヒント

- **Dry run** は実行前に「何が起こるか」を確認したいときに使う
- **個別実行** vs **一括実行**: 個別実行の方が柔軟で、途中で止めやすい
- **Plate-aware** vs **通常**: Plate-aware の方がプレート間の系統誤差を考慮できるため推奨
