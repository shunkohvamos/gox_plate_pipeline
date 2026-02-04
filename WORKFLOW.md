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

---

### ステップ 2: Extract → Fit rates+REA（各 run ごと）

**方法 A: 個別に実行（推奨）**
- **「Extract clean CSV ({run_id})」** を実行
- 続けて **「Fit rates+REA ({run_id})」** を実行
- これを各 run ごとに繰り返す

**方法 B: まとめて実行（round に含まれる run のみ）**
- **「Fit+REA 全run → Round平均FoGまとめ」** を実行
- Round に round_id が付いている run について、extract → fit を自動実行
- ⚠️ **注意**: 既に fit が完了している run はスキップされる（`fog_summary__{run_id}.csv` が存在する場合）

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

## 🎯 よく使う設定（必須）

| 設定名 | いつ使う | 頻度 |
|--------|----------|------|
| **Extract clean CSV ({run_id})** | 新規 run の extract | 新規 run ごと |
| **Fit rates+REA ({run_id})** | 新規 run の fit | 新規 run ごと |
| **全フォルダ–Round対応TSVを出力** | Round 割り当てを設定/更新 | per_polymer 確認後 |
| **FoG（同一プレート→同一ラウンド）計算** | Round 平均 FoG を計算 | Round 割り当て確定後 |
| **BO学習データ作成（Plate-aware Round平均FoG）** | BO 学習データを作成 | FoG 計算後 |

---

## 🔍 確認・デバッグ用（必要に応じて）

| 設定名 | いつ使う | 説明 |
|--------|----------|------|
| **FoG（同一プレート→同一ラウンド）Dry run** | FoG 計算前に確認 | どの run に `rates_with_rea.csv` があるか確認 |
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
| **Fit+REA 全run → Round平均FoGまとめ** | 既に fit が完了している場合、個別実行の方が柔軟 |
| **BO学習データ作成（Round平均FoG）** | Plate-aware 版を使う方が適切（プレート間の系統誤差を考慮） |

---

## 📝 典型的なワークフロー例

### シナリオ 1: 新しい実験データを追加した

1. `data/raw/{new_run_id}/` に CSV を入れる
2. `data/meta/{new_run_id}.tsv` を用意（なければ「Generate TSV template from raw」）
3. 「Generate launch.json from data」で launch を更新
4. 「Extract clean CSV ({new_run_id})」を実行
5. 「Fit rates+REA ({new_run_id})」を実行
6. per_polymer の曲線と t50 を確認
7. 「全フォルダ–Round対応TSVを出力」で round を設定
8. 「FoG（同一プレート→同一ラウンド）計算」を実行
9. 「BO学習データ作成（Plate-aware Round平均FoG）」を実行

### シナリオ 2: 既存データで round を再設定したい

1. per_polymer の曲線と t50 を確認
2. 「全フォルダ–Round対応TSVを出力」で round を再設定
3. 「FoG（同一プレート→同一ラウンド）計算」を実行
4. 「BO学習データ作成（Plate-aware Round平均FoG）」を実行

---

## 💡 ヒント

- **Dry run** は実行前に「何が起こるか」を確認したいときに使う
- **個別実行** vs **一括実行**: 個別実行の方が柔軟で、途中で止めやすい
- **Plate-aware** vs **通常**: Plate-aware の方がプレート間の系統誤差を考慮できるため推奨
