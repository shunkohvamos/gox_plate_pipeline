# GOx plate pipeline

酵素熱耐性（t50 / REA / FoG）解析パイプライン。生データから初速フィット・REA・t50・FoG まで一貫して実行できる。

---

## 生データを入れた後の実行順（推奨フロー）

実験の生データを **日付フォルダ**（例: `data/raw/260204-1/`）に入れたら、次の順で実行する。

### 1. 生データと row map を用意する

- **生データ**: `data/raw/{run_id}/` に CSV を入れる（例: `data/raw/260204-1/1-testdata.csv`）。
- **row map**: `data/meta/{run_id}.tsv` を用意する（行・ウェルと polymer_id の対応）。  
  - まだ無い場合は **「Generate TSV template from raw」** でテンプレートを出し、編集して `data/meta/{run_id}.tsv` として保存。

### 2. launch.json を更新する（新規 run を追加したとき）

- **「Generate launch.json from data」** を実行する。  
- `data/raw` と `data/meta` をスキャンし、以下が Run and Debug に追加される。  
  - `Extract clean CSV ({run_id})`  
  - `Fit rates+REA [t50=y0/2] ({run_id})`（通常運用: well単位fit画像なし、plate統合図は出力）  
  - `Fit rates+REA [t50=REA50] ({run_id})`（通常運用: well単位fit画像なし、plate統合図は出力）  
  - `Well plots only ({run_id})`（必要時のみ: well単位fit画像のみを生成）  
  - `Well plots only (Debug) ({run_id})`（上記 + 除外理由カウント）
  - `Group mean plots+ranking [t50=y0/2] ({run_id})`（グループ横断: 平均フィット + エラーバー + 平均t50/FoGランキング）
  - `Group mean plots+ranking [t50=REA50] ({run_id})`（同上）

### 3. Round 対応を決める（BO に使う run だけ）

- **「全フォルダ–Round対応TSVを出力」** を実行する。  
- `meta/bo_run_round_map.tsv` が更新される（全 run_id と round_id の一覧）。  
- エディタで `meta/bo_run_round_map.tsv` を開き、**BO に使う run** には `R1`, `R2`, … を、**使わない run** には `—` を入れる。

### 3.5 run集計グループを決める（任意）

- **「全フォルダ–集計グループTSVを出力」** を実行する。  
- `meta/run_group_map.tsv` が更新される（raw/processed から run_id を自動収集）。  
- `group_id` を編集して、日付が違っても同じ条件のrunを同一グループにまとめられる。  
- `include_in_group_mean` を `True/False` で編集し、グループ横断（エラーバー・平均棒グラフ）に含めるrunを制御する。  
  - このTSVは `data/meta/{run_id}.tsv`（row map）や `meta/bo_run_round_map.tsv` とは別ファイル。

### 4. Extract → Fit rates+REA を実行する

**方法 A（1 run ずつ）**

- **「Extract clean CSV ({run_id})」** でその run の tidy/wide を出力。  
- 続けて **「Fit rates+REA [t50=y0/2] ({run_id})」** または **「Fit rates+REA [t50=REA50] ({run_id})」** で初速・REA・t50・fog_summary を出力（well単位fit画像なし、plate統合図あり）。  
  - 同時に run 単位ランキング（`fit/ranking/t50_ranking__{run_id}.csv/.png`, `fit/ranking/fog_ranking__{run_id}.csv/.png`）も生成される。  
- well単位fit画像が必要なときだけ **「Well plots only ({run_id})」**（必要なら **Debug** 版）を使う。
- 複数runをまとめて平均化したい場合は **「Group mean plots+ranking [t50=...] ({run_id})」** を実行すると、
  - `data/processed/across_runs/{group_id}-group_mean/plots/` にポリマーごとの `abs_activity / REA`（平均値フィット + SEMエラーバー）PNG
  - `data/processed/across_runs/{group_id}-group_mean/ranking/` に平均t50/FoGのランキングCSV/PNG
  - `data/processed/across_runs/{group_id}-group_mean/fog_summary__{group_id}-group_mean.csv`
  が出力される。
  - 使うrunは `meta/run_group_map.tsv` の `group_id` が一致し、`include_in_group_mean=True` の行（`--run_ids` 指定時はそちらを優先）。

**方法 B（Round に含まれる run をまとめて）**

- **「Fit+REA 全run → Round平均FoGまとめ [t50=y0/2]」** または **「Fit+REA 全run → Round平均FoGまとめ [t50=REA50]」** を実行する。  
- Round に round_id が付いている run について、extract が無ければ extract → 続けて fit を実行し、最後に round 平均 FoG と GOx 追跡 CSV を出す。

**方法 C（全rawフォルダを一括で）**

- **「Extract clean CSV 全run」** で全rawフォルダを一括extract。  
- **「Fit rates+REA 全run [t50=y0/2]」** または **「Fit rates+REA 全run [t50=REA50]」** で全runを一括fit。
- round指定runだけで BO まで一括実行したい場合は **「Round指定全run → BO一括 [t50=...]」** を使う。

### 5. FoG をどう使うかで分岐

**A. 従来どおり「同じ run の GOx」で FoG を使う場合**

- 4 までで各 run の `data/processed/{run_id}/fit/fog_summary__{run_id}.csv` ができている。  
- Round 平均だけ欲しいとき:  
  `python scripts/build_round_averaged_fog.py --run_round_map meta/bo_run_round_map.tsv --processed_dir data/processed --out data/processed/fog_round_averaged.csv`  
  で `fog_round_averaged.csv` と `fog_round_gox_traceability.csv` ができる。

**B. 「同じプレート → 同じラウンド」の GOx で FoG を使う場合**

- 4 までで各 run の `rates_with_rea.csv` ができている前提。  
- **「FoG（同一プレート→同一ラウンド）計算」** を実行する。  
- `data/processed/fog_plate_aware.csv`、`fog_plate_aware_round_averaged.csv`、`fog_round_gox_traceability.csv` が出力される。

### 6. BO 学習データを作る（必要なら）

- Round 平均 FoG（A または B のどちらか）ができている状態で、  
  `python scripts/build_bo_learning_data.py --catalog meta/bo_catalog_bma.csv --fog_round_averaged data/processed/fog_round_averaged.csv`  
  などで BO 用学習 CSV を生成する。  
- plate-aware 版を使う場合は、`fog_plate_aware_round_averaged.csv` を BO 入力に合わせて利用する（スクリプトのオプションは要確認）。

### 7. ベイズ最適化を実行する

- Run and Debug で **「Bayesian Optimization（Pure Regression / Plate-aware）」** を実行する（ワンクリック）。
  - この設定は `bo_learning_plate_aware.csv` を再生成してから BO を実行する。
- 既存の学習 CSV をそのまま使いたい場合は **「Bayesian Optimization（Pure Regression / 既存学習データ）」** を実行する。
- 出力先: `data/processed/bo_runs/{bo_run_id}/`
  - 三角図（mean/std/EI/UCB）
  - 提案候補ログ・提案一覧
  - 次実験向け上位5提案: `next_experiment_top5__{bo_run_id}.csv`
    - `priority_rank`（優先順位）, `priority_score`（重み付きスコア）, `recommended_top3`（次ラウンド実施推奨）
    - 重みは既定で `FoG 0.45 / t50 0.45 / EI 0.10`
  - t50 / FoG ランキング表（round別・全体）
  - BO 実行マニフェスト（traceability）

---

## 実行順のまとめ（チェックリスト）

| 順番 | やること | 実行方法 |
|------|----------|----------|
| 1 | 生データを `data/raw/{run_id}/` に置く | 手動 |
| 2 | row map `data/meta/{run_id}.tsv` を用意 | テンプレート生成 → 編集 |
| 3 | launch に新 run を反映 | 「Generate launch.json from data」 |
| 4 | Round 割り当てを決める | 「全フォルダ–Round対応TSVを出力」→ `bo_run_round_map.tsv` を編集 |
| 5 | Extract + Fit rates+REA | 「Extract clean CSV」→「Fit rates+REA [t50=y0/2] / [t50=REA50]（通常: well図なし）」を実行。well図が必要なら別途「Well plots only」を実行 |
| 5.5 | runグループ横断の平均可視化（必要時） | 「Group mean plots+ranking [t50=...] ({run_id})」で平均フィット + エラーバー + 平均t50/FoG棒グラフ |
| 6 | FoG（必要に応じて） | 「FoG（同一プレート→同一ラウンド）計算」または build_round_averaged_fog |
| 7 | BO 学習データ（必要に応じて） | build_bo_learning_data.py |
| 8 | ベイズ最適化実行 | 「Bayesian Optimization（Pure Regression / Plate-aware）」 |

---

## デバッグ・確認用

- **「Fit+REA 全run → Round平均FoGまとめ (Dry run)」**: 何を extract/fit するかだけ表示（dry/debugは `t50=y0/2` 設定）。  
- **「FoG（同一プレート→同一ラウンド）Dry run」**: どの run に rates_with_rea があるかと出力パスだけ表示。
- **「Extract clean CSV 全run (Dry run) / Fit rates+REA 全run (Dry run) / Round指定全run → BO一括 (Dry run)」**: 実行せず、対象runと実行予定コマンドだけを確認。
