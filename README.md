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
- `data/raw` と `data/meta` をスキャンし、`Extract clean CSV ({run_id})` と `Fit rates+REA ({run_id})` が Run and Debug に追加される。

### 3. Round 対応を決める（BO に使う run だけ）

- **「全フォルダ–Round対応TSVを出力」** を実行する。  
- `meta/bo_run_round_map.tsv` が更新される（全 run_id と round_id の一覧）。  
- エディタで `meta/bo_run_round_map.tsv` を開き、**BO に使う run** には `R1`, `R2`, … を、**使わない run** には `—` を入れる。

### 4. Extract → Fit rates+REA を実行する

**方法 A（1 run ずつ）**

- **「Extract clean CSV ({run_id})」** でその run の tidy/wide を出力。  
- 続けて **「Fit rates+REA ({run_id})」** で初速・REA・t50・fog_summary を出力。

**方法 B（Round に含まれる run をまとめて）**

- **「Fit+REA 全run → Round平均FoGまとめ」** を実行する。  
- Round に round_id が付いている run について、extract が無ければ extract → 続けて fit を実行し、最後に round 平均 FoG と GOx 追跡 CSV を出す。

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

---

## 実行順のまとめ（チェックリスト）

| 順番 | やること | 実行方法 |
|------|----------|----------|
| 1 | 生データを `data/raw/{run_id}/` に置く | 手動 |
| 2 | row map `data/meta/{run_id}.tsv` を用意 | テンプレート生成 → 編集 |
| 3 | launch に新 run を反映 | 「Generate launch.json from data」 |
| 4 | Round 割り当てを決める | 「全フォルダ–Round対応TSVを出力」→ `bo_run_round_map.tsv` を編集 |
| 5 | Extract + Fit rates+REA | 「Extract clean CSV」→「Fit rates+REA」または「Fit+REA 全run → Round平均FoGまとめ」 |
| 6 | FoG（必要に応じて） | 「FoG（同一プレート→同一ラウンド）計算」または build_round_averaged_fog |
| 7 | BO 学習データ（必要に応じて） | build_bo_learning_data.py |

---

## デバッグ・確認用

- **「Fit+REA 全run → Round平均FoGまとめ (Dry run)」**: 何を extract/fit するかだけ表示。  
- **「FoG（同一プレート→同一ラウンド）Dry run」**: どの run に rates_with_rea があるかと出力パスだけ表示。
