# meta/ — プロジェクト共通のマスター・設定

生データフォルダを追加したら **すぐに** ここにあるマスターシートを編集します。  
Fit rates+REA の実行後に慌てて入力する必要はありません。

---

## 生データフォルダを追加した後の手順

1. **生データを置く**  
   `data/raw/{run_id}/` に CSV を入れる（例: `data/raw/260203-3/`）。run_id はフォルダ名（日付など）。

2. **ポリマー名簿・組成を用意する（BO 用）**  
   - **`meta/bo_catalog_bma.csv`** を開く。**ポリマーと組成の対応表だけ**（round_id 列は使わない・入れない）。  
   - 新規ポリマー（例: PMBTA-12）を追加するときは、行を足して `polymer_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`（と必要なら `x`, `y`）を書く。組成の合計は 1。

3. **ウェル対応 TSV を用意する**  
   - Run and Debug で **「Generate TSV template from raw」** を実行し、`data/meta/{run_id}.tsv` のテンプレを出す。  
   - ウェルと polymer_id の対応を編集して保存。

4. **launch.json を更新する（任意）**  
   - Run and Debug で **「Generate launch.json from data」** を実行。  
   - 新しい run_id 用の「Extract clean CSV」「Fit rates+REA」が Run and Debug に追加される。

5. **Extract → Fit を実行する**  
   - **「Extract clean CSV ({run_id})」** を実行 → `data/processed/{run_id}/extract/tidy.csv` などができる。  
   - **「Fit rates+REA ({run_id})」** を実行 → 初速・REA・t50・FoG・プロットが `data/processed/{run_id}/fit/` にできる。

6. **BO に使う run だけ Round を振る**  
   - Run and Debug で **「全フォルダ–Round対応TSVを出力」** を実行 → `meta/bo_run_round_map.tsv` が更新される。  
   - TSV を開き、**BO に使う run** だけ `round_id` を R1, R2, … に書き換える。**使わない run** は `—` のまま。

7. **BO 学習データを作る（BO を回すとき）**  
   `python scripts/build_bo_learning_data.py --catalog meta/bo_catalog_bma.csv --run_round_map meta/bo_run_round_map.tsv --out data/processed/bo_learning.csv --exclusion_report data/processed/bo_learning_excluded.csv`  
   → round_id が付いた run だけから t50/FoG を集めた BO 用 CSV ができる。

---

## ポリマーID と BO 用組成比（マスターシート）

| ファイル | 役割 | いつ入力するか |
| -------- | ----- | -------------- |
| **`bo_catalog_bma.csv`** または **`bo_catalog_bma.tsv`** | ポリマーID と BMA 三元組成比（MPC/BMA/MTAC）の **マスター**。BO が参照する唯一の組成ソース。 | **生データフォルダを追加した直後**（または新規ポリマーを追加したとき）。どの実行でも **生成されない**。手で編集する。 |

- **必須列**: `polymer_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC`（順序固定。組成の合計は 1、各 frac は [0,1]）
- **任意列**: `x`, `y`（三元組成の 2D 射影。空欄なら組成比から計算される）
- **round_id はカタログには入れない**。このファイルは **ポリマーと組成の対応表だけ**。Round は実験フォルダごとに `bo_run_round_map.tsv` 等で指定する。
  - 定義（BO で使用）: `x = [BMA] / ([BMA] + [MTAC])`, `y = ([BMA] + [MTAC]) / ([MPC] + [BMA] + [MTAC])`。三元の順序は常に [MPC, BMA, MTAC]。
  - `(frac_BMA + frac_MTAC) == 0` のときは `x` は NaN（例: PMPC）。
- **参照されるタイミング**: BO 学習データ作成時（Fit 実行の末尾で自動作成される処理）、および BO 実行時のみ。Fit rates+REA の「初速・REA・t50・FoG」計算では参照しない。
- **テンプレ**: 初回や上書き用に `bo_catalog_bma_template.csv` と `bo_catalog_bma_template.tsv` を用意してある。どちらかをコピーして `bo_catalog_bma.csv` または `bo_catalog_bma.tsv` として使うか、そのまま編集してよい。PMBTA-12 以降は行を追加する。Fit 実行時は `meta/bo_catalog_bma.csv` を優先し、無ければ `meta/bo_catalog_bma.tsv` を参照する。

### 「どの実行で生成されるどのファイルに入力するか」

**どれでもありません。** 上記マスターシート（`meta/bo_catalog_bma.csv` または `.tsv`）は **実行の出力ではなく、プロジェクトに最初から置いておく入力ファイル** です。生データを追加したらこのファイルを開いてポリマーID・組成比を追加・修正します。

---

## BO で使う実験の指定（フォルダ名 → Round）

BO では「どの実験日（run_id）をどの Round として使うか」を明示します。Fit rates+REA は日付ごと（run_id ごと）に実行し、t50/FoG は run_id ごとに計算済みです。BO 学習データを作るときに、**フォルダ名（run_id）と Round の対応** を書いたファイルを渡します。

| ファイル | 役割 |
| -------- | ----- |
| **`bo_run_round_map.yml`**（または **`bo_run_round_map.tsv`** / `.csv`） | **run_id → round_id** の対応。例: `251118` → R1、`260201` → R2。**Round に使わない run** は round_id を空欄または **`—`**（全角ダッシュ）や **NA** にしておく。読み込み時はこれらは「BO に使用しない」としてスキップされる。 |

- **YAML 例**: `run_round_map: { "251118": R1, "260201": R2, "260203": R3 }`（キーはフォルダ名＝run_id、値は Round）
- **CSV 例**: 列 `run_id`, `round_id` の表
- **運用**: 実験日ごとに `data/processed/{run_id}/` ができたら、その run_id を何 Round にするかをこのファイルに追記する。BO 学習データ作成時は `--run_round_map meta/bo_run_round_map.tsv` を指定し、その run_id だけから fog_summary を集め、各行に round_id を付与する。カタログは polymer_id のみで照合（カタログには round_id は含まれない）。
- **カタログとの関係**: カタログは polymer_id と組成・x,y の対応のみ（round_id なし）。照合は polymer_id のみ。round_id は run_round_map から付与されるので、同じポリマーが R1 と R3 の run で測定されていれば、BO 学習データには (PMBTA-1, R1, log_fog from run R1) と (PMBTA-1, R3, log_fog from run R3) の 2 行が入る。

### 全フォルダ×Round 対応 TSV を出力する（実行とデバッグ）

- **Run and Debug** で **「全フォルダ–Round対応TSVを出力」** を実行すると、`data/raw` と `data/processed` にあるすべての run_id（生データフォルダ）を列挙した **`meta/bo_run_round_map.tsv`** が生成される。
- 列: `run_id`, `round_id`。Round に**使わない** run は `round_id` を **`—`** のまま（または空欄・NA）にしておく。BO に使う run だけ R1, R2, … に書き換える。
- 既に `bo_run_round_map.yml` や `bo_run_round_map.tsv` がある場合は、そこから round を引き継いで TSV を更新する。再実行すると編集内容（R1/R2/…）は保持される。

### BO 学習データの作り方（推奨オペレーション）

1. **Fit rates+REA**: 日付ごと（run_id ごと）に実行 → 各 `data/processed/{run_id}/fit/fog_summary__{run_id}.csv` と t50 ができる。
2. **run_round_map を書く**: 「全フォルダ–Round対応TSVを出力」で TSV を出し、BO に使う run だけ round_id を R1, R2, … に編集。使わない run は `—` のまま。
3. **BO 学習データを生成**:  
   `python scripts/build_bo_learning_data.py --catalog meta/bo_catalog_bma.csv --run_round_map meta/bo_run_round_map.tsv --out data/processed/bo_learning.csv --exclusion_report data/processed/bo_learning_excluded.csv`  
   これで、round_id が有効な run だけから fog_summary を集め、round_id を付けて既存の t50/FoG を使った BO 用 CSV ができる。

---

## MolLogP マスター（BO 三角図の Weighted MolLogP 用）

| ファイル | 役割 |
| -------- | ----- |
| **`mol_logp_master.csv`** | モノマー（MPC, MTAC, BMA）ごとの **MolLogP** のマスター。BO の三角図で Weighted MolLogP を描くときに参照する。 |
| **`mol_logp_master_template.csv`** | 上記のテンプレート。初回や上書き用。 |

- **必須列**: `monomer_id`, `MolLogP`（monomer_id は MPC / MTAC / BMA など、三元組成のモノマー名と一致させる）
- **任意列**: `note`（出典・メモなど）
- **運用**: 値は手で編集する。実行で上書きされない。BO とは別の Run and Debug「**MolLogP マスター確認**」で存在・列・数値をチェックできる。
- **参照されるタイミング**: BO で三角図（Weighted MolLogP）を出す機能を実装するときに読み込む。

---

## その他

- **`config.yml`**: 熱処理時間など。Fit 実行時に参照。
- **`polymer_colors.yml`**: プロット用の polymer_id → 色。必要に応じて編集。
