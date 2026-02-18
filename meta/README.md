# 入力ファイルはすべてここ（meta/）にまとまっています

**あなたが入力・編集するファイルは、すべて `meta/` の下にあります。** コード側のパスは `src/gox_plate_pipeline/meta_paths.py` で一括定義しています。

---

## meta/ の構成

| ディレクトリ / ファイル | 内容 |
|------------------------|------|
| **row_maps/** | ランごとの row map: `{run_id}.tsv`（例: `260216-1.tsv`, `Organic_Concentration.tsv`）。plate/well → polymer_id、use_for_bo などのフラグ。 |
| **run_groups/** | グループ集約用。同じ `group_id` の run をまとめて group mean を計算。 |
| `run_groups/run_group_map.tsv` | 列: `run_id`, `group_id`, `include_in_group_mean`, `notes`。 |
| **bo/** | BO 用: ラウンド割り当てと組成カタログ。 |
| `bo/run_round_map.tsv` | 列: `run_id`, `round_id`。 |
| `bo/catalog_bma.csv` | BO 設計空間: `polymer_id`, `frac_MPC`, `frac_BMA`, `frac_MTAC` など。 |
| **polymers/** | 図の色指定とポリマー溶媒メタ。 |
| `polymers/colors.yml` | `polymer_id` →  hex 色。同じ ID は全図で同じ色。 |
| `polymers/stock_solvent.tsv` | `polymer_id` ごとの `stock_solvent` と `objective_control_group`。目的関数の絶対活性側（加点/減点）の基準選択に使用。 |
| **chemistry/** | 化学マスタ。 |
| `chemistry/mol_logp_master.csv` | `monomer_id`, `MolLogP` など。 |
| **config.yml** | 測定設定（heat_times など）。extract / fit で使用。 |

---

## row_maps/（ランごとの TSV）

各ランに **1 本**の row map を置きます。どのウェルがどの polymer か・BO に使うかなどを指定します。

| 項目 | 説明 |
|------|------|
| **パス** | `meta/row_maps/{run_id}.tsv`（例: `meta/row_maps/260216-1.tsv`）。別名: `meta/row_maps/{run_id}_row_map.tsv`。 |
| **主な列** | `plate`, `row`, `polymer_id`, `sample_name`, `use_for_bo`, `include_in_all_polymers` など（テンプレート参照）。 |
| **テンプレート** | `scripts/generate_row_map_template.py` で `meta/row_maps/{run_id}.tsv` を生成してから編集。 |
| **参照する処理** | extract / fit（`extract_clean_csv.py`, `fit_initial_rates.py` および run_fit_then_round_fog など）。launch.json は data/raw と meta/row_maps のペアから生成。 |

---

## 新しいランを追加するとき

1. **meta/row_maps/{run_id}.tsv** — row map を作成（テンプレート生成後に polymer_id や use_for_bo などを編集）。
2. **meta/bo/run_round_map.tsv** — 行を追加: `run_id`, `round_id`。
3. **meta/run_groups/run_group_map.tsv** — その run を入れたい `group_id` の行を追加。`scripts/generate_same_date_include_tsv.py` で TSV を更新してから編集してもよい。
