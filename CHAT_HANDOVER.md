# チャット引継ぎメモ（プロジェクト状況・このチャットでやったこと）

他のチャットや後から見たときに、**今何をしていたか・どこまでできているか**が分かるようにまとめたメモです。

---

## 1. プロジェクトの目的と流れ

- **目的**: プレート測定データから「初速フィット → 絶対活性 → REA(%)」まで一貫して算出し、BO（ベイズ最適化）の準備まで行う。
- **実行の流れ**:

  1. **Extract**: 生データ CSV → `data/processed/{run_id}/extract/` に tidy.csv, wide.csv
  2. **Fit**: tidy → ウェルごと初期速度・REA・QC・per-polymer 図・t50 など → `data/processed/{run_id}/fit/`
  3. **集計のみ**: `aggregate_polymer_heat.py` で summary_simple + BO 用 + per_polymer 図 + t50 を再出力可能

- **生データとの対応**: 出力はすべて `data/processed/{run_id}/` 以下。run_id = 生データファイル名（拡張子なし）。extract / fit で段階ごとにフォルダ分け。

---

## 2. このチャットで実施・合意した主なこと

### 2.1 図の体裁（per_polymer 統合図）

- **Absolute（左）と REA（右）を 1 枚に統合**: `per_polymer__{run_id}/{polymer_stem}__{run_id}.png`
- **情報ボックス（info 窓）**:
  - フィット曲線の式は **e の累乗**で表記: `y = y0 e^{-k x}`（x = Heat time）。`exp(...)` ではなく **x の関数**として表示。
  - **R²**: mathtext で上付き 2（`$R^2$`）。STIX fontset 使用。
  - **t₅₀**: 50 は t の右下に下付き（`$t_{50}$`）。
  - **t50 の出所**: `(exp fit)` = 指数減衰フィットから得た t50、`(linear interp)` = 観測曲線の線形補間から得た t50。
- **Heat time 軸**: 0–60 min、目盛り 7 点（0, 10, 20, 30, 40, 50, 60）
- **スタイル**: `apply_paper_style()` + `mathtext.fontset: stix`。フォントは Arial 優先（図は英語のみ）。指数減衰曲線の線幅 1.6、alpha 0.65。PNG は dpi=600、`pil_kwargs={"compress_level": 1}` で画質維持。

### 2.2 フィッティングまわり（ウェル単位）

- 初速を取りたいので「**直線っぽさ（R²）**」より「**反応の妥当性・初期らしさ**」を優先する設計。
- 特定のウェル名や「step jump」など**データ固有の用語はルールに書かず、抽象化**してある（AGENTS.md / .cursor/rules）。
- **救済は局所的**に。正常に動いているケースの前提を崩さない。「前半を含める」「1 点だけ trim で外す」などは、型判定・ガードレール付きで行う方針。
- R² 閾値は**グローバルに緩めない**。救済は「型」で限定。

### 2.3 出力レイアウト

- **processed**: `{run_id}/extract/` と `{run_id}/fit/` で段階ごとに分離。BO 用は `fit/bo/`。
- **summary_simple.csv**: fit フォルダ直下（`fit/summary_simple.csv`）。BO 用は別途後工程で用意。
- **per_polymer 図**: `fit/per_polymer__{run_id}/` に統合 PNG のみ。旧 `per_polymer_abs__` / `per_polymer_rea__` は実行時に削除する仕様。

### 2.4 ルール・AGENTS.md

- フィッティングの**根幹方針**（初速・型分類・救済の限定・ガードレール）は `.cursor/rules` と `AGENTS.md` に記載。
- **具体的なウェル名は使わない**。データの「型」や「制約条件」で抽象化してある。
- **注意書き**: 「特定データに合わせた極端なロジック変更をしない」「どんなデータが来ても最適なフィッティングになるロジックを目指す」を明記。

---

## 3. 主要なファイル・場所

| 役割                                 | パス・ファイル                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| 生データ                             | `data/raw/*.csv`                                                                                             |
| メタ（well ↔ polymer_id, heat_time） | `data/meta/*.tsv` など row map                                                                               |
| Extract 出力                         | `data/processed/{run_id}/extract/tidy.csv`, `wide.csv`                                                       |
| Fit 出力                             | `data/processed/{run_id}/fit/`（rates\__.csv, summary_simple.csv, bo/, per_polymer\_\__, t50/, qc/, plots/） |
| 図のスタイル                         | `src/gox_plate_pipeline/fitting/core.py` の `apply_paper_style()`                                            |
| per_polymer 図・t50                  | `src/gox_plate_pipeline/polymer_timeseries.py`（`plot_per_polymer_timeseries`）                              |
| フィット・選択ロジック               | `src/gox_plate_pipeline/fitting/`（pipeline, selection, candidates, plotting など）                          |
| 実行・デバッグ                       | `.vscode/launch.json`（Extract / Fit rates+REA など）                                                        |
| プロジェクトルール                   | `.cursor/rules/`、`AGENTS.md`                                                                                |
| 出力の説明                           | `data/processed/INDEX.md`                                                                                    |

---

## 4. 直近のコード変更（このチャット）

- **polymer_timeseries.py**
  - 式表示を **t → x** に変更: `e^{-k x}`（x = Heat time、曲線の説明用）。
  - t50 のラベルを **"exp" → "exp fit"**、「linear」→ **"linear interp"** に変更し、出所が分かるようにした。
- 上記以外は、既に「e の累乗・R²・t₅₀・STIX・0–60 min 軸」が入った状態で説明済み。

---

## 5. 他チャットに渡すときのポイント

- **図**: すべて `apply_paper_style()` を通し、PNG 必須。フォント Arial 優先、数式は STIX mathtext（e^{...}, R², t₅₀）。
- **フィット**: 「初速」「型に応じた救済」「正常系を壊さない」が設計思想。ルールと AGENTS.md に抽象化してある。
- **出力**: run_id 単位で `data/processed/{run_id}/extract|fit/` に整理。INDEX.md で一覧できる。
- **t50 (exp fit / linear interp)**: exp fit = 指数減衰フィットから算出、linear interp = 観測 REA 曲線の線形補間から算出。

このファイルは**状況のスナップショット**です。実装の細部は `AGENTS.md` と `.cursor/rules/` を参照してください。
