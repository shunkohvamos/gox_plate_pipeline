# 現状把握：per-polymer 時系列グラフと t50

## 1. フォルダ配置の現状

### data/processed 全体

- **トップ**: `INDEX.md` と `{run_id}/` のみ。整理されている。
- **{run_id}/**: `extract/` と `fit/` の 2 段階。生データ・実行段階ごとの分け方は一貫している。

### fit 直下（synthetic_GOx_rows_test の例）

| 種類     | 例                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| ファイル | rates_selected.csv, rates_with_rea.csv, summary_simple.csv                     |
| フォルダ | bo/, per_polymer_abs**{run_id}/, per_polymer_rea**{run_id}/, plots/, qc/, t50/ |

**指摘（散らかり・一貫性）**

- fit 直下に**フォルダが 6 種類**あり、やや多い。
- **命名の混在**: `per_polymer_abs__{run_id}` は run_id を名前に含むが、`t50/`, `bo/`, `qc/`, `plots/` は含まない。現状は「1 run あたり 1 fit」なので実害はないが、将来 run をまたぐ場合は t50 も `t50__{run_id}.csv` のように run_id を付けた方が一貫する（ファイル名には既に `t50__{run_id}.csv` と付いている）。
- **INDEX.md**: per*polymer*\* と t50 の説明が**未記載**。追記した方がよい。

---

## 2. 既存実装の評価（polymer_timeseries.py）

### やっていること

- `summary_simple.csv` を読み、polymer_id ごとに heat_min vs abs_activity / REA_percent の時系列を描画。
- **折れ線**: 観測点は必ず scatter、つなぎは「折れ線」または「指数減衰曲線」。
- **回帰**: 指数減衰は「データ点が十分（min_points=4 以上、ユニーク t が 3 以上）」「全体が減少傾向（Spearman ρ < -0.2）」「y が正」のときのみ試行。R²≥0.7 のときだけ曲線を描き t50(exp) を採用。
- **t50**: REA (%) に対して「50% に落ちる時間」。指数モデルなら半減期 ln(2)/k または plateau 版の解析式。フォールバックは線形補間（t50_linear）。
- **永続カラーマップ**: `meta/polymer_colors.yml` に polymer_id → 色を保存。既存 ID は同色、新規 ID はパレットから追加。
- **出力**: `per_polymer_abs__{run_id}/{stem}__{run_id}.png`, `per_polymer_rea__{run_id}/...`, `t50/t50__{run_id}.csv`。run_id はファイル名・フォルダ名に含まれている。
- **図**: 英語ラベル（Heat time (min), Absolute activity (a.u./s), REA (%)）、paper-style（apply_paper_style）, DPI 600。

### 良い点

- 折れ線を主、回帰は optional にしている。
- 指数減衰の制約（y0>0, k≥0, 減少トレンド、非正値はスキップ）で破綻を防いでいる。
- t50 は「半減期」の意味（exp: ln(2)/k）と plateau 版の 50% 到達時間が正しく実装されている。
- 永続カラーマップで ID ごとに色が固定されている。

### 改善の余地（必須ではない）

- **y 軸ラベル**: 現状 "Absolute activity (a.u./s)"。論文で単位を外したい場合は "Absolute activity" のみにするオプションがあってもよい。
- **1 枚に上下 2 段**: 要望の「1 枚にまとめても OK」には未対応。abs と REA は別 PNG。必要なら「1 枚・上段 abs・下段 REA」のオプションを追加可能。
- **GOx_Concentration_afterheat**: この run の fit 直下には summary*simple.csv, bo/, per_polymer*\*, t50/ が**ない**。古い Fit 実行のため。Fit を再実行すれば summary_simple → aggregate → plot_per_polymer_timeseries の流れで全て出力される。

---

## 3. t50 計算の整理

- **定義**: REA (%) は 0 min で 100% なので、t50 =「50% に落ちるまでの時間（分）」。
- **指数減衰 y = y0·exp(-k t)**
  - 半減期: t50 = ln(2)/k。意味は「初期値の半分になる時間」で破綻しない。
- **指数＋ plateau y = c + (y0-c)·exp(-k t)**
  - 50% に達する t は、target = 0.5·y0 として (target - c)/(y0 - c) = frac のとき t = -ln(frac)/k。
  - plateau c が 50% 以上なら「50% に落ちない」ので t50 は None。
- **制約**: y0 は t=0 の観測で固定、k≥0、減少トレンド（Spearman）で無意味なフィットを避けている。現状のロジックで妥当。

---

## 4. まとめ

| 項目         | 状態                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------- |
| フォルダ配置 | fit 直下はやや多いが、run 単位では整理されている。INDEX に per_polymer / t50 を追記するとよい。 |
| グラフ       | paper-grade・英語・永続カラーマップ・run_id 付きで実装済み。                                    |
| t50          | 定義・制約とも妥当。CSV は fit 実行時に t50/ に出力済み。                                       |
| 致命的な不備 | なし。GOx_Concentration_afterheat は Fit 再実行で揃う。                                         |

**推奨**: INDEX.md に per*polymer*\* と t50 の行を追加する。コードは現状のままでよいが、必要なら「1 枚・上下 2 段」オプションや y 軸ラベル表記のオプションを後から追加できる。
