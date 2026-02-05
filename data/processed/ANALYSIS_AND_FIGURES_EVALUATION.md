# 解析全体・ベイズ最適化周辺のフラット評価

**評価日**: 2026-02-05  
**目的**: 研究に必要な図がすべて出せているか、正しく出せているか（グラデーション・軸・カラーバー衝突）、体裁、BOロジックの妥当性をフラットに評価する。  
**ゴール**: FoG を最大化する組成比を見つける。

---

## 1. 現状の出力図の一覧と役割

### 1.1 BO 実行で出る図（`bo_runs/{bo_run_id}/`）

| ファイル | 内容 | 用途 |
|----------|------|------|
| `ternary_mean_log_fog__*.png` | 三角図：予測平均 log(FoG) | 組成空間での期待性能 |
| `ternary_std_log_fog__*.png` | 三角図：予測標準偏差 | 不確実性の可視化 |
| `ternary_ei__*.png` | 三角図：Expected Improvement | 獲得関数（探索候補） |
| `ternary_ucb__*.png` | 三角図：UCB | 活用・探索のバランス |
| `xy_2x2_mean_std_ei_ucb__*.png` または `bma_mtac_2x2_*` | 2×2 パネル（Mean/Std/EI/UCB） | x,y または BMA–MTAC 平面での同内容 |
| `fog_ranking_all__*.png` | FoG ランキング（横棒） | 全ラウンドでの FoG 順位 |
| `t50_ranking_all__*.png` | t50 ランキング（横棒） | 全ラウンドでの t50 順位 |

### 1.2 その他パイプラインで出る図

- **Fitting**: per_polymer 曲線、plate_grid、QC（t_end ヒストグラム、R²、SNR 等）
- **FoG**: 図は直接出力していない（CSV のみ）

---

## 2. 図の品質（正しさ・体裁・衝突）

### 2.1 グラデーション（つぶれ）

- **Mean / EI / UCB**: `Normalize(vmin, vmax)` のみ。データ範囲が狭いと差が分かりにくいが、意図的なクリップはしていない。問題なし。
- **Std**: `PowerNorm(gamma=8)` を使用（`_STD_COLOR_GAMMA = 8.0`）。  
  - **評価**: 低い std を強調する設計で、std がほぼ一定に近い run では「高 std 側が一色に潰れて見える」ことがある。  
  - **推奨**: 論文用には gamma を設定可能にするか、線形ノルムをオプションで選べるとよい。現状でも「低不確実性域を見やすくする」という意味では妥当。

### 2.2 軸ラベル・体裁

- **三角図**: 頂点ラベルは英語（MPC/MTAC/BMA）、各辺に 0–100% の目盛りと矢印あり。`apply_paper_style()` を利用。問題なし。
- **2×2 xy / BMA–MTAC**: 軸ラベルは数式（$x$, $y$ または [BMA], [MTAC]）、カラーバーラベルも $\log(\mathrm{FoG})$ 等で統一。`constrained_layout=True` で余白調整。体裁は良好。
- **ランキング棒グラフ**: 単色（#4C78A8）。軸・タイトルは英語。polymer_id の色は `meta/polymer_colors.yml` の永続マップがあるが、**ランキング図では未使用**。ルール上は「polymer_id の色は永続マップで固定」のため、ここで polymer ごとに色を付けると一貫する。

### 2.3 ラベルとカラーバーの衝突

- **三角図**:  
  - カラーバー: `fig.colorbar(..., fraction=0.045, pad=0.05)` で軸の右側に配置。  
  - 「BMA (100%)」は (1.03, -0.03)、xlim は (-0.13, 1.13) なので、カラーバーは軸外右側にあり、通常はラベルと重ならない。  
  - **注意**: figsize やアスペクトを変えた場合、pad が小さいと右端ラベルとカラーバーが近づく可能性がある。**推奨**: `pad=0.08` 程度に増やすと安全。
- **2×2**: 各サブプロットに `fraction=0.046, pad=0.03` でカラーバー。`constrained_layout=True` により、通常はラベルとの重なりは起きにくい。現状で問題なし。

### 2.4 フォント・解像度

- `apply_paper_style()` で Arial → Helvetica → DejaVu Sans、フォントサイズ 6–7 pt、線 0.6–0.8 pt、savefig は 600 dpi PNG。Figure-rules に沿っている。
- 2×2 の数式は `_XY_MATH_RC` で DejaVu Sans の mathtext に統一。他図は paper_style のみで一貫している。

---

## 3. 研究で欲しくなるが「今は出ていない」図

以下は論文・報告でよく使うが、現状は未実装。

1. **Observed vs Predicted**  
   - 学習点での実測 log(FoG) vs GP の予測平均。  
   - モデルの当てはまり・バイアス・外れ値の確認に必須に近い。

2. **Residual の可視化**  
   - 残差を三角図または xy 平面上にプロット（色またはマーカーサイズ）。  
   - どの組成域で過大/過小予測かを見るため。

3. **ラウンド別・時系列のサンプリング位置**  
   - 各ラウンドで「どこを測ったか」を三角図上に round_id 別にプロット。  
   - BO の探索履歴として説得力がある。

4. **「次にやる組成」の図上ハイライト**  
   - next_experiment_top5 の上位 1–3 点を、ternary_mean や ternary_ucb の上にマーカーで明示。  
   - 表だけだと場所のイメージが伝わりにくい。

5. **FoG vs t50 のトレードオフ**  
   - 現在は priority_score で FoG/t50/EI を線形結合しているが、2 軸（FoG vs t50）の散布図がない。  
   - 多目的として扱う場合、Pareto  front の可視化があるとよい。

6. **モデル診断**  
   - length scale やノイズの要約は `bo_map_quality__*.json` に入っているが、図（例: ハイパーパラメータの要約パネル）はない。  
   - 監査用には現状の JSON で足りるが、論文用には簡易パネルがあると便利。

---

## 4. ベイズ最適化ロジックの評価

### 4.1 目的・設計

- **目的**: FoG 最大化 → 目的変数は `log_fog`（正の FoG なので log で扱うのは妥当）。
- **設計**: 2D GP（x,y または BMA–MTAC）、Matern-5/2、ARD/等方の切り替え、スパース時は等方＋トレンド。獲得関数は EI、EI が崩れたら UCB でランク。アンカー・リプリケート・exploit/explore 割合あり。**FoG 最大化として筋は通っている。**

### 4.2 現状の挙動（直近 run の例）

- `bo_map_quality__bo_2026-02-05_13-27-28.json` では `ei_collapsed: true`、`ei_max` が極小。  
  → 学習点が少なくモデルがほぼ「どこも同じ」と見ているか、すでに最良付近で EI がほぼ 0 になっている。
- コード上は `ei_max < 1e-8` のとき **exploit_ucb_fallback** で UCB 順に候補を選んでおり、提案は出ている。  
→ **EI 崩れ時のフォールバックは正しく動いている。**

### 4.3 「ロジックを丸ごと変える」必要があるか

- **現状の結論**: 「FoG を最大化する組成を探す」というゴールに対して、現行の GP + EI/UCB + バッチ多様性は標準的で妥当。**丸ごと変える必要はない**。
- **検討の余地がある点**:
  - **データが少ないとき**: 等方＋トレンドで安定化しているが、さらに慎重にするなら、事前平均を線形や 2 次に固定した GP も選択肢。
  - **獲得関数**: PI（Probability of Improvement）や Thompson サンプリングをオプションで追加する程度で十分。EI のままでも問題ない。
  - **多目的**: FoG と t50 を同時に最適化したいなら、Pareto 獲得関数や重み付きスカラー化を「オプション」として入れる価値はある。現状は priority_score で実務的に重み付けしているだけなので、図（FoG–t50 散布）を足すだけでも効果が大きい。

---

## 5. 追跡可能性・提案理由ログ（BO ルール）

- **run_id / bo_run_id**: 全 CSV と manifest に付与されている。OK。
- **提案理由ログ**:  
  - `bo_suggestions__*.csv`: selection_reason, pred_* , ei, ucb, constraint_* あり。  
  - `next_experiment_top5__*.csv`: priority_rank, selection_reason, 予測・重み・baseline あり。  
  → **BO-rules で求めている「候補ごとの設計変数・予測・獲得関数・制約・採否理由」は満たしている。**

---

## 6. まとめと推奨アクション

### 6.1 図の品質

- グラデーション: Std の PowerNorm は意図的。潰れが気になる場合は gamma の設定 or 線形オプションを検討。
- 軸・ラベル: 英語・論文向け体裁で問題なし。
- 衝突: 三角図のカラーバーを少し右にずらす（pad 増）と安心。2×2 は現状で可。

### 6.2 足りない図（優先度順）

1. **Observed vs Predicted**（必須に近い）  
2. **次実験候補の三角図上のハイライト**（説得力向上）  
3. **ラウンド別サンプリング位置**（BO 履歴）  
4. Residual 図、FoG–t50 散布、モデル診断パネルは「あるとよい」レベル。

### 6.3 BO ロジック

- FoG 最大化として現行設計でよい。**丸ごと変える必要はない。**  
- オプションで PI/Thompson、多目的用の重み・図を足す程度で十分。

### 6.4 その他

- ランキング棒グラフに `polymer_colors.yml` を反映すると、他図と色の一貫性が取れる。
- 図の出力は現状すべて PNG。ルールでは PNG 必須で PDF は補助のため、現状のままでよい。

---

## 7. 実装反映（2026-02-05）

上記の推奨をすべて反映した。

- **三角図カラーバー**: `pad=0.08` に変更（ラベルとの衝突防止）。
- **Std の色スケール**: `BOConfig.std_color_gamma` を追加。`None` で線形、数値で PowerNorm(gamma)。CLI は `--std_color_gamma`（既定 8.0）と `--std_color_linear`。
- **Observed vs Predicted**: `observed_vs_predicted__{bo_run_id}.png` を出力。
- **次実験候補のハイライト**: `ternary_mean_with_top_candidates__{bo_run_id}.png`（上位3点を星マーカーで表示）。
- **ラウンド別サンプリング**: `ternary_sampling_by_round__{bo_run_id}.png`。
- **残差図**: `ternary_residual__{bo_run_id}.png`（学習点を残差で色分け）。
- **FoG vs t50 散布図**: `fog_vs_t50_scatter__{bo_run_id}.png`。
- **モデル診断パネル**: `bo_model_diagnostic__{bo_run_id}.png`。
- **ランキング棒グラフ**: `--polymer_colors`（既定 `meta/polymer_colors.yml`）で polymer_id ごとの色を適用。FoG/t50 ランキングと FoG–t50 散布図で利用。

---

*このドキュメントは解析パイプラインと bo_engine/scripts のコード・実際の出力を確認したうえで記載した。*
