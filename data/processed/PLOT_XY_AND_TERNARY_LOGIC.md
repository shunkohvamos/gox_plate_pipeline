# xy 2x2 と三角図の描画・計算ロジック整理

計算と描画の対応関係を、コードを追って整理したメモ。不具合調査用。

---

## 1. 座標の定義（共通）

- **組成 (frac_MPC, frac_BMA, frac_MTAC)**: 合計 1。三角図の「中身」。
- **bo_data.xy_from_frac**（および bo_engine._xy_from_frac）:
  - `y = frac_BMA + frac_MTAC`（BMA+MTAC の合計、0〜1）
  - `x = frac_BMA / (frac_BMA + frac_MTAC)`（BMA の割合）。`y==0` のとき x は NaN（PMPC）。
- **三角図用 2D 座標**（bo_engine._ternary_xy_from_frac）:
  - `tx = frac_bma + 0.5 * frac_mpc`
  - `ty = frac_mpc * (sqrt(3)/2)`
  - 頂点: (0,0)=MTAC 100%, (1,0)=BMA 100%, (0.5, √3/2)=MPC 100%。

---

## 2. xy 2x2 パネル（_plot_xy_2x2_panels）

**使う条件**: `use_bma_mtac_coords=False`（legacy xy 座標で GP を学習したとき）。

- **グリッド**
  - `x_grid = y_grid = np.linspace(0, 1, 241)`
  - `X1, X2 = np.meshgrid(x_grid, y_grid)`（デフォルト `indexing='xy'`）
    - 形状: (n_grid, n_grid)。`X1[i,j] = x_grid[j]`, `X2[i,j] = y_grid[i]`
    - つまり **配列の (i, j) = (y のインデックス, x のインデックス)** → 行 = y、列 = x。
- **predict 用 1D グリッド**
  - `grid = np.column_stack([X1.T.ravel(), X2.T.ravel()])`
  - `(X1.T)[i,j] = x_grid[i]`, `(X2.T)[i,j] = y_grid[j]`
  - 行優先 ravel → `grid[k]` の k = i*n_grid + j のとき `grid[k] = (x_grid[i], y_grid[j])`。**x が遅い・y が速い**。
- **予測と reshape**
  - `mu, std = model.predict(grid)` → `mu[k]` は `(x_grid[i], y_grid[j])` での値（k = i*n_grid + j）。
  - `MU = mu.reshape((n_grid, n_grid))` → **MU[i,j] = 点 (x_grid[i], y_grid[j]) の値**。第1軸=x、第2軸=y。
- **pcolormesh**
  - `pcolormesh(X1, X2, Z)` では「四角形 (i,j) の色 = Z[i,j]」「その角の座標 = (X1[i,j], X2[i,j])」。
  - 欲しい対応: 画面の (x, y) = (x_grid[j], y_grid[i]) に値「点 (x_grid[j], y_grid[i]) での予測」を表示。
  - つまり **Z[i,j] = 値 at (x_grid[j], y_grid[i]) = MU[j,i] = (MU.T)[i,j]**。
  - コードでは `Z = MU`（または SD/EI/UCB）を渡し、**Z.T** を渡している → `(MU.T)[i,j]` で正しい。

**結論（xy 2x2）**: グリッド・reshape・Z.T の対応は一貫している。横軸=x、縦軸=y で、(x,y) と色の対応は意図どおり。

---

## 3. BMA–MTAC 2x2 パネル（_plot_bma_mtac_2x2_panels）

**使う条件**: `use_bma_mtac_coords=True`（通常。GP は (frac_BMA, frac_MTAC) で学習）。

- **グリッド**
  - `bma_grid = mtac_grid = np.linspace(0, 1, 241)`
  - `B, M = np.meshgrid(bma_grid, mtac_grid)` → `B[i,j]=bma_grid[j]`, `M[i,j]=mtac_grid[i]`。行=MTAC、列=BMA。
- **有効域**
  - `valid = (B + M <= 1) & (B>=0) & (M>=0)`。三角領域内だけ predict。
- **予測**
  - `X_pred = np.column_stack([B.ravel()[valid.ravel()], M.ravel()[valid.ravel()]])` → 有効な (bma, mtac) の並び。
  - `mu, std = model.predict(X_pred)`。この並びは `valid.ravel()` の True の並びと一致。
- **Z への代入**
  - `Z_mu.ravel()[valid.ravel()] = mu`。なので **Z_mu[i,j] が valid[i,j] のとき、その値は (B[i,j], M[i,j]) = (bma_grid[j], mtac_grid[i]) での予測**。第1軸=MTAC、第2軸=BMA。
- **pcolormesh**
  - `pcolormesh(B, M, Z_masked)`。`(B[i,j], M[i,j])` と `Z_masked[i,j]` が同じ (bma, mtac) を指しており、横=BMA・縦=MTAC で一致。

**結論（BMA–MTAC 2x2）**: 有効マスクと ravel の順序の対応も取れており、描画の対応は正しい。

---

## 4. 三角図（_plot_ternary_field）

- **データソース**
  - 渡しているのは **plot_cand = _build_plot_frame(gp, learning, cfg)**（`cand` ではない）。
  - `_build_plot_frame` は `_generate_simplex_grid(step=ternary_plot_step, min_component=0)` で **密なシンプレックスグリッド** を生成し、同じ model で `pred_log_fog_mean` / `pred_log_fog_std` / `ei` / `ucb` を付けている。
- **座標と値**
  - `mpc, bma, mtac = cand["frac_MPC"], cand["frac_BMA"], cand["frac_MTAC"]`（列をそのまま配列に）。
  - `tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)` → 行 i の組成に対応する (tx[i], ty[i])。
  - `z = cand[value_col]` → 行 i の値。**同じ行 i で (tx[i], ty[i]) と z[i] は必ず同じ組成・同じ予測**。
- **Triangulation と描画**
  - `tri = mtri.Triangulation(tx, ty)` → (tx, ty) の Delaunay 三角分割。頂点番号は配列のインデックス i。
  - `ax.tripcolor(tri, z, ...)` → 頂点 i に値 z[i] を付与して補間。**(tx[i], ty[i]) と z[i] の対応は崩れない**。
- **頂点の意味**
  - (0,0)=MTAC, (1,0)=BMA, (0.5, √3/2)=MPC とラベルされており、`_ternary_xy_from_frac` の式と一致。

**結論（三角図）**: 行インデックス i で (frac_MPC/BMA/MTAC) → (tx, ty) → z が一対一で紐づいており、Triangulation も同じインデックスを使うため、**計算・描画の対応は正しい**。

---

## 5. 潜在的な「見た目」要因（バグではなく）

- **三角図**
  - 点が (mpc, bma) の等間隔グリッドを (tx, ty) に写しただけなので、(tx, ty) 空間では**均一でない**。端で細長い三角形ができると、線形補間の向きでグラデーションが歪んで見えることがある。
  - **levels / 解像度**: 以前は tricontourf の levels 数で帯状に見えた可能性。現在は tripcolor(gouraud) で連続的に塗っている。
- **xy / BMA–MTAC**
  - グリッドは矩形で均一。軸と Z の対応は上記の通り。もし「勾配の向きがおかしい」と感じる場合は、**モデルが (x,y) または (BMA,MTAC) のどちらで学習されているか**と、**描画で使っている座標がそれと一致しているか**を再確認するとよい（現状は use_bma_mtac_coords 時は BMA–MTAC 2x2 のみ使用し、xy 2x2 は legacy 用）。

---

## 6. 参照コード位置（bo_engine.py）

| 内容 | 行付近 |
|------|--------|
| _xy_from_frac | 537 |
| _ternary_xy_from_frac | 545 |
| _generate_simplex_grid | 554 |
| _build_candidate_frame（候補に pred_* を付与） | 1218 |
| _build_plot_frame（三角図用の密グリッド） | 1267 |
| _plot_ternary_field（tripcolor） | 1450 |
| _plot_xy_2x2_panels（meshgrid, Z.T, pcolormesh） | 1608 |
| _plot_bma_mtac_2x2_panels（B,M, valid, pcolormesh） | 1702 |
| run_bo 内で plot_cand を生成して _plot_ternary_field に渡す箇所 | 1915, 2010 付近 |

---

## 7. チェックリスト（不具合を疑うとき）

1. **xy 2x2**: 使用しているか？ 通常は BMA–MTAC 2x2 のみ。xy は `use_xy_coords` のときだけ。
2. **学習座標と描画座標**: BMA–MTAC 学習なら BMA–MTAC 2x2 と三角図の pred は同じ (frac_BMA, frac_MTAC) から計算されているか → どちらも cand/plot_df の frac_* と model.predict(frac_BMA, frac_MTAC) で一致。
3. **三角図のデータ**: 三角図は `plot_cand`（_build_plot_frame）か `cand`（_build_candidate_frame）か → **plot_cand**。より細かい ternary_plot_step のグリッド。
4. **numpy.meshgrid**: デフォルトは `indexing='xy'`。`indexing='ij'` にすると (i,j) と (x,y) の対応が変わるので、変更時は要確認。

このドキュメントは、計算ロジックと描画の対応を追った結果をまとめたものです。実際に「ここがおかしい」という箇所があれば、上記の対応関係と照らしてインデックス・座標のどれがずれているかを切り分けできます。
