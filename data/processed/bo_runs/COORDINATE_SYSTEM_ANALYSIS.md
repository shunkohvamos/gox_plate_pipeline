# BO座標系の分析と改善案

## 問題の整理

### 現在の(x, y)座標系の問題点

現在の実装では：
- **x** = frac_BMA / (frac_BMA + frac_MTAC) → BMA/MTAC比
- **y** = frac_BMA + frac_MTAC = 1 - frac_MPC → 非MPC量

**問題**: `y` が小さい領域（MPC≈1、ホモポリマー近傍）で、`x` は物理的にほぼ意味を持たないのに、RBFカーネルだと `x` 方向の差も「差」として効いてしまう。

**例**:
- MPC=0.95, y=0.05 の場合
  - x1=0.2 → BMA=0.01, MTAC=0.04
  - x2=0.8 → BMA=0.04, MTAC=0.01
  - x1とx2の差: 0.6（見かけ上大きい）
  - 実際の組成差（BMA）: 0.03（非常に小さい）

→ RBFカーネルは `x` 方向の差を「大きい差」として扱うため、**距離感が歪む**。これが端（ホモポリマー近傍）や辺でEI地形が変にギザる/探索が偏る原因になりうる。

---

## 推奨される代替案

### 代替案A: (BMA, MTAC)を直接使う（**最も推奨**）

**メリット**:
- **最も手堅い**: 設計空間がそのまま三角形（制約つき2D）
- **距離感が自然**: 実際の組成差に比例した距離
- **実装が簡単**: 既存のコード資産を壊しにくい
- **可視化は既存のternary mapでOK**: 変更不要

**実装**:
- GPの入力: `(frac_BMA, frac_MTAC)` の2次元
- MPCは `1 - frac_BMA - frac_MTAC` で自動決定
- 候補生成: 既存の `_generate_simplex_grid()` をそのまま使用
- 可視化: 既存のternary mapをそのまま使用

**変更箇所**:
- `GPModel2D.fit()` の入力: `learning[["frac_BMA", "frac_MTAC"]]`
- `_build_candidate_frame()` の `X_grid`: `cand[["frac_BMA", "frac_MTAC"]]`
- 距離計算: `(frac_BMA, frac_MTAC)` 空間で計算

---

### 代替案B: log-ratio (ILR/ALR)変換

**メリット**:
- compositional dataの定石
- 距離の解釈が「比」に沿って自然

**デメリット**:
- 0があると∞になるので、端点（ホモポリマー）にεを入れる設計が必要
- 実装がやや複雑

**実装**:
- ILR (Isometric Log-Ratio) または ALR (Additive Log-Ratio) 変換
- 2次元の制約なし空間でGPを学習
- 予測時に逆変換して組成に戻す

---

### 代替案C: 現在の(x,y)を残す + カーネル調整

**メリット**:
- 既存コードの変更が最小限

**デメリット**:
- 実装が中途半端だと逆にバグ温床
- 「xの無意味領域」を殺すロジックが必要（例: yが小さいときはx方向の長さスケールを極端に大きくする）

**実装**:
- ARDカーネルで `length_scale_x` を `y` の関数にする
- 例: `length_scale_x(y) = base_scale * (1 + alpha / (y + epsilon))`

---

## 推奨: 代替案Aを採用

**理由**:
1. **実装が最も簡単**: 既存の候補生成ロジックをそのまま使える
2. **距離感が自然**: 実際の組成差に比例した距離
3. **可視化は変更不要**: 既存のternary mapをそのまま使用
4. **既存コード資産を壊しにくい**: 変更箇所が限定的

**実装手順**:
1. `BOConfig` に `use_bma_mtac_coords: bool = False` を追加（デフォルトはFalseで後方互換性を保つ）
2. `GPModel2D.fit()` で、`use_bma_mtac_coords=True` のときは `(frac_BMA, frac_MTAC)` を入力とする
3. `_build_candidate_frame()` で、`use_bma_mtac_coords=True` のときは `cand[["frac_BMA", "frac_MTAC"]]` を `X_grid` とする
4. 距離計算も `(frac_BMA, frac_MTAC)` 空間で行う
5. 可視化は既存のternary mapをそのまま使用（変更不要）

**注意点**:
- `(frac_BMA, frac_MTAC)` 空間は制約 `frac_BMA + frac_MTAC ≤ 1` があるが、候補生成は既に `_generate_simplex_grid()` で制約を満たしているので問題なし
- 距離計算も制約内で行うので、自然な距離になる

---

## まとめ

- **2Dにすること自体は正しい**: 三元系は自由度2なので、2D表現は数学的に正当
- **問題は「(x, y)の取り方」**: 現在の(x, y)は距離感を歪めうる
- **推奨**: 代替案A `(BMA, MTAC)` を直接使う
- **3次元BOは不要**: 実質2自由度なので、3次元にする意味は薄い
