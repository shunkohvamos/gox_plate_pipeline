# (BMA, MTAC)座標系の再評価：ternary mapとの対応

## 重要な気づき

**現在のxy_2x2パネルでも、各モノマーの量が直感的にわからないから、ternary mapを別途用意して対応させている。**

→ ならば、**(BMA, MTAC)座標系でも、同様にternary mapを用意すれば良い**。

## 現在の実装の確認

### xy_2x2パネルとternary mapの関係

1. **xy_2x2パネル**: `(x, y)` 空間での2Dヒートマップ
   - x = BMA/(BMA+MTAC), y = BMA+MTAC = 1-MPC
   - 各モノマーの量が直感的にわからない

2. **ternary map**: `(frac_MPC, frac_BMA, frac_MTAC)` から直接描画
   - 各モノマーの量が頂点からの距離で直感的にわかる
   - **既に実装済み**

3. **両方とも同じ候補データ(`cand`)から生成**
   - `cand` には `frac_MPC`, `frac_BMA`, `frac_MTAC` が含まれている
   - ユーザーは両方を見て対応関係を理解

## (BMA, MTAC)座標系での対応

### 実装の確認

1. **2Dヒートマップ**: `(BMA, MTAC)` 空間
   - MPCの量が直感的にわからない（MPC = 1 - BMA - MTAC）
   - **新規実装が必要**（`_plot_bma_mtac_2x2_panels()` のような関数）

2. **ternary map**: 既存のまま（変更不要）
   - `_plot_ternary_field()` は既に `(frac_MPC, frac_BMA, frac_MTAC)` から描画
   - 各モノマーの量が頂点からの距離で直感的にわかる
   - **変更不要**

3. **両方とも同じ候補データ(`cand`)から生成**
   - `cand` には `frac_MPC`, `frac_BMA`, `frac_MTAC` が含まれている
   - ユーザーは両方を見て対応関係を理解

### 対応関係の例

```
(BMA, MTAC) = (0.2, 0.3)
→ MPC = 1 - 0.2 - 0.3 = 0.5
→ ternary map上では: MPC=0.5, BMA=0.2, MTAC=0.3 の点
```

2Dヒートマップ上の点 `(BMA, MTAC)` と ternary map上の点は、**同じ組成 `(frac_MPC, frac_BMA, frac_MTAC)` に対応している**。

→ ユーザーは両方を見て、対応関係を理解できる。

## デメリットの再評価

### 以前の懸念：「MPCがどれくらいかが直感的にわからない」

**再評価**: 
- **xy_2x2パネルでも、各モノマーの量が直感的にわからない**
- **ternary mapで対応している**
- **(BMA, MTAC)座標系でも、ternary mapで対応できる**

→ **この懸念は解消される**

### 残るデメリット

1. **実装の追加が必要**
   - `_plot_bma_mtac_2x2_panels()` のような関数を新規実装
   - ただし、`_plot_xy_2x2_panels()` をベースにすれば、変更箇所は限定的

2. **解釈性の問題（軽微）**
   - MPCが主要な設計変数の場合、影響を直接評価できない
   - ただし、ternary mapで確認できるので、実用上の問題は小さい

3. **既存コードとの互換性**
   - xy_2x2パネルの可視化コードの変更が必要
   - ただし、ternary mapは変更不要

## 結論

**最大のデメリット（MPCが直感的にわからない）は、ternary mapで解決できる。**

**推奨**: 
- **(BMA, MTAC)座標系 + ternary map** の組み合わせは実現可能で、実用的
- 実装の追加は必要だが、既存のternary mapをそのまま使える
- 距離感が自然になるメリットが大きい

**実装方針**:
1. GPの入力: `(frac_BMA, frac_MTAC)` の2次元
2. 2Dヒートマップ: `(BMA, MTAC)` 空間で描画（新規実装）
3. ternary map: 既存のまま（変更不要）
4. 両方とも同じ候補データから生成
