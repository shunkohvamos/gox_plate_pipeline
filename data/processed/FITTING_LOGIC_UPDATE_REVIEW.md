# フィッティングロジック更新レビュー

コード変更を確認し、前回指摘した2つの問題点に対する改善を評価しました。

---

## 改善1: Step jump 検出の改善

### 変更前の問題点
- 1ステップの増加が range の 25% 超なら検出
- **1点の外れ値でも打ち切られる**可能性
- 「本当の step jump」と「1点の外れ値」を区別できない

### 変更後の実装 (`core.py:289-338`)

**改善点**:

1. **外れ値に頑健な range 計算**:
   ```python
   rng = _percentile_range(y)  # 5-95パーセンタイルを使用
   ```
   - `y.max() - y.min()` ではなく、5-95パーセンタイルの範囲を使用
   - 外れ値の影響を軽減

2. **セグメントベースの判定**:
   ```python
   k = 3 if n >= 8 else 2
   pre_seg = y[max(0, i - k + 1) : i + 1]
   post_seg = y[i + 1 : i + 1 + k]
   pre = float(np.median(pre_seg))
   post = float(np.median(post_seg))
   ```
   - 前後のセグメント（k=3または2点）の**中央値**で判定
   - 1点のスパイクの影響を軽減

3. **Level-shift-like の判定**:
   ```python
   pre_span = float(abs(pre_seg[-1] - pre_seg[0]))
   post_span = float(abs(post_seg[-1] - post_seg[0]))
   trend_guard = 0.35 * threshold
   if pre_span > trend_guard or post_span > trend_guard:
       continue
   ```
   - 前後のセグメントが**平坦かどうか**をチェック
   - 自然な上昇（trend）と level-shift を区別

4. **Persistency check**:
   ```python
   post_min = float(np.min(post_seg))
   if post_min > (pre + 0.5 * threshold):
       return int(i)
   ```
   - ジャンプ後のセグメントの**最小値**が閾値以上であることを確認
   - 1点のスパイクを無視

### 評価

**✅ 大幅に改善されています**

- **1点の外れ値**は、前後のセグメントの中央値で判定されるため、影響が軽減される
- **Level-shift-like の判定**により、自然な上昇と区別できる
- **Persistency check**により、1点のスパイクを無視できる

**残存する可能性のある問題**:
- 外れ値が**連続2点以上**で、かつ前後のセグメントが平坦な場合、誤検出の可能性は残る（ただし、これは稀なケース）

---

## 改善2: `find_fit_with_outlier_removal` の優先順位改善

### 変更前の問題点
- 優先順位: Progressive rescue → `find_best_short_window` → `find_fit_with_outlier_removal`
- **「少ない点でのフィット」が「外れ値を除外したより多くの点でのフィット」より優先される**状態

### 変更後の実装 (`pipeline.py:882-904`)

**改善点**:

```python
# Before short-window rescue, try robust outlier-removal first.
# This avoids preferring a tiny short window when a larger
# outlier-cleaned fit is available.
sel = find_fit_with_outlier_removal(
    t=t_arr,
    y=y_arr,
    min_points=6,
    r2_min=0.80,
    slope_min=slope_min,
    min_snr=2.0,
    outlier_sigma=3.0,
)
if sel is not None:
    used_params = {...}
else:
    sel = None
for rescue_max_points, rescue_r2_min in rescue_attempts:
    if sel is not None:
        break  # find_fit_with_outlier_removal が成功したら progressive rescue をスキップ
    # ...
```

**新しい優先順位**:
1. **`find_fit_with_outlier_removal`** (先に試す)
2. Progressive rescue (5点 → 4点 → 3点)
3. `find_best_short_window` (最後の手段)

### 評価

**✅ 改善されています**

- **外れ値を除外したより多くの点でのフィット**が優先される
- コメントも明確で、意図が伝わる

**注意点**:
- `find_fit_with_outlier_removal` は `min_points=6` なので、**7点以上のデータ**でないと試されない
- 6点以下のデータでは、progressive rescue が先に試される（これは適切）

**動作フロー**:
1. **7点以上のデータ**: `find_fit_with_outlier_removal` を先に試す → 成功すれば終了、失敗すれば progressive rescue → `find_best_short_window`
2. **6点以下のデータ**: progressive rescue を先に試す → `find_best_short_window`

---

## 総合評価

### ✅ 両方の問題点が改善されています

1. **Step jump 検出**: 1点の外れ値による誤検出を大幅に軽減
2. **除外フィットの優先順位**: 「外れ値を除外したより多くの点でのフィット」が優先される

### 推奨事項

- 変更内容は適切で、前回指摘した問題点を解決している
- コードの可読性も向上している（コメントが明確）
- 実データでの動作確認を推奨（特に step jump 検出の改善が実際のデータでどの程度効果的か）

---

## 確認事項

1. **`find_fit_with_outlier_removal` が成功した場合、progressive rescue をスキップする設計**:
   - これは意図的な設計か？
   - もし `find_fit_with_outlier_removal` が r2=0.80 で成功したが、progressive rescue で r2=0.90 の5点フィットが得られる可能性がある場合、どちらを優先すべきか？
   - 現状の設計では、**「より多くの点でのフィット」を優先**している（r2 が低くても）

2. **`find_fit_with_outlier_removal` の r2_min=0.80 と `find_best_short_window` の r2_min=0.55**:
   - `find_fit_with_outlier_removal` の方が厳しい条件だが、先に試されるため、適切なバランスになっている

---

（コードは一切変更していません。レビューのみです。）
