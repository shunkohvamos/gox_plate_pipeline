# 初速フィッティング・ロジック分析（コード変更なし）

現行フィッティングの「分類ロジック」と「どんなデータでも正しくフィットするか」を整理したドキュメントです。

---

## 1. 全体フロー（1ウェルあたり）

```
[入力] (plate_id, well) の time_s, signal
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 前処理 (candidates.py)                                          │
│   ・ inf/NaN 除去、time_s でソート                                  │
│   ・ step jump 検出 → 検出時は「ジャンプ前まで」にデータを truncate     │
│   ・ 点数 n が min_points 未満 → 4点以上かつ先頭の正のトレンドなら      │
│     候補1つ (全区間) を生成して rescue、否则 候補なし → 後段で excluded  │
│   ・ それ以外: find_start で start_idx を決定 → 候補窓を生成           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 候補からの選択 (selection.select_fit)                            │
│    複数パラメータセットで試行 (fallback_attempts)                     │
│    → 全部 FitSelectionError → 3. Rescue へ                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Rescue (候補なし or select_fit が全て失敗)                         │
│    ・ 短い窓で再候補生成 (5点→4点→3点、r2 段階緩和)                    │
│    ・ まだ失敗 → find_best_short_window (先頭〜前半中心、r2≥0.55)       │
│    ・ まだ失敗 → find_fit_with_outlier_removal (1〜2点除外)           │
│    ・ まだ失敗 → FitSelectionError → 4. Column-1 救済 or excluded     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 選択後の後処理 (Step 1〜7)                                        │
│    Step 1: full-range outlier skip が良ければ採用                   │
│    Step 2: try_extend_fit                                            │
│    Step 3: try_skip_extend (末尾外れ値スキップして拡張)                │
│    Step 4: try_extend_fit 再度                                       │
│    Step 5: detect_internal_outliers                                  │
│    Step 6: detect_curvature_and_shorten (曲率で接線窓に短縮)          │
│    Step 7: apply_conservative_long_override                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 最終 safety と出力                                               │
│    _enforce_final_safety(r2_min, slope_min, ...)                    │
│    → 通過すれば status=ok、否則 ここで例外なら excluded               │
│    excluded かつ col==1 のときのみ 2〜3 点で再フィットして ok に       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 分類の軸（どんなロジックで分かれているか）

### 2.1 データの「形」による分岐

| 分岐 | 条件 | 結果 |
|------|------|------|
| **点数不足** | n < min_points | 4点以上かつ先頭が正トレンドなら候補1つ、否则 候補なし → rescue または excluded |
| **Step jump** | 1ステップが range の 25% 超 | ジャンプ前まで truncate、start_idx=0 で全区間を候補に含める |
| **増加トレース** | 全体の net が正 | _find_start_index は「立ち上がり前のラグ」を検出して start_idx を遅らせうる |
| **減少トレース** | 全体の net が負 | ラグ判定・ガードレールのあと、y を反転して同様の最小値探索。浅いディップは start_idx=0 のまま |

### 2.2 候補生成の「開始点」

- **start_idx = 0**: step jump 時、または _find_start_index が「ラグなし／浅いディップ」と判断したとき。
- **start_idx > 0**: 「本当のラグ」（持続的平坦・ノイズの後の立ち上がり）があると判断したとき。窓はすべて start_idx 以降からしか始まらない。

→ 「初速」の定義は「start_idx 以降の、できるだけ早い線形区間」に依存している。

### 2.3 選択メソッド (select_fit) による分類

- **force_whole が True かつ条件満たす**: 同じ start_idx のうち最も長い窓を採用（slope 落ち・R²・trim 条件あり）。→ **force_whole** / **force_whole_trim1** / **force_whole_trim2**
- **initial_positive**:
  - 傾きが max の (1 - slope_drop_frac) 以上の窓だけ残す（曲率ガード）。
  - 「早い窓の傾きが遅い窓の 1.3 倍超」なら、早い窓を優先（**initial_positive_early_steep**）。
  - 否则、**t_end が最も小さい窓**を採用（同点なら r2↑, n↑, t_start↑）。
- **best_r2**: 合格窓のうち R² 最大。
- **best_score**: R² / SNR / n / t_end / slope_se 等のスコア加重和で最大。

→ 通常は **initial_positive** なので、「合格窓のうち t_end が早いほう」が選ばれ、その結果として「初速に近い短い窓」か「やや長い窓」かが決まる。

### 2.4 パラメータ緩和の段階 (fallback_attempts)

- r2_min: 0.98 → 0.92 → 0.98(min_dy=0) → 0.88(mono 緩和) → 0.80(pos_steps=0).
- どれかで select_fit が通れば、その used_params が後段（Step 7 の safety 含む）まで使われる。

→ 「きれいなデータ」は厳しめ、「ノイジーなデータ」は緩い閾値で 1 本のフィットを選ぶ設計。

### 2.5 Rescue による分類

- **Progressive rescue**: 5点/4点/3点の固定長窓で候補を再生成し、r2_min を 0.80〜0.50 で段階緩和。通れば **select_method_used** は initial_positive 等のまま。
- **find_best_short_window**: 先頭〜前半（max_start_frac=0.5）、4〜8 点、r2≥0.55。→ **last_resort**
- **find_fit_with_outlier_removal**: 1〜2 点除外で線形フィット。→ **outlier_removed**

→ ここまでで「候補が空」または「全 fallback で不合格」なら excluded（col==1 なら 2〜3 点救済あり）。

### 2.6 選択後の後処理による「置き換え」

| Step | 条件のイメージ | 置き換え先 |
|------|----------------|------------|
| 1 | full-range で 1〜2 点除外したフィットの R² が、現在 sel の R² と同程度以上 | **full_range_outlier_skip** |
| 2, 4 | 窓を右に伸ばしても R² がほぼ落ちない | **_ext** |
| 3 | 窓の直後の点が外れ値 → スキップして拡張 | **skip** |
| 5 | 窓内 1〜2 点を外すと R² が 0.005 以上改善 | **_intskip** |
| 6 | 窓の前半と後半で傾きが 15% 以上落ちる → 前半だけの接線窓に短縮 | **_tangent** |
| 7 | 中間短窓の過大評価・last_resort の過小評価などを救済 | **post_full_ext** / **post_long_ext** / **post_early_ext** / **post_full_planA** 等 |

→ 同じ「元の select_method_used」からでも、後処理で **full / long / early** に差し替わるかどうかで最終的な「分類」が変わる。

### 2.7 最終的な「分類」の出方（select_method_used の例）

- **normal**: initial_positive, initial_positive_early_steep, force_whole, best_r2, best_score
- **rescue**: last_resort, outlier_removed
- **full-range**: full_range_outlier_skip
- **extend/skip**: *_ext, *skip*, *_intskip*
- **curvature**: *_tangent
- **override**: *post_full_ext*, *post_long_ext*, *post_early_ext*, *post_full_planA*
- **column-1**: *_col1_rescue*（excluded からの救済時のみ）

「データの形」を直接名付けた列はなく、**上記のフローと閾値の組み合わせで**結果が決まる。

---

## 3. 「どんなデータが来ても正しくフィットするか」

### 3.1 正しくフィットしやすいデータ

- 点数が十分（≥ min_points）。
- 全体が増加トレンドで、初速に相当する区間がはっきり線形（R² 高め）。
- 先頭がラグなら「持続的平坦」で、_find_start_index が適切に start_idx を遅らせる。
- ノイズが小さく、down_steps / mono_frac / snr が閾値を満たす。

→ こうしたデータは **initial_positive**（または force_whole）で「初速に近い窓」が選ばれ、後処理でも大きく壊されない設計になっている。

### 3.2 正しくフィットしづらい・失敗しうるパターン

1. **中間の短い窓だけ R² が高く、初速は R² が低い**
   - initial_positive は「傾きが floor 以上」の窓のうち **t_end が早い**ものを取るが、早い窓が r2_min や snr で落ちると、遅い窓（中間）が選ばれる。
   - Step 7 の override で full/long に差し替えられる**が**、差し替え後の _enforce_final_safety で r2_min=0.98 を要求するため、全体 R² が 0.98 未満だと却下され、**中間窓のまま**になる（過大評価の可能性）。

2. **ラグのあと一気に立ち上がる（G7 型）**
   - 先頭がノイジーで、last_resort の短い窓だけが通ると、傾きが過小になる。
   - Step 7 の「last_resort かつ full の傾きが 1.25 倍」で full に替わる**が**、last_resort 時は used_params の r2_min が 0.55 なので、full の R² が 0.55 以上なら通過。**ここは救済されうる**。

3. **全区間がかなり線形だが、初速窓より全体の傾きが小さい（熱失活など）**
   - 「初速優先」なので、早い窓が選ばれ、全体は採用されない。意図どおりの挙動。

4. **点数が少ない（4 点以上だが min_points 未満）**
   - 先頭の正トレンドがなければ候補なし → rescue または excluded。4 点ちょうどでノイズが大きいと、rescue も通らず excluded になりうる。

5. **減少トレース**
   - _find_start_index の減少用ガードレール（y_range、net_change、early_decline、noise_ratio）で「ラグなし」とされると start_idx=0。減少トレース向けの「初速」定義が増加と完全同一ではないため、意図しない窓が選ばれる可能性はある。

6. **Step jump が 25% 未満の「段差」**
   - 検出されず、全区間が候補に入る。その後の select_fit で「早い窓」が選ばれれば大きな問題にはならないが、 jump 直後の高い傾きが選ばれる可能性はある。

### 3.3 結論

- **「どんなデータが来ても必ず正しくフィットする」とは言えない。**
  - 正しくフィットするかは、「初速」の定義（start_idx 以降のできるだけ早い線形区間）と、閾値（r2_min, snr, mono_frac, slope_drop_frac 等）の組み合わせに強く依存する。
  - 以下は**現状の設計では保証されていない**:
    - 中間だけ R² が高い「偶然の上昇」を避けて、常に全体 or 初速窓を選ぶこと（Step 7 の safety が厳しい）。
    - ラグ型で「中間以降 or 全体」を常に選ぶこと（last_resort 経路では救済されうるが、条件が揃わないとそのまま）。
    - 点数が少ない・ノイズが大きい場合の「必ず 1 本出す」こと（col==1 以外は excluded になりうる）。
- **意図どおりに働きやすいデータ**は、**単調増加で初速区間がはっきり線形なもの**。ラグがある場合は「持続的平坦」なら start_idx が遅れ、last_resort に落ちた場合は Step 7 で full に替わる余地がある。
- **分類**は、ファイル名や well 名ではなく、**点数・トレンド・R²・傾き・t_ratio・method 名**などの**汎用条件**で分岐している。ただし閾値（0.98, 0.85, 1.25, 1.40 等）が固定のため、**データのスケールやノイズレベルが想定外だと**、同じ「形」でも別の枝に乗り、望まない結果になる可能性はある。

---

## 4. まとめ

- **分類の仕方**: データの「形」を直接ラベルするのではなく、**前処理（step jump / 点数 / start_idx）→ 候補生成 → select_fit（force_whole / initial_positive / best_r2 等）→ 複数段の後処理（full-range skip, extend, skip, internal outlier, curvature, long override）** の結果として、**select_method_used** や **窓の位置・長さ** が決まる。
- **どんなデータでも正しくフィットするか**: **いいえ**。初速定義と閾値に依存し、中間窓の過大評価・ラグ型の過小評価・少点数・高ノイズなどでは、現状の閾値のままでは望ましくない or excluded になるケースがある。**「どんなデータが来ても」を目指すなら、閾値の見直しや Step 7 の safety 緩和、さらに条件の明確化が必要。**

（コードは一切変更していません。）
