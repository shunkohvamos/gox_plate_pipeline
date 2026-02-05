# Codex 修正の意図通り性の確認と、現設計の弱点

## 1. 意図通りかどうかの確認

分析で挙がった **4 方針** と、実装の対応を事実ベースで照合した。

---

### 方針 1: 分母ガード（FoG 層）

**意図**: same_plate GOx を無条件採用せず、round 中央値比で QC し、異常なら same_round（中央値/MAD ベース）へフォールバック。denominator_source で監査可能に。

**実装確認**:

- `fog.py` の `build_fog_plate_aware`:
  - 引数 `gox_guard_same_plate=True`（デフォルト）、`gox_guard_low_threshold` / `gox_guard_high_threshold`（未指定時は outlier 閾値と同一）。
  - round 代表 GOx を `round_gox_median`（または mean）で計算し、`guard_low_threshold = round_gox_median * guard_low_mult`、`guard_high_threshold = round_gox_median * guard_high_mult` を算出。
  - 各 (run_id, plate_id, polymer_id) で、same_plate GOx がある場合に `gox_guard_same_plate` が True なら、`gox_t50 < guard_low_threshold` または `gox_t50 > guard_high_threshold` のとき `use_same_plate = False` とし、`gox_t50 = round_gox_fallback`、`denominator_source = "same_round"`。
  - フォールバックした plate は `warning_info.guarded_same_plate` に (round_id, run_id, plate_id, same_plate_gox_t50_min, fallback_gox_t50_min, guard_*_threshold_min) を記録。
  - `per_row_df` に `denominator_source`（`"same_plate"` / `"same_round"`）が必ず入る。

**判定**: 意図通り。通常データでは same_plate を維持し、異常時のみ same_round に切り替えている。

---

### 方針 2: round 集約のロバスト化（学習データ層）

**意図**: (round_id, polymer_id) の目的値は「平均」ではなく median / trimmed mean を検討し、分散（不確かさ）を保持して BO に渡す。

**実装確認**:

- `fog.py` の round-averaged 集計（約 750 行付近）:
  - 各 (round_id, polymer_id) で `robust_fog = nanmedian(g["fog"])`、`robust_log_fog = nanmedian(g["log_fog"])`、`log_fog_mad = nanmedian(|log_fog - robust_log_fog|)` を計算。
  - `mean_fog` / `mean_log_fog` も従来通り出力（互換性のため）。
  - `round_averaged_df` の列: `round_id`, `polymer_id`, `mean_fog`, `mean_log_fog`, `robust_fog`, `robust_log_fog`, `log_fog_mad`, `n_observations`, `run_ids`。
- `bo_data.py` の `build_bo_learning_data_from_round_averaged`:
  - `has_robust_cols = ("robust_fog" in df.columns) and ("robust_log_fog" in df.columns)` で判定。
  - ある場合: `fog_col = "robust_fog"`, `log_col = "robust_log_fog"`、`objective_source = "robust_round_aggregated"`。
  - 学習用の目的値は `log_fog` 列に `log_col` の値を書き、`n_observations` と `log_fog_mad` を学習行にそのまま渡す。
- `bo_engine.py` の `_load_bo_learning`:
  - CSV に `n_observations` / `log_fog_mad` があれば数値化して保持し、行フィルタで落とさない。

**判定**: 意図通り。round 集約は median ベースのロバスト列を追加し、BO 学習では robust を優先し、不確かさ（n_observations, log_fog_mad）を保持している。

---

### 方針 3: モデルをノイズ前提に（BO 層）

**意図**: 最低限、heteroskedastic / noisy 対応（qNEI 寄り）。観測ごとの相対ノイズを入れる。

**実装確認**:

- `bo_engine.py` の `GPModel2D`:
  - `obs_noise_rel: np.ndarray` を保持。
  - `fit(..., obs_noise_rel=None)` で、None なら `noise_rel = ones(n)`、否则 `noise_rel` を検証（長さ n、有限かつ >0）。
  - 共分散の対角: `K[i,i] += (noise**2) * noise_rel[i] + jitter`（NLL と最終 K の両方）。
- `_estimate_obs_noise_rel(learning_df, min_rel, max_rel)`:
  - 初期 `rel = ones(n)`。
  - `n_observations` があれば `rel *= 1/sqrt(n_observations)`。
  - `log_fog_mad` があれば、MAD の中央値で正規化し 0.5〜3 にクリップした MAD 比を `rel` に乗算。
  - 中央値で正規化して median=1 にし、`min_rel`〜`max_rel` にクリップして返す。
- `run_bo`:
  - `cfg.enable_heteroskedastic_noise` が True なら `obs_noise_rel = _estimate_obs_noise_rel(learning, ...)`、否则 `ones(len(learning))`。
  - `GPModel2D.fit(X, y, ..., obs_noise_rel=obs_noise_rel)` に渡す。
  - manifest に `obs_noise_rel_min/median/max` を記録。

**判定**: 意図通り。観測ごとの相対ノイズを n_observations と log_fog_mad から推定し、GP の対角ノイズに反映している（獲得関数は従来 EI/UCB のままだが、予測分散はノイズ付きで計算される）。

---

### 方針 4: バッチを exploit / explore / control / replicate の明示配分に

**意図**: control は anchor 組成の再提案、replicate は選定候補の複製を追加。24 本を内訳で固定配分する運用。

**実装確認**:

- `BOConfig`:
  - `control_fraction`, `replicate_fraction`, `control_count`, `replicate_count`（Optional）、`control_anchor_ids`。
- `_resolve_batch_counts(cfg)`:
  - `n_total = n_suggestions`。`control_count` が指定されていればそれ、否则 `n_control = round(n_total * control_fraction)`（n_total>=6 のとき）。`replicate_count` 同様。`n_unique = n_total - n_control - n_replicate`。その中で `n_exploit` / `n_explore` を `exploration_ratio` で分割。
- `_select_diverse_batch`:
  - まず exploit（EI、落ちていれば UCB）で n_exploit 件選択 → 続けて explore（pred_log_fog_std）で n_explore 件 → その後 `_control_targets_from_learning(..., control_anchor_ids, n_controls=n_control)` で anchor 等を優先し、候補グリッド中で最も近い点を「control_anchor_*」として追加 → 足りなければ balanced_combo で埋める。
  - `cand` に `policy_n_control`, `policy_n_replicate` を書き込む。
- `_append_replicates(selected, n_replicate)`:
  - `selection_reason` が "exploit" を含む行をプール（なければ全 selected）。そのプールから循環で n_replicate 件をコピーし、`selection_reason = "replicate_of_<order>"`、`selection_order` を新規で付与して連結。
- `run_bo`:
  - `selected_unique = cand[cand["selected"]==1]`、`n_replicate = cand["policy_n_replicate"].iloc[0]`、`selected = _append_replicates(selected_unique, n_replicate)`。提案 CSV は `selected` を保存。

**判定**: 意図通り。control は anchor 組成の再測定用、replicate は選定候補の複製として明示され、バッチ内訳が設定で制御されている。

---

### CLI・テスト

- **build_fog_plate_aware.py**: `--disable_gox_guard`, `--gox_guard_low_threshold`, `--gox_guard_high_threshold`, `--gox_round_fallback_stat` を追加。デフォルトはガード有効。
- **run_bayesian_optimization.py**: `--control_count`, `--replicate_count`, `--disable_heteroskedastic_noise`, `noise_rel_min`, `noise_rel_max`、および `control_fraction` / `replicate_fraction` を BOConfig に渡している。
- **test_fog_and_bo_data_robust.py**: (1) 極端な same_plate GOx でガードが発動し denominator_source が same_round になること、(2) round_averaged に robust_* と log_fog_mad があること、(3) BO 学習が robust_log_fog を優先し objective_source と log_fog_mad が正しく渡ること、を検証。

**判定**: 意図した動作をテストで担保している。

---

## 2. 結論：意図通りに修正できているか

- **方針 1〜4 はいずれも「事実ベース」で意図通りに実装されている。**
- 分析で指摘した「間違いの中心＝分母品質が悪い GOx でも same_plate を無条件採用」「round 内集約が単純平均」「anchor は round 一括シフトのみ」に対して、
  - 分母ガードで same_plate を条件付きにし、
  - round 集約に robust 列を追加し BO はそれを優先し、
  - ノイズを観測ごとに GP に取り込み、
  - バッチを exploit/explore/control/replicate に明示配分している。

---

## 3. 現設計の弱点（追加で思う点）

以下は、今回の修正とは別に、現状の設計・運用で気になる点です。

1. **獲得関数は依然として EI/UCB**
   - ノイズは GP の対角にしか入っておらず、**qNEI / Noisy EI** のような「バッチ・ノイズを明示した獲得関数」にはなっていない。複数点同時提案の最適化や、ノイズ付き EI の理論的な利点はまだ取りにきれていない。

2. **Round をランダム効果として入れていない**
   - anchor 補正は「round ごとの加算シフト」のみ。**Round を階層モデル（ランダム効果）として GP に組み込む**案は未実装。inter-round をやめる運用にすると、Round 差はノイズか別要因として残る。

3. **control の「再提案」が候補グリッドの最近傍のみ**
   - control は「学習データにいる anchor 組成」に最も近い**グリッド点**を選んでいる。実組成がカタログと完全一致しない場合、厳密には「同じ組成の再測定」ではなく「近い組成の再測定」になる。NMR 等で実組成を入れる場合は、その点を errors-in-variables と組み合わせる余地がある。

4. **replicate の元は exploit プールのみ**
   - 複製は「exploit で選んだ候補」から循環で取っている。分析で言う「選定候補の複製」には合っているが、「探索候補の複製」でノイズを推定する選択肢はない。用途に応じて replicate の取り元を設定で切り替えられるとよい。

5. **目的関数は依然「単一時点の log FoG」**
   - AUC や下側分位点などの目的は未対応。bo_learning の列は「1 つの log_fog」のみ。AUC/分位点を入れるには、FoG 計算〜round 集約〜BO 学習のパイプライン拡張が必要。

6. **FoG 計算の round 代表 GOx が「除外後」の median/mean**
   - ガードの閾値は `round_gox_median` ベースで、outlier 除外は round 代表計算の前に適用されている。一方、**round 内の plate 数が少ない場合**、1 plate の異常 GOx が round_median を大きく引きずる可能性は残る。plate 数・サンプル数が少ない round では、代表値のロバスト性（例: trimmed mean）のオプションがあるとよい。

7. **warnings / manifest の一元化**
   - ガード発動は `warning_info.guarded_same_plate` と warnings.md に書かれる。BO 側の manifest には FoG の denominator_source や guard 発動数は直接入っていない。**「どの BO 実行がどの FoG ビルド（どのガード設定・発動件数）に依存したか」**を manifest で一意にたどれると、再現性・監査がさらに明確になる。

8. **回帰テストが 1 本**
   - `test_fog_and_bo_data_robust` はガードと robust 列・BO 学習の優先を検証しているが、**bo_engine の run_bo 全体（heteroskedastic + control + replicate）** をモックで回す統合テストは別途あると安心。既存の test_bo_engine が run_bo を叩いているなら、そこに robust 列・n_observations/log_fog_mad を含む学習データで length_scale や提案数が妥当かを見るケースを足すとよい。

---

## 4. 次の一手（分析・Codex の提案と一致）

- 実データで `build_fog_plate_aware.py` を再実行し、warnings.md の guard 発動件数を確認する。
- その出力（round_averaged 含む）で `run_bayesian_optimization.py` を回し、提案バッチ（control / replicate 含む）と bo_summary の length_scale・obs_noise_rel が意図通りか確認する。

以上で、Codex の修正は意図通りであり、現設計の弱点を列挙した。
