# ベイズ最適化実装の監査レポート

Codex により実装された BO が、ご要望およびプロジェクトルールに沿っているかを確認した結果です。**コードの変更は行わず、確認のみ**です。

---

## 1. ご要望との対応

| 要望 | 実装状況 | 備考 |
|------|----------|------|
| 今までの BO ロジックを参考にしつつ、よりアップデートした BO を構築 | ✅ | GPR（Matern52 ARD）、log(FoG) 目的、アンカー補正オプション、batch 提案（exploit/explore）を実装。旧コードの思想を引き継ぎつつ、bo_learning / fog_plate_aware に合わせて整理されている。 |
| 実行とデバッグからワンクリックで BO 実行 | ✅ | `.vscode/launch.json` に「Bayesian Optimization（Plate-aware）」と「Bayesian Optimization（既存学習データ）」の 2 構成があり、`run_bayesian_optimization.py` を正しく起動する。 |
| 三角図をしっかり出力 | ✅ | 4 種類の三角図を PNG 出力：Predicted mean log(FoG)、Predictive std、EI、UCB。`apply_paper_style()` 使用・英語ラベルのみ。 |
| t50 のランキング表 | ✅ | CSV: `t50_ranking_by_round__{bo_run_id}.csv`、`t50_ranking_all__{bo_run_id}.csv`。図: `t50_ranking_all__{bo_run_id}.png`（全ラウンド集約の横棒グラフ）。 |
| FoG のランキング表 | ✅ | CSV: `fog_ranking_by_round__{bo_run_id}.csv`、`fog_ranking_all__{bo_run_id}.csv`。図: `fog_ranking_all__{bo_run_id}.png`（全ラウンド集約の横棒グラフ）。 |
| MolLogP はあとからでいい | ✅ | MolLogP / 重み付き MolLogP の三角図は未実装。要望通り後回しで問題なし。 |

---

## 2. 実装の流れ（意図どおりか）

1. **データ入力**
   - BO 学習: `bo_learning_plate_aware.csv`（`polymer_id`, `round_id`, `frac_MPC/BMA/MTAC`, `log_fog`）。
   - ランキング用: `fog_plate_aware.csv`（round_id, run_id, polymer_id, t50_min, fog, log_fog）。
2. **アンカー補正**（オプション）  
   ラウンド間で共通するポリマーの log_fog 中央値でシフトし、`log_fog_corrected` を生成。補正メタは `bo_summary__{bo_run_id}.json` の `anchor_correction` に保存。
3. **GPR**  
   (x, y) を正規化し、Matern52 ARD で `log_fog_corrected` をフィット。ハイパーパラメータは NLL で最適化。
4. **候補・獲得関数**  
   単体グリッド上で予測平均・標準偏差・EI・UCB を計算。制約（組成和=1、最小成分、既測点・候補間の最小距離）を満たす点のみ採用。
5. **batch 提案**  
   exploit（EI または UCB）と explore（標準偏差）の割合を `exploration_ratio` で制御し、多様な batch を選択。`selection_reason`（exploit_ei / exploit_ucb_fallback / explore_std / balanced_combo）を記録。
6. **出力**
   - 提案理由ログ: `bo_candidate_log__{bo_run_id}.csv`、`bo_suggestions__{bo_run_id}.csv`、`next_experiment_top5__{bo_run_id}.csv`。
   - ランキング: 上記 t50 / FoG の CSV と、全ラウンド集約の棒グラフ PNG。
   - 三角図: mean / std / EI / UCB の 4 枚。
   - 追跡用: `bo_summary__{bo_run_id}.json`、`bo_manifest__{bo_run_id}.json`（参照 run_id 一覧含む）。

旧コードでやっていた「時系列 REA から t50 → FoG」は、本プロジェクトでは既に `fog_plate_aware` / `bo_learning` に集約されているため、BO はその結果だけを読む形になっており、意図に合っています。

---

## 3. BO ルール（.cursor/rules/BO-rules.mdc）との整合

| ルール | 対応 |
|--------|------|
| 追跡可能性（run_id / bo_run_id、参照 run_id 一覧） | ✅ 全出力に `run_id`/`bo_run_id` を付与。manifest に `referenced_run_ids` を保存。 |
| 提案理由ログ（設計変数、予測・分散、獲得関数、制約、採否理由） | ✅ `bo_candidate_log` に全候補の制約・距離・EI/UCB、`bo_suggestions` に選ばれた候補と `selection_reason`。`next_experiment_top5` に優先度スコア・ランク・信頼区間等。 |
| アンカー補正の前提と記録 | ✅ 共通アンカーによるラウンドシフト。適用可否・アンカー名・シフト量を `bo_summary` の `anchor_correction` に保存。 |
| 目的関数（t50/FoG 中心、定義の固定） | ✅ 目的は log(FoG)。FoG は既存パイプラインの定義（t50_polymer / t50_GOx）をそのまま使用。 |
| 制約（組成和=1、最小成分） | ✅ 候補グリッドと選別ロジックで制約を満たす点のみ採用。`constraint_sum_ok` / `constraint_bounds_ok` をログに記録。 |
| batch 提案と多様性 | ✅ `min_distance_between` / `min_distance_to_train` で既測・候補間の距離を制約。exploit/explore の比率は `exploration_ratio` で明示。 |
| 可視化（英語のみ、bo_run_id をファイル名に） | ✅ 図のテキストは英語。ファイル名に `{bo_run_id}` を含む。 |
| テスト（特定生データに依存しない） | ✅ `test_bo_engine.py` は合成の learning/fog CSV で `run_bo` を実行し、必須出力の存在・提案数・next_experiment の列を検証。unittest で OK。 |

---

## 4. データ経路の確認

- **Plate-aware 実行時**  
  launch「Bayesian Optimization（Plate-aware）」は `--rebuild_learning` を付け、  
  `fog_plate_aware_round_averaged.csv` → `build_bo_learning_data_from_round_averaged` → `bo_learning_plate_aware.csv` を更新したうえで、  
  `bo_learning_plate_aware.csv` と `fog_plate_aware.csv` を `run_bo` に渡している。  
  → 意図どおり。
- **既存学習データ実行時**  
  「Bayesian Optimization（既存学習データ）」は `--rebuild_learning` なしで、既存の `bo_learning_plate_aware.csv` と `fog_plate_aware.csv` をそのまま使用。  
  → 意図どおり。
- **FoG ランキング用 CSV**  
  `_load_fog_plate_aware` は `round_id`, `run_id`, `polymer_id`, `t50_min`, `fog`, `log_fog` を必須としている。  
  `fog_plate_aware.csv`（`build_fog_plate_aware` の per-row 出力）はこれらをすべて持つ。  
  → 問題なし。

---

## 5. 図のスタイル

- `bo_engine` は `fitting.core.apply_paper_style()` を `plt.rc_context` で使用。
- 三角図のコーナーラベルは "MPC", "MTAC", "BMA"（英語）。
- ランキング棒グラフのタイトル・軸ラベルも英語。  
→ Figure-rules / BO-rules の「英語のみ」「paper-grade」に沿っている。

---

## 6. 軽微な確認事項（任意の改善）

- **ランキング図のラウンド別**  
  現状、棒グラフは「全ラウンド集約」のみ（`fog_ranking_all__*.png`, `t50_ranking_all__*.png`）。  
  ラウンド別のランキング表は CSV（`*_by_round__*.csv`）で出力されているが、ラウンド別のランキング棒グラフは出していない。  
  → 必要であれば、後から「ラウンド別 FoG/t50 ランキング図」を追加する形で対応可能。
- **x, y の学習データ**  
  `bo_learning_plate_aware.csv` に `x`, `y` が無くても、`bo_engine` は `_xy_from_frac(frac_MPC, frac_BMA, frac_MTAC)` で内部計算している。  
  → 現状の学習データのままでも BO は正しく動作する。

---

## 7. 総合判定

- **ワンクリック実行**: launch 2 種で「Plate-aware」と「既存学習データ」の両方が用意されている。
- **三角図**: mean / std / EI / UCB の 4 枚を出力している。
- **t50 / FoG ランキング**: 両方とも「ラウンド別・全ラウンド」の CSV と、全ラウンド集約の棒グラフを出力している。
- **旧 BO の分析とプロジェクトへの適合**: 旧コードの GPR・EI/UCB・三角図・ランキングの流れを汲みつつ、bo_learning / fog_plate_aware / 追跡可能性・提案理由ログに合わせて整理されている。

**結論: ご要望およびプロジェクトの BO ルールに沿って正しく実装されていると判断できます。**  
上記のとおり、コード変更は行わず確認のみで完了しています。ラウンド別ランキング図が必要になった場合のみ、追加実装を検討すれば十分です。
