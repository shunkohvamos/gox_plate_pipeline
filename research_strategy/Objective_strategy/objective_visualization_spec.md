# Objective Visualization Spec

## A) 目的関数候補
- 可視化対象の主スコアは `objective_loglinear_main = log(FoG*) + 1.0*log(U0*)` を固定。
- 比較表示として既存 `fog_activity_bonus_penalty` は維持するが、主図の順位付けは `objective_loglinear_main` を優先。

## B) Methods 文
We visualized the pre-specified primary score \(S_{\text{main}}=\log(\mathrm{FoG}^*)+1.0\log(U_0^*)\) by plotting \(\log(U_0^*)\) against \(\log(\mathrm{FoG}^*)\), overlaying iso-score lines \(y=S-\lambda x\), and ranking polymers by descending \(S_{\text{main}}\) with pre-fixed tie-breakers (\(|\Delta S|<0.02\): higher \(U_0^*\), then higher \(\mathrm{FoG}^*\), then smaller SEM, then polymer ID), while infeasible or QC-failed points were tracked as constraints and not numerically penalized in the objective.

## C) 感度分析
- 重み感度図: `supp_weight_sensitivity__{run_id}.png`
  - x: lambda
  - y: ranked polymer count, top5 overlap, Kendall tau, Spearman rho
- 閾値感度図: `supp_threshold_sensitivity__{run_id}.png`
  - x: U0* threshold (`0.7/0.8/0.9`)
  - y: ranked polymer count, top5 overlap, Kendall tau, Spearman rho
- 相関CSV: `supp_rank_correlation__{run_id}.csv`

## D) 実装対応
- Main図:
  - `mainA_log_u0_vs_log_fog_iso_score__{run_id}.png`
    - x: `log(U0*)`, y: `log(FoG*)`
    - 参照線: `(U0*=1, FoG*=1)` に対応する `x=0, y=0`
    - 直線等スコア: `y = S - lambda*x`
    - Pareto front 重ね書き
  - `mainB_u0_vs_fog_tradeoff_with_pareto__{run_id}.png`
    - x: `U0*`, y: `FoG*`
    - Pareto front
    - top-k ラベル
    - PMTAC型 (`U0*<1 and FoG*>1`) を明示
- Ranking図:
  - `objective_loglinear_main_ranking__{run_id}.png`
- CSV:
  - `objective_loglinear_main_ranking__{run_id}.csv`
  - `primary_objective_table__{run_id}.csv` (新列追加)
