# Objective Methods Paragraph

## A) 目的関数候補
- Primary score: `objective_loglinear_main = log(fog_star) + 1.0*log(u0_star)`
- Readable scale: `objective_loglinear_main_exp = exp(objective_loglinear_main)`
- Comparator (legacy): `fog_activity_bonus_penalty`

## B) Methods 文
Primary endpoints were defined as solvent-matched initial activity \(U_0^*\) (`abs0_vs_solvent_control`) and solvent-matched thermal stability \(\mathrm{FoG}^*\) (`fog_vs_solvent_control`); before analysis, we fixed the primary objective to \(S_{\text{main}}=\log(\mathrm{FoG}^*)+1.0\cdot\log(U_0^*)\), computed only for positive finite endpoint values, and ranked candidates by descending \(S_{\text{main}}\) using a pre-declared tie-break sequence (if \(|\Delta S|<0.02\): higher \(U_0^*\), then higher \(\mathrm{FoG}^*\), then smaller SEM when available, then polymer ID); all infeasible measurements (QC failure, non-measurable values, missing values) were treated as constraints and reported separately, rather than being embedded as penalties in the objective function.

## C) 感度分析
- Lambda sweep: `{0.5, 0.75, 1.0, 1.25, 1.5}`
- Threshold sweep (secondary): `U0* >= {0.7, 0.8, 0.9}`
- Metrics: Kendall tau, Spearman rho, top-k overlap

## D) 実装対応
- Code mapping:
  - `src/gox_plate_pipeline/fog.py` (objective columns, ranking, figures)
  - `src/gox_plate_pipeline/bo_data.py` (BO learning column passthrough)
  - `scripts/run_bayesian_optimization.py` (parallel objective comparison)
- Output mapping:
  - `objective_loglinear_main_ranking__{run_id}.csv`
  - `mainA_log_u0_vs_log_fog_iso_score__{run_id}.png`
  - `mainB_u0_vs_fog_tradeoff_with_pareto__{run_id}.png`
  - `supp_weight_sensitivity__{run_id}.png`
  - `supp_threshold_sensitivity__{run_id}.png`
  - `supp_rank_correlation__{run_id}.csv`
