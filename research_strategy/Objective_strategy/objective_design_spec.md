# Objective Function Design Spec

## Research philosophy (研究思想)

- **ゴール**: GOx の活性を邪魔することなく、耐熱性（t50）を上げること。
- **絶対活性（初期活性）の扱い**:
  - 下げないことが望ましい（下げることはマイナス）。
  - 何なら上げることはプラスに扱う。
- 目的関数設計では「t50 を上げる」ことに加え、「絶対活性を維持する／上げる」ことを正に評価し、下げることを負に評価する方針とする。

---

## A) 目的関数候補
- Primary endpoints:
  - `u0_star = abs0_vs_solvent_control` (initial activity vs solvent-matched control)
  - `fog_star = fog_vs_solvent_control` (stability FoG vs solvent-matched control)
- 主解析（採用）:
  - `objective_loglinear_main = log(fog_star) + 1.0 * log(u0_star)`
  - `objective_loglinear_main_exp = exp(objective_loglinear_main)`
- 候補（比較用、主解析では非採用）:
  - 候補A: `fog_star + 1.0 * log(u0_star)`
  - 候補B: `log(fog_star) + 1.0 * log(u0_star)` (採用)
  - 候補C: `fog_activity_bonus_penalty` (既存ベースライン)
- 計算条件:
  - `u0_star > 0` かつ `fog_star > 0` のときのみ算出
  - 非有限値は `NaN` とし、制約/QCとして別列で管理
- 事前固定タイブレーク:
  - 1位条件: `objective_loglinear_main` 降順
  - 同点条件: `|ΔS| < 0.02` の場合 `u0_star` が高い方
  - 次条件: `fog_star` が高い方
  - 次条件: SEM があれば小さい方、なければ `polymer_id` 昇順
- PMTAC型対策:
  - Pareto front を明示可視化
  - 最終順位は上記の固定ルールで決定（後付け禁止）

## B) Methods 文
Primary endpoints were pre-defined as solvent-matched initial activity \(U_0^* = \mathrm{abs0\_vs\_solvent\_control}\) and solvent-matched stability \(\mathrm{FoG}^* = \mathrm{fog\_vs\_solvent\_control}\); the primary analysis used a pre-specified log-linear score \(S_{\text{main}}=\log(\mathrm{FoG}^*)+1.0\cdot\log(U_0^*)\), evaluated only when both endpoints were positive and finite, with infeasible or non-measurable cases (QC failure, missingness, out-of-range measurements) handled as constraints rather than encoded in the score; candidates were ranked by descending \(S_{\text{main}}\), and when \(|\Delta S|<0.02\), ties were resolved in a fixed order by higher \(U_0^*\), then higher \(\mathrm{FoG}^*\), then smaller SEM when available, and finally lexicographic polymer ID.

## C) 感度分析
- 重み感度: `lambda in {0.5, 0.75, 1.0, 1.25, 1.5}`
  - top-k overlap、Kendall tau、Spearman rho を `lambda=1.0` 基準で算出
- 閾値感度（補助解析）: `U0* >= {0.7, 0.8, 0.9}`
  - 再ランキング、top-k overlap、Kendall tau、Spearman rho を無閾値主解析基準で算出
- 出力:
  - `supp_weight_sensitivity__{run_id}.png`
  - `supp_threshold_sensitivity__{run_id}.png`
  - `supp_rank_correlation__{run_id}.csv`

## D) 実装対応
- 追加列（解析I/F）:
  - `objective_loglinear_main`
  - `objective_loglinear_main_exp`
  - `rank_objective_loglinear_main`
- 追加CSV:
  - `objective_loglinear_main_ranking__{run_id}.csv`
  - `primary_objective_table__{run_id}.csv` に上記列を追加
- BO比較運用:
  - 主: `log_fog_activity_bonus_penalty`
  - 比較: `objective_loglinear_main`
  - 差分: `bo_comparison__{bo_run_id}.csv`

## 参照文献運用（高確度/要再確認）
- 高確度:
  - Hickman et al., Anubis: Bayesian optimization with unknown feasibility constraints. DOI: `10.1039/D5DD00018A` URL: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00018a
  - Schmitzer et al., Thermal Stabilization of Enzymes with Molecular Brushes. DOI: `10.1021/acscatal.7b03138` URL: https://pubs.acs.org/doi/abs/10.1021/acscatal.7b03138
  - Wang et al., Machine learning-driven multi-objective optimization of enzyme combinations. DOI: `10.3390/pr13061936` URL: https://www.mdpi.com/2227-9717/13/6/1936
- 要再確認:
  - Griffiths et al., Bayesian optimization with known experimental and design constraints for chemistry applications. URL: https://dspace.mit.edu/bitstream/handle/1721.1/146011/d2dd00028h.pdf
  - Berkenkamp et al., Safe Bayesian optimization with safety constraints (robotics adaptation). URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485113/
