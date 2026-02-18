# Next Tasks Roadmap (Decision-Critical)

## What was learned from simulation
- Raw FoG (`fog_raw`) selects PMTAC top in 3 run(s) among evaluable runs.
- Constrained objective with `theta=0.70` reduces PMTAC top picks to 2 and selects PMBTA top in 6 run(s).
- Overly strict setting (`theta=0.90`) increases PMPC top picks (1 run(s)) and can suppress stability gains.
- Best PMBTA beats PMPC in 8/8 runs; beats PMTAC in 5/8 runs.

## Recommended default policy
1. Objective for selection/BO: `fog_native_constrained` with `theta=0.70` (main).
2. Keep diagnostics: `fog_raw` and `fog_native_constrained_tradeoff` (supplementary only).
3. Keep quality flag (not hard exclusion by default): highlight rows with low `fit_r2`.

## Expected behavior after implementation
- Low-native PMTAC-like spikes should stop dominating top rank.
- Top picks should concentrate on PMBTA candidates with both usable native activity and improved FoG.
- Figure narrative shifts from rank inversion confusion to feasible high-stability selection.

## Remaining risks to manage
- Threshold sensitivity: top candidate may switch near the boundary (`theta` sensitivity must be reported).
- Reference fallback bias: when same-run reference is missing, same-round substitution can shift native_0.
- Fit uncertainty: low-SNR traces can inflate t50/FoG; keep fit-quality columns in review tables.
- Mechanistic ambiguity: low native_0 may be inhibition, viscosity, or assay interference (not always true instability).
- External validity: current conclusions are run-limited; avoid universal claims across all enzymes/polymer families.

## Figure plan for manuscript
1. Main: `fog_native_constrained_decision__{run}.png` (native gate + FoG ranking).
2. Main support: `all_polymers__{run}.png` with right decision chart panel.
3. Supplement: `fog_native_constrained_tradeoff__{run}.png` (trade-off scatter).
4. Supplement: representative REA+t50 curves (GOx / PMPC / PMTAC / best PMBTA).

## Immediate decision table (raw vs constrained)
| run_id | top_raw | top_cons_070 |
| --- | --- | --- |
| 260209-1 | PMBTA-3 | PMBTA-3 |
| 260209-2 | PMBTA-10 | PMBTA-10 |
| 260209-3 | PMBTA-11 | PMBTA-11 |
| 260211-1 | PMTAC | PMBTA-11 |
| 260211-2 | PMBTA-10 | PMBTA-11 |
| 260216-1 | PMTAC | PMTAC |
| 260216-2 | PMTAC | PMTAC |
| 260216-3 | PMBTA-10 | PMBTA-10 |