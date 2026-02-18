# Fit QC Report

- Generated: 2026-02-18 10:24:10.044864

## (a) OK / EXCLUDED
- Total wells: 56
- OK: 56
- EXCLUDED: 0
- OK rate: 100.0%

- CSV: csv/fit_qc_summary_overall.csv
- CSV (by plate): csv/fit_qc_summary_by_plate.csv
- CSV (by heat): csv/fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 96 / 606 s
- q10: 111 s
- q25: 156 s
- q50: 216 s
- q75: 606 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 10.7%
- t_end ≤ 240 s : 51.8%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 56
- Pearson r: -0.4892
- Spearman ρ: -0.2618

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: csv/fit_qc_select_method_counts.csv
- initial_positive_promote_long_ext: 24 (42.9%)
- initial_positive: 19 (33.9%)
- initial_positive_tangent: 5 (8.9%)
- initial_positive_ext: 2 (3.6%)
- initial_positive_early_steep_delayed_steep: 1 (1.8%)
- initial_positive_skip2: 1 (1.8%)
- initial_positive_intskip1: 1 (1.8%)
- initial_positive_promote_long_ext_tangent: 1 (1.8%)
- initial_positive_post_tangent_under: 1 (1.8%)
- initial_positive_post_broad_overfit_ext: 1 (1.8%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.6311 / 0.9999
- R² q10: 0.9564
- R² q25: 0.9875
- R² q50: 0.999
- R² q75: 0.9995
- R² q90: 0.9997

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 0.9474 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 3.408 / 260.3
- snr q10: 9.661
- snr q25: 26.6
- snr q50: 76.08
- snr q75: 133.8
- snr q90: 174.4

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- excluded wells: 0
