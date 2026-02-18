# Methods (one-page summary)

Single-page summary of definitions and settings for **thermal stability assay**, **initial-rate fitting**, **t50 / FoG**, and **Bayesian optimization**. Suitable for paper Methods or SI. Values below are filled from pipeline code defaults when you run **Update Methods one-page** (Run and Debug) or `scripts/update_methods_one_page.py`.

---

## 1. Thermal stability assay and REA

- **Signal**: Product accumulation over time (e.g. Amplex Red / resorufin). Per-well time series are recorded at several heating times (e.g. 0, 5, 10, 15, 20, 30 min at target temperature).
- **Absolute activity**: Initial rate (slope of the linear phase) at each heating time, from the selected window (see §2).
- **REA (Retained Enzyme Activity), %**: For each (plate, polymer, heat_min), REA_percent = 100 × (absolute activity at that heat_min) / (absolute activity at heat_min = 0). So 100% = no loss; lower values indicate thermal inactivation.

---

## 2. Initial-rate window selection (design principles)

- **Goal**: Estimate the **initial rate** as the slope of a **local linear approximation** in the **initial regime** (before substrate depletion, product inhibition, or strong curvature). Do not fit a straight line into an already curved or flat region.
- **Window selection priority** (in order):
  1. **Earlier start** is preferred (late start is penalized).
  2. **Higher linearity** (e.g. R²) within that early region.
  3. **Sufficient point count** (minimum number of points respected).
- **Leading-point skip**: Allowed only to skip a **true lag** (sustained flat or noisy start, e.g. mixing or equilibration). A shallow dip or single-point noise is not treated as lag; the first points remain eligible. A **skip cap** (number of points or time) is enforced.
- **Default implementation parameters** (from `compute_rates_and_rea` / `select_fit`):
  - Method: `{{ initial_rate.select_method }}`.
  - Minimum points in window: {{ initial_rate.min_points }} (rescue down to 4–5 with relaxed R²).
  - Maximum points: {{ initial_rate.max_points }}.
  - R² minimum: {{ initial_rate.r2_min }} (relaxed in rescue steps, e.g. 0.80 → 0.70 → 0.60 → 0.50 for shorter windows).
  - Start detection: `find_start={{ initial_rate.find_start }}`, `start_max_shift={{ initial_rate.start_max_shift }}`, `start_window={{ initial_rate.start_window }}`.
  - Monotonicity: `mono_min_frac={{ initial_rate.mono_min_frac }}`, `min_pos_steps={{ initial_rate.min_pos_steps }}`, `min_snr={{ initial_rate.min_snr }}`.
  - Maximum t_end: {{ initial_rate.max_t_end_s }} s (optional cap so that only the initial phase is used).
- **Audit**: When the start index is advanced (leading points skipped), the skip count or first fitted time is recorded in the output (CSV/log).

---

## 3. t50 (half-life) definition

- **t50**: Heating time (minutes) at which REA reaches a defined **threshold**. Unit: **{{ t50.unit }}** (throughout the pipeline).
- **Two canonical modes** (selected at run time; default: **{{ t50.default_mode }}**):

  | Mode      | Threshold              | Typical use |
  |-----------|------------------------|-------------|
  | **y0_half** | {{ t50.y0_half_description }} | Same-run relative; adapts if baseline differs. |
  | **rea50**   | Fixed REA = 50%        | Cross-run / cross-study comparison. |

- **Fitting**: REA vs. heating time (min) is fitted per (run, plate, polymer). Preferred: exponential decay (or exp with plateau); fallback: linear interpolation to threshold. Minimum 3 points; at least 2 distinct heating times.
- **Default in scripts**: `--t50_definition {{ t50.default_mode }}` (e.g. `fit_initial_rates.py`, `build_fog_plate_aware.py`, `run_bayesian_optimization.py` uses FoG built with the same definition).

---

## 4. FoG (Fold over GOx)

- **Definition**:  
  **{{ fog.formula }}**  
  (same run, or same plate / same round when using plate-aware FoG).
- **Interpretation**: FoG > 1 means the polymer–enzyme sample retains activity longer than bare GOx under the same conditions; FoG is a dimensionless thermal-stability ratio.
- **Constrained objective (BO)**: Only polymers with **sufficient native (unheated) activity** are included in the optimization objective. Default: `native_activity_rel_at_0 >= {{ fog.native_activity_min_rel }}` ({{ fog.native_activity_min_rel_pct }}% of GOx baseline at heat = 0). The objective column is **{{ fog.objective_column }}** (log(FoG) where the constraint is satisfied; NaN otherwise).
- **Plate-aware denominator**: For each (run, plate, polymer), the GOx t50 used in the denominator is from the same (run, plate) when available; otherwise a round-level representative (e.g. median) is used. Outlier GOx t50 values can be detected and optionally excluded; same-plate values outside a guard range fall back to the round representative.

---

## 5. Bayesian optimization (BO) settings

- **Objective**: Maximize **log(FoG)** ({{ bo.objective_column }}) on the design space of **ternary composition** (MPC, BMA, MTAC; fractions sum to 1).
- **Model**: Gaussian process regression (GPR), **{{ bo.kernel }}** kernel. Default: **ARD** (one length scale per component); for sparse design points (e.g. ≤{{ bo.sparse_isotropic_max_unique_points }} unique points), **isotropic** kernel is used by default to avoid unstable anisotropy (optional: `--disable_sparse_isotropic`).
- **Acquisition**: **EI (Expected Improvement)** by default; optional **UCB**. Batch proposals combine **exploit** (EI or UCB) and **explore** (high predictive uncertainty).
  - Default: `--acquisition {{ bo.acquisition }}`, `--exploration_ratio {{ bo.exploration_ratio }}`, `--n_suggestions {{ bo.n_suggestions }}`.
  - EI: `--ei_xi {{ bo.ei_xi }}` (jitter). UCB: `--ucb_kappa {{ bo.ucb_kappa }}` (or `--ucb_beta` if set).
- **Batch composition**: Part of the batch is reserved for **anchor** (fixed-composition re-measurements, e.g. {{ bo.anchor_polymer_ids }}) and **replicates** (re-measurements of high-value compositions). Default: `--anchor_fraction {{ bo.anchor_fraction }}`, `--replicate_fraction {{ bo.replicate_fraction }}`, `--anchor_polymer_ids {{ bo.anchor_polymer_ids }}`.
- **Constraints**: Composition fractions in [{{ bo.min_component }}, {{ bo.max_component }}] per component (default `--min_component {{ bo.min_component }}`, `--max_component {{ bo.max_component }}`); sum = 1; **minimum distance** between suggested points and between suggestions and existing training points (`--min_distance_between {{ bo.min_distance_between }}`, `--min_distance_to_train {{ bo.min_distance_to_train }}`).
- **Round correction**: **Anchor-based correction** of log(FoG) across rounds is **off by default** (`--anchor_correction` to enable), with a minimum number of shared anchor polymers (default: {{ bo.min_anchor_polymers }}).
- **Proposal grid**: Candidates are evaluated on a simplex grid (e.g. `--candidate_step {{ bo.candidate_step }}`); `--n_random_candidates {{ bo.n_random_candidates }}` for random (x,y) sampling where applicable.
- **Outputs**: Each BO run produces a unique **bo_run_id**; outputs include proposal log (candidates, acquisition values, constraints), **next_experiment_top5** table, surrogate maps (ternary: mean, std, EI, UCB; optional 2×2 panels), Observed vs. Predicted plot, and ranking tables (t50, FoG). All referenced run_ids and parameters are stored in the run manifest.

---

## 6. Traceability

- Every derived output (CSV, figures) is tied to a **run_id** (and **bo_run_id** for BO). A **run manifest** (JSON) records run_id, timestamp, git commit, input paths and hashes, and main parameters.
- Lineage (e.g. well-level ancestry) is kept in separate CSVs where needed; long lineage strings are not embedded in summary tables. Figure filenames include run_id or bo_run_id; raw paths are not written in figures.

---

## Regenerating this page from code

This file (**METHODS_ONE_PAGE.md**) is generated from **METHODS_ONE_PAGE_TEMPLATE.md** and the default values in code.

- **Run/debug で更新**: VS Code の Run and Debug で **「Update Methods one-page」** を実行すると、このファイルがコードのデフォルトで上書きされます。
- **手動**: リポジトリルートで `PYTHONPATH=src .venv/bin/python scripts/update_methods_one_page.py` を実行しても同じです。
- **コード変更時に更新** (optional): `pre-commit install` でフックを有効にすると、t50/FoG/初速/BO のデフォルトを定義しているファイルを変更してコミットするときに、自動でこのページが再生成されます。変更後に `research strategy/METHODS_ONE_PAGE.md` を add してから再度コミットしてください。

Defaults are defined in:

- **t50**: `src/gox_plate_pipeline/polymer_timeseries.py`; `src/gox_plate_pipeline/fog.py`.
- **Initial rate**: `src/gox_plate_pipeline/fitting/pipeline.py`, `selection.py`; `.cursor/rules/` (design rules).
- **BO**: `scripts/run_bayesian_optimization.py`; `src/gox_plate_pipeline/bo_engine.py`.
