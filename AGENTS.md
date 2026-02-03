# Project rules for AI agents (Codex, etc.)

This file is the single source of project premises. Codex and other extensions that do not read Cursor Rules should use this file (e.g. `@file AGENTS.md` or add to context). Cursor Rules live under `.cursor/rules/`; this document is a consolidated copy of the essential rules.

---

## 1. Core design (provenance & reproducibility)

- **Purpose**: Reproducible evaluation of enzyme thermal stability (t50, REA, etc.) and optimization.
- **Provenance**: Every derived output (CSV/TSV/JSON/figures) must be traceable to raw data.
  - Each run has a `run_id`; all output CSVs must include a `run_id` column.
  - One **run manifest (JSON)** per run: run_id, timestamp, git commit, input paths/sha256/mtime/size, CLI/params.
  - Lineage: one row per well in a separate CSV; do not embed long lineage strings in summary CSVs.
- **Figures**: All in-plot text (title/axis/legend/annotation) must be **English only**. Do not embed raw file paths in figures; use run_id in filenames and track via manifest.
- **Numerical definitions**: Do not change definitions (e.g. initial slope → absolute activity → REA) without explicit reason, impact analysis, and test updates. Do not change numeric logic for "looks better" only.

---

## 2. Initial rate fitting (design)

### Note (generality)

- **Do not make extreme logic changes tailored to specific data.** Do not add thresholds or exceptions that fit only a particular plate, well, or file. Keep rules **data- and terminology-agnostic** (abstract).
- **The logic must be such that optimal fitting is achieved regardless of what data comes in the future.** Base behavior on the **ideal fit** (local linear approximation in the initial regime) and **constraints** (skip only true lag, contiguous segment → all points eligible by default, etc.) so that new data is handled by the same principles.

### 2.1 Ideal fitting (what we want)

- **Definition of initial rate**: The slope of the **local linear approximation** in a region where the reaction is in its **initial regime** — i.e. substrate depletion, product inhibition, and background effects are still small, so the slope is interpretable as rate.
- **Background**: Signal often rises as product accumulates; slope may decrease over time (e.g. thermal inactivation), so the **earliest** valid linear window captures initial rate before curvature. Do not fit a straight line into an already curved region.
- **Ideal choice of window**: The **earliest** contiguous window that (1) is sufficiently linear (e.g. R² above threshold), (2) has sufficient points, and (3) is **reaction-valid** (the slope reflects rate, not artifact). Prefer **reaction validity over "looking linear"** (e.g. do not prefer a later, flatter window with higher R² if it no longer represents initial rate).
- **Window selection priority** (in order): (1) Earlier start (penalize late start), (2) Higher linearity within that early region, (3) Sufficient point count.

### 2.2 Constraint: when may we move the start index forward?

- **Only to skip a true lag.** A **true lag** is a **sustained** flat or noisy section at the beginning that is **not** part of the initial reaction (e.g. mixing, equilibration, dead time). Do **not** move the start forward for a **shallow dip** — a small local minimum that is already **well above the first point** (e.g. >12% of signal range above y[0]). That indicates we are already in the rising phase with noise; treat it as noise, not lag → keep start_idx=0 so the first points remain eligible.
- **Do not impose "first N points always in" globally.** Traces with a true lag would break. Allow skipping leading points only when there is a sustained flat/noisy start, with a **defined cap** (count or time).

### 2.3 Constraint: when the fitting domain is a single contiguous segment

- **Setting**: Sometimes the pipeline (or a prior step) **restricts** the data to one **contiguous segment** (e.g. points before a detected discontinuity, or points in a chosen regime). That segment is the **fitting domain**.
- **Rule**: In that segment, **all points are eligible by default** (start_idx=0 within the segment). Do **not** run start-point detection on that segment in a way that excludes the beginning of the segment, unless there is a **true lag within the segment itself**. The intent is: "use this whole segment for fitting" unless the segment clearly starts with a lag.
- **Exception within the segment**: If the segment itself has a noisy/flat lead-in followed by a clear linear rise, it is correct to take only the **clear initial-rate portion** (start after the noise), not necessarily from the very first index of the segment.

### 2.4 Design principle: rescue locally, keep normal case unchanged

- **"救済は局所的に、正常系は不変"**: When fixing edge cases, do not change behavior for wells that already fit correctly. After changes, check that only the targeted trace types change; if many wells change, the design is invading the normal case.

### 2.5 Decreasing traces (if supported)

- First ask: is the negative slope **reaction** or **measurement artifact** (drift, bleach, etc.)? Rescue only with guardrails: (1) physical validity, (2) control consistency, (3) minimum dynamic range, (4) consistency across replicates. Do **not** lower R² threshold globally; limit rescue to specific trace types.

### 2.6 Forbidden

- Do not extend a linear fit into a clearly curved region just for high R².
- Do not prefer a "clean" short window at the end of the trace (anti-initial-rate).
- Do not allow unbounded leading-point skip.
- Do not change start/selection logic in a way that breaks traces that currently fit correctly (e.g. true-lag traces, or segments where only the clear initial-rate portion should be taken).

### 2.7 Audit (logging)

- When leading points are skipped (start index > 0), record the skip (e.g. start_idx or number of points skipped, or time of first fitted point) in the output (CSV or log) so that results are auditable.

### 2.8 Tests

- Use synthetic/small data; no hard dependence on real file/well names. Verify: (1) early linear window wins on monotonic concave data, (2) limited leading skip when true lag at start, (3) no escape to short end window for R², (4) shallow-dip-only: first points eligible (start_idx=0), (5) when domain is one contiguous segment: all points in that segment eligible by default (start_idx=0), unless true lag within the segment.

---

## 3. Figures (paper-grade, PNG, English)

- **Policy**: All figures are **paper-grade**, regardless of use (diagnostic/debug/final). No default matplotlib look.
- **Language**: In-plot text **English only**. No full-width Japanese in figures.
- **Typography**: Prefer **Arial** → Helvetica → DejaVu Sans. One font family for the whole figure. Final size ~5–8 pt (aim 6–7 pt). Panel labels (a,b,c…) 8 pt bold.
- **Lines & markers**: Line width ~0.6–0.9 pt; markers minimal and distinguishable when printed; grid minimal.
- **Layout**: Aspect ratio is **not** fixed; choose by content. Prefer ~90 mm single-column width; minimize margins (e.g. bbox_inches="tight"); legend must not hide data.
- **Color**: No red-green-only or jet-like colormaps. Use only meaningful colors. **polymer_id** colors from a single persistent map (e.g. yml/json); same ID → same color across figures.
- **Output**: **PNG only** at 300–600 dpi (600 for line/text-heavy). Figure filenames must include run_id (or bo_run_id). Do not embed raw paths in figures.
- **Implementation**: Use one central style (e.g. `apply_paper_style()` via rcParams); every plot path must use it. No "this figure only" exceptions unless the user explicitly asks.

---

## 4. Bayesian Optimization (when applicable)

- BO must be traceable: inputs tied to run_id; outputs carry run_id/bo_run_id and list of referenced run_ids; **proposal reason log** is required (per candidate: design vars, predicted mean/var, acquisition value, constraint result, selection reason).
- Normalize/align with **anchor** samples across rounds/plates; document correction and conditions in manifest.
- Objective centered on t50 (or equivalent); do not swap objective for "looks better"; any change must come with rules, tests, and comparison to prior results.
- Constraints: only propose feasible designs (composition sum, bounds, solubility, etc.); hard or soft constraints must be explicit.
- Batch proposals with diversity; exploration/exploitation ratio explicit and recorded per round.
- Do not drop outliers silently; record reason and detection logic; prefer modeling as uncertainty.
- Figures: English only; polymer_id colors from persistent map; filenames include run_id/bo_run_id.
- Tests: Use synthetic/small data; no dependence on specific real wells/files; verify constraints, proposal log columns, and diversity.

---

## 5. Change checklist

- run_id / manifest / lineage remain consistent.
- Output schema (column names) of main CSVs is unchanged (use a normalization layer if needed).
- All figures remain English-only and paper-style.
