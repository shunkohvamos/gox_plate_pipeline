# Project Instructions (gox_plate_pipeline)

For **Codex** and other tools that don’t read Cursor Rules: use this file (e.g. `@file AGENTS.md`).  
Detailed rules live in `.cursor/rules/`; refer to them when needed (e.g. fitting, BO).

---

## 0. Serena MCP — use first to reduce context usage

**Goal**: Use Serena MCP for navigation and editing so you avoid loading whole files into context; that keeps context consumption low.

- **Before reading a file**: Prefer Serena tools — `get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`, `find_file`, `list_dir`. Only open full file contents as a **last resort**.
- **When editing**: Prefer symbolic edits — `replace_symbol_body`, `insert_after_symbol`, `insert_before_symbol`. For multi-file changes, find impacted symbols/references with Serena first, then edit one file at a time.
- **Session start** (if using Serena): `check_onboarding_performed` → if needed `onboarding`; then `activate_project`; then `initial_instructions`.
- **User overrides**: If the user explicitly asks to read a full file or do something Serena can’t do, follow the user. User instructions take precedence.

---

## 1. Core design — research philosophy & provenance

- **Purpose**: Reproducible evaluation of enzyme thermal stability (t50, REA, etc.) and optimization.
- **Research philosophy**:
  - Goal: **improve thermostability (e.g. t50) without compromising GOx activity**.
  - **Absolute activity (e.g. U0)**: keep at least unchanged; increasing it is a plus. Objective functions should reward “maintain or increase initial activity,” not only “avoid dropping below a threshold.”
- **Provenance**: Every derived output must be traceable to raw data.
  - Each run has a `run_id`; all output CSVs include a `run_id` column.
  - One **run manifest (JSON)** per run: run_id, timestamp, git commit, input paths/sha256/mtime/size, CLI/params.
  - **Lineage**: one row per well in a separate CSV; do not embed long lineage strings in summary CSVs.
- **Numerical definitions**: Do not change definitions (e.g. initial slope → absolute activity → REA) without explicit reason, impact analysis, and test updates. Do not change numeric logic for “looks better” only.

---

## 2. Figures (paper-grade, PNG, English)

All figures are **paper-grade**; no default matplotlib look.

- **Language**: In-plot text **English only**. No raw paths or full-width Japanese in figures.
- **Typography**: Font **Arial** → Helvetica → DejaVu Sans; one family per figure. Size ~5–8 pt (aim 6–7 pt); panel labels (a,b,c…) **8 pt bold**.
- **Lines & markers**: Line width ~0.6–0.9 pt; minimal, distinguishable markers; minimal grid.
- **Layout**: Aspect ratio flexible; prefer ~90 mm single-column width; minimize margins (e.g. `bbox_inches="tight"`); legend must not hide data.
- **Color**: No red–green-only or jet-like colormaps. **polymer_id** colors from a single persistent map (yml/json); same ID → same color across figures.
- **Output**: **PNG only**, 300–600 dpi (600 for line/text-heavy). Filenames must include `run_id` or `bo_run_id`.
- **Implementation**: One central style (e.g. `apply_paper_style()` via rcParams); every plot path uses it. No “this figure only” exceptions unless the user asks.
- **Consistency**: Same figure elements across all figures (error bars, axes, spines, fonts). When adding or changing a figure, match existing same-type figures; if you change an element (e.g. error-bar style), apply it to all figures that use it. Keep definitions in rcParams or shared constants.

---

## 3. Initial rate fitting (principles only)

- **Initial rate** = slope of the earliest valid linear window in the initial regime; do not fit into an already curved region. Prefer **reaction validity over high R²** (e.g. don’t prefer a late, flatter window).
- **Window choice**: (1) Earlier start, (2) Higher linearity in that early region, (3) Sufficient point count.
- **Start index**: Move forward **only to skip a true lag** (sustained flat/noisy start). Do **not** move for a shallow dip (e.g. one point above y[0]); keep start_idx=0 so early points stay eligible.
- **Rescue locally**: When fixing edge cases, don’t change behavior for wells that already fit correctly.
- When leading points are skipped, record the skip (e.g. start_idx or time) in output (CSV/log).  
  Full design: `.cursor/rules/Fitting-logic-addition.mdc`, `2-Auto-attached.mdc`.

---

## 4. Bayesian Optimization (when applicable)

- **Surrogate map** = BO gradient colormap figures (ternary mean/std/EI/UCB and 2×2 xy or BMA–MTAC). Use this term for “those gradient maps.”
- Traceable: inputs → run_id; outputs → run_id/bo_run_id and **proposal reason log** (per candidate: design vars, predicted mean/var, acquisition, constraint, selection reason).
- Normalize with **anchor** samples across rounds/plates; document in manifest. Objective centered on t50; only feasible designs; batch with diversity. Don’t drop outliers silently; record reason and prefer modeling as uncertainty.  
  Full: `.cursor/rules/BO-rules.mdc`, `BO-surrogate-map-terminology.mdc`.

---

## 5. Change checklist

- run_id / manifest / lineage remain consistent.
- Output schema (column names) of main CSVs unchanged (use a normalization layer if needed).
- All figures remain English-only and paper-style.

---

## 6. WIP workflow (wip/agent branch)

When asked for incremental commits: Edit → Test → Stage → `git commit -m "wip: <short description>"` in small chunks. Commit prefix `wip:`; scope to wip/agent only. Details: `.cursor/rules/Agent-wip-workflow.mdc`.
