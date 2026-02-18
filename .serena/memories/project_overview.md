# Project overview: gox_plate_pipeline

## Purpose
- Pipeline for enzyme thermal stability analysis from plate raw data to initial-rate fitting, REA, t50, FoG, and Bayesian Optimization inputs/proposals.
- Designed for reproducibility and provenance: outputs are tied to `run_id`, manifest JSON, and lineage tables.

## Tech stack
- Language: Python (requires >=3.12).
- Packaging/build: `uv`, `uv_build`.
- Core libs: `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`.
- Testing: `pytest` (dev dependency).
- Main execution style: script-driven (`scripts/*.py`) and VS Code Run-and-Debug configurations.

## High-level workflow
1. Put raw CSV under `data/raw/{run_id}/`.
2. Prepare row map `data/meta/{run_id}.tsv` (template can be generated).
3. Extract clean CSV.
4. Fit rates + REA + t50; optionally generate well plots.
5. Build round-aware FoG summaries.
6. Build BO learning data.
7. Run Bayesian optimization.

## Repository structure (rough)
- `src/gox_plate_pipeline/`: core library code (loading, fitting, fog, BO data/engine, summaries/plotting).
- `scripts/`: operational entry scripts (extract, fit, aggregation, BO, launch generation, diagnostics).
- `tests/`: unit/integration tests around fitting guards, BO data/engine, ranking outputs, etc.
- `data/`: raw and processed datasets.
- `meta/`: run/round/group metadata, configs, catalogs.
- `.vscode/launch.json`: large generated set of Run-and-Debug tasks for routine operations.

## Key project premises
- Keep provenance strict (`run_id`, manifest, lineage).
- Figure in-plot text must be English only.
- Do not change numeric definitions (initial rate/REA/t50/FoG logic) without explicit rationale and test updates.