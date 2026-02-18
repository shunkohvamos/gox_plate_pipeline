# Style and conventions

## Python code style observed
- Function and variable names use `snake_case`.
- Test classes use `PascalCase`; test methods use descriptive `test_...` names.
- Type hints are used broadly (including `-> None`, `Path`, `list[...]`, `tuple[...]`, union syntax).
- Small helper functions and script-local private helpers frequently use leading underscore names (`_helper`).
- Docstrings are present in many modules/functions; language may be mixed in code comments/docs, but figure text requirements are strict (see below).

## Project-level behavioral conventions
- Reproducibility/provenance is mandatory:
  - keep `run_id` in outputs,
  - keep manifest JSON per run,
  - keep lineage in separate CSV tables.
- Numerical definitions (initial-rate/REA/t50/FoG) are treated as contract-level behavior; do not tweak for aesthetics.
- Edge-case fixes should be local and should not perturb normal cases broadly.

## Figure conventions
- In-plot text must be English only.
- Paper-grade styling expected consistently across outputs.
- PNG output with run_id/bo_run_id in filenames.
- Avoid embedding raw filesystem paths in figures.

## Editing workflow preferences
- Prefer targeted, scoped edits.
- Prefer symbolic/code-aware navigation/editing when available.
- Avoid reading whole files unless needed.
- For multi-file refactors: identify impacted symbols/files first, then edit in small steps.