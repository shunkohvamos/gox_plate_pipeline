# Suggested commands

## Environment and dependency
- `uv sync`
- `source .venv/bin/activate`

## Tests
- Canonical CI-equivalent test command:
  - `uv run --extra dev pytest -q`
- Run subset during development:
  - `uv run --extra dev pytest -q tests/test_fitting_logic_guards.py`

## Pre-commit / docs sync
- Install hook:
  - `pre-commit install`
- Hook in this repo updates methods one-page doc when specific code files change:
  - `bash scripts/pre-commit-update-methods.sh`

## Common pipeline script commands (CLI)
- Generate launch configs from discovered runs:
  - `python scripts/generate_launch_json.py`
- Generate row-map template from raw:
  - `python scripts/generate_row_map_template.py`
- Extract one run:
  - `python scripts/extract_clean_csv.py ...`
- Fit one run:
  - `python scripts/fit_initial_rates.py ...`
- Extract all runs:
  - `python scripts/run_extract_all.py ...`
- Fit all runs:
  - `python scripts/run_fit_all.py ...`
- Build plate-aware FoG:
  - `python scripts/build_fog_plate_aware.py ...`
- Build BO learning data:
  - `python scripts/build_bo_learning_data.py ...`
- Run BO:
  - `python scripts/run_bayesian_optimization.py ...`

Note: In day-to-day operation, this project often uses VS Code Run-and-Debug entries from `.vscode/launch.json` instead of typing long arg lists manually.

## Useful Linux utility commands
- `git status`, `git diff`, `git add -A`, `git commit -m "..."`
- `ls -la`, `cd <dir>`, `pwd`
- `rg "pattern" <path>` (preferred fast search)
- `find <path> -name "*.py"`
- `sed -n 'start,endp' <file>`