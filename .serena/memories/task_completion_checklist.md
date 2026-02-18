# Task completion checklist

## Always
- Run relevant tests before finishing changes.
  - Baseline: `uv run --extra dev pytest -q`
- Ensure no unintended schema regressions in key outputs.
- Ensure provenance remains intact (`run_id`, manifest, lineage outputs).

## If figures are touched
- Verify in-plot text is English only.
- Verify paper-style consistency across related figures.
- Verify PNG output and filename includes `run_id`/`bo_run_id`.

## If fitting/BO logic is touched
- Validate behavior with targeted tests (especially fitting guards and BO data/engine tests).
- Avoid broad behavior drift; verify normal-case wells/runs are unchanged unless intentionally modified.

## If methods/defaults files are touched
- Ensure methods one-page update hook/script is run if needed:
  - `bash scripts/pre-commit-update-methods.sh`

## Branch workflow note
- On `wip/agent` branch with incremental workflow requests: edit -> test/run -> `git add -A` -> `git commit -m "wip: ..."` in small logical chunks.