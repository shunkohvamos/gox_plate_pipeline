#!/usr/bin/env bash
# Pre-commit hook: regenerate METHODS_ONE_PAGE.md when default-defining code changed.
# Used by .pre-commit-config.yaml (optional). Run from repo root.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts"
python="${ROOT}/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Warning: .venv/bin/python not found; skipping Methods one-page update."
  exit 0
fi
"$python" scripts/update_methods_one_page.py
if ! git diff --quiet "research strategy/METHODS_ONE_PAGE.md" 2>/dev/null; then
  echo "METHODS_ONE_PAGE.md was regenerated. Please: git add 'research strategy/METHODS_ONE_PAGE.md' and commit again."
  exit 1
fi
exit 0
