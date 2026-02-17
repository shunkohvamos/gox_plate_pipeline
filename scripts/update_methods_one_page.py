#!/usr/bin/env python3
"""
Regenerate research strategy/METHODS_ONE_PAGE.md from template and code defaults.

Run from repo root (e.g. via launch "Update Methods one-page"):
  PYTHONPATH=src python scripts/update_methods_one_page.py

Reads METHODS_ONE_PAGE_TEMPLATE.md and replaces {{ section.key }} placeholders
with values from dump_methods_defaults.get_all_defaults(), then writes METHODS_ONE_PAGE.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESEARCH_STRATEGY = REPO_ROOT / "research strategy"
TEMPLATE_PATH = RESEARCH_STRATEGY / "METHODS_ONE_PAGE_TEMPLATE.md"
OUTPUT_PATH = RESEARCH_STRATEGY / "METHODS_ONE_PAGE.md"


def _flatten(obj: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    """Flatten nested dict to dot-separated keys. Lists are joined to string."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = "\n".join(f"- {x}" if not str(x).startswith("-") else str(x) for x in v)
        elif v is None:
            out[key] = ""
        else:
            out[key] = v
    return out


def main() -> None:
    from dump_methods_defaults import get_all_defaults

    defaults = get_all_defaults()
    flat: dict[str, str | int | float | bool] = {}
    for section, data in defaults.items():
        flat.update(_flatten({section: data}))

    if not TEMPLATE_PATH.is_file():
        print(f"Template not found: {TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)

    text = TEMPLATE_PATH.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        if key in flat:
            val = flat[key]
            if isinstance(val, bool):
                return "true" if val else "false"
            return str(val)
        return m.group(0)

    text = re.sub(r"\{\{\s*([^}]+)\s*\}\}", repl, text)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"Updated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
