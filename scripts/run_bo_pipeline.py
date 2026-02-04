#!/usr/bin/env python3
"""
BO pipeline: (1) Fit rates + REA + FoG, (2) Bayesian optimization (when implemented).

One Run and Debug button runs this script to do the full calculation and then BO.
Same arguments as fit_initial_rates.py; they are forwarded to the fit step.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FIT_SCRIPT = REPO_ROOT / "scripts" / "fit_initial_rates.py"


def main() -> int:
    # Step 1: Fit rates + REA + FoG (same as Fit rates+REA)
    if not FIT_SCRIPT.is_file():
        print(f"Error: {FIT_SCRIPT} not found.", file=sys.stderr)
        return 1

    # Forward all our args to fit_initial_rates.py (drop our script name)
    fit_args = [str(Path(sys.executable)), str(FIT_SCRIPT)] + sys.argv[1:]
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(SRC_DIR)}

    print("--- Step 1: Fit rates + REA + FoG ---")
    ret = subprocess.run(fit_args, cwd=REPO_ROOT, env=env)
    if ret.returncode != 0:
        print("Fit step failed.", file=sys.stderr)
        return ret.returncode
    if "--help" in sys.argv or "-h" in sys.argv:
        return 0

    # Step 2: Bayesian optimization (when implemented)
    print("--- Step 2: Bayesian optimization ---")
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    try:
        from gox_plate_pipeline import bo_engine
        if hasattr(bo_engine, "run_bo") and callable(bo_engine.run_bo):
            # Derive run_id from --tidy (e.g. data/processed/260203-1/extract/tidy.csv -> 260203-1)
            run_id = _run_id_from_argv(sys.argv)
            if run_id:
                bo_engine.run_bo(run_id=run_id, repo_root=REPO_ROOT)
            else:
                print("BO step: run_id could not be derived from args; skipping.")
        else:
            print("BO step: not yet implemented (FoG summary and bo_output.json are ready for BO).")
    except Exception as e:
        print(f"BO step: not yet implemented ({e}). FoG summary and bo_output.json are ready for BO.")

    return 0


def _run_id_from_argv(argv: list[str]) -> str | None:
    """Derive run_id from --tidy path (e.g. .../260203-1/extract/tidy.csv -> 260203-1)."""
    for i, a in enumerate(argv):
        if a == "--tidy" and i + 1 < len(argv):
            p = Path(argv[i + 1])
            # .../run_id/extract/tidy.csv or .../run_id/tidy.csv
            if p.name == "tidy.csv":
                parent = p.parent
                if parent.name == "extract" and parent.parent.name:
                    return parent.parent.name
                if parent.name:
                    return parent.name
            return p.stem.replace("__tidy", "") if p.stem.endswith("__tidy") else None
    return None


if __name__ == "__main__":
    sys.exit(main())
