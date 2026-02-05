#!/usr/bin/env python3
"""
Validate (and optionally initialize) the MolLogP master sheet.

This script is independent of the Bayesian optimization code. It ensures
meta/mol_logp_master.csv exists with required columns (monomer_id, MolLogP)
and prints a summary. Used for Run and Debug as "MolLogP マスター確認".

Usage:
  python scripts/validate_mol_logp_master.py
  python scripts/validate_mol_logp_master.py --master meta/mol_logp_master.csv
  python scripts/validate_mol_logp_master.py --init   # create from template if missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "meta" / "mol_logp_master.csv"
DEFAULT_TEMPLATE = REPO_ROOT / "meta" / "mol_logp_master_template.csv"

REQUIRED_COLS = ["monomer_id", "MolLogP"]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate MolLogP master sheet (monomer_id, MolLogP)."
    )
    p.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help=f"Path to MolLogP master CSV. Default: {DEFAULT_MASTER.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Path to template CSV used by --init. Default: {DEFAULT_TEMPLATE.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="If master does not exist, create it from template and exit.",
    )
    args = p.parse_args()

    master = Path(args.master)
    if not master.is_absolute():
        master = REPO_ROOT / master
    template = Path(args.template)
    if not template.is_absolute():
        template = REPO_ROOT / template

    if args.init and not master.is_file():
        if not template.is_file():
            print(f"Error: Template not found: {template}", file=sys.stderr)
            return 1
        master.parent.mkdir(parents=True, exist_ok=True)
        content = template.read_text(encoding="utf-8")
        master.write_text(content, encoding="utf-8")
        print(f"Created: {master}")
        print("Edit meta/mol_logp_master.csv as needed, then run without --init to validate.")
        return 0

    if not master.is_file():
        print(f"Error: MolLogP master not found: {master}", file=sys.stderr)
        print("Run with --init to create from template.", file=sys.stderr)
        return 1

    import pandas as pd

    df = pd.read_csv(master)
    df["monomer_id"] = df["monomer_id"].astype(str).str.strip()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"Error: Master missing required columns: {missing}", file=sys.stderr)
        return 1

    mol_logp = pd.to_numeric(df["MolLogP"], errors="coerce")
    bad = mol_logp.isna() & df["MolLogP"].notna()
    if bad.any():
        print(f"Error: Non-numeric MolLogP in rows: {df.index[bad].tolist()}", file=sys.stderr)
        return 1

    dup = df["monomer_id"].duplicated()
    if dup.any():
        print(f"Warning: Duplicate monomer_id: {df.loc[dup, 'monomer_id'].tolist()}", file=sys.stderr)

    rel = master.relative_to(REPO_ROOT) if REPO_ROOT in master.parents else master
    print(f"MolLogP master: {rel}")
    print(df[REQUIRED_COLS + ([c for c in df.columns if c not in REQUIRED_COLS])].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
