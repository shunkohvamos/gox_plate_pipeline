"""
Central definitions for user-editable meta file paths.

All paths that users edit (run groups, BO round map, polymer colors, etc.)
live under meta/ in a structured layout. Scripts should use get_meta_paths(repo_root)
so that moving files only requires changing this module.

Layout (all user inputs in one place: meta/):
  meta/
    row_maps/      - per-run row maps: {run_id}.tsv (e.g. 260216-1.tsv); plate/well → polymer_id, flags
    run_groups/    - run_group_map.tsv (which runs belong to which group for group-mean)
    bo/            - run_round_map.tsv, catalog_bma.csv (BO round assignment and composition catalog)
    polymers/      - colors.yml (polymer_id → plot color), stock_solvent.tsv (polymer_id → stock solvent/control group)
    chemistry/     - mol_logp_master.csv (monomer_id → MolLogP)
    config.yml     - assay config (heat_times, etc.)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def get_meta_paths(repo_root: Path) -> SimpleNamespace:
    """Return paths to user-editable meta files under repo_root/meta/."""
    root = Path(repo_root)
    meta = root / "meta"
    return SimpleNamespace(
        # Per-run row maps: meta/row_maps/{run_id}.tsv
        row_maps_dir=meta / "row_maps",
        # Run groups: which run_id belongs to which group_id for group-mean aggregation
        run_group_map=meta / "run_groups" / "run_group_map.tsv",
        # BO: run_id → round_id; composition catalog for BO space
        run_round_map=meta / "bo" / "run_round_map.tsv",
        bo_catalog_bma=meta / "bo" / "catalog_bma.csv",
        bo_catalog_bma_template=meta / "bo" / "catalog_bma_template.csv",
        # Polymers: polymer_id → hex color for plots
        polymer_colors=meta / "polymers" / "colors.yml",
        # Polymers: polymer_id → stock solvent / objective control group
        polymer_stock_solvent=meta / "polymers" / "stock_solvent.tsv",
        # Assay config (heat_times, etc.)
        config=meta / "config.yml",
        # Chemistry: monomer_id → MolLogP
        mol_logp_master=meta / "chemistry" / "mol_logp_master.csv",
        mol_logp_master_template=meta / "chemistry" / "mol_logp_master_template.csv",
    )
