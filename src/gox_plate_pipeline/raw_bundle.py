from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd


_PLATE_NUM_RE = re.compile(r"(\d+)")
_FILE_PLATE_PREFIX_RE = re.compile(r"^(\d+)-")


def list_raw_csv_files(raw_input: Path) -> list[Path]:
    """
    Resolve raw input into one or more CSV files.

    - file path  -> [file]
    - directory  -> sorted direct children *.csv
    """
    raw_input = Path(raw_input)
    if raw_input.is_file():
        if raw_input.suffix.lower() != ".csv":
            raise ValueError(f"Raw file must be a .csv: {raw_input}")
        return [raw_input]

    if raw_input.is_dir():
        csvs = sorted(p for p in raw_input.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            raise ValueError(f"No CSV files found in raw folder: {raw_input}")
        return csvs

    raise FileNotFoundError(f"Raw input not found: {raw_input}")


def derive_run_id_from_raw_input(raw_input: Path) -> str:
    """
    Run ID convention:
      - raw file   -> file stem
      - raw folder -> folder name
    """
    raw_input = Path(raw_input)
    return raw_input.stem if raw_input.is_file() else raw_input.name


def _plate_sort_key(plate_id: str) -> tuple[int, str]:
    s = str(plate_id)
    m = _PLATE_NUM_RE.search(s)
    if m:
        return (int(m.group(1)), s)
    return (10**9, s)


def parse_plate_start_from_filename(raw_file: Path) -> Optional[int]:
    """
    Parse filename prefix like "2-something.csv" -> 2.
    Returns None when the prefix is absent.
    """
    m = _FILE_PLATE_PREFIX_RE.match(Path(raw_file).name)
    if not m:
        return None
    return int(m.group(1))


def remap_plate_ids_for_file(
    tidy: pd.DataFrame,
    *,
    raw_file: Path,
    used_plate_ids: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Remap plate IDs for one raw file.

    Rule:
      - If filename starts with N- (e.g. 2-raw.csv), map that file's first plate
        to plateN, second to plateN+1, ... in internal plate order.
      - Otherwise keep internal plate IDs as-is.

    When used_plate_ids is provided, collision across files is treated as an error.
    """
    if "plate_id" not in tidy.columns:
        raise KeyError("tidy must contain 'plate_id'")

    out = tidy.copy()
    internal_ids = sorted(out["plate_id"].astype(str).unique().tolist(), key=_plate_sort_key)

    start = parse_plate_start_from_filename(raw_file)
    if start is None:
        mapping = {pid: pid for pid in internal_ids}
    else:
        mapping = {pid: f"plate{start + i}" for i, pid in enumerate(internal_ids)}

    out["plate_id"] = out["plate_id"].astype(str).map(mapping)

    if used_plate_ids is not None:
        remapped_ids = set(mapping.values())
        overlap = sorted(remapped_ids & used_plate_ids, key=_plate_sort_key)
        if overlap:
            raise ValueError(
                "Plate ID collision across files after mapping. "
                f"raw_file={raw_file.name}, overlapping={overlap}. "
                "If multiple CSVs each start from plate1, prefix filenames with "
                "'1-', '2-', '3-' ... so they map to distinct plate IDs."
            )
        used_plate_ids.update(remapped_ids)

    return out, mapping


def remap_plate_row_pairs_for_file(
    plate_row_pairs: Sequence[Tuple[str, str]],
    *,
    raw_file: Path,
) -> list[Tuple[str, str]]:
    """
    Apply the same filename-prefix plate remapping to inferred (plate_id, row) pairs.
    Used when generating row-map templates for raw folders.
    """
    pairs = [(str(p), str(r).upper()) for (p, r) in plate_row_pairs]
    if not pairs:
        return []

    start = parse_plate_start_from_filename(raw_file)
    if start is None:
        return pairs

    internal_ids = sorted({p for (p, _r) in pairs}, key=_plate_sort_key)
    mapping = {pid: f"plate{start + i}" for i, pid in enumerate(internal_ids)}
    return [(mapping[p], r) for (p, r) in pairs]


def sort_plate_row_pairs(pairs: Iterable[Tuple[str, str]]) -> list[Tuple[str, str]]:
    uniq = sorted({(str(p), str(r).upper()) for (p, r) in pairs}, key=lambda x: (_plate_sort_key(x[0]), x[1]))
    return uniq
