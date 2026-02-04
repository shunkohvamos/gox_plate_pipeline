from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


WELL_RE = re.compile(r"^([A-H])0*([1-9]|1[0-2])$", re.IGNORECASE)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def normalize_plate_id(s: str) -> str:
    s2 = str(s).strip().lower()
    m = re.search(r"(\d+)", s2)
    if m:
        return f"plate{int(m.group(1))}"
    if s2.startswith("plate"):
        return s2
    return f"plate{s2}"


def normalize_well(well: str) -> str:
    w = str(well).strip().upper()
    m = re.match(r"^([A-H])0*([0-9]+)$", w)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return w


def parse_well(well: str) -> Tuple[str, int]:
    w = normalize_well(well)
    m = WELL_RE.match(w)
    if not m:
        raise ValueError(f"Invalid well: {well}")
    return m.group(1).upper(), int(m.group(2))


def time_to_seconds(t: Any) -> Optional[float]:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip()
    parts = s.split(":")
    if len(parts) == 3:
        hh, mm, ss = parts
        try:
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        except ValueError:
            return None
    if len(parts) == 2:
        mm, ss = parts
        try:
            return int(mm) * 60 + float(ss)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def read_row_map_tsv(path: Path) -> pd.DataFrame:
    """
    TSV with columns:
      - required: row, polymer_id
      - optional: plate, sample_name
    Blank lines are allowed.
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    if "row" not in df.columns or "polymer_id" not in df.columns:
        raise ValueError(f"row_map must include 'row' and 'polymer_id': {path}")

    df["row"] = df["row"].str.strip().str.upper()

    if "plate" in df.columns:
        def _norm_plate_or_wildcard(v: object) -> str:
            s = "" if v is None else str(v).strip()
            if s == "" or s == "*":
                return ""
            return normalize_plate_id(s)

        df["plate"] = df["plate"].apply(_norm_plate_or_wildcard)
    else:
        df["plate"] = ""  # wildcard

    if "sample_name" not in df.columns:
        df["sample_name"] = ""

    # empty polymer_id means "no sample" -> keep for explicit empties, handled later
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    df["sample_name"] = df["sample_name"].astype(str).str.strip()

    return df[["plate", "row", "polymer_id", "sample_name"]]


def infer_plate_row_from_synergy(raw_path: Path) -> List[Tuple[str, str]]:
    """
    Infer (plate_id, row) pairs from a Synergy H1 export file.
    Used to generate a row-map TSV template (plate, row, polymer_id, sample_name)
    with empty polymer_id/sample_name for the user to fill in.
    """
    well_re = re.compile(r"^[A-H](?:[1-9]|1[0-2])$")
    data = raw_path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            continue
    else:
        text = data.decode("latin-1", errors="ignore")
    lines = text.splitlines()

    header_idxs = []
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("Time") and ("A1" in s):
            header_idxs.append(i)
    if not header_idxs:
        return []

    def _plate_id_from_context(start_idx: int) -> str:
        for j in range(start_idx, max(-1, start_idx - 200), -1):
            line = lines[j]
            if line.startswith("Plate Number"):
                parts = re.split(r"[\t,]", line)
                if len(parts) >= 2:
                    name = parts[1].strip()
                    m = re.search(r"(\d+)", name)
                    if m:
                        return f"plate{m.group(1)}"
        return "plate1"

    result: List[Tuple[str, str]] = []
    for hidx in header_idxs:
        plate_id = _plate_id_from_context(hidx)
        header_line = lines[hidx]
        sep = "\t" if header_line.count("\t") >= header_line.count(",") else ","
        parts = [p.strip() for p in header_line.split(sep)]
        well_cols = [p for p in parts if well_re.match(p)]
        rows = sorted(set(w[0].upper() for w in well_cols))
        for row in rows:
            result.append((plate_id, row))
    return result


def _extract_plate_blocks(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Returns list of (plate_id, block_lines).
    If file contains only one plate, returns one block with plate1.
    Detects 'Plate Number\tPlate X' lines.
    """
    plate_indices: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if ln.startswith("Plate Number"):
            parts = ln.split("\t")
            if len(parts) >= 2:
                plate_id = normalize_plate_id(parts[1])
                plate_indices.append((i, plate_id))

    if not plate_indices:
        return [("plate1", lines)]

    blocks = []
    for idx, (start_i, plate_id) in enumerate(plate_indices):
        end_i = plate_indices[idx + 1][0] if idx + 1 < len(plate_indices) else len(lines)
        blocks.append((plate_id, lines[start_i:end_i]))
    return blocks


def _extract_data_table_from_block(block_lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Find the line starting with 'Time\t' that contains 'A1' and parse until 'Results' or blank section.
    """
    header_i = None
    for i, ln in enumerate(block_lines):
        if ln.startswith("Time\t") and "\tA1\t" in ln:
            header_i = i
            break
    if header_i is None:
        return None

    header = block_lines[header_i].rstrip("\n")
    cols = header.split("\t")

    rows = []
    for ln in block_lines[header_i + 1 :]:
        if ln.startswith("Results"):
            break
        if ln.strip() == "":
            # allow blank lines inside, but if we already collected data, break on consecutive empties
            if rows:
                break
            continue

        vals = ln.rstrip("\n").split("\t")
        # pad
        if len(vals) < len(cols):
            vals = vals + [""] * (len(cols) - len(vals))
        row = dict(zip(cols, vals))
        # data rows must have Time like 0:00:06
        if not str(row.get("Time", "")).strip():
            continue
        rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df


def extract_tidy_from_synergy_export(raw_path: Path, config: dict) -> pd.DataFrame:
    """
    Parse Synergy H1 export text (often .csv extension but actually tab-delimited with a long header)
    and return tidy dataframe: [plate_id, time_s, well, signal]
    Supports multiple plate blocks in a single file.
    """

    import re
    from io import StringIO

    def _read_text_flexible(p: Path) -> str:
        data = p.read_bytes()
        for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("latin-1", errors="ignore")

    text = _read_text_flexible(raw_path)
    lines = text.splitlines()

    # find table header lines (Time ... A1 ...)
    header_idxs = []
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("Time") and ("A1" in s):
            header_idxs.append(i)
    
    # Debug output removed for memory optimization

    if not header_idxs:
        # debugging help: show candidates that contain 'Time'
        candidates = []
        for i, line in enumerate(lines):
            if "Time" in line:
                candidates.append((i + 1, line[:200]))
            if len(candidates) >= 10:
                break
        raise ValueError(
            f"Could not find any data table (header like 'Time ... A1') in: {raw_path}\n"
            f"First Time-containing lines (up to 10): {candidates}"
        )

    # layout filter (optional)
    layout = config.get("layout", {})
    rows = layout.get("rows", list("ABCDEFGH"))
    max_col = int(layout.get("max_col", 12))

    well_re = re.compile(r"^[A-H](?:[1-9]|1[0-2])$")

    def _plate_id_from_context(start_idx: int) -> str:
        # scan upwards to find "Plate Number\tPlate 1"
        for j in range(start_idx, max(-1, start_idx - 200), -1):
            line = lines[j]
            if line.startswith("Plate Number"):
                parts = re.split(r"[\t,]", line)
                if len(parts) >= 2:
                    name = parts[1].strip()
                    # e.g., "Plate 1" -> "plate1"
                    m = re.search(r"(\d+)", name)
                    if m:
                        return f"plate{m.group(1)}"
        # fallback if not found
        return "plate1"

    blocks = []
    for k, hidx in enumerate(header_idxs):
        plate_id = _plate_id_from_context(hidx)
        # Debug output removed for memory optimization

        # determine end of block
        # End at the next header (if exists) or at Results/Cutoffs/empty section
        end = len(lines)
        # Check if there's a next header_idx
        if k + 1 < len(header_idxs):
            # End before the next header
            next_hidx = header_idxs[k + 1]
            end = next_hidx
        else:
            # Last block: find Results/Cutoffs or empty section
            for j in range(hidx + 1, len(lines)):
                s = lines[j].strip()
                if s == "":
                    # Check if this empty line is followed by Results/Cutoffs or another plate section
                    # Look ahead a few lines to see if this is a real break
                    is_real_break = False
                    for look_ahead in range(j + 1, min(j + 10, len(lines))):
                        look_line = lines[look_ahead].strip()
                        if look_line.startswith("Results") or look_line.startswith("Cutoffs"):
                            is_real_break = True
                            break
                        if look_line.startswith("Plate Number") or (look_line.startswith("Time") and "A1" in look_line):
                            is_real_break = True
                            break
                    if is_real_break:
                        end = j
                        break
                elif s.startswith("Results") or s.startswith("Cutoffs"):
                    end = j
                    break

        block_text = "\n".join(lines[hidx:end])

        # detect delimiter: prefer tabs
        header_line = lines[hidx]
        sep = "\t" if header_line.count("\t") >= header_line.count(",") else ","

        try:
            df = pd.read_csv(StringIO(block_text), sep=sep, engine="python")
        except Exception:
            # fallback try the other separator
            alt = "," if sep == "\t" else "\t"
            df = pd.read_csv(StringIO(block_text), sep=alt, engine="python")

        if "Time" not in df.columns:
            continue

        # parse time
        time_td = pd.to_timedelta(df["Time"], errors="coerce")
        df = df.assign(time_s=time_td.dt.total_seconds())
        df = df.dropna(subset=["time_s"])

        # pick well columns only
        well_cols = [c for c in df.columns if isinstance(c, str) and well_re.match(c)]
        # layout filter
        allowed = []
        for w in well_cols:
            r = w[0]
            cnum = int(w[1:])
            if (r in rows) and (cnum <= max_col):
                allowed.append(w)

        if not allowed:
            continue

        # Data shift correction: Due to instrument specification change, data columns are shifted left by 1 position.
        # A1's data is in the column one position to the left of the "A1" column (e.g., "590" column).
        # A2's data is in the column one position to the left of the "A2" column (i.e., "A1" column).
        # A3's data is in the column one position to the left of the "A3" column (i.e., "A2" column).
        # Strategy: For each well, read data from the column that is one position to the left of that well's column.
        # Example: Well "A1" -> read from column one position left of "A1", Well "A2" -> read from "A1" column, etc.
        
        # Get all columns in the dataframe to find the position of each well column
        all_cols = list(df.columns)
        
        # Sort well columns in order (A1, A2, ..., A12, B1, B2, ..., H12)
        well_cols_sorted = sorted(allowed, key=lambda x: (x[0], int(x[1:])))
        
        # Create mapping: well_name -> source_column_name (one position to the left)
        shifted_mapping = {}
        for well in well_cols_sorted:
            # Find the index of this well column in the dataframe
            if well in all_cols:
                well_idx = all_cols.index(well)
                # Read from the column one position to the left
                if well_idx > 0:
                    source_col = all_cols[well_idx - 1]
                    shifted_mapping[well] = source_col
                # Skip if this is the first column (no column to the left)
        
        # Create a new dataframe with shifted data
        df_shifted = df[["time_s"]].copy()
        for well, source_col in shifted_mapping.items():
            if source_col in df.columns:
                df_shifted[well] = df[source_col]

        long = df_shifted.melt(
            id_vars=["time_s"],
            value_vars=list(shifted_mapping.keys()),
            var_name="well",
            value_name="signal",
        )
        long["signal"] = pd.to_numeric(long["signal"], errors="coerce")
        long = long.dropna(subset=["signal"])

        long["plate_id"] = plate_id
        blocks.append(long[["plate_id", "time_s", "well", "signal"]])

    if not blocks:
        raise ValueError(f"Could not parse any valid table block in: {raw_path}")

    tidy = pd.concat(blocks, ignore_index=True)
    return tidy


    return pd.concat(out_rows, ignore_index=True)


def attach_row_map(df: pd.DataFrame, row_map: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    row_map = row_map.copy()

    # --- normalize column names in row_map ---
    if "plate" in row_map.columns and "plate_id" not in row_map.columns:
        row_map = row_map.rename(columns={"plate": "plate_id"})

    # --- derive row from well if missing ---
    if "row" not in df.columns:
        if "well" not in df.columns:
            raise KeyError("attach_row_map requires 'row' or 'well' column in df.")
        df["row"] = df["well"].astype(str).str.extract(r"^([A-H])", expand=False)

    # --- normalize keys ---
    df["plate_id"] = df["plate_id"].astype(str).str.strip().str.lower()
    row_map["plate_id"] = row_map["plate_id"].astype(str).str.strip().str.lower()

    df["row"] = df["row"].astype(str).str.strip().str.upper()
    row_map["row"] = row_map["row"].astype(str).str.strip().str.upper()

    # keep only expected columns (optional but safer)
    keep_cols = [c for c in ["plate_id", "row", "polymer_id", "sample_name"] if c in row_map.columns]
    row_map = row_map[keep_cols]

    # plate_id == "" in row_map means wildcard (applies to all plates for that row)
    exact_map = row_map[row_map["plate_id"].astype(str).str.strip() != ""].copy()
    wild_map = row_map[row_map["plate_id"].astype(str).str.strip() == ""].copy()

    if not wild_map.empty:
        dup_rows = wild_map["row"][wild_map["row"].duplicated()].unique().tolist()
        if dup_rows:
            raise ValueError(
                "row_map wildcard rows must be unique when plate is blank. "
                f"Duplicated rows: {dup_rows}"
            )
        wild_map = wild_map.drop(columns=["plate_id"])

    if not exact_map.empty:
        exact_map = exact_map.rename(
            columns={
                "polymer_id": "polymer_id_exact",
                "sample_name": "sample_name_exact",
            }
        )
        df = df.merge(
            exact_map,
            on=["plate_id", "row"],
            how="left",
            validate="m:1",
        )
    else:
        df["polymer_id_exact"] = ""
        df["sample_name_exact"] = ""

    if not wild_map.empty:
        wild_map = wild_map.rename(
            columns={
                "polymer_id": "polymer_id_wild",
                "sample_name": "sample_name_wild",
            }
        )
        df = df.merge(
            wild_map,
            on=["row"],
            how="left",
            validate="m:1",
        )
    else:
        df["polymer_id_wild"] = ""
        df["sample_name_wild"] = ""

    for c in ["polymer_id", "sample_name"]:
        exact_c = f"{c}_exact"
        wild_c = f"{c}_wild"
        exact_v = df[exact_c] if exact_c in df.columns else ""
        wild_v = df[wild_c] if wild_c in df.columns else ""
        df[c] = exact_v
        # fallback to wildcard when exact match is missing
        if isinstance(df[c], pd.Series):
            miss = df[c].isna() | (df[c].astype(str).str.strip() == "")
            df.loc[miss, c] = wild_v[miss] if isinstance(wild_v, pd.Series) else wild_v
        if exact_c in df.columns:
            df = df.drop(columns=[exact_c])
        if wild_c in df.columns:
            df = df.drop(columns=[wild_c])

    # 見栄えと後段処理の安定化（NaNを空文字へ）
    for c in ["polymer_id", "sample_name"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df
