from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.raw_bundle import (  # noqa: E402
    derive_run_id_from_raw_input,
    list_raw_csv_files,
    remap_plate_ids_for_file,
)


class RawBundleTests(unittest.TestCase):
    def test_list_raw_csv_files_for_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "2-b.csv").write_text("x\n", encoding="utf-8")
            (root / "1-a.csv").write_text("x\n", encoding="utf-8")
            (root / "note.txt").write_text("x\n", encoding="utf-8")

            files = list_raw_csv_files(root)
            self.assertEqual([p.name for p in files], ["1-a.csv", "2-b.csv"])
            self.assertEqual(derive_run_id_from_raw_input(root), root.name)

    def test_remap_plate_ids_with_filename_prefix(self) -> None:
        tidy = pd.DataFrame(
            {
                "plate_id": ["plate1", "plate2", "plate1"],
                "time_s": [0.0, 0.0, 30.0],
                "well": ["A1", "A1", "A1"],
                "signal": [1.0, 2.0, 3.0],
            }
        )
        out, mapping = remap_plate_ids_for_file(tidy, raw_file=Path("3-sample.csv"))
        self.assertEqual(mapping, {"plate1": "plate3", "plate2": "plate4"})
        self.assertEqual(sorted(out["plate_id"].unique().tolist()), ["plate3", "plate4"])

    def test_collision_without_prefix_raises(self) -> None:
        tidy = pd.DataFrame({"plate_id": ["plate1"], "time_s": [0.0], "well": ["A1"], "signal": [1.0]})
        used = {"plate1"}
        with self.assertRaises(ValueError):
            remap_plate_ids_for_file(tidy, raw_file=Path("sample.csv"), used_plate_ids=used)


if __name__ == "__main__":
    unittest.main()
