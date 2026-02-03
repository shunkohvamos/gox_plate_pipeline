from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.fitting.plotting import write_plate_grid  # noqa: E402


def _write_dummy_well_png(path: Path, rgb: tuple[int, int, int]) -> None:
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    arr[..., 0] = rgb[0]
    arr[..., 1] = rgb[1]
    arr[..., 2] = rgb[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


class WritePlateGridTests(unittest.TestCase):
    def test_multi_plate_outputs_are_plate_specific(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_dummy_well_png(root / "plate1" / "A1.png", (255, 0, 0))
            _write_dummy_well_png(root / "plate2" / "A1.png", (0, 255, 0))

            out = write_plate_grid(root, "runX")
            names = sorted(p.name for p in out)
            self.assertEqual(names, ["plate_grid__runX__plate1.png", "plate_grid__runX__plate2.png"])
            self.assertFalse((root / "plate_grid__runX.png").exists())

    def test_single_plate_keeps_legacy_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_dummy_well_png(root / "plate1" / "A1.png", (255, 0, 0))

            out = write_plate_grid(root, "runY")
            self.assertEqual(len(out), 1)
            self.assertTrue((root / "plate_grid__runY__plate1.png").exists())
            self.assertTrue((root / "plate_grid__runY.png").exists())


if __name__ == "__main__":
    unittest.main()
