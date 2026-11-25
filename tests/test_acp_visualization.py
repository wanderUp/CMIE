"""Regression test for ACP panel generation and lossless PDF assembly."""

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acp_visualization


class ACPVisualizationTest(unittest.TestCase):
    def test_two_panels_are_exported_then_assembled_as_one_pdf(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            root = Path(tmp_dir)
            paths = []
            for name, offset in (("Alpha", 0.0), ("Beta", 0.2)):
                data = pd.DataFrame(
                    {
                        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                        "x1": np.array([1.0, 2.0, 3.0, 4.0]) + offset,
                        "x2": np.array([1.2, 2.1, 3.2, 4.1]) + offset,
                        "x3": np.array([4.0, 3.0, 2.0, 1.0]) - offset,
                    }
                )
                path = root / f"{name}.csv"
                data.to_csv(path, index=False)
                paths.append(path)

            combined = root / "acp_groups.pdf"
            panel_paths = acp_visualization.export_and_combine(
                data_paths=paths,
                dataset_names=["Alpha", "Beta"],
                output_path=combined,
                threshold=0.5,
                dpi=600,
            )

            self.assertEqual(len(panel_paths), 2)
            self.assertTrue(all(path.is_file() for path in panel_paths))
            self.assertTrue(all(path.suffix == ".png" for path in panel_paths))
            self.assertTrue((root / "acp_groups.png").is_file())
            self.assertTrue(combined.is_file())
            self.assertEqual(len(PdfReader(combined).pages), 1)


if __name__ == "__main__":
    unittest.main()
