from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from longitudinal_dbt.preprocessing.dicom_to_tiff import (
    SplitSpec,
    atomic_tiff_write,
    build_path_map,
    is_up_to_date,
    label_from_row,
    process_split,
    sanitize_name,
)


class DicomToTiffTests(unittest.TestCase):
    def test_sanitize_name(self) -> None:
        self.assertEqual(sanitize_name(" patient / view "), "patient_view")
        self.assertEqual(sanitize_name("..."), "...")

    def test_label_modes(self) -> None:
        benign = pd.Series({"Cancer": 0, "Benign": 1, "Actionable": 0})
        self.assertEqual(label_from_row(benign, "cancer"), 0)
        self.assertEqual(label_from_row(benign, "abnormal"), 1)

    def test_build_path_map_accepts_aliases(self) -> None:
        frame = pd.DataFrame(
            [{"PatientID": "P1", "StudyInstanceUID": "S1", "View": "LCC", "file_path": "study/series"}]
        )
        self.assertEqual(build_path_map(frame), {("P1", "S1", "lcc"): "study/series"})

    def test_atomic_tiff_write_and_smart_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dcm"
            source.write_bytes(b"source")
            source_mtime = source.stat().st_mtime
            output = root / "volume.tif"
            volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

            atomic_tiff_write(volume, output, "none", source_mtime)

            np.testing.assert_array_equal(tifffile.imread(output), volume)
            self.assertTrue(is_up_to_date(output, [source]))
            self.assertFalse(output.with_suffix(".tif.tmp").exists())

            os.utime(source, (source_mtime + 10, source_mtime + 10))
            self.assertFalse(is_up_to_date(output, [source]))

    def test_process_split_writes_metadata_named_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dicom_root = root / "dicom"
            dicom_root.mkdir()
            source = dicom_root / "view.dcm"
            source.write_bytes(b"synthetic-dicom")
            labels_csv = root / "labels.csv"
            paths_csv = root / "paths.csv"
            pd.DataFrame(
                [{"PatientID": "P1", "StudyUID": "S1", "View": "LCC", "Cancer": 1}]
            ).to_csv(labels_csv, index=False)
            pd.DataFrame(
                [{"PatientID": "P1", "StudyUID": "S1", "View": "LCC", "descriptive_path": "view.dcm"}]
            ).to_csv(paths_csv, index=False)
            split = SplitSpec("train", labels_csv, paths_csv, dicom_root)
            output_dir = root / "output"
            failures = []

            with patch(
                "longitudinal_dbt.preprocessing.dicom_to_tiff.convert_single",
                return_value=np.ones((2, 3, 4), dtype=np.uint16),
            ):
                stats = process_split(split, output_dir, "cancer", "none", True, False, failures)

            self.assertEqual(stats.requested, 1)
            self.assertEqual(stats.written, 1)
            self.assertEqual(stats.failed, 0)
            self.assertEqual(failures, [])
            output = output_dir / "P1" / "P1_S1_LCC_lbl1.tif"
            self.assertTrue(output.is_file())
            self.assertEqual(tifffile.imread(output).shape, (2, 3, 4))

    def test_process_split_rejects_missing_target_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dicom_root = root / "dicom"
            dicom_root.mkdir()
            labels_csv = root / "labels.csv"
            paths_csv = root / "paths.csv"
            pd.DataFrame([{"PatientID": "P1", "StudyUID": "S1", "View": "LCC"}]).to_csv(
                labels_csv, index=False
            )
            pd.DataFrame(
                [{"PatientID": "P1", "StudyUID": "S1", "View": "LCC", "descriptive_path": "view.dcm"}]
            ).to_csv(paths_csv, index=False)
            split = SplitSpec("train", labels_csv, paths_csv, dicom_root)

            with self.assertRaisesRegex(ValueError, "must contain Cancer"):
                process_split(split, root / "output", "cancer", "none", True, False, [])


if __name__ == "__main__":
    unittest.main()
