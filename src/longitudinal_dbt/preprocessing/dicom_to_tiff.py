"""Convert metadata-indexed DBT DICOM studies into multi-page TIFF volumes.

The implementation supports a single multi-frame DICOM file or a directory
containing one or more DICOM series. Writes are atomic and retain the newest
source modification time so interrupted and resumed conversions are safe.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - exercised by environments without the optional extra
    sitk = None


DICOM_EXTENSIONS = {".dcm", ".dicom", ".dcmdecoded", ".dicomdecoded"}


@dataclass(frozen=True)
class SplitSpec:
    """Input files needed to convert one dataset split."""

    name: str
    labels_csv: Path
    paths_csv: Path
    dicom_root: Path


@dataclass
class ConversionStats:
    """Per-split conversion counters."""

    split: str
    requested: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0


Failure = Tuple[str, str, str, str, str]


def _require_simpleitk() -> None:
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is required for DICOM conversion. Install the optional dependency with "
            "'python -m pip install -e .[dicom]'."
        )


def sanitize_name(value: object) -> str:
    """Return a filesystem-safe identifier without changing letters or digits."""
    normalized = re.sub(r"[^\w.-]+", "_", str(value).strip())
    return normalized or "series"


def newest_mtime(paths: Iterable[Path]) -> float:
    """Return the newest modification time among existing paths."""
    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def is_up_to_date(output_path: Path, sources: Iterable[Path]) -> bool:
    """Return whether an output exists and is no older than every source."""
    if not output_path.is_file():
        return False
    try:
        return output_path.stat().st_mtime >= newest_mtime(sources) - 1e-3
    except OSError:
        return False


def atomic_tiff_write(
    volume: np.ndarray,
    output_path: Path,
    compression: str,
    source_mtime: float,
) -> None:
    """Write a TIFF through a sibling temporary file and atomically replace it."""
    volume = np.squeeze(volume)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)
    if volume.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D image or 3D volume, received shape {volume.shape}")

    options = {"photometric": "minisblack", "bigtiff": True}
    if compression.lower() != "none":
        options["compression"] = compression.lower()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tifffile.imwrite(str(temporary_path), volume, **options)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if source_mtime > 0:
        try:
            os.utime(output_path, (source_mtime, source_mtime))
        except OSError:
            pass


def _candidate_source_paths(dicom_root: Path, patient_id: str, descriptive_path: str) -> List[Path]:
    candidate = Path(str(descriptive_path))
    if candidate.is_absolute():
        return [candidate]
    return [
        dicom_root / candidate,
        dicom_root / "Breast-Cancer-Screening-DBT" / candidate,
        dicom_root / "Breast-Cancer-Screening-DBT" / patient_id / candidate,
    ]


def resolve_dicom_source(
    dicom_root: Path,
    patient_id: str,
    descriptive_path: str,
) -> Tuple[str, List[str], List[Path]]:
    """Resolve metadata to a single DICOM file or the largest series in a directory."""
    _require_simpleitk()
    base = next((path for path in _candidate_source_paths(dicom_root, patient_id, descriptive_path) if path.exists()), None)
    if base is None:
        return "", [], []

    if base.is_file() and base.suffix.lower() in DICOM_EXTENSIONS:
        return "single", [str(base)], [base]

    if not base.is_dir():
        return "", [], []

    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(base)) or []
    except Exception:
        series_ids = []

    largest_series: List[str] = []
    for series_id in series_ids:
        try:
            files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(base), series_id))
        except Exception:
            continue
        if len(files) > len(largest_series):
            largest_series = files

    if largest_series:
        sources = [Path(filename) for filename in largest_series]
        mode = "single" if len(sources) == 1 else "series"
        return mode, largest_series, sources

    files = sorted(
        path for path in base.iterdir() if path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS
    )
    if not files:
        return "", [], []
    mode = "single" if len(files) == 1 else "series"
    return mode, [str(path) for path in files], files


def convert_series(files: Sequence[str]) -> np.ndarray:
    """Read an ordered DICOM series into a NumPy volume."""
    _require_simpleitk()
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(files))
    return sitk.GetArrayFromImage(reader.Execute())


def convert_single(filename: str) -> np.ndarray:
    """Read a single multi-frame DICOM file into a NumPy volume."""
    _require_simpleitk()
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


def _binary_flag(row: pd.Series, column: str) -> int:
    value = row.get(column, 0)
    if pd.isna(value) or str(value).strip() == "":
        return 0
    return int(float(value) > 0)


def label_from_row(row: pd.Series, label_mode: str) -> int:
    """Resolve a binary cancer or abnormal label from a metadata row."""
    cancer = _binary_flag(row, "Cancer")
    if label_mode == "cancer":
        return cancer
    if label_mode == "abnormal":
        return int(cancer or _binary_flag(row, "Benign") or _binary_flag(row, "Actionable"))
    raise ValueError(f"Unsupported label mode: {label_mode}")


def pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Select the first case-insensitive column name found in a dataframe."""
    available = {column.lower(): column for column in frame.columns}
    return next((available[name.lower()] for name in candidates if name.lower() in available), None)


def build_path_map(paths: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    """Index descriptive DICOM paths by patient, study, and normalized view."""
    patient_column = pick_column(paths, ["PatientID"])
    study_column = pick_column(paths, ["StudyUID", "StudyInstanceUID"])
    view_column = pick_column(paths, ["View"])
    path_column = pick_column(paths, ["descriptive_path", "DescriptivePath", "path", "file_path", "filepath"])
    required = (patient_column, study_column, view_column, path_column)
    if not all(required):
        raise ValueError(
            "Paths CSV must contain patient, study, view, and descriptive-path columns; "
            f"received {list(paths.columns)}"
        )

    indexed: Dict[Tuple[str, str, str], str] = {}
    for patient, study, view, source in paths[
        [patient_column, study_column, view_column, path_column]
    ].itertuples(index=False, name=None):
        if pd.isna(source) or not str(source).strip():
            continue
        key = (str(patient), str(study), str(view).strip().lower())
        indexed.setdefault(key, str(source))
    return indexed


def _load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"PatientID", "StudyUID", "View"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV {path} is missing columns: {', '.join(sorted(missing))}")
    labels = labels.copy()
    labels["PatientID"] = labels["PatientID"].astype(str)
    labels["StudyUID"] = labels["StudyUID"].astype(str)
    labels["View"] = labels["View"].astype(str).str.strip().str.lower()
    return labels


def process_split(
    split: SplitSpec,
    output_dir: Path,
    label_mode: str,
    compression: str,
    smart_update: bool,
    overwrite: bool,
    failures: List[Failure],
) -> ConversionStats:
    """Convert every metadata-resolved view in one split."""
    for path_name, path in (
        ("labels CSV", split.labels_csv),
        ("paths CSV", split.paths_csv),
        ("DICOM root", split.dicom_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{split.name} {path_name} does not exist: {path}")

    labels = _load_labels(split.labels_csv)
    if label_mode == "cancer" and "Cancer" not in labels.columns:
        raise ValueError(f"Labels CSV {split.labels_csv} must contain Cancer for cancer label mode")
    abnormal_columns = {"Cancer", "Benign", "Actionable"}
    if label_mode == "abnormal" and not abnormal_columns.intersection(labels.columns):
        raise ValueError(
            f"Labels CSV {split.labels_csv} must contain Cancer, Benign, or Actionable for abnormal label mode"
        )
    path_map = build_path_map(pd.read_csv(split.paths_csv))
    labels["_resolved_label"] = labels.apply(label_from_row, axis=1, label_mode=label_mode)
    labels = labels.sort_values("_resolved_label", ascending=False)
    stats = ConversionStats(split=split.name, requested=len(labels))

    for patient_id, study_uid, view, label in labels[
        ["PatientID", "StudyUID", "View", "_resolved_label"]
    ].itertuples(index=False, name=None):
        patient_id = str(patient_id)
        study_uid = str(study_uid)
        view = str(view)
        label = int(label)
        descriptive_path = path_map.get((patient_id, study_uid, view))
        if descriptive_path is None:
            failures.append((split.name, patient_id, study_uid, view, "no_matching_path"))
            stats.failed += 1
            continue

        mode, files, sources = resolve_dicom_source(split.dicom_root, patient_id, descriptive_path)
        if mode not in {"single", "series"} or not sources:
            failures.append((split.name, patient_id, study_uid, view, "dicom_not_found"))
            stats.failed += 1
            continue

        output_name = "_".join(
            (
                sanitize_name(patient_id),
                sanitize_name(study_uid),
                sanitize_name(view.upper()),
                f"lbl{label}",
            )
        ) + ".tif"
        output_path = output_dir / sanitize_name(patient_id) / output_name

        if output_path.exists() and not overwrite:
            if not smart_update or is_up_to_date(output_path, sources):
                stats.skipped += 1
                continue

        try:
            volume = convert_series(files) if mode == "series" else convert_single(files[0])
            atomic_tiff_write(volume, output_path, compression, newest_mtime(sources))
            stats.written += 1
        except Exception as error:  # conversion errors are persisted and processing continues
            failures.append((split.name, patient_id, study_uid, view, f"conversion_error:{type(error).__name__}:{error}"))
            stats.failed += 1

    return stats


def write_failures(path: Path, failures: Sequence[Failure]) -> None:
    """Atomically write a tab-separated conversion failure table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("split", "PatientID", "StudyUID", "View", "reason"))
        writer.writerows(dict.fromkeys(failures))
    os.replace(temporary_path, path)


def _parse_split(values: Sequence[str]) -> SplitSpec:
    name, labels_csv, paths_csv, dicom_root = values
    return SplitSpec(name=sanitize_name(name), labels_csv=Path(labels_csv), paths_csv=Path(paths_csv), dicom_root=Path(dicom_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longitudinal-dbt convert-dicom",
        description="Convert metadata-indexed DBT DICOM studies to lossless multi-page TIFF volumes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination root for TIFF volumes.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "LABELS_CSV", "PATHS_CSV", "DICOM_ROOT"),
        help="Add a dataset split. May be specified more than once.",
    )
    parser.add_argument("--label-mode", choices=("cancer", "abnormal"), default="cancer")
    parser.add_argument("--compression", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing TIFF files.")
    parser.add_argument(
        "--no-smart-update",
        dest="smart_update",
        action="store_false",
        help="Skip every existing output without comparing modification times.",
    )
    parser.set_defaults(smart_update=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run DICOM conversion and print a concise per-split summary."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    _require_simpleitk()
    splits = [_parse_split(values) for values in args.split]
    if len({split.name for split in splits}) != len(splits):
        raise ValueError("Split names must be unique within a conversion run")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Failure] = []
    summaries = [
        process_split(
            split=split,
            output_dir=output_dir,
            label_mode=args.label_mode,
            compression=args.compression,
            smart_update=args.smart_update,
            overwrite=args.overwrite,
            failures=failures,
        )
        for split in splits
    ]
    failure_path = output_dir / "failed_views.tsv"
    write_failures(failure_path, failures)

    for summary in summaries:
        fields = asdict(summary)
        print(" ".join(f"{key}={value}" for key, value in fields.items()))
    print(f"failure_report={failure_path} failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
