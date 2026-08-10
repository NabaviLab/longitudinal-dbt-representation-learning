# Data contract

This repository operates on de-identified metadata and image paths. Input CSV files must not contain names, dates of birth, medical-record numbers, or other direct identifiers.

## DICOM conversion inputs

The labels CSV requires `PatientID`, `StudyUID`, and `View`. A cancer-mode run also requires `Cancer`; abnormal mode can additionally use `Benign` and `Actionable`. Missing optional flags are interpreted as zero.

The paths CSV requires the same identity triplet and a source-path column named `descriptive_path`, `path`, `file_path`, or `filepath`. Relative paths are resolved under the supplied split DICOM root, including the standard `Breast-Cancer-Screening-DBT` nesting. Absolute paths are accepted for managed internal runs but should not be committed.

Duplicate path rows are resolved deterministically by keeping the first occurrence. Cohort preparation should remove contradictory duplicates before conversion.

## Graph manifest

Every row represents one current or prior DBT view.

| Field | Requirement | Notes |
|---|---|---|
| `PatientID` | Required | De-identified stable identifier. |
| `View` | Required | Must distinguish current and prior records in the surrounding experiment metadata. |
| `Raw_path` | Required | TIFF file or directory of TIFF slices. |
| `GraphLabel` | Recommended | Normal `0`, malignant `1`, benign `2`. |
| `Normal`, `Cancer`, `Benign` | Conditional | Binary fallback columns when `GraphLabel` is absent. |
| `Mask_path` | Optional | Used only by the legacy Normal/Benign overlap rule; masks are not consumed as graph node labels. |

Only `GraphLabel` values `0`, `1`, and `2` are accepted. Actionable or unknown targets must be isolated before graph construction.

## Protected data boundaries

Generated graph artifacts can retain patient identifiers and labels in filenames and object attributes. Treat them as controlled research data. Do not publish manifests, TIFFs, graphs, logs, summaries, or checkpoints until their de-identification and data-use status have been reviewed.
