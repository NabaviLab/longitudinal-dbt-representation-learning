# Contributing

Contributions should preserve reproducibility, patient privacy, and the documented graph artifact contract.

1. Create a focused branch from `main`.
2. Keep data locations and credentials outside source files and configuration templates.
3. Add or update tests for behavior changes.
4. Run `make test` and `make lint`.
5. Use a Conventional Commit subject such as `fix(preprocessing): reject invalid graph labels`.

Do not attach DICOM files, TIFF volumes, patient-level manifests, graph artifacts, clinical logs, checkpoints, or screenshots containing protected information to commits or issues. Synthetic fixtures must not be derived from identifiable data.

Changes to label semantics, cohort rules, protected-test handling, graph topology, features, QC thresholds, augmentation, or serialization are scientific protocol changes. Describe their rationale and expected compatibility impact explicitly in the pull request.
