# Longitudinal DBT Representation Learning

Reproducible preprocessing for graph-based learning from digital breast tomosynthesis (DBT). The repository converts DBT DICOM studies into lossless multi-page TIFF volumes and constructs quality-controlled, multiscale PyTorch Geometric graphs for current and prior views.

The code is dataset-path agnostic: local storage locations, manifests, and model checkpoints are supplied at runtime. No patient data, annotations, pretrained checkpoints, or institution-specific paths are included.

## Pipeline

```text
DICOM series + metadata CSVs
            │
            ▼
  lossless multi-page TIFF
            │
            ▼
 per-view construction manifest
            │
            ▼
 fine / medium / coarse supervoxels
            │
            ▼
 PyTorch Geometric graph + label volumes + QC summary
```

Graph construction is performed independently for each DBT view. Prior–current pairing should be defined in the downstream experiment manifest so that the same per-view graph artifacts can be reused without rebuilding them.

## What is included

- DICOM-to-TIFF conversion for single multi-frame files and DICOM series.
- Metadata-driven source resolution using patient, study, and view identifiers.
- Atomic TIFF writes, resumable smart updates, and a machine-readable failure report.
- Three-scale CUDA supervoxel construction with handcrafted and optional 2.5D ResNet features.
- Spatial adjacency, feature-space k-nearest-neighbor, and cross-scale hierarchy edges.
- Clean, AugMix, diffusion-like, and model-free adversarial graph variants.
- Persisted graph geometry, supervoxel label maps, provenance fields, and QC metrics.

## Installation

Python 3.9 or newer and a CUDA-capable PyTorch environment are recommended for graph construction.

```bash
git clone https://github.com/NabaviLab/longitudinal-dbt-representation-learning.git
cd longitudinal-dbt-representation-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dicom,graphs]"
```

For documentation and unit-test tooling:

```bash
python -m pip install -e ".[dev]"
```

The graph extra installs PyTorch from the default Python package index. On managed GPU systems, install the CUDA-matched PyTorch build first, then install this package.

## 1. Convert DICOM studies to TIFF

Each dataset split needs three inputs:

1. a labels CSV containing `PatientID`, `StudyUID`, and `View`;
2. a paths CSV containing the same identifiers and one of `descriptive_path`, `path`, `file_path`, or `filepath`;
3. the split's DICOM root directory.

Run one or more splits in a single command:

```bash
longitudinal-dbt convert-dicom \
  --output-dir /path/to/tiff-volumes \
  --label-mode cancer \
  --split train /path/to/train-labels.csv /path/to/train-paths.csv /path/to/train-dicom \
  --split validation /path/to/validation-labels.csv /path/to/validation-paths.csv /path/to/validation-dicom \
  --split test /path/to/test-labels.csv /path/to/test-paths.csv /path/to/test-dicom
```

`--label-mode cancer` writes binary labels (`0` non-cancer, `1` cancer). `--label-mode abnormal` maps cancer, benign, or actionable findings to `1`. Existing outputs are skipped when they are newer than their DICOM sources; use `--overwrite` to force replacement.

Output layout:

```text
/path/to/tiff-volumes/
├── <PatientID>/
│   └── <PatientID>_<StudyUID>_<VIEW>_lbl<LABEL>.tif
└── failed_views.tsv
```

The failure table records split, patient, study, view, and a stable reason code. A non-empty failure table does not automatically invalidate successful conversions, but every row should be reviewed before defining a study cohort.

## 2. Prepare the graph manifest

Graph construction reads one CSV row per DBT view. Required fields are:

| Column | Description |
|---|---|
| `PatientID` | Stable de-identified patient identifier. |
| `View` | View identifier such as `LCC`, `LMLO`, `RCC`, or `RMLO`. |
| `Raw_path` | Absolute or run-directory-relative path to a TIFF volume. |
| `GraphLabel` | Recommended class target: normal `0`, malignant `1`, benign `2`. |

`GraphLabel` is the preferred target. If it is absent, the constructor can resolve targets from binary `Normal`, `Cancer`, and `Benign` columns. Actionable or otherwise unresolved cases should be excluded from a three-class manifest rather than silently folded into a class.

Minimal example:

```csv
PatientID,View,Raw_path,GraphLabel
DBT0001,LCC,/data/tiff/DBT0001/DBT0001_STUDY1_LCC_lbl0.tif,0
DBT0001,LCC_PRIOR,/data/tiff/DBT0001/DBT0001_STUDY0_LCC_lbl0.tif,0
```

## 3. Construct multiscale graphs

```bash
longitudinal-dbt build-graphs \
  --manifest /path/to/graph-manifest.csv \
  --output-dir /path/to/graphs \
  --config configs/graph.yaml \
  --embedding-weights /path/to/resnet50-checkpoint.pth
```

Useful run controls:

```bash
# Process a comma-separated subset or a text file with one PatientID per line.
longitudinal-dbt build-graphs \
  --manifest /path/to/graph-manifest.csv \
  --output-dir /path/to/graphs \
  --patients DBT0001,DBT0002

# Construct clean graphs without learned embeddings or augmented variants.
longitudinal-dbt build-graphs \
  --manifest /path/to/graph-manifest.csv \
  --output-dir /path/to/graphs \
  --no-embeddings \
  --no-augmentations
```

Graph construction requires CUDA. If `--embedding-weights` is omitted while embeddings are enabled, the implementation requests torchvision's ImageNet ResNet-50 weights and falls back to random initialization only if those weights cannot be loaded. For a reproducible experiment, provide a fixed checkpoint and record its checksum.

## Graph artifact contract

For each accepted variant, the constructor writes a PyTorch Geometric `Data` object and a compressed supervoxel-label archive:

```text
<PatientID>__<View>__lbl<Class>__aug_<Variant><Index>.pt
<PatientID>__<View>__lbl<Class>__aug_<Variant><Index>__labels.npz
```

The clean variant uses `aug_clean0`; other default variants are `augmix1`, `augmix2`, `diffusion1`, and `adv1`. Set their counts in [configs/graph.yaml](configs/graph.yaml), or use `--no-augmentations` for clean-only construction.

Important serialized fields:

| Field | Meaning |
|---|---|
| `x` | Node features, stored as `float16`. |
| `pos` | Supervoxel centroids in millimetres, ordered `(x, y, z)`. |
| `pos_vox` | The same centroids in voxel coordinates. |
| `edge_index` | Directed graph connectivity. |
| `edge_attr` | Spatial displacement, distance, contact, size, and feature-distance attributes. |
| `edge_type` | `0` spatial adjacency, `1` feature kNN, `2` cross-scale hierarchy. |
| `node_scale` | `0` fine, `1` medium, `2` coarse. |
| `y` | Graph-level class target. |
| `spacing_mm` | Voxel spacing used to derive physical coordinates. |
| `parent_fine_to_med`, `parent_med_to_coarse` | Explicit hierarchy mappings. |

The companion `__labels.npz` contains `lab_fine`, `lab_med`, and `lab_coarse`. `summary.csv` records graph dimensions, QC decisions, stage timing, output paths, and peak GPU memory.

An artifact is saved only when it passes the configured tissue-coverage, largest-connected-component, average-degree, edge-index, self-loop, hierarchy, and reverse-adjacency checks.

## Reproducibility checklist

- Keep source TIFFs and accepted graph artifacts immutable within an experiment.
- Record the Git commit, graph configuration, input-manifest checksum, checkpoint checksum, CUDA/PyTorch versions, and GPU model.
- Define train, validation, and protected test partitions before model selection.
- Pair prior and current views by explicit identifiers; do not infer longitudinal pairing from filenames alone.
- Review `failed_views.tsv`, rejected rows in `summary.csv`, and missing graph variants before training.
- Never commit patient data, DICOM headers, institutional paths, credentials, or model checkpoints.

## Development

```bash
make test
make lint
```

The lightweight test suite exercises the public CLI, conversion helpers, metadata contracts, and graph source compilation without requiring patient data or a GPU. Full graph construction is an integration workload and must be validated in a CUDA environment with a representative de-identified volume.

## Repository layout

```text
configs/                         Graph-construction defaults
docs/                            Data and artifact specifications
src/longitudinal_dbt/            Installable Python package
  preprocessing/dicom_to_tiff.py DICOM conversion implementation
  preprocessing/graph_construction.py
tests/                           Unit and smoke tests
```

## License

The software is released under the [MIT License](LICENSE). Dataset access and use remain subject to each data provider's terms and applicable privacy requirements.

## Citation and contributions

Repository citation metadata are provided in [CITATION.cff](CITATION.cff). Replace or extend the citation metadata with the permanent article record when it becomes available. Contributions are welcome under the privacy and reproducibility requirements in [CONTRIBUTING.md](CONTRIBUTING.md).
