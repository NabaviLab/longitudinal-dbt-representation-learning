# tmi-graph

This repository contains the training and graph-construction entrypoints used in our IEEE TMI submission.

The project is organized so that your original research scripts remain **drop-in**: you can paste your existing code into the files under `src/tmi_graph/experiments/` with no algorithmic changes beyond the required filename updates.

## What was changed (and what was not)

Only these changes are assumed:

- **File renames** (see the mapping below).
- **Repository organization** around those scripts (CLI runner, config files, packaging).

No model, loss, preprocessing, augmentation, or graph logic is modified here, because the actual implementation should remain exactly as in your existing scripts.

If your scripts import one another by the old filenames (e.g., `import super_voxel_slic_aug_prev`), you will need to update those import statements to the new module names. That is the only situation where code edits beyond renaming are typically required.

## Script rename mapping

| Original filename | New filename (in this repo) | Purpose |
|---|---|---|
| `TR_pretrain_v2.py` | `pretrain_tr.py` | Pretraining entrypoint |
| `super_voxel_slic_aug.py` | `graph_current.py` | Graph construction for current scans |
| `super_voxel_slic_aug_prev.py` | `graph_prior.py` | Graph construction for prior scans |
| `super_voxel_slic_aug_BCS_higher_res.py` | `bcs_highres_pipeline.py` | Public BCS graph construction and the final fine-tuning entrypoint |

In this repo, those files live at:

- `src/tmi_graph/experiments/pretrain_tr.py`
- `src/tmi_graph/experiments/graph_current.py`
- `src/tmi_graph/experiments/graph_prior.py`
- `src/tmi_graph/experiments/bcs_highres_pipeline.py`

## Quick start

### 1) Create an environment

Option A (pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Option B (conda):

```bash
conda env create -f environment.yml
conda activate tmi-graph
pip install -e .
```

### 2) Paste your existing scripts

Copy the contents of your original files into the renamed counterparts under `src/tmi_graph/experiments/`.

Only the filenames must change; keep the contents identical unless you need to update imports that reference the old filenames.

### 3) Configure your run commands

The runner reads YAML config files under `configs/`. Each config specifies:

- the script to run
- the argument list to pass to that script
- optional environment variables

Edit the `args:` list in each config to match your script's CLI.

### 4) Run

Pretraining:

```bash
tmi-graph pretrain -- --help
tmi-graph pretrain -- --your --script --args
```

Graph construction:

```bash
tmi-graph graph-current -- --help
tmi-graph graph-prior -- --help
tmi-graph graph-bcs -- --help
```

Final fine-tuning:

```bash
tmi-graph finetune -- --help
```

You can also run a config directly:

```bash
tmi-graph run --config configs/pretrain.yaml -- --any --extra --args
```

## Repository layout

- `src/tmi_graph/`: importable Python package
- `src/tmi_graph/experiments/`: **your main research scripts (renamed)**
- `configs/`: YAML run configurations for the CLI runner
- `tools/`: utilities for migrating scripts and checking docstring coverage
- `tests/`: lightweight smoke tests for the runner/CLI

## Notes on docstrings

This repository avoids inline comments in the runner code and uses docstrings for function documentation.

Your experiment scripts are kept as drop-in modules. If you want docstring coverage checks across the entire repo (including your scripts), run:

```bash
python tools/check_docstrings.py --paths src/tmi_graph/experiments
```

## Citation

A `CITATION.cff` file is provided as a template; fill in the final metadata when your manuscript details are finalized.
