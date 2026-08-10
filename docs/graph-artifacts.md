# Graph artifact specification

Each view is represented at fine, medium, and coarse supervoxel scales. Nodes are serialized in `[fine, medium, coarse]` order.

## Nodes

Handcrafted features include intensity statistics, gradient statistics, voxel count, compactness, elongation, distance to the breast centroid, normalized location, texture statistics, and a scale code. When enabled, a deterministic random projection of 2.5D ResNet-50 features is appended.

`pos_vox` stores region centroids in voxel `(x, y, z)` order. `pos` stores the corresponding physical coordinates after applying `spacing_mm`. Per-scale slice membership, node bounds, supervoxel counts, and parent mappings are retained on the `Data` object.

## Edges

- Type `0`: region-adjacency edges derived from touching supervoxels.
- Type `1`: within-scale feature-space k-nearest-neighbor edges.
- Type `2`: bidirectional fine-to-medium and medium-to-coarse hierarchy edges.

Edge attributes contain physical displacement, Euclidean distance, contact-derived and size-derived quantities where available, and standardized feature distance.

## Companion files

The compressed `__labels.npz` archive maps every covered voxel back to its supervoxel identifier at all three scales. `summary.csv` is append-safe across worker processes and provides the primary construction/QC ledger.

Consumers should validate graph and label-archive pairs together. A `.pt` file without its matching `__labels.npz`, or vice versa, is incomplete.
