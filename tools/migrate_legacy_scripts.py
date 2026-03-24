"""Utilities for migrating original research scripts into this repository layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

from tmi_graph.paths import ensure_dir, resolve_repo_path


RENAME_MAP: Dict[str, str] = {
    "TR_pretrain_v2.py": "src/tmi_graph/experiments/pretrain_tr.py",
    "super_voxel_slic_aug.py": "src/tmi_graph/experiments/graph_current.py",
    "super_voxel_slic_aug_prev.py": "src/tmi_graph/experiments/graph_prior.py",
    "super_voxel_slic_aug_BCS_higher_res.py": "src/tmi_graph/experiments/bcs_highres_pipeline.py",
}


def migrate_scripts(source_dir: Path, *, overwrite: bool = False) -> None:
    """Copy and rename legacy scripts into the repository structure.

    Args:
        source_dir: Directory containing the original scripts.
        overwrite: If True, overwrites existing destination files.

    Returns:
        None

    Raises:
        FileNotFoundError: If a required source script is missing.
    """
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for src_name, dst_rel in RENAME_MAP.items():
        src_path = source_dir / src_name
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source script: {src_path}")

        dst_path = resolve_repo_path(dst_rel)
        ensure_dir(dst_path.parent)

        if dst_path.exists() and not overwrite:
            continue

        shutil.copy2(src_path, dst_path)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this tool."""
    p = argparse.ArgumentParser(description="Migrate legacy scripts into the repo layout.")
    p.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing TR_pretrain_v2.py and super_voxel_slic_aug*.py files.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist.")
    return p


def main() -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args()
    migrate_scripts(Path(args.source_dir), overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
