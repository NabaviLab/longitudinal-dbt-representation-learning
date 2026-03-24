"""Path utilities for resolving repository-relative files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def repo_root(start: Optional[Path] = None) -> Path:
    """Resolve the repository root directory.

    The root is identified by walking up parent directories until one of the following
    markers is found:

    - `pyproject.toml`
    - `.git/`

    Args:
        start: Starting path for the search. If omitted, the search begins from the
            current file location.

    Returns:
        The resolved repository root directory.

    Raises:
        RuntimeError: If the repository root cannot be located.
    """
    anchor = (start or Path(__file__)).resolve()
    for parent in [anchor, *anchor.parents]:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (missing pyproject.toml or .git).")


def resolve_repo_path(relative_path: PathLike, *, root: Optional[Path] = None) -> Path:
    """Resolve a path relative to the repository root.

    Args:
        relative_path: Path relative to the repository root.
        root: Explicit repository root. If omitted, `repo_root()` is used.

    Returns:
        An absolute `Path` pointing to the requested location.
    """
    base = (root or repo_root()).resolve()
    return (base / Path(relative_path)).resolve()


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved directory path.
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
