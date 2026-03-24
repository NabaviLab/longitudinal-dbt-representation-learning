"""Docstring coverage checker for Python functions."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MissingDocstring:
    """Record describing a function without a docstring."""

    file: Path
    line: int
    qualname: str


def iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    """Yield Python files under the provided paths.

    Args:
        paths: Files or directories to scan.

    Yields:
        Resolved `.py` file paths.
    """
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file() and p.suffix == ".py":
            yield p
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))


def _is_dunder(name: str) -> bool:
    """Return True if a name is a dunder (e.g., __init__)."""
    return name.startswith("__") and name.endswith("__")


def find_missing_docstrings(
    file_path: Path,
    *,
    ignore_dunder: bool = False,
) -> List[MissingDocstring]:
    """Find functions without docstrings in a Python source file.

    Args:
        file_path: File to analyze.
        ignore_dunder: If True, ignores dunder methods and functions.

    Returns:
        List of missing-docstring records.

    Raises:
        SyntaxError: If the file cannot be parsed.
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    missing: List[MissingDocstring] = []

    def visit(node: ast.AST, parents: Tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if ignore_dunder and _is_dunder(name):
                return
            if ast.get_docstring(node) is None:
                qual = ".".join([*parents, name]) if parents else name
                missing.append(MissingDocstring(file=file_path, line=int(node.lineno), qualname=qual))
        if isinstance(node, ast.ClassDef):
            next_parents = (*parents, node.name)
        else:
            next_parents = parents

        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree, ())
    return missing


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(description="Check that functions have docstrings.")
    p.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more files/directories to scan.",
    )
    p.add_argument("--ignore-dunder", action="store_true", help="Ignore dunder methods/functions.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    files = list(iter_python_files([Path(x) for x in args.paths]))

    all_missing: List[MissingDocstring] = []
    for f in files:
        all_missing.extend(find_missing_docstrings(f, ignore_dunder=bool(args.ignore_dunder)))

    if not all_missing:
        return 0

    for item in all_missing:
        print(f"{item.file}:{item.line}: missing docstring: {item.qualname}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
