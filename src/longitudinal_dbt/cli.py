"""Public command-line interface."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from longitudinal_dbt import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longitudinal-dbt",
        description="Convert DBT DICOM studies and construct multiscale graph artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("convert-dicom", "build-graphs"),
        help="Command to run. Use '<command> --help' for command-specific options.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch a public subcommand without importing optional heavy dependencies."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "--version"}:
        namespace = _parser().parse_args(args)
        if namespace.command is None and "--version" not in args:
            _parser().print_help()
        return 0

    command, command_args = args[0], args[1:]
    if command == "convert-dicom":
        from longitudinal_dbt.preprocessing.dicom_to_tiff import main as convert_main

        return convert_main(command_args)
    if command == "build-graphs":
        from longitudinal_dbt.preprocessing.graph_construction import main as graph_main

        return graph_main(command_args)

    _parser().error(f"unknown command: {command}")
    return 2
