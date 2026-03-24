"""Command-line interface for running experiment scripts."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from tmi_graph.config import load_script_config
from tmi_graph.runners.script_runner import run_script


def _strip_passthrough_separator(args: Sequence[str]) -> List[str]:
    """Normalize passthrough arguments captured by `argparse.REMAINDER`.

    Args:
        args: Raw captured remainder arguments.

    Returns:
        Normalized list where a leading `--` separator is removed if present.
    """
    if not args:
        return []
    if args[0] == "--":
        return list(args[1:])
    return list(args)


def _add_common_runner_flags(parser: argparse.ArgumentParser) -> None:
    """Attach common runner flags to a parser.

    Args:
        parser: Parser instance to modify.

    Returns:
        None
    """
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the runner (e.g., INFO, DEBUG).",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

    Returns:
        Configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(prog="tmi-graph", description="Run TMI experiment scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run a script specified by a YAML config.")
    run_p.add_argument("--config", required=True, help="Path to a YAML config under configs/ or an absolute path.")
    run_p.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed through to the script.")
    _add_common_runner_flags(run_p)

    def _add_named_command(name: str, default_config: str, help_text: str) -> None:
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument(
            "--config",
            default=default_config,
            help=f"Config file to use (default: {default_config}).",
        )
        p.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed through to the script.")
        _add_common_runner_flags(p)

    _add_named_command("pretrain", "configs/pretrain.yaml", "Run the pretraining entrypoint.")
    _add_named_command("graph-current", "configs/graph_current.yaml", "Run graph construction for current scans.")
    _add_named_command("graph-prior", "configs/graph_prior.yaml", "Run graph construction for prior scans.")
    _add_named_command("graph-bcs", "configs/graph_bcs_highres.yaml", "Run graph construction for public BCS data.")
    _add_named_command("finetune", "configs/finetune.yaml", "Run the final fine-tuning entrypoint.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the `tmi-graph` command.

    Args:
        argv: Optional argument sequence. If omitted, arguments are read from `sys.argv`.

    Returns:
        Process exit code (0 indicates success).
    """
    parser = _build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_script_config(ns.config)
    passthrough = _strip_passthrough_separator(ns.script_args)

    return run_script(cfg, extra_args=passthrough, dry_run=bool(ns.dry_run), log_level=str(ns.log_level))
