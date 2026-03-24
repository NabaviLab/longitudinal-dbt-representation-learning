"""Run graph construction for current scans using the repository CLI."""

from __future__ import annotations

import sys

from tmi_graph.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["graph-current", *sys.argv[1:]]))
