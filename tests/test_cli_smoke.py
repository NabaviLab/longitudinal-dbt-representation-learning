from __future__ import annotations

from tmi_graph.cli import main


def test_cli_run_dry_run() -> None:
    """Ensure the generic runner subcommand supports dry-run execution."""
    code = main(["run", "--config", "configs/pretrain.yaml", "--dry-run"])
    assert code == 0


def test_cli_named_command_dry_run() -> None:
    """Ensure the named pretrain subcommand resolves its default config."""
    code = main(["pretrain", "--dry-run"])
    assert code == 0
