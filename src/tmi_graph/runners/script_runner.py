"""Subprocess runner for executing experiment scripts."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

from tmi_graph.config import ScriptConfig, merge_args
from tmi_graph.paths import repo_root
from tmi_graph.utils.logging import configure_logging


def build_command(config: ScriptConfig, extra_args: Sequence[str]) -> Sequence[str]:
    """Build the command line for executing a configured script.

    Args:
        config: Script execution configuration.
        extra_args: Additional args to append at runtime.

    Returns:
        The full command list suitable for `subprocess.run`.
    """
    python_exe = config.python or sys.executable
    script_path = str(config.resolved_script())
    args = merge_args(config.args, extra_args)
    return [python_exe, script_path, *args]


def build_environment(env_overrides: Mapping[str, str]) -> Mapping[str, str]:
    """Build a subprocess environment based on the current process environment.

    Args:
        env_overrides: Environment variable overrides.

    Returns:
        A mapping representing the subprocess environment.
    """
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in env_overrides.items()})
    return env


def run_script(
    config: ScriptConfig,
    *,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    log_level: str = "INFO",
) -> int:
    """Execute the configured experiment script in a subprocess.

    Args:
        config: Script execution configuration.
        extra_args: Additional args appended to the configured `args`.
        dry_run: If True, prints the resolved command and exits with code 0.
        log_level: Logging level for the runner.

    Returns:
        Process return code.

    Raises:
        subprocess.CalledProcessError: If the subprocess exits with a non-zero status
            and `check=True` is used internally.
    """
    logger = configure_logging(log_level, name="tmi_graph.runner")
    cmd = list(build_command(config, extra_args))
    env = build_environment(config.env)

    cwd = config.resolved_cwd()
    if cwd is None:
        cwd = repo_root()

    if dry_run:
        rendered = " ".join(shlex.quote(x) for x in cmd)
        logger.info("Dry run command: %s", rendered)
        logger.info("Working directory: %s", str(cwd))
        return 0

    logger.info("Executing: %s", " ".join(shlex.quote(x) for x in cmd))
    logger.info("Working directory: %s", str(cwd))

    completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return int(completed.returncode)
