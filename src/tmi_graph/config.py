"""Configuration loading for script execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from tmi_graph.paths import resolve_repo_path


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ScriptConfig:
    """Configuration describing how to execute a Python script.

    Attributes:
        script: Path (relative to repo root or absolute) to the Python script.
        args: Argument list passed to the script.
        env: Environment variable overrides applied to the subprocess environment.
        cwd: Working directory for the subprocess (relative to repo root or absolute).
        python: Python executable to use. If omitted, the current interpreter is used.
    """

    script: str
    args: List[str]
    env: Dict[str, str]
    cwd: Optional[str] = None
    python: Optional[str] = None

    def resolved_script(self) -> Path:
        """Resolve the configured script path to an absolute path."""
        return resolve_repo_path(self.script)

    def resolved_cwd(self) -> Optional[Path]:
        """Resolve the configured working directory to an absolute path."""
        if self.cwd is None:
            return None
        return resolve_repo_path(self.cwd)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScriptConfig":
        """Create a `ScriptConfig` from a raw mapping.

        Args:
            mapping: Parsed configuration mapping (typically from YAML).

        Returns:
            A validated `ScriptConfig`.

        Raises:
            TypeError: If required fields are missing or have invalid types.
            ValueError: If the config contains invalid values.
        """
        if "script" not in mapping:
            raise TypeError("Config is missing required field: 'script'.")

        script = mapping["script"]
        if not isinstance(script, str) or not script.strip():
            raise TypeError("Field 'script' must be a non-empty string.")

        args_raw = mapping.get("args", [])
        if args_raw is None:
            args_raw = []
        if not isinstance(args_raw, list) or not all(isinstance(x, (str, int, float)) for x in args_raw):
            raise TypeError("Field 'args' must be a list of strings/numbers.")

        args = [str(x) for x in args_raw]

        env_raw = mapping.get("env", {}) or {}
        if not isinstance(env_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in env_raw.items()
        ):
            raise TypeError("Field 'env' must be a mapping of string keys to string/number values.")

        env = {k: str(v) for k, v in env_raw.items()}

        cwd = mapping.get("cwd", None)
        if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
            raise TypeError("Field 'cwd' must be a string path or null.")

        python = mapping.get("python", None)
        if python is not None and (not isinstance(python, str) or not python.strip()):
            raise TypeError("Field 'python' must be a string path or null.")

        return cls(script=script, args=args, env=env, cwd=cwd, python=python)


def load_script_config(path: PathLike) -> ScriptConfig:
    """Load a `ScriptConfig` from a YAML file.

    Args:
        path: YAML file path (relative to repo root or absolute).

    Returns:
        The validated `ScriptConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML content is empty or invalid.
        TypeError: If the YAML root is not a mapping.
    """
    config_path = resolve_repo_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty YAML config: {config_path}")
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(raw).__name__} in {config_path}")

    cfg = ScriptConfig.from_mapping(raw)

    script_path = cfg.resolved_script()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    return cfg


def merge_args(config_args: Sequence[str], extra_args: Sequence[str]) -> List[str]:
    """Merge configured arguments with extra arguments.

    Args:
        config_args: Arguments loaded from configuration.
        extra_args: Additional arguments supplied at runtime.

    Returns:
        Combined argument list.
    """
    return [*list(config_args), *list(extra_args)]
