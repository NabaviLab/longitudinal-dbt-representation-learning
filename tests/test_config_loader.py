from __future__ import annotations

from pathlib import Path

import pytest

from tmi_graph.config import load_script_config


def test_load_script_config_valid(tmp_path: Path) -> None:
    """Load a minimal YAML config and verify that script resolution works."""
    repo_tmp = tmp_path / "repo"
    repo_tmp.mkdir()
    (repo_tmp / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.0'\n")
    script_path = repo_tmp / "script.py"
    script_path.write_text("print('ok')\n")
    cfg_path = repo_tmp / "cfg.yaml"
    cfg_path.write_text("script: script.py\nargs: ['--help']\nenv: {}\ncwd: .\npython: null\n")

    from tmi_graph import paths as paths_mod

    original_repo_root = paths_mod.repo_root
    paths_mod.repo_root = lambda start=None: repo_tmp
    try:
        cfg = load_script_config("cfg.yaml")
        assert cfg.resolved_script() == script_path.resolve()
    finally:
        paths_mod.repo_root = original_repo_root


def test_load_script_config_missing_raises(tmp_path: Path) -> None:
    """Verify that missing configured scripts raise FileNotFoundError."""
    repo_tmp = tmp_path / "repo"
    repo_tmp.mkdir()
    (repo_tmp / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.0'\n")
    cfg_path = repo_tmp / "cfg.yaml"
    cfg_path.write_text("script: does_not_exist.py\nargs: []\nenv: {}\n")

    from tmi_graph import paths as paths_mod

    original_repo_root = paths_mod.repo_root
    paths_mod.repo_root = lambda start=None: repo_tmp
    try:
        with pytest.raises(FileNotFoundError):
            load_script_config("cfg.yaml")
    finally:
        paths_mod.repo_root = original_repo_root
