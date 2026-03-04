"""Offline packaging metadata tests.

Verify pyproject.toml declares the correct structure and matches the runtime
version constant.  These tests parse pyproject.toml with tomllib and import
hypofuse -- they never build a wheel or require network access.
"""

from __future__ import annotations

import tomllib
from functools import cache
from pathlib import Path

import hypofuse


@cache
def _meta() -> dict:  # type: ignore[type-arg]
    """Load and cache pyproject.toml metadata."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


def test_requires_python_at_least_311() -> None:
    """requires-python must be >=3.11."""
    assert _meta()["project"]["requires-python"] == ">=3.11"


def test_numpy_is_only_required_dependency() -> None:
    """numpy must be the sole required runtime dependency."""
    deps: list[str] = _meta()["project"]["dependencies"]
    names = [d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for d in deps]
    assert names == ["numpy"], f"expected only numpy, got {names}"


def test_torch_only_in_optional_extra() -> None:
    """torch must not be a required dep; it lives under the torch extra only."""
    required: list[str] = _meta()["project"]["dependencies"]
    assert not any("torch" in d.lower() for d in required)
    extras: dict[str, list[str]] = _meta()["project"]["optional-dependencies"]
    assert "torch" in extras
    assert any("torch" in d.lower() for d in extras["torch"])


def test_console_script_entry_point() -> None:
    """The hypofuse console script must point at hypofuse.cli:main."""
    scripts: dict[str, str] = _meta()["project"]["scripts"]
    assert scripts["hypofuse"] == "hypofuse.cli:main"


def test_hatchling_wheel_target_packages() -> None:
    """The hatchling wheel target must package src/hypofuse."""
    targets = _meta()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert targets["packages"] == ["src/hypofuse"]


def test_declared_version_matches_runtime() -> None:
    """pyproject.toml version must equal hypofuse.__version__."""
    declared: str = _meta()["project"]["version"]
    assert declared == hypofuse.__version__
