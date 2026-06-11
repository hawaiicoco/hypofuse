"""Version and changelog consistency checks.

Parses CHANGELOG.md and pyproject.toml to verify the declared version
matches hypofuse.__version__ and follows the 0.1.N patch format.
"""

from __future__ import annotations

import re
import tomllib
from functools import cache
from pathlib import Path

import hypofuse

_ROOT = Path(__file__).resolve().parent.parent


@cache
def _pyproject_version() -> str:
    """Read the version field from pyproject.toml."""
    path = _ROOT / "pyproject.toml"
    with path.open("rb") as fh:
        meta = tomllib.load(fh)
    return str(meta["project"]["version"])


def _changelog_versions() -> list[str]:
    """Return every ## [x.y.z] heading from CHANGELOG.md, top first."""
    path = _ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^## \[([\d]+\.[\d]+\.[\d]+)\]", re.MULTILINE)
    return pattern.findall(text)


def test_changelog_newest_matches_runtime() -> None:
    """The newest CHANGELOG heading must equal hypofuse.__version__."""
    versions = _changelog_versions()
    assert versions, "CHANGELOG.md has no version headings"
    assert versions[0] == hypofuse.__version__, (
        f"CHANGELOG newest is {versions[0]}, runtime is {hypofuse.__version__}"
    )


def test_pyproject_matches_runtime() -> None:
    """pyproject.toml version must equal hypofuse.__version__."""
    assert _pyproject_version() == hypofuse.__version__


def test_version_is_patch_format() -> None:
    """Version must match ^0\\.1\\.\\d+$ (patch-only releases)."""
    assert re.fullmatch(r"0\.1\.\d+", hypofuse.__version__), (
        f"version {hypofuse.__version__} does not match 0.1.N"
    )
