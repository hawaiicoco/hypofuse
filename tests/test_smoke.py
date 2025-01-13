"""Bootstrap smoke test: package imports and exposes a version string."""

from __future__ import annotations

import re

import hypofuse
from hypofuse import _version


def test_version_is_semver() -> None:
    assert isinstance(hypofuse.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", hypofuse.__version__), hypofuse.__version__


def test_version_constant_matches_package() -> None:
    assert hypofuse.__version__ == _version.__version__
