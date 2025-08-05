"""Tests for config.merge_configs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hypofuse.config import merge_configs


@dataclass(frozen=True)
class Inner:
    x: int
    y: str


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner


def test_empty_override_returns_equal() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    result = merge_configs(base, {})
    assert result == base


def test_simple_override() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    result = merge_configs(base, {"name": "new"})
    assert result.name == "new"
    assert result.inner == base.inner


def test_dotted_path_override() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    result = merge_configs(base, {"inner.x": 99})
    assert result.inner.x == 99
    assert result.inner.y == "a"
    assert result.name == "test"


def test_unknown_path_raises() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    with pytest.raises(ValueError, match="unknown path"):
        merge_configs(base, {"nonexistent": 5})


def test_unknown_nested_path_raises() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    with pytest.raises(ValueError, match="unknown path"):
        merge_configs(base, {"inner.z": 5})


def test_base_unchanged() -> None:
    base = Outer(name="test", inner=Inner(x=1, y="a"))
    merge_configs(base, {"name": "changed"})
    assert base.name == "test"
