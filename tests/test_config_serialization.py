"""Tests for config.to_dict and config.from_dict helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hypofuse.config import from_dict, to_dict


@dataclass(frozen=True)
class Inner:
    x: int
    y: str


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner
    values: tuple[int, ...]


def test_to_dict_converts_nested_dataclass() -> None:
    cfg = Outer(name="test", inner=Inner(x=1, y="a"), values=(10, 20))
    result = to_dict(cfg)
    assert result == {"name": "test", "inner": {"x": 1, "y": "a"}, "values": [10, 20]}


def test_to_dict_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError, match="expects a dataclass"):
        to_dict({"not": "a dataclass"})


def test_from_dict_restores_tuples() -> None:
    data = {"name": "test", "inner": {"x": 1, "y": "a"}, "values": [10, 20]}
    cfg = from_dict(Outer, data)
    assert cfg.values == (10, 20)
    assert isinstance(cfg.values, tuple)


def test_from_dict_reconstructs_nested() -> None:
    data = {"name": "test", "inner": {"x": 1, "y": "a"}, "values": [10, 20]}
    cfg = from_dict(Outer, data)
    assert isinstance(cfg.inner, Inner)
    assert cfg.inner.x == 1


def test_from_dict_rejects_unknown_keys() -> None:
    data = {"name": "test", "inner": {"x": 1, "y": "a"}, "values": [], "extra": "bad"}
    with pytest.raises(ValueError, match=r"unknown keys.*extra"):
        from_dict(Outer, data)


def test_from_dict_rejects_missing_required() -> None:
    data = {"name": "test", "inner": {"x": 1, "y": "a"}}
    with pytest.raises(ValueError, match=r"missing required key.*values"):
        from_dict(Outer, data)


def test_roundtrip_preserves_structure() -> None:
    original = Outer(name="round", inner=Inner(x=99, y="z"), values=(1, 2, 3))
    restored = from_dict(Outer, to_dict(original))
    assert restored == original
