"""Tests for config.config_hash."""

from __future__ import annotations

from dataclasses import dataclass

from hypofuse.config import config_hash


@dataclass(frozen=True)
class Cfg:
    a: int
    b: str
    values: tuple[int, ...]


def test_equal_configs_equal_hash() -> None:
    c1 = Cfg(a=1, b="x", values=(10, 20))
    c2 = Cfg(a=1, b="x", values=(10, 20))
    assert config_hash(c1) == config_hash(c2)


def test_changed_field_different_hash() -> None:
    c1 = Cfg(a=1, b="x", values=(10, 20))
    c2 = Cfg(a=2, b="x", values=(10, 20))
    assert config_hash(c1) != config_hash(c2)


def test_hash_is_hex_digest() -> None:
    c = Cfg(a=1, b="x", values=(1, 2))
    h = config_hash(c)
    # SHA-256 produces 64 hex characters
    assert len(h) == 64
    assert all(ch in "0123456789abcdef" for ch in h)
