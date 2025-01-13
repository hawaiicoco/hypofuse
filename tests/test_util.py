"""Tests for hypofuse.util helpers."""

from __future__ import annotations

import pytest

from hypofuse.util import relative_within, safe_join, seeded, stable_hash


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_changes_with_value() -> None:
    assert stable_hash({"x": 1}) != stable_hash({"x": 2})


def test_stable_hash_accepts_nested() -> None:
    a = stable_hash({"nested": {"list": [1, 2, 3]}})
    b = stable_hash({"nested": {"list": [1, 2, 3]}})
    assert a == b


def test_seeded_is_deterministic() -> None:
    a = [seeded(123).random() for _ in range(10)]
    b = [seeded(123).random() for _ in range(10)]
    assert a == b


def test_seeded_none_still_independent() -> None:
    rng = seeded(None)
    first = rng.random()
    second = rng.random()
    # two draws from one rng should still differ
    assert first != second


def test_relative_within_returns_relative(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    inner = base / "x" / "y.txt"
    inner.parent.mkdir()
    inner.write_text("hi")
    assert relative_within(base, inner) == "x/y.txt"


def test_relative_within_rejects_escape(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("hi")
    with pytest.raises(ValueError, match="outside base"):
        relative_within(base, outside)


def test_safe_join_rejects_traversal(tmp_path) -> None:
    base = tmp_path / "safe"
    base.mkdir()
    with pytest.raises(ValueError, match="escapes base"):
        safe_join(base, "..", "etc", "passwd")
