"""Tests for config.describe."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hypofuse.config import describe


@dataclass(frozen=True)
class Sample:
    alpha: float
    beta: int
    name: str


def test_describe_lists_all_fields() -> None:
    cfg = Sample(alpha=1.5, beta=42, name="test")
    text = describe(cfg)
    assert "alpha" in text
    assert "beta" in text
    assert "name" in text


def test_describe_each_field_exactly_once() -> None:
    cfg = Sample(alpha=1.5, beta=42, name="test")
    text = describe(cfg)
    for field_name in ["alpha", "beta", "name"]:
        assert text.count(field_name) == 1


def test_describe_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError, match="expects a dataclass"):
        describe({"not": "a dataclass"})


def test_describe_multiline() -> None:
    cfg = Sample(alpha=1.5, beta=42, name="test")
    text = describe(cfg)
    assert "\n" in text
    lines = text.split("\n")
    assert len(lines) == 3
