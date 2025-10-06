"""Gap symbol guard: input tokens must not equal the gap symbol."""

from __future__ import annotations

import pytest

from hypofuse.multi_align import GAP, progressive_align


def test_gap_symbol_value() -> None:
    assert GAP == "*"


def test_gap_in_first_hypothesis_raises() -> None:
    with pytest.raises(ValueError, match="gap symbol"):
        progressive_align([("*", "a", "b")])


def test_gap_in_second_hypothesis_raises() -> None:
    with pytest.raises(ValueError, match="gap symbol"):
        progressive_align([("a", "b"), ("a", "*", "b")])


def test_gap_in_third_hypothesis_raises() -> None:
    with pytest.raises(ValueError, match="gap symbol"):
        progressive_align([("a", "b"), ("a", "b"), ("*",)])


def test_normal_tokens_pass() -> None:
    grid = progressive_align([("a", "b"), ("a", "x")])
    assert grid.width >= 2
