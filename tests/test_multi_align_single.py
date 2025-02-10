"""Single hypothesis grid behaves like an identity mapping."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import AlignmentError
from hypofuse.multi_align import progressive_align


def test_single_hypothesis_columns_match_input() -> None:
    grid = progressive_align([("a", "b", "c")])
    assert grid.width == 3
    assert grid.depth == 1
    assert grid.columns == (("a",), ("b",), ("c",))


def test_single_hypothesis_back_pointers() -> None:
    grid = progressive_align([("a", "b")])
    bp = grid.back_pointers[0]
    assert bp == ((0, "MATCH"), (1, "MATCH"))


def test_empty_input_raises() -> None:
    with pytest.raises(AlignmentError):
        progressive_align([])
