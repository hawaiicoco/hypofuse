"""Column order must follow the sequence order, not the merge order.

Insertions used to be appended after every existing column, so a grid built
from ``("a", "b")`` and ``("a", "x", "y", "b")`` read back as ``a b x y``.
Anything that walks columns left to right -- ROVER voting, confusion-network
slots, 1-best extraction -- inherited that wrong order.
"""

from __future__ import annotations

from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import GAP, progressive_align


def test_columns_follow_sequence_order() -> None:
    grid = progressive_align([("a", "b"), ("a", "x", "y", "b")])
    assert [col[1] for col in grid.columns] == ["a", "x", "y", "b"]
    assert [col[0] for col in grid.columns] == ["a", GAP, GAP, "b"]


def test_fused_output_keeps_insertion_position() -> None:
    grid = progressive_align([("a", "b"), ("a", "x", "y", "b"), ("a", "x", "y", "b")])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    assert tuple(result.tokens) == ("a", "x", "y", "b")


def test_insertion_after_deletion_keeps_order() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "z", "c")])
    assert [col[1] for col in grid.columns] == ["a", "z", "c"]
    assert [col[0] for col in grid.columns] == ["a", "b", "c"]
