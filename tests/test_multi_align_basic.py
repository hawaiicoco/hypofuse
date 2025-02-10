"""Pairwise grid alignment basics."""

from __future__ import annotations

from hypofuse.multi_align import GAP, grid_row, progressive_align


def test_two_identical_hypotheses_full_match() -> None:
    grid = progressive_align([("a", "b"), ("a", "b")])
    assert grid.width == 2
    assert grid.depth == 2
    assert grid.columns == (("a", "a"), ("b", "b"))


def test_grid_row_returns_gap_for_missing_position() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    # Second hypothesis is shorter, so the missing position becomes a gap.
    row = grid_row(grid, hypothesis_index=1)
    assert row[0] == "a"
    assert row[2] == "c"
    assert row[1] == GAP


def test_grid_records_insertion_column() -> None:
    grid = progressive_align([("a", "b"), ("a", "x", "b")])
    # The 'x' should appear in its own column.
    cols_with_x = [col for col in grid.columns if "x" in col]
    assert cols_with_x, grid.columns


def test_grid_three_hypotheses_alignment_width() -> None:
    grid = progressive_align([("a", "b"), ("a", "b"), ("a", "b")])
    assert grid.width == 2
    assert grid.depth == 3


def test_grid_returns_pair_alignments() -> None:
    grid = progressive_align([("a", "b"), ("a", "x")])
    assert len(grid.pair_alignments) == 1
