"""The token grid must stay rectangular.

Every column holds exactly one cell per hypothesis: a token where that
hypothesis contributes and the gap symbol elsewhere. Insertion columns used
to be shorter than the rest of the grid, which made per-row views (and any
vote counted over a column) read the wrong hypothesis.
"""

from __future__ import annotations

from hypofuse.multi_align import GAP, grid_row, progressive_align


def test_insertion_columns_are_padded_with_gaps() -> None:
    grid = progressive_align([("a", "b"), ("a", "x", "y", "b")])
    assert grid.width == 4
    for col in grid.columns:
        assert len(col) == grid.depth, col
    # The two inserted tokens sit in their own columns, gap-padded for row 0.
    assert grid.columns[1] == (GAP, "x")
    assert grid.columns[2] == (GAP, "y")


def test_rows_read_back_the_original_hypotheses() -> None:
    hyps = [("a", "b"), ("a", "x", "y", "b"), ("a", "b", "c")]
    grid = progressive_align(hyps)
    for index, hyp in enumerate(hyps):
        assert tuple(tok for tok in grid_row(grid, index) if tok != GAP) == hyp


def test_three_hypothesis_grid_is_rectangular() -> None:
    grid = progressive_align([("a",), ("a", "b"), ("c", "b", "d")])
    assert all(len(col) == grid.depth for col in grid.columns)
