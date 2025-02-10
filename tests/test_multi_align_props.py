"""Property-style invariants for the alignment grid."""

from __future__ import annotations

from hypofuse.multi_align import grid_row, progressive_align


def test_grid_row_consistent_with_columns() -> None:
    hyps = [("a", "b", "c"), ("a", "x", "c"), ("a", "b", "c")]
    grid = progressive_align(hyps)
    for h_idx in range(grid.depth):
        row = grid_row(grid, h_idx)
        for col_idx, token in enumerate(row):
            if token == "*":
                continue
            assert token in grid.columns[col_idx]


def test_grid_covers_every_input_position() -> None:
    hyps = [("a", "b", "c", "d"), ("a", "c", "d"), ("a", "b", "x", "d")]
    grid = progressive_align(hyps)
    for h_idx, hyp in enumerate(hyps):
        bp = grid.back_pointers[h_idx]
        assert len(bp) == len(hyp)
        consumed = sorted(p[0] for p in bp)
        assert min(consumed) >= 0
        assert max(consumed) < grid.width


def test_pair_alignment_count_matches_input_count() -> None:
    hyps = [("a", "b"), ("a", "c"), ("a", "b")]
    grid = progressive_align(hyps)
    assert len(grid.pair_alignments) == len(hyps) - 1


def test_insertions_add_columns() -> None:
    hyps = [("a", "b"), ("a", "x", "y", "b")]
    grid = progressive_align(hyps)
    # Two extra tokens in the second hypothesis must add columns.
    assert grid.width >= 4
