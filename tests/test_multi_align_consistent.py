"""Grid consistency check: valid and corrupted grids."""

from __future__ import annotations

from hypofuse.multi_align import (
    TokenGrid,
    grid_is_consistent,
    progressive_align,
)


def test_consistent_grid_from_progressive_align() -> None:
    hyps = [("a", "b", "c"), ("a", "x", "c"), ("a", "b")]
    grid = progressive_align(hyps)
    assert grid_is_consistent(grid) is True


def test_consistent_single_hypothesis() -> None:
    grid = progressive_align([("a", "b")])
    assert grid_is_consistent(grid) is True


def test_corrupted_grid_bad_pointer() -> None:
    grid = TokenGrid(
        columns=(("a", "a"), ("b", "b")),
        hypotheses=(("a", "b"), ("a", "b")),
        back_pointers=(
            ((0, "MATCH"), (1, "MATCH")),
            ((0, "MATCH"), (5, "MATCH")),
        ),
        pair_alignments=(),
    )
    assert grid_is_consistent(grid) is False


def test_corrupted_grid_wrong_length() -> None:
    grid = TokenGrid(
        columns=(("a", "a"), ("b", "b")),
        hypotheses=(("a", "b"), ("a", "b")),
        back_pointers=(
            ((0, "MATCH"),),
            ((0, "MATCH"), (1, "MATCH")),
        ),
        pair_alignments=(),
    )
    assert grid_is_consistent(grid) is False
