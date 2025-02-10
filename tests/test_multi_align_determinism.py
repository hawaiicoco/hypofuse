"""Progressive alignment must be deterministic for the same input."""

from __future__ import annotations

from hypofuse.multi_align import grid_determinism_check, progressive_align


def test_same_input_same_output() -> None:
    hyps = [("a", "b", "c"), ("a", "b"), ("a", "b", "x")]
    grid = progressive_align(hyps)
    grid2 = progressive_align(hyps)
    assert grid.columns == grid2.columns
    assert grid.back_pointers == grid2.back_pointers


def test_permutation_changes_grid() -> None:
    hyps_a = [("a", "b"), ("a", "b", "c")]
    hyps_b = [("a", "b", "c"), ("a", "b")]
    grid_a = progressive_align(hyps_a)
    grid_b = progressive_align(hyps_b)
    # Different anchor changes column layout.
    assert grid_a.columns != grid_b.columns


def test_determinism_helper_returns_true() -> None:
    assert grid_determinism_check([("a", "b"), ("a", "x", "b")]) is True
