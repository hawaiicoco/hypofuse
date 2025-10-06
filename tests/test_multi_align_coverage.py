"""Grid coverage: non-gap cell counts per row."""

from __future__ import annotations

from hypofuse.multi_align import grid_coverage, progressive_align


def test_coverage_sum_equals_total_tokens() -> None:
    hyps = [("a", "b", "c"), ("a", "x", "c"), ("a", "b")]
    grid = progressive_align(hyps)
    cov = grid_coverage(grid)
    total_tokens = sum(len(h) for h in hyps)
    assert sum(cov) == total_tokens


def test_coverage_identical_hypotheses() -> None:
    hyps = [("a", "b"), ("a", "b"), ("a", "b")]
    grid = progressive_align(hyps)
    cov = grid_coverage(grid)
    assert cov == [2, 2, 2]


def test_coverage_single_hypothesis() -> None:
    grid = progressive_align([("a", "b", "c")])
    cov = grid_coverage(grid)
    assert cov == [3]


def test_coverage_with_insertions() -> None:
    hyps = [("a", "b"), ("a", "x", "y", "b")]
    grid = progressive_align(hyps)
    cov = grid_coverage(grid)
    assert sum(cov) == 6
    assert cov[0] == 2
    assert cov[1] == 4
