"""Tests for fuse_many and fusion_summary batch operations."""

from __future__ import annotations

import pytest

from hypofuse.fusion import (
    FusionConfig,
    FusionSummary,
    fuse_many,
    fusion_summary,
)
from hypofuse.multi_align import GAP, TokenGrid, progressive_align


def test_fuse_many_empty_input() -> None:
    assert fuse_many([]) == []


def test_fuse_many_single_grid() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    results = fuse_many([grid])
    assert len(results) == 1
    assert results[0].tokens == ("a", "b") or results[0].tokens[0] == "a"


def test_fuse_many_multiple_grids() -> None:
    g1 = progressive_align([("a",), ("a",)])
    g2 = progressive_align([("b",), ("c",)])
    results = fuse_many([g1, g2])
    assert len(results) == 2


def test_fusion_summary_empty() -> None:
    summary = fusion_summary([])
    assert summary.token_count == 0
    assert summary.mean_confidence == pytest.approx(0.0)
    assert summary.unanimous_fraction == pytest.approx(0.0)
    assert summary.gap_fraction == pytest.approx(0.0)


def test_fusion_summary_single_result() -> None:
    grid = progressive_align([("a",), ("a",)])
    results = fuse_many([grid])
    summary = fusion_summary(results)
    assert summary.token_count == 1
    assert summary.mean_confidence == pytest.approx(1.0)
    assert summary.unanimous_fraction == pytest.approx(1.0)
    assert summary.gap_fraction == pytest.approx(0.0)


def test_fusion_summary_seeded_batch() -> None:
    """Hand-checkable summary over a small seeded batch."""
    # 3 grids, each unanimous with 1 token => total_tokens=3
    # mean_confidence=1.0, unanimous_fraction=1.0, gap_fraction=0.0
    grids = [
        progressive_align([("a",), ("a",)]),
        progressive_align([("b",), ("b",)]),
        progressive_align([("c",), ("c",)]),
    ]
    results = fuse_many(grids)
    summary = fusion_summary(results)
    assert summary.token_count == 3
    assert summary.mean_confidence == pytest.approx(1.0)
    assert summary.unanimous_fraction == pytest.approx(1.0)
    assert summary.gap_fraction == pytest.approx(0.0)


def test_fusion_summary_with_gap_columns() -> None:
    """A column where one hypothesis has a token is not a gap column."""
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    results = fuse_many([grid], config=FusionConfig(null_policy="keep"))
    summary = fusion_summary(results)
    # Columns are ("a","a"), ("b",GAP), ("c","c"): "b" still wins the middle.
    assert summary.token_count == 3
    assert summary.gap_fraction == pytest.approx(0.0)


def test_fusion_summary_counts_all_gap_columns() -> None:
    """An all-gap column survives under 'keep' and counts as a gap."""
    grid = TokenGrid(
        columns=(("a", "a"), (GAP, GAP), ("c", "c")),
        hypotheses=(("a", "c"), ("a", "c")),
        back_pointers=(((0, "MATCH"), (2, "MATCH")), ((0, "MATCH"), (2, "MATCH"))),
        pair_alignments=(),
    )
    summary = fusion_summary(fuse_many([grid], config=FusionConfig(null_policy="keep")))
    assert summary.gap_fraction == pytest.approx(1 / 3)


def test_fusion_summary_is_dataclass() -> None:
    summary = FusionSummary(
        token_count=10, mean_confidence=0.8, unanimous_fraction=0.5, gap_fraction=0.1
    )
    assert summary.token_count == 10
    assert summary.mean_confidence == pytest.approx(0.8)
