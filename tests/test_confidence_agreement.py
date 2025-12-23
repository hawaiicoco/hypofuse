"""Agreement confidence from fusion grid columns."""

from __future__ import annotations

import pytest

from hypofuse.confidence import agreement_confidence
from hypofuse.fusion import fuse
from hypofuse.multi_align import GAP, progressive_align


def test_unanimous_column() -> None:
    assert agreement_confidence([["a", "a", "a"]]) == [pytest.approx(1.0)]


def test_split_column() -> None:
    # 2 out of 3 agree -> 2/3
    result = agreement_confidence([["a", "a", "b"]])
    assert result == [pytest.approx(2 / 3)]


def test_all_gap_column() -> None:
    assert agreement_confidence([["*", "*", "*"]], gap="*") == [0.0]


def test_matches_fusion_majority_confidences() -> None:
    grid = progressive_align([("a", "b"), ("a", "b"), ("x", "y")])
    result = fuse(grid)
    ag = agreement_confidence(grid.columns, gap=GAP)
    for a, c in zip(ag, result.confidences, strict=True):
        assert a == pytest.approx(c)


def test_multiple_columns() -> None:
    cols = [["a", "a", "a"], ["a", "b", "c"]]
    result = agreement_confidence(cols)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1 / 3)
