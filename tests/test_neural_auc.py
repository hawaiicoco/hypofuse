"""AUC on tiny hand-computed cases."""

from __future__ import annotations

import pytest

from hypofuse.neural import auc_from_scores


def test_perfect_separation() -> None:
    # ranks [1, 2, 3, 4]; positive sum = 3+4 = 7; U = 7-3 = 4; AUC = 4/4 = 1.0
    assert auc_from_scores([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_all_tied() -> None:
    # all ranks average to 2.5; positive sum = 5.0; U = 5-3 = 2; AUC = 2/4 = 0.5
    assert auc_from_scores([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_partial_overlap() -> None:
    # scores [0.4, 0.6, 0.3, 0.7] labels [1, 0, 0, 1]
    # sorted: (2,0.3) (0,0.4) (1,0.6) (3,0.7) -> ranks [2,3,1,4]
    # positive sum = 2+4 = 6; U = 6-3 = 3; AUC = 3/4 = 0.75
    assert auc_from_scores([0.4, 0.6, 0.3, 0.7], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_inverse_separation() -> None:
    # positives have lowest scores -> AUC = 0.0
    assert auc_from_scores([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_single_pair() -> None:
    assert auc_from_scores([0.3, 0.7], [0, 1]) == pytest.approx(1.0)
    assert auc_from_scores([0.7, 0.3], [0, 1]) == pytest.approx(0.0)


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        auc_from_scores([0.5], [1, 0])


def test_single_class_raises() -> None:
    with pytest.raises(ValueError, match="both classes"):
        auc_from_scores([0.5, 0.6], [1, 1])


def test_with_ties_at_boundary() -> None:
    # scores [0.5, 0.5, 0.3, 0.7] labels [1, 0, 0, 1]
    # sorted: (2,0.3) (0,0.5) (1,0.5) (3,0.7) -> ranks [2.5, 2.5, 1.0, 4.0]
    # positive sum = 2.5+4.0 = 6.5; U = 6.5-3 = 3.5; AUC = 3.5/4 = 0.875
    assert auc_from_scores([0.5, 0.5, 0.3, 0.7], [1, 0, 0, 1]) == pytest.approx(0.875)
