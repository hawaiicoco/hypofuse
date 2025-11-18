"""Baseline accuracy on simple inputs."""

from __future__ import annotations

import pytest

from hypofuse.neural import baseline_accuracy


def test_perfect_predictions() -> None:
    assert baseline_accuracy([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_all_wrong() -> None:
    assert baseline_accuracy([0.9, 0.1], [0, 1]) == pytest.approx(0.0)


def test_half_correct() -> None:
    assert baseline_accuracy([0.9, 0.9], [1, 0]) == pytest.approx(0.5)


def test_empty_input() -> None:
    assert baseline_accuracy([], []) == 0.0


def test_boundary_at_half() -> None:
    # score exactly 0.5 is treated as positive (>= 0.5)
    assert baseline_accuracy([0.5], [1]) == pytest.approx(1.0)
    assert baseline_accuracy([0.5], [0]) == pytest.approx(0.0)


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        baseline_accuracy([0.5], [1, 0])
