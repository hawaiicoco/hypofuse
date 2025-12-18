"""Brier score and log loss metrics."""

from __future__ import annotations

import math

import pytest

from hypofuse.confidence import brier_score, log_loss


def test_brier_score_hand_computed() -> None:
    # (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5; mean = 0.25
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_brier_score_perfect() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


def test_brier_score_worst() -> None:
    # (1-0)^2 + (0-1)^2 = 2; mean = 1.0
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_brier_score_three_elements() -> None:
    # (0.8-1)^2 + (0.2-0)^2 + (0.5-1)^2 = 0.04 + 0.04 + 0.25 = 0.33
    # mean = 0.33 / 3 = 0.11
    assert brier_score([0.8, 0.2, 0.5], [1, 0, 1]) == pytest.approx(0.11)


def test_brier_score_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"same length"):
        brier_score([0.5], [1, 0])


def test_brier_score_non_binary_labels() -> None:
    with pytest.raises(ValueError, match=r"binary"):
        brier_score([0.5, 0.5], [0, 2])


def test_brier_score_empty() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        brier_score([], [])


def test_log_loss_hand_computed() -> None:
    # -ln(0.5) for both -> mean = ln(2)
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))


def test_log_loss_perfect_near_zero() -> None:
    assert log_loss([1.0, 0.0], [1, 0]) < 1e-10


def test_log_loss_three_element() -> None:
    # all p=0.5, all give -ln(0.5) = ln(2); mean = ln(2)
    assert log_loss([0.5, 0.5, 0.5], [1, 0, 1]) == pytest.approx(math.log(2))


def test_log_loss_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"same length"):
        log_loss([0.5], [1, 0])


def test_log_loss_non_binary_labels() -> None:
    with pytest.raises(ValueError, match=r"binary"):
        log_loss([0.5], [0.5])


def test_log_loss_empty() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        log_loss([], [])
