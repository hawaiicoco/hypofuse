"""Tests for duration_buckets label classification."""

from __future__ import annotations

from hypofuse.timings import duration_buckets


def test_below_first_edge() -> None:
    assert duration_buckets(0.5) == "<1.0s"


def test_at_first_edge_upper_bucket() -> None:
    # Boundary lands in the upper bucket
    assert duration_buckets(1.0) == "1.0-3.0s"


def test_between_first_and_second() -> None:
    assert duration_buckets(2.0) == "1.0-3.0s"


def test_at_second_edge_upper_bucket() -> None:
    assert duration_buckets(3.0) == "3.0-8.0s"


def test_between_second_and_third() -> None:
    assert duration_buckets(5.0) == "3.0-8.0s"


def test_just_below_third_edge() -> None:
    assert duration_buckets(7.9) == "3.0-8.0s"


def test_at_third_edge_upper_bucket() -> None:
    assert duration_buckets(8.0) == ">=8.0s"


def test_above_last_edge() -> None:
    assert duration_buckets(100.0) == ">=8.0s"


def test_zero_duration() -> None:
    assert duration_buckets(0.0) == "<1.0s"


def test_custom_edges() -> None:
    assert duration_buckets(0.5, edges=(0.5, 2.0)) == "0.5-2.0s"
    assert duration_buckets(0.3, edges=(0.5, 2.0)) == "<0.5s"
    assert duration_buckets(3.0, edges=(0.5, 2.0)) == ">=2.0s"
