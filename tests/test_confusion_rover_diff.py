"""Tests for rover_diff and consistent_with_rover."""

from __future__ import annotations

from hypofuse.confusion import (
    build_confusion_network,
    consistent_with_rover,
    rover_diff,
)
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_rover_diff_no_mismatch() -> None:
    grid = progressive_align([("a", "b"), ("a", "b")])
    net = build_confusion_network(grid)
    assert rover_diff(net, list(net.one_best())) == []


def test_rover_diff_reports_positions() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "x", "c")])
    net = build_confusion_network(grid)
    best = list(net.one_best())
    # Flip the middle token to create a mismatch
    altered = best[:]
    altered[1] = "FLIP"
    diffs = rover_diff(net, altered)
    assert 1 in diffs


def test_rover_diff_length_mismatch() -> None:
    grid = progressive_align([("a", "b"), ("a", "b")])
    net = build_confusion_network(grid)
    diffs = rover_diff(net, ["a"])
    assert len(diffs) > 0


def test_consistent_with_rover_still_bool() -> None:
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    net = build_confusion_network(grid)
    rover = fuse(grid, config=FusionConfig(policy="majority"))
    assert consistent_with_rover(net, rover.tokens) is True


def test_rover_diff_identical_settings_no_mismatch() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("a", "b")])
    net = build_confusion_network(grid)
    rover = fuse(grid, config=FusionConfig(policy="majority"))
    diffs = rover_diff(net, rover.tokens)
    assert diffs == []


def test_rover_diff_deliberate_tiebreak_diff() -> None:
    """Force a tie-break difference by using first vs lexicographic."""
    grid = progressive_align([("b",), ("a",)])
    net = build_confusion_network(grid)
    # net pivot is "a" (lexicographic tie-break)
    # Simulate a rover that picks "b" (first tie-break)
    diffs = rover_diff(net, ["b"])
    assert 0 in diffs
