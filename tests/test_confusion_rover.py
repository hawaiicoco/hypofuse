"""Confusion network consistency with ROVER-style fusion."""

from __future__ import annotations

from hypofuse.confusion import build_confusion_network, consistent_with_rover
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_confusion_matches_majority_rover() -> None:
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    net = build_confusion_network(grid)
    rover = fuse(grid, config=FusionConfig(policy="majority"))
    assert consistent_with_rover(net, rover.tokens)


def test_confusion_matches_score_weighted_rover() -> None:
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    net = build_confusion_network(grid)
    rover = fuse(grid, config=FusionConfig(policy="score_weighted"))
    # Networks and ROVER with identical scoring should agree on 1-best.
    assert net.one_best() == rover.tokens
