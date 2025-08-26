"""Confusion network construction tests."""

from __future__ import annotations

import pytest

from hypofuse.confusion import (
    Arc,
    build_confusion_network,
    confusion_from_json,
    confusion_to_json,
    minimal_cut_one_best,
)
from hypofuse.multi_align import progressive_align


def test_pivot_is_most_common_token() -> None:
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    net = build_confusion_network(grid)
    assert net.slots[0].pivot == "the"
    assert net.slots[1].pivot == "cat"


def test_arcs_sum_to_one() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    for slot in net.slots:
        assert slot.is_consistent(), slot


def test_one_best_picks_pivot() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("b", "c")])
    net = build_confusion_network(grid)
    assert net.one_best() == tuple(s.pivot for s in net.slots)
    assert minimal_cut_one_best(net) == net.one_best()


def test_gap_column_is_dropped_from_arc_set() -> None:
    grid = progressive_align([("a", "b"), ("a",)])
    net = build_confusion_network(grid)
    # Gaps are excluded from the arc set; non-gap token becomes pivot.
    for slot in net.slots:
        assert all(a.token != "*" for a in slot.arcs)


def test_completely_empty_column_becomes_gap_slot() -> None:
    from hypofuse.multi_align import TokenGrid

    grid = TokenGrid(
        columns=(("*",),),
        hypotheses=(("a",), ("b",)),
        back_pointers=(((0, "DEL"),),),
        pair_alignments=(),
    )
    net = build_confusion_network(grid)
    assert net.slots[0].arcs == (Arc("*", 1.0),)


def test_arc_posterior_normalization_uses_weights() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid, weights=[[10.0, 0.1], [0.1, 10.0]])
    # First slot has 10.1 to "a" vs 0.1 to "a" -> posterior ~ 0.99
    pivot_slot = net.slots[0]
    pivot_arc = next(a for a in pivot_slot.arcs if a.token == "a")
    assert pivot_arc.posterior == pytest.approx(1.0, abs=1e-2)


def test_json_round_trip() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    payload = confusion_to_json(net)
    parsed = confusion_from_json(payload)
    assert parsed.to_dict() == net.to_dict()
    assert confusion_to_json(parsed) == payload
