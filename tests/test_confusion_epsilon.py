"""Epsilon/gap arc policy tests for build_confusion_network."""

from __future__ import annotations

import pytest

from hypofuse.confusion import build_confusion_network
from hypofuse.multi_align import GAP, progressive_align


def test_default_excludes_gap_arcs() -> None:
    # hyp 2 is shorter, producing a gap column
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    net = build_confusion_network(grid)
    for slot in net.slots:
        for arc in slot.arcs:
            assert arc.token != GAP or slot.pivot == GAP


def test_keep_epsilon_false_matches_default() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    net_default = build_confusion_network(grid)
    net_false = build_confusion_network(grid, keep_epsilon=False)
    assert net_default.to_dict() == net_false.to_dict()


def test_keep_epsilon_true_includes_gap_arc() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    net = build_confusion_network(grid, keep_epsilon=True)
    # At least one slot should have a gap arc
    has_gap_arc = any(
        any(a.token == GAP for a in slot.arcs) for slot in net.slots if slot.pivot != GAP
    )
    assert has_gap_arc, "expected at least one non-pivot gap arc"


def test_keep_epsilon_true_gap_not_pivot_when_alternatives_exist() -> None:
    grid = progressive_align([("a", "b"), ("a",)])
    net = build_confusion_network(grid, keep_epsilon=True)
    for slot in net.slots:
        non_gap = [a for a in slot.arcs if a.token != GAP]
        if non_gap:
            assert slot.pivot != GAP


def test_keep_epsilon_true_all_gap_column_pivot_is_gap() -> None:
    from hypofuse.multi_align import TokenGrid

    grid = TokenGrid(
        columns=((GAP, GAP),),
        hypotheses=(("a",), ("b",)),
        back_pointers=(((0, "DEL"),),),
        pair_alignments=(),
    )
    net = build_confusion_network(grid, keep_epsilon=True)
    assert net.slots[0].pivot == GAP
    assert net.slots[0].arcs[0].token == GAP


def test_keep_epsilon_true_posteriors_still_normalize() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    net = build_confusion_network(grid, keep_epsilon=True)
    for slot in net.slots:
        total = sum(a.posterior for a in slot.arcs)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_one_best_excludes_gap_when_possible() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    net = build_confusion_network(grid, keep_epsilon=True)
    best = net.one_best()
    # No column of this grid is entirely gap, so one_best must avoid GAP.
    assert all(tok != GAP for tok in best)
