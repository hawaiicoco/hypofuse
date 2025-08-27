"""Numerical edge cases for confusion network construction."""

from __future__ import annotations

import pytest

from hypofuse.confusion import build_confusion_network
from hypofuse.multi_align import TokenGrid, progressive_align


def test_single_hypothesis_all_mass_on_one_arc() -> None:
    grid = progressive_align([("hello", "world")])
    net = build_confusion_network(grid)
    for slot in net.slots:
        assert len(slot.arcs) == 1
        assert slot.arcs[0].posterior == pytest.approx(1.0)
    net.validate()


def test_all_different_uniform_posteriors() -> None:
    # Every hypothesis contributes a different token to a single column
    grid = progressive_align([("a",), ("b",), ("c",)])
    net = build_confusion_network(grid)
    slot = net.slots[0]
    # Three different tokens, each with posterior 1/3
    assert len(slot.arcs) == 3
    for arc in slot.arcs:
        assert arc.posterior == pytest.approx(1.0 / 3.0)
    net.validate()


def test_empty_grid_returns_empty_network() -> None:
    """An empty grid (zero columns) produces an empty network, not an error."""
    grid = TokenGrid(
        columns=(),
        hypotheses=((),),
        back_pointers=(),
        pair_alignments=(),
    )
    net = build_confusion_network(grid)
    assert net.slots == ()
    assert net.one_best() == ()
    net.validate()


def test_single_token_single_hypothesis() -> None:
    grid = progressive_align([("only",)])
    net = build_confusion_network(grid)
    assert len(net.slots) == 1
    assert net.slots[0].pivot == "only"
    assert net.slots[0].arcs[0].posterior == pytest.approx(1.0)
    net.validate()


def test_identical_hypotheses_give_full_agreement() -> None:
    grid = progressive_align([("x", "y"), ("x", "y"), ("x", "y")])
    net = build_confusion_network(grid)
    for slot in net.slots:
        assert len(slot.arcs) == 1
        assert slot.arcs[0].posterior == pytest.approx(1.0)
    net.validate()
