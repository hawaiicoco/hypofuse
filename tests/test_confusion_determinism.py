"""Deterministic arc ordering and serialization tests."""

from __future__ import annotations

from hypofuse.confusion import (
    Arc,
    ConfusionSlot,
    build_confusion_network,
    confusion_to_json,
)
from hypofuse.multi_align import progressive_align


def test_arcs_sorted_by_posterior_desc_then_token_lex() -> None:
    # Hand-build a slot with known posteriors
    slot = ConfusionSlot(
        pivot="b",
        arcs=(Arc("b", 0.5), Arc("a", 0.3), Arc("c", 0.2)),
    )
    # Verify ordering: posterior desc, then token lex for ties
    for i in range(len(slot.arcs) - 1):
        a, b = slot.arcs[i], slot.arcs[i + 1]
        assert a.posterior >= b.posterior
        if a.posterior == b.posterior:
            assert str(a.token) <= str(b.token)


def test_build_produces_sorted_arcs() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("b", "c")])
    net = build_confusion_network(grid)
    for slot in net.slots:
        for i in range(len(slot.arcs) - 1):
            a, b = slot.arcs[i], slot.arcs[i + 1]
            assert a.posterior >= b.posterior
            if a.posterior == b.posterior:
                assert str(a.token) <= str(b.token)


def test_same_grid_same_network() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("b", "c")])
    net1 = build_confusion_network(grid)
    net2 = build_confusion_network(grid)
    assert net1.to_dict() == net2.to_dict()


def test_serialization_deterministic() -> None:
    grid = progressive_align([("x", "y"), ("x", "z"), ("w", "y")])
    net = build_confusion_network(grid)
    s1 = confusion_to_json(net)
    s2 = confusion_to_json(net)
    assert s1 == s2


def test_tie_broken_lexicographically() -> None:
    # Two hypotheses with equal weight, different tokens
    grid = progressive_align([("a",), ("b",)])
    net = build_confusion_network(grid)
    slot = net.slots[0]
    # Both have posterior 0.5; "a" < "b" so "a" is pivot
    assert slot.pivot == "a"
    assert slot.arcs[0].token == "a"
    assert slot.arcs[1].token == "b"
