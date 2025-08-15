"""ConfusionNetwork.validate checks posterior invariants."""

from __future__ import annotations

import pytest

from hypofuse.confusion import (
    Arc,
    ConfusionNetwork,
    ConfusionSlot,
    build_confusion_network,
)
from hypofuse.multi_align import progressive_align


def _net(slots: list[ConfusionSlot]) -> ConfusionNetwork:
    return ConfusionNetwork(slots=tuple(slots))


def test_valid_network_passes() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("a", "b")])
    net = build_confusion_network(grid)
    net.validate()  # must not raise


def test_valid_hand_built_network() -> None:
    net = _net(
        [
            ConfusionSlot(pivot="x", arcs=(Arc("x", 0.6), Arc("y", 0.4))),
            ConfusionSlot(pivot="a", arcs=(Arc("a", 1.0),)),
        ]
    )
    net.validate()


def test_empty_network_passes() -> None:
    net = ConfusionNetwork(slots=())
    net.validate()


def test_empty_slot_rejected() -> None:
    net = _net([ConfusionSlot(pivot="x", arcs=())])
    with pytest.raises(ValueError, match="slot 0"):
        net.validate()


def test_negative_posterior_rejected() -> None:
    net = _net(
        [
            ConfusionSlot(pivot="x", arcs=(Arc("x", -0.1), Arc("y", 1.1))),
        ]
    )
    with pytest.raises(ValueError, match=r"slot 0.*negative"):
        net.validate()


def test_unnormalized_slot_rejected() -> None:
    net = _net(
        [
            ConfusionSlot(pivot="x", arcs=(Arc("x", 0.5), Arc("y", 0.3))),
        ]
    )
    with pytest.raises(ValueError, match=r"slot 0.*posteriors sum"):
        net.validate()


def test_error_names_slot_index() -> None:
    net = _net(
        [
            ConfusionSlot(pivot="a", arcs=(Arc("a", 1.0),)),
            ConfusionSlot(pivot="b", arcs=(Arc("b", 0.5), Arc("c", 0.3))),
        ]
    )
    with pytest.raises(ValueError, match="slot 1"):
        net.validate()


def test_tolerance_boundary() -> None:
    # sum = 1.0 + 1e-7 which is within default tol=1e-6
    net = _net(
        [
            ConfusionSlot(pivot="a", arcs=(Arc("a", 1.0 + 1e-7),)),
        ]
    )
    net.validate()
    # sum = 1.0 + 1e-5 which exceeds default tol
    net2 = _net(
        [
            ConfusionSlot(pivot="a", arcs=(Arc("a", 1.0 + 1e-5),)),
        ]
    )
    with pytest.raises(ValueError, match="slot 0"):
        net2.validate()
