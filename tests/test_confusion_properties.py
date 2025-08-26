"""Property invariants over seeded random confusion networks."""

from __future__ import annotations

import random

from hypofuse.confusion import build_confusion_network
from hypofuse.multi_align import GAP, progressive_align

_VOCAB = ["alpha", "beta", "gamma", "delta", "zeta"]


def _random_hyps(rng: random.Random, n: int) -> list[tuple[str, ...]]:
    hyps: list[tuple[str, ...]] = []
    for _ in range(n):
        length = rng.randint(1, 5)
        hyps.append(tuple(rng.choice(_VOCAB) for _ in range(length)))
    return hyps


def test_slot_count_le_grid_width() -> None:
    rng = random.Random(42)
    for _ in range(200):
        hyps = _random_hyps(rng, rng.randint(2, 5))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        assert len(net.slots) <= grid.width


def test_no_invented_tokens() -> None:
    """Every non-null token in the grid appears in at least one arc."""
    rng = random.Random(43)
    for _ in range(200):
        hyps = _random_hyps(rng, rng.randint(2, 5))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        grid_tokens: set[str] = set()
        for col in grid.columns:
            for tok in col:
                if tok != GAP:
                    grid_tokens.add(str(tok))
        arc_tokens: set[str] = set()
        for slot in net.slots:
            for arc in slot.arcs:
                if str(arc.token) != GAP:
                    arc_tokens.add(str(arc.token))
        assert grid_tokens.issubset(arc_tokens)


def test_posteriors_in_unit_interval() -> None:
    rng = random.Random(44)
    for _ in range(200):
        hyps = _random_hyps(rng, rng.randint(2, 5))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        for slot in net.slots:
            for arc in slot.arcs:
                assert 0.0 <= arc.posterior <= 1.0


def test_one_best_length_equals_slot_count() -> None:
    rng = random.Random(45)
    for _ in range(200):
        hyps = _random_hyps(rng, rng.randint(2, 5))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        assert len(net.one_best()) == len(net.slots)


def test_validate_passes_for_every_built_network() -> None:
    rng = random.Random(46)
    for _ in range(200):
        hyps = _random_hyps(rng, rng.randint(2, 5))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        net.validate()
