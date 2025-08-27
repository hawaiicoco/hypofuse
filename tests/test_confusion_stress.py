"""Slow stress test: 500 eight-hypothesis grids with invariants."""

from __future__ import annotations

import random
import time

import pytest

from hypofuse.confusion import build_confusion_network, minimal_cut_one_best
from hypofuse.multi_align import GAP, progressive_align

_VOCAB = ["alpha", "beta", "gamma", "delta", "zeta", "eta", "theta"]


@pytest.mark.slow
def test_stress_500_eight_hyp_grids() -> None:
    rng = random.Random(2024)
    start = time.monotonic()
    for _ in range(500):
        hyps: list[tuple[str, ...]] = []
        for _ in range(8):
            length = rng.randint(2, 8)
            hyps.append(tuple(rng.choice(_VOCAB) for _ in range(length)))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        # Invariant: slot count <= grid width
        assert len(net.slots) <= grid.width
        # Invariant: posteriors in [0, 1]
        for slot in net.slots:
            for arc in slot.arcs:
                assert 0.0 <= arc.posterior <= 1.0
        # Invariant: validate passes
        net.validate()
        # Invariant: minimal_cut == one_best
        assert minimal_cut_one_best(net) == net.one_best()
        # Invariant: one_best length == slot count
        assert len(net.one_best()) == len(net.slots)
        # Invariant: no invented tokens
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
    elapsed = time.monotonic() - start
    # Bounded runtime: 500 grids should complete well under 60 seconds
    assert elapsed < 60.0, f"stress test took {elapsed:.1f}s"
