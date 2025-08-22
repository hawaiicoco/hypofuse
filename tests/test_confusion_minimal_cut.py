"""Minimal-cut one-best equivalence to per-slot argmax."""

from __future__ import annotations

import random

from hypofuse.confusion import (
    build_confusion_network,
    minimal_cut_one_best,
)
from hypofuse.multi_align import progressive_align


def test_minimal_cut_equals_one_best_basic() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("b", "c")])
    net = build_confusion_network(grid)
    assert minimal_cut_one_best(net) == net.one_best()


def test_minimal_cut_equals_one_best_exhaustive() -> None:
    """Per-slot argmax equals minimal-cost path when no cross-slot
    constraints exist (the chain factorizes). 200 seeded cases."""
    rng = random.Random(2024)
    vocab = ["alpha", "beta", "gamma", "delta"]
    for _ in range(200):
        n_hyps = rng.randint(2, 5)
        hyps = []
        for _ in range(n_hyps):
            length = rng.randint(1, 4)
            hyps.append(tuple(rng.choice(vocab) for _ in range(length)))
        grid = progressive_align(hyps)
        net = build_confusion_network(grid)
        assert minimal_cut_one_best(net) == net.one_best()


def test_minimal_cut_returns_tuple_of_strings() -> None:
    grid = progressive_align([("x",), ("y",)])
    net = build_confusion_network(grid)
    result = minimal_cut_one_best(net)
    assert isinstance(result, tuple)
    for tok in result:
        assert isinstance(tok, str)
