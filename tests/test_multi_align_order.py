"""Permutation policy: given vs sorted order."""

from __future__ import annotations

import random

import pytest

from hypofuse.multi_align import progressive_align


def test_given_order_not_permutation_invariant() -> None:
    hyps_a = [("a", "b"), ("a", "b", "c")]
    hyps_b = [("a", "b", "c"), ("a", "b")]
    grid_a = progressive_align(hyps_a, order="given")
    grid_b = progressive_align(hyps_b, order="given")
    assert grid_a.columns != grid_b.columns


def test_sorted_order_permutation_invariant() -> None:
    hyps_a = [("a", "b"), ("a", "b", "c")]
    hyps_b = [("a", "b", "c"), ("a", "b")]
    grid_a = progressive_align(hyps_a, order="sorted")
    grid_b = progressive_align(hyps_b, order="sorted")
    assert grid_a.columns == grid_b.columns


def test_sorted_order_seeded_permutations() -> None:
    """Synthetic seeded permutations: sorted order is invariant."""
    rng = random.Random(99)
    base_hyps = [("a", "b", "c"), ("a", "x", "c"), ("a", "b")]
    reference = progressive_align(base_hyps, order="sorted")
    for _ in range(20):
        perm = list(base_hyps)
        rng.shuffle(perm)
        grid = progressive_align(perm, order="sorted")
        assert grid.columns == reference.columns


def test_unknown_order_raises() -> None:
    with pytest.raises(ValueError, match="unknown order"):
        progressive_align([("a",)], order="random")


def test_default_order_is_given() -> None:
    hyps = [("a", "b"), ("a", "x")]
    default = progressive_align(hyps)
    explicit = progressive_align(hyps, order="given")
    assert default.columns == explicit.columns
