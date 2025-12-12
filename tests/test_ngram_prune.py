"""Tests for NgramLM.prune(min_count)."""

from __future__ import annotations

from hypofuse.ngram import NgramLM


def test_prune_removes_singletons_at_min_count_2() -> None:
    """Bigrams with count 1 disappear when min_count=2."""
    lm = NgramLM.train([["a", "b", "c"], ["a", "b", "c"]], order=2)
    pruned = lm.prune(min_count=2)
    for ng, c in pruned.counts[2].items():
        assert c >= 2, f"{ng} has count {c} < 2 after pruning"


def test_prune_preserves_unigrams() -> None:
    """Unigrams are never pruned regardless of min_count."""
    lm = NgramLM.train([["a", "b"]], order=2)
    pruned = lm.prune(min_count=100)
    assert set(pruned.counts[1].keys()) == set(lm.counts[1].keys())


def test_prune_is_idempotent() -> None:
    """Pruning twice with the same threshold gives the same result."""
    lm = NgramLM.train([["a", "b", "c"], ["a", "b"]], order=3)
    p1 = lm.prune(min_count=2)
    p2 = p1.prune(min_count=2)
    for n in range(1, lm.order + 1):
        assert dict(p1.counts[n]) == dict(p2.counts[n])


def test_prune_empty_lm_does_not_raise() -> None:
    """Pruning an LM trained on an empty sentence list does not raise."""
    lm = NgramLM.train([["a"]], order=2)
    pruned = lm.prune(min_count=999)
    assert pruned.order == lm.order
    # All bigrams gone, but unigrams remain
    assert len(pruned.counts[2]) == 0
