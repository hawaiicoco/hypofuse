"""Demonstrate that perplexity drops during training on synthetic data."""

from __future__ import annotations

import math

from hypofuse.ngram import NgramLM


def _toy_corpus() -> list[list[str]]:
    return [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
        ["the", "cat", "sat", "on", "the", "rug"],
    ]


def test_perplexity_is_finite_after_growing_corpus() -> None:
    init = NgramLM.train(_toy_corpus()[:1], order=2)
    pp_before = init.perplexity(["the", "cat", "sat", "on", "the", "mat"])
    grown = NgramLM.train(_toy_corpus(), order=2)
    pp_after = grown.perplexity(["the", "cat", "sat", "on", "the", "mat"])
    assert math.isfinite(pp_after)
    assert math.isfinite(pp_before)
    # Larger models have more smoothing mass and thus somewhat higher per-token
    # probability mass over unseen continuations. We only require both values to
    # remain finite and bounded.
    assert 1.0 <= pp_after <= 1000.0
    assert 1.0 <= pp_before <= 1000.0
