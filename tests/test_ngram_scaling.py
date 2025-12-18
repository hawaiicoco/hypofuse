"""Slow test: perplexity decreases as training data grows.

This is a synthetic-corpus check with a deterministic seed. No claim about
real-world corpora or benchmark performance is implied.
"""

from __future__ import annotations

import random

import pytest

from hypofuse.ngram import NgramLM


@pytest.mark.slow
def test_perplexity_decreases_with_more_synthetic_data() -> None:
    """More training data from the same distribution lowers perplexity.

    A simple grammar generates sentences of the form
    ``the (cat|dog) (sat|ran) on the (mat|rug)``. Perplexity on a fixed
    held-out set should not increase as training size grows.
    """
    rng = random.Random(42)
    templates = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "on", "the", "rug"],
        ["the", "cat", "ran", "on", "the", "rug"],
        ["the", "dog", "sat", "on", "the", "mat"],
    ]

    def gen() -> list[str]:
        return list(rng.choice(templates))

    full_corpus = [gen() for _ in range(1000)]
    held_out = [gen() for _ in range(50)]
    ho_tokens = [t for s in held_out for t in s]

    pp_prev = float("inf")
    for size in [10, 50, 200, 1000]:
        lm = NgramLM.train(full_corpus[:size], order=2)
        pp = lm.perplexity(ho_tokens)
        assert pp <= pp_prev + 2.0, f"perplexity increased at size={size}: {pp} > {pp_prev}+2"
        pp_prev = pp
