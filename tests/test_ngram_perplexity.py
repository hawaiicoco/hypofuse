"""Perplexity and OOV behaviour."""

from __future__ import annotations

import math

from hypofuse.ngram import NgramLM


def test_perplexity_at_least_one() -> None:
    sents = [["a", "b", "c"]]
    lm = NgramLM.train(sents, order=2)
    pp = lm.perplexity(["a", "b", "c"])
    assert pp >= 1.0


def test_perplexity_lower_on_training_than_unseen() -> None:
    sents = [["a", "b", "a", "b"], ["a", "b", "a"]]
    lm = NgramLM.train(sents, order=2)
    pp_train = lm.perplexity(["a", "b", "a", "b"])
    pp_unseen = lm.perplexity(["a", "c", "e", "z", "q", "x"])
    assert pp_train <= pp_unseen


def test_perplexity_handles_empty() -> None:
    sents = [["a"]]
    lm = NgramLM.train(sents, order=2)
    assert lm.perplexity([]) == 1.0


def test_oov_uses_unk_token() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    pp = lm.perplexity(["never", "seen"])
    assert math.isfinite(pp)
