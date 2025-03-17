"""Smoothing probability behaviour."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import LanguageModelError
from hypofuse.ngram import NgramLM


def test_katz_known_bigram_has_nonzero_prob() -> None:
    sents = [["the", "cat"], ["the", "dog"]]
    lm = NgramLM.train(sents, order=2)
    p = lm.prob_katz(("the", "cat"))
    assert p > 0
    assert p < 1


def test_katz_unknown_falls_back_to_unigram() -> None:
    sents = [["the", "cat"]]
    lm = NgramLM.train(sents, order=3)
    p = lm.prob_katz(("the", "cat", "sat"))
    assert p > 0


def test_jelinek_lambda_sum_must_equal_one() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    with pytest.raises(LanguageModelError):
        lm.prob_jelinek(("a", "b"), lambdas=(0.7, 0.7))


def test_jelinek_default_lambdas_sum_to_one() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=3)
    p = lm.prob_jelinek(("a", "b"))
    assert p > 0


def test_probability_zero_for_unseen_unigram_without_backoff() -> None:
    sents = [["a"]]
    lm = NgramLM.train(sents, order=1)
    p = lm.prob_katz(("zzz",))
    assert 0 < p <= 1
