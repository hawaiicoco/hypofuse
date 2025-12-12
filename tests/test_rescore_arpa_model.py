"""Rescoring with a model loaded from ARPA.

A loaded model carries log-probability tables instead of counts, so scoring
goes through ``NgramLM.prob`` -- which selects the tables when
``source == "arpa"`` -- rather than ``prob_katz``, which needs counts.
"""

from __future__ import annotations

import math

import pytest

from hypofuse.ngram import UNK, NgramLM
from hypofuse.rescore import ScoredHypothesis, rescore_lm, rescore_nbest

CORPUS = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat", "ran"]]


def _trained() -> NgramLM:
    return NgramLM.train(CORPUS, order=2)


def _loaded() -> NgramLM:
    return NgramLM.from_arpa(_trained().to_arpa())


def test_prob_auto_dispatch_per_source() -> None:
    trained, loaded = _trained(), _loaded()
    ngram = ("the", "cat")
    assert trained.prob(ngram) == pytest.approx(trained.prob(ngram, method="katz"))
    assert loaded.prob(ngram) == pytest.approx(loaded.prob(ngram, method="arpa"))


def test_arpa_tables_survive_the_roundtrip() -> None:
    """Export then import must not change any probability the tables define."""
    trained, loaded = _trained(), _loaded()
    ngrams = (
        ("the",),
        ("cat",),
        ("</s>",),
        ("the", "cat"),
        ("cat", "sat"),
        ("dog", "ran"),
        ("<s>", "the"),
    )
    for ngram in ngrams:
        # The ARPA text writes log10 probabilities with four decimals, so the
        # roundtrip is lossless up to that documented precision.
        assert math.log10(loaded.prob(ngram, method="arpa")) == pytest.approx(
            math.log10(trained.prob(ngram, method="arpa")), abs=1e-4
        ), ngram


def test_unseen_bigram_backs_off_to_its_unigram() -> None:
    loaded = _loaded()
    # ("sat", "the") was never seen. The exported backoff weight for "sat" is
    # one, so the estimate is exactly the unigram probability of "the".
    assert loaded.prob(("sat", "the"), method="arpa") == pytest.approx(
        loaded.prob(("the",), method="arpa"), rel=1e-9
    )


def test_unlisted_unigram_is_not_certain() -> None:
    loaded = _loaded()
    p_unk = loaded.prob((UNK,), method="arpa")
    assert 0.0 < p_unk <= 1.0 / len(loaded.vocab) + 1e-12
    assert p_unk < 0.5


def test_rescore_nbest_prefers_in_corpus_hypothesis() -> None:
    loaded = _loaded()
    nbest = [
        ScoredHypothesis.from_tokens(["zebra", "zebra"]),
        ScoredHypothesis.from_tokens(["the", "cat"]),
    ]
    ranked = rescore_nbest(nbest, loaded, lm_weight=1.0)
    assert ranked[0].tokens == ("the", "cat")


def test_rescore_lm_is_finite_for_both_sources() -> None:
    hyp = ScoredHypothesis.from_tokens(["the", "cat"])
    assert math.isfinite(rescore_lm(hyp, _trained()))
    assert math.isfinite(rescore_lm(hyp, _loaded()))


def test_empty_hypothesis_scores_zero_for_both_sources() -> None:
    empty = ScoredHypothesis.from_tokens([])
    assert rescore_lm(empty, _trained()) == 0.0
    assert rescore_lm(empty, _loaded()) == 0.0
