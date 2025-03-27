"""Rescoring basics."""

from __future__ import annotations

import math

from hypofuse.ngram import NgramLM
from hypofuse.rescore import ScoredHypothesis, rescore_lm, shallow_fusion_score


def test_rescore_lm_is_finite() -> None:
    sents = [["the", "cat", "sat"], ["the", "dog", "ran"]]
    lm = NgramLM.train(sents, order=2)
    hyp = ScoredHypothesis.from_tokens(["the", "cat"])
    score = rescore_lm(hyp, lm)
    assert math.isfinite(score)


def test_rescore_lm_higher_for_in_corpus() -> None:
    sents = [["the", "cat", "sat"]]
    lm = NgramLM.train(sents, order=2)
    in_corpus = ScoredHypothesis.from_tokens(["the", "cat"])
    out_of_corpus = ScoredHypothesis.from_tokens(["never", "seen"])
    s_in = rescore_lm(in_corpus, lm)
    s_out = rescore_lm(out_of_corpus, lm)
    # Both are negative log-probs; the in-corpus one is less negative.
    assert s_in > s_out


def test_shallow_fusion_combines_weights() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    hyp = ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-1.0)
    score = shallow_fusion_score(hyp, lm, lm_weight=2.0, acoustic_weight=1.0)
    assert math.isfinite(score)
    assert score == -1.0 + 2.0 * rescore_lm(hyp, lm)
