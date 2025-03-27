"""LM weight sweep and invariance."""

from __future__ import annotations

from hypofuse.ngram import NgramLM
from hypofuse.rescore import (
    ScoredHypothesis,
    invariance_under_acoustic_change,
    lm_weight_sweep,
    rescore_nbest,
)


def _two_hyp_nbest() -> list[ScoredHypothesis]:
    return [
        ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-1.0),
        ScoredHypothesis.from_tokens(["a", "c"], acoustic_log10=-1.0),
    ]


def test_weight_sweep_returns_one_row_per_weight() -> None:
    sents = [["a", "b"], ["a", "c"]]
    lm = NgramLM.train(sents, order=2)
    rows = lm_weight_sweep(_two_hyp_nbest(), ["a", "b"], lm, weights=[0.0, 0.5, 1.0])
    assert len(rows) == 3


def test_weight_sweep_zero_weight_picks_acoustic_top() -> None:
    nbest = [
        ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-0.1),
        ScoredHypothesis.from_tokens(["a", "c"], acoustic_log10=-5.0),
    ]
    sents = [["a", "b"], ["a", "c"]]
    lm = NgramLM.train(sents, order=2)
    rows = lm_weight_sweep(nbest, ["a", "b"], lm, weights=[0.0])
    assert rows[0][2] == "a b"


def test_invariance_returns_true() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    assert invariance_under_acoustic_change(_two_hyp_nbest(), lm) is True


def test_rescore_returns_top_hypothesis() -> None:
    sents = [["the", "cat"], ["the", "dog"]]
    lm = NgramLM.train(sents, order=2)
    nbest = [
        ScoredHypothesis.from_tokens(["the", "cat"], acoustic_log10=-1.0),
        ScoredHypothesis.from_tokens(["the", "dog"], acoustic_log10=-1.0),
    ]
    ranked = rescore_nbest(nbest, lm, lm_weight=0.5)
    assert len(ranked) == 2
    # The chosen text must come from the n-best list.
    texts = {hyp.text for hyp in nbest}
    assert ranked[0].text in texts
