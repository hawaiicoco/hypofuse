"""Tests for insertion_penalty in shallow_fusion_score and rescore_nbest."""

from __future__ import annotations

from hypofuse.ngram import NgramLM
from hypofuse.rescore import (
    ScoredHypothesis,
    rescore_nbest,
    shallow_fusion_score,
)


def test_penalty_zero_leaves_order_unchanged() -> None:
    """insertion_penalty=0.0 must produce the same ranking as the default."""
    lm = NgramLM.train([["a", "b"]], order=2)
    nbest = [
        ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-1.0),
        ScoredHypothesis.from_tokens(["a"], acoustic_log10=-2.0),
    ]
    ranked_default = rescore_nbest(nbest, lm, lm_weight=0.5)
    ranked_zero = rescore_nbest(nbest, lm, lm_weight=0.5, insertion_penalty=0.0)
    assert [h.text for h in ranked_default] == [h.text for h in ranked_zero]


def test_penalty_flips_ranking_between_short_and_long() -> None:
    """A large insertion_penalty should favour the shorter hypothesis.

    Long hypothesis "the cat sat" has 3 tokens and better acoustic (-1.0).
    Short hypothesis "the" has 1 token and worse acoustic (-5.0).
    Without penalty the long hypothesis ranks first (acoustic dominates at w=0).
    With penalty=5.0 at lm_weight=0.5:
        long score  = -1.0 + 0.5*lm_long - 5.0*3 = -1.0 + 0.5*lm_long - 15.0
        short score = -5.0 + 0.5*lm_short - 5.0*1 = -5.0 + 0.5*lm_short - 5.0
    The penalty gap (10.0) overwhelms the acoustic gap (4.0), flipping the order.
    """
    lm = NgramLM.train([["the", "cat", "sat", "on", "the", "mat"]], order=2)
    short = ScoredHypothesis.from_tokens(["the"], acoustic_log10=-5.0)
    long = ScoredHypothesis.from_tokens(["the", "cat", "sat"], acoustic_log10=-1.0)
    nbest = [long, short]

    ranked_no_penalty = rescore_nbest(nbest, lm, lm_weight=0.5, insertion_penalty=0.0)
    assert ranked_no_penalty[0].text == "the cat sat"

    ranked_with_penalty = rescore_nbest(nbest, lm, lm_weight=0.5, insertion_penalty=5.0)
    assert ranked_with_penalty[0].text == "the"


def test_shallow_fusion_penalty_subtracts_per_token() -> None:
    """shallow_fusion_score subtracts penalty * len(tokens)."""
    lm = NgramLM.train([["a", "b"]], order=2)
    hyp = ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-1.0)
    score_no_pen = shallow_fusion_score(hyp, lm, lm_weight=1.0, insertion_penalty=0.0)
    score_pen = shallow_fusion_score(hyp, lm, lm_weight=1.0, insertion_penalty=0.5)
    # 2 tokens, penalty 0.5 each -> subtract 1.0
    assert score_no_pen - score_pen == 1.0
