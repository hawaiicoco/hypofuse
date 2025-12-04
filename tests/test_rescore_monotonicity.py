"""Tests for lm_weight_monotonicity: satisfy and violate."""

from __future__ import annotations

from hypofuse.ngram import NgramLM
from hypofuse.rescore import (
    ScoredHypothesis,
    lm_weight_monotonicity,
    rescore_lm,
)


def test_monotonicity_satisfied_with_correct_lm_log10() -> None:
    """When lm_log10 matches the actual LM, monotonicity always holds."""
    lm = NgramLM.train([["a", "b"], ["a", "c"]], order=2)
    hyps = [
        ScoredHypothesis.from_tokens(["a", "b"], acoustic_log10=-1.0),
        ScoredHypothesis.from_tokens(["a", "c"], acoustic_log10=-1.0),
    ]
    scored = [
        ScoredHypothesis(
            text=h.text,
            tokens=h.tokens,
            acoustic_log10=h.acoustic_log10,
            lm_log10=rescore_lm(h, lm),
        )
        for h in hyps
    ]
    assert lm_weight_monotonicity(scored, lm) is True


def test_monotonicity_violated_with_wrong_lm_log10() -> None:
    """If caller-provided lm_log10 disagrees with the LM, violation is detected.

    Hypothesis a claims lm_log10=0.0 (good) but the LM gives it a poor score.
    Hypothesis b claims lm_log10=-1.0 (worse) but the LM gives it a better score.
    Condition: a.lm_log10 (0.0) >= b.lm_log10 (-1.0), same acoustic.
    At w > 0, b ranks above a because b has better actual LM score -> violation.
    """
    lm = NgramLM.train([["the", "cat", "sat"]], order=2)
    a = ScoredHypothesis(
        text="never seen",
        tokens=("never", "seen"),
        acoustic_log10=-1.0,
        lm_log10=0.0,
    )
    b = ScoredHypothesis(
        text="the cat",
        tokens=("the", "cat"),
        acoustic_log10=-1.0,
        lm_log10=-1.0,
    )
    assert lm_weight_monotonicity([a, b], lm) is False


def test_monotonicity_empty_nbest() -> None:
    """Empty n-best list trivially satisfies monotonicity."""
    lm = NgramLM.train([["a"]], order=2)
    assert lm_weight_monotonicity([], lm) is True
