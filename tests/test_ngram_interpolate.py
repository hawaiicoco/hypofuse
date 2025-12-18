"""Tests for NgramLM.interpolate and MixtureLM."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import LanguageModelError
from hypofuse.ngram import NgramLM


def _two_models() -> tuple[NgramLM, NgramLM]:
    lm_a = NgramLM.train([["a", "b", "c"]], order=2)
    lm_b = NgramLM.train([["x", "y", "z"]], order=2)
    return lm_a, lm_b


def test_weight_one_reduces_to_self() -> None:
    """weight=1.0 gives probabilities from model_a only."""
    lm_a, lm_b = _two_models()
    mix = lm_a.interpolate(lm_b, weight=1.0)
    ng = ("a", "b")
    p_mix = 10 ** mix.log_prob(ng)
    p_a = lm_a.prob_jelinek(ng)
    assert p_mix == pytest.approx(p_a, rel=1e-6)


def test_weight_zero_reduces_to_other() -> None:
    """weight=0.0 gives probabilities from model_b only."""
    lm_a, lm_b = _two_models()
    mix = lm_a.interpolate(lm_b, weight=0.0)
    ng = ("x", "y")
    p_mix = 10 ** mix.log_prob(ng)
    p_b = lm_b.prob_jelinek(ng)
    assert p_mix == pytest.approx(p_b, rel=1e-6)


def test_weight_half_is_arithmetic_mean() -> None:
    """weight=0.5 gives the arithmetic mean of the two probabilities."""
    lm_a, lm_b = _two_models()
    mix = lm_a.interpolate(lm_b, weight=0.5)
    ng = ("a", "b")
    p_a = lm_a.prob_jelinek(ng)
    p_b = lm_b.prob_jelinek(ng)
    expected = 0.5 * p_a + 0.5 * p_b
    p_mix = 10 ** mix.log_prob(ng)
    assert p_mix == pytest.approx(expected, rel=1e-6)


def test_invalid_weight_raises() -> None:
    """Weights outside [0, 1] raise LanguageModelError."""
    lm_a, lm_b = _two_models()
    with pytest.raises(LanguageModelError):
        lm_a.interpolate(lm_b, weight=1.5)
    with pytest.raises(LanguageModelError):
        lm_a.interpolate(lm_b, weight=-0.1)
