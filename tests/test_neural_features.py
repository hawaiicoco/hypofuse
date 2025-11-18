"""Hand-computed confidence feature values."""

from __future__ import annotations

import math

import pytest

from hypofuse.neural import confidence_features


def test_uniform_posteriors_entropy_and_margin() -> None:
    # H = -4 * 0.25 * ln(0.25) = ln(4)
    feat = confidence_features([0.25, 0.25, 0.25, 0.25])
    assert feat.entropy == pytest.approx(math.log(4))
    assert feat.posterior_margin == pytest.approx(0.0)
    assert feat.max_posterior == pytest.approx(0.25)
    assert feat.mean_posterior == pytest.approx(0.25)


def test_peaked_posterior() -> None:
    # H = -(0.9 * ln(0.9) + 0.1 * ln(0.1))
    expected_entropy = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    feat = confidence_features([0.9, 0.1])
    assert feat.entropy == pytest.approx(expected_entropy)
    assert feat.posterior_margin == pytest.approx(0.8)
    assert feat.max_posterior == pytest.approx(0.9)
    assert feat.mean_posterior == pytest.approx(0.5)


def test_single_element_posterior() -> None:
    # H = -(1.0 * ln(1.0)) = 0.0; margin = 1.0 - 0.0 = 1.0
    feat = confidence_features([1.0])
    assert feat.entropy == pytest.approx(0.0)
    assert feat.posterior_margin == pytest.approx(1.0)
    assert feat.n_tokens == 1


def test_empty_posteriors_raises() -> None:
    with pytest.raises(ValueError, match="posteriors must not be empty"):
        confidence_features([])


def test_mismatched_vote_counts_raises() -> None:
    with pytest.raises(ValueError, match="vote_counts"):
        confidence_features([0.5, 0.5], vote_counts=[1])


def test_vote_counts_agreement() -> None:
    # 3 systems vote; winner gets 2 votes out of 3 total
    feat = confidence_features([0.6, 0.4], vote_counts=[2, 1])
    assert feat.agreement_ratio == pytest.approx(2 / 3)


def test_default_agreement_uses_max_posterior() -> None:
    feat = confidence_features([0.7, 0.3])
    assert feat.agreement_ratio == pytest.approx(0.7)


def test_acoustic_and_lm_default_to_zero() -> None:
    feat = confidence_features([0.5, 0.5])
    assert feat.acoustic_log10 == pytest.approx(0.0)
    assert feat.lm_log10 == pytest.approx(0.0)


def test_acoustic_and_lm_supplied() -> None:
    feat = confidence_features([0.5, 0.5], acoustic_log10=-20.0, lm_log10=-5.0)
    assert feat.acoustic_log10 == pytest.approx(-20.0)
    assert feat.lm_log10 == pytest.approx(-5.0)
