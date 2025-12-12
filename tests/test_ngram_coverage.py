"""Tests for NgramLM.coverage() and VocabCoverage."""

from __future__ import annotations

import pytest

from hypofuse.ngram import NgramLM


def test_coverage_known_fixture() -> None:
    """Exact coverage arithmetic on a known vocabulary.

    LM trained on [["a", "b"]] has vocab = {<unk>, <s>, </s>, a, b}.
    Tokens ["a", "b", "c", "d"]:
        total = 4
        in-vocab: a, b (2 tokens)
        oov: c, d (2 tokens)
        oov_rate = 2/4 = 0.5
    """
    lm = NgramLM.train([["a", "b"]], order=2)
    cov = lm.coverage(["a", "b", "c", "d"])
    assert cov.total == 4
    assert cov.oov == 2
    assert cov.oov_rate == pytest.approx(0.5)


def test_coverage_all_in_vocab() -> None:
    """All tokens in vocab gives oov_rate = 0."""
    lm = NgramLM.train([["a", "b"]], order=2)
    cov = lm.coverage(["a", "b"])
    assert cov.oov == 0
    assert cov.oov_rate == pytest.approx(0.0)


def test_coverage_empty_tokens() -> None:
    """Empty token list gives total=0, oov=0, oov_rate=0."""
    lm = NgramLM.train([["a"]], order=2)
    cov = lm.coverage([])
    assert cov.total == 0
    assert cov.oov == 0
    assert cov.oov_rate == pytest.approx(0.0)
