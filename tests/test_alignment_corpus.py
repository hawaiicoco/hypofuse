"""Corpus error rates: micro vs macro on a two-utterance example."""

from __future__ import annotations

import pytest

from hypofuse.alignment import corpus_error_rates


def test_corpus_rates_micro_ne_macro() -> None:
    # Utterance 1: ref=["a","b"], hyp=["a","b"] -> 0 errors, rate=0/2=0.0
    # Utterance 2: ref=["c"], hyp=["x","y"] -> SUB(c,x)+INS(y)=2 errors
    #   rate = 2/1 = 2.0
    # Micro: (0+2) / (2+1) = 2/3
    # Macro: (0.0 + 2.0) / 2 = 1.0
    pairs = [
        (["a", "b"], ["a", "b"]),
        (["c"], ["x", "y"]),
    ]
    rates = corpus_error_rates(pairs, level="word")
    assert rates.n == 2
    assert rates.micro == pytest.approx(2 / 3)
    assert rates.macro == pytest.approx(1.0)


def test_corpus_rates_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        corpus_error_rates([])


def test_corpus_rates_char_level() -> None:
    # "abc" vs "axc": 1 sub in 3 chars -> rate = 1/3
    pairs = [("abc", "axc")]
    rates = corpus_error_rates(pairs, level="char")
    assert rates.micro == pytest.approx(1 / 3)
    assert rates.macro == pytest.approx(1 / 3)


def test_corpus_rates_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        corpus_error_rates([(["a"], ["a"])], level="phone")
