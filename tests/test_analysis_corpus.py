"""Corpus-level error rate aggregation."""

from __future__ import annotations

import pytest

from hypofuse.analysis import UtteranceScore, corpus_error_rate


def test_micro_error_rate() -> None:
    # utt1: ref=("a",) hyp=("x",) -> 1 error, 1 ref token
    # utt2: ref=("a","b","c","d") hyp=("a","b","c","d") -> 0 errors, 4 ref tokens
    # micro = total_errors / total_ref = 1 / 5 = 0.2
    scores = [
        UtteranceScore("u1", ("a",), ("x",)),
        UtteranceScore("u2", ("a", "b", "c", "d"), ("a", "b", "c", "d")),
    ]
    assert corpus_error_rate(scores, average="micro") == pytest.approx(0.2)


def test_macro_error_rate() -> None:
    # utt1: WER = 1/1 = 1.0
    # utt2: WER = 0/4 = 0.0
    # macro = mean(1.0, 0.0) = 0.5
    scores = [
        UtteranceScore("u1", ("a",), ("x",)),
        UtteranceScore("u2", ("a", "b", "c", "d"), ("a", "b", "c", "d")),
    ]
    assert corpus_error_rate(scores, average="macro") == pytest.approx(0.5)


def test_micro_and_macro_differ() -> None:
    scores = [
        UtteranceScore("u1", ("a",), ("x",)),
        UtteranceScore("u2", ("a", "b", "c", "d"), ("a", "b", "c", "d")),
    ]
    micro = corpus_error_rate(scores, average="micro")
    macro = corpus_error_rate(scores, average="macro")
    assert micro != macro


def test_empty_corpus() -> None:
    assert corpus_error_rate([], average="micro") == 0.0
    assert corpus_error_rate([], average="macro") == 0.0


def test_unknown_average_raises() -> None:
    import pytest as pt

    with pt.raises(ValueError, match="unknown average"):
        corpus_error_rate([], average="weighted")
