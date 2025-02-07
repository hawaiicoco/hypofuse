"""CER and WER definitions on golden cases."""

from __future__ import annotations

import pytest

from hypofuse.alignment import character_error_rate, word_error_rate


def test_wer_perfect() -> None:
    assert word_error_rate(["hello", "world"], ["hello", "world"]) == 0.0


def test_wer_one_substitution() -> None:
    assert word_error_rate(["hello", "world"], ["hello", "there"]) == pytest.approx(1 / 2)


def test_wer_one_insertion() -> None:
    assert word_error_rate(["a", "b"], ["a", "x", "b"]) == pytest.approx(1 / 2)


def test_wer_one_deletion() -> None:
    assert word_error_rate(["a", "b", "c"], ["a", "c"]) == pytest.approx(1 / 3)


def test_wer_empty_reference_returns_one_when_hyp_nonempty() -> None:
    assert word_error_rate([], ["x"]) == 1.0


def test_wer_both_empty_returns_zero() -> None:
    assert word_error_rate([], []) == 0.0


def test_cer_perfect() -> None:
    assert character_error_rate("hello", "hello") == 0.0


def test_cer_one_substitution() -> None:
    assert character_error_rate("cat", "car") == pytest.approx(1 / 3)


def test_cer_chinese() -> None:
    # Two errors in three characters.
    assert character_error_rate("你好世", "你好界") == pytest.approx(1 / 3)


def test_cer_empty_reference() -> None:
    assert character_error_rate("", "abc") == 1.0
