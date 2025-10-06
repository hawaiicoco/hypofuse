"""Property invariants for seeded random alignment sequences."""

from __future__ import annotations

import random

import pytest

from hypofuse.alignment import (
    DEL,
    INS,
    MATCH,
    SUB,
    character_error_rate,
    edit_alignment,
    word_error_rate,
)


def _random_tokens(rng: random.Random, length: int, alphabet: list[str]) -> list[str]:
    """Synthetic: random token list from *alphabet*."""
    return [rng.choice(alphabet) for _ in range(length)]


def test_errors_equal_error_ops_count() -> None:
    rng = random.Random(7)
    alphabet = ["a", "b", "c", "d"]
    for _ in range(100):
        n = rng.randint(0, 15)
        m = rng.randint(0, 15)
        ref = _random_tokens(rng, n, alphabet)
        hyp = _random_tokens(rng, m, alphabet)
        align = edit_alignment(ref, hyp)
        error_ops = sum(1 for op in align.ops if op.op != MATCH)
        assert align.errors == error_ops


def test_path_monotone_and_complete() -> None:
    rng = random.Random(11)
    alphabet = ["a", "b", "c"]
    for _ in range(100):
        n = rng.randint(0, 15)
        m = rng.randint(0, 15)
        ref = _random_tokens(rng, n, alphabet)
        hyp = _random_tokens(rng, m, alphabet)
        align = edit_alignment(ref, hyp)
        ref_pos = hyp_pos = 0
        for op in align.ops:
            if op.op in (MATCH, SUB):
                ref_pos += 1
                hyp_pos += 1
            elif op.op == INS:
                hyp_pos += 1
            elif op.op == DEL:
                ref_pos += 1
        assert ref_pos == n
        assert hyp_pos == m


def test_cer_equals_wer_on_single_char_tokens() -> None:
    """When each token is a single character, CER and WER must agree."""
    rng = random.Random(23)
    alphabet = ["a", "b", "c"]
    for _ in range(50):
        n = rng.randint(1, 10)
        m = rng.randint(1, 10)
        ref = _random_tokens(rng, n, alphabet)
        hyp = _random_tokens(rng, m, alphabet)
        wer = word_error_rate(ref, hyp)
        cer = character_error_rate("".join(ref), "".join(hyp))
        assert wer == pytest.approx(cer)


def test_cer_identical_strings_is_zero() -> None:
    assert character_error_rate("abcde", "abcde") == 0.0


def test_cer_empty_hypothesis_is_one() -> None:
    # ref="abc", hyp="": 3 deletions, ref_length=3, CER = 3/3 = 1.0
    # Empty-reference policy (alignment.py): when ref_length == 0,
    # return 0.0 if hyp is also empty, else 1.0.
    # For non-empty ref with empty hyp, all tokens are deleted,
    # giving errors/ref_length = 1.0.
    assert character_error_rate("abc", "") == 1.0


def test_cer_both_empty_is_zero() -> None:
    assert character_error_rate("", "") == 0.0


def test_cer_empty_reference_nonempty_hyp_is_one() -> None:
    # Empty-reference policy: ref_length == 0 and hyp non-empty -> 1.0
    assert character_error_rate("", "abc") == 1.0
