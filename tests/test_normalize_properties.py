"""Determinism and idempotence property tests."""

from __future__ import annotations

import random

from hypofuse.normalize import NormalizationConfig, normalize, tokenize

_NASTY_ALPHABET = (
    "abcABC012\u4e00\u4e01\u4e02\uff10\uff11\uff12\u0301\u0300\u200b\u200d\ufeff\x00\x01 ,.!?\t\n "
)


def _generate_nasty(rng: random.Random, max_len: int = 50) -> str:
    length = rng.randint(0, max_len)
    return "".join(rng.choice(_NASTY_ALPHABET) for _ in range(length))


def test_normalize_never_raises_on_seeded_inputs() -> None:
    rng = random.Random(42)
    for _ in range(200):
        text = _generate_nasty(rng)
        result = normalize(text)
        assert isinstance(result, str)


def test_idempotence_on_seeded_inputs() -> None:
    rng = random.Random(42)
    for _ in range(200):
        text = _generate_nasty(rng)
        once = normalize(text)
        twice = normalize(once)
        assert once == twice


def test_tokenize_join_equals_normalized_word_mode() -> None:
    cfg = NormalizationConfig(tokenization="word")
    rng = random.Random(42)
    for _ in range(100):
        text = _generate_nasty(rng)
        normalized = normalize(text, cfg)
        tokens = tokenize(text, cfg)
        joined = " ".join(tokens)
        assert joined == normalized


def test_determinism_same_input_same_output() -> None:
    rng = random.Random(99)
    for _ in range(100):
        text = _generate_nasty(rng)
        a = normalize(text)
        b = normalize(text)
        assert a == b


def test_idempotence_with_all_options() -> None:
    cfg = NormalizationConfig(
        digits_to="spoken",
        keep_punctuation=("'", "-"),
        case_mode="lower",
    )
    rng = random.Random(77)
    for _ in range(100):
        text = _generate_nasty(rng)
        once = normalize(text, cfg)
        twice = normalize(once, cfg)
        assert once == twice
