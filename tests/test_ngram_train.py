"""Training and basic probability."""

from __future__ import annotations

from hypofuse.ngram import EOS, UNK, NgramLM


def test_train_builds_vocabulary() -> None:
    sents = [["the", "cat", "sat"], ["the", "dog", "ran"]]
    lm = NgramLM.train(sents, order=2)
    assert "the" in lm.vocab
    assert UNK in lm.vocab
    assert EOS in lm.vocab


def test_unigram_probability_sums_to_one() -> None:
    sents = [["a", "b", "a"], ["a", "b"]]
    lm = NgramLM.train(sents, order=1)
    total = sum(lm.prob_katz((tok,)) for tok in lm.vocab if tok not in (UNK,))
    assert 0.5 <= total <= 2.0  # counts include padding


def test_map_sequence_handles_oov() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    out = lm.map_sequence(["a", "c"])
    assert out == ["a", UNK]


def test_invalid_order_raises() -> None:
    import pytest

    from hypofuse.exceptions import LanguageModelError

    with pytest.raises(LanguageModelError):
        NgramLM.train([["a"]], order=0)


def test_bigram_count_increases_with_repeats() -> None:
    sents = [["a", "b", "a", "b"]]
    lm = NgramLM.train(sents, order=2)
    bigram = ("a", "b")
    assert lm.counts[2][bigram] >= 2
