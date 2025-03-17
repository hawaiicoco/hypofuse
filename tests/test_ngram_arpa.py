"""ARPA export/import round-trip."""

from __future__ import annotations

from hypofuse.ngram import NgramLM


def test_arpa_roundtrip_preserves_vocab() -> None:
    sents = [["a", "b", "c"], ["a", "b"]]
    lm = NgramLM.train(sents, order=2)
    payload = lm.to_arpa()
    parsed = NgramLM.from_arpa(payload)
    assert parsed.vocab == lm.vocab


def test_arpa_roundtrip_preserves_order() -> None:
    sents = [["a", "b", "c"]]
    lm = NgramLM.train(sents, order=3)
    payload = lm.to_arpa()
    parsed = NgramLM.from_arpa(payload)
    assert parsed.order == lm.order


def test_arpa_starts_with_data_header() -> None:
    sents = [["a"]]
    lm = NgramLM.train(sents, order=2)
    payload = lm.to_arpa()
    assert payload.startswith("\\data\\")
    assert payload.rstrip().endswith("\\end\\")


def test_arpa_contains_ngram_sections() -> None:
    sents = [["a", "b"]]
    lm = NgramLM.train(sents, order=3)
    payload = lm.to_arpa()
    assert "\\1-grams:" in payload
    assert "\\2-grams:" in payload
    assert "\\3-grams:" in payload
