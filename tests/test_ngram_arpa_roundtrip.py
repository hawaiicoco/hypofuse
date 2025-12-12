"""ARPA roundtrip: probability and byte-identical text checks.

The ARPA format uses ``%.4f`` for log10 probabilities and backoff weights,
with tab separators between fields. These tests verify that:
1. prob(ng, "arpa") on the parsed model matches the original for every n-gram.
2. to_arpa(from_arpa(text)) produces byte-identical text.
"""

from __future__ import annotations

import pytest

from hypofuse.ngram import NgramLM


def _tiny_lm() -> NgramLM:
    """Train a small bigram LM for roundtrip tests."""
    return NgramLM.train([["a", "b"]], order=2)


def test_arpa_prob_roundtrip_matches_for_every_ngram() -> None:
    """lm.prob(ng, 'arpa') must approx-equal parsed.prob(ng, 'arpa')."""
    lm = _tiny_lm()
    text = lm.to_arpa()
    parsed = NgramLM.from_arpa(text)
    for n in range(1, lm.order + 1):
        for ng in lm.counts[n]:
            original_p = lm.prob(ng, "arpa")
            parsed_p = parsed.prob(ng, "arpa")
            assert original_p == pytest.approx(parsed_p, rel=1e-3), (
                f"mismatch for {ng}: {original_p} vs {parsed_p}"
            )


def test_arpa_byte_identical_text_roundtrip() -> None:
    """to_arpa(from_arpa(text)) must be byte-identical to the original text.

    Format contract: log10 probabilities use ``%.4f``, fields separated by
    tabs, n-grams sorted lexicographically within each section.
    """
    lm = _tiny_lm()
    text = lm.to_arpa()
    parsed = NgramLM.from_arpa(text)
    text2 = parsed.to_arpa()
    assert text == text2


def test_arpa_format_uses_tabs() -> None:
    """Every data line in an ARPA section uses tab separators."""
    lm = _tiny_lm()
    text = lm.to_arpa()
    in_section = False
    for line in text.splitlines():
        if line.startswith("\\") and line.endswith("-grams:"):
            in_section = True
            continue
        if line.startswith("\\"):
            in_section = False
            continue
        if in_section and line.strip():
            assert "\t" in line, f"expected tab in line: {line!r}"
