"""Top confused token pair mining."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, substitution_pairs


def test_substitution_pair_counts_match_alignment() -> None:
    items = [
        UtteranceScore("u1", ("a", "b", "c"), ("a", "x", "c")),
    ]
    pairs = substitution_pairs(items)
    assert pairs[(("b",), ("x",))] == 0  # format check
    # Pairs are keyed by (str(ref), str(hyp)).
    assert pairs[("b", "x")] == 1


def test_no_substitutions_means_empty_counter() -> None:
    items = [UtteranceScore("u1", ("a", "b"), ("a", "b"))]
    pairs = substitution_pairs(items)
    assert len(pairs) == 0


def test_multiple_substitutions_in_one_utterance() -> None:
    items = [UtteranceScore("u1", ("a", "b", "c"), ("x", "y", "z"))]
    pairs = substitution_pairs(items)
    assert sum(pairs.values()) == 3
