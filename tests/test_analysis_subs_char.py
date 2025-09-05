"""Character-level substitution mining for CJK text."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, substitution_pairs


def test_char_level_splits_tokens() -> None:
    # ref = ("\u4e2d\u6587",) hyp = ("\u4e2d\u56fd",)
    # Word-level: one sub (\u4e2d\u6587 -> \u4e2d\u56fd)
    # Char-level: \u4e2d matches, \u6587 -> \u56fd
    items = [
        UtteranceScore("u0", ("\u4e2d\u6587",), ("\u4e2d\u56fd",)),
    ]
    word_pairs = substitution_pairs(items, level="word")
    assert len(word_pairs) == 1
    char_pairs = substitution_pairs(items, level="char")
    assert ("\u6587", "\u56fd") in char_pairs
    assert char_pairs[("\u6587", "\u56fd")] == 1


def test_char_level_multiple_chars() -> None:
    # ref = ("abc",) hyp = ("axc",) -> at char level: a=a, b->x, c=c
    items = [UtteranceScore("u0", ("abc",), ("axc",))]
    pairs = substitution_pairs(items, level="char")
    assert pairs[("b", "x")] == 1
    assert len(pairs) == 1
