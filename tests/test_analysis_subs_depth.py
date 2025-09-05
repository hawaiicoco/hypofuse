"""Substitution pair mining with depth and direction controls."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, substitution_pairs


def _items() -> list[UtteranceScore]:
    # u0: ref=("a","b","c") hyp=("x","b","c") -> sub a->x
    # u1: ref=("a","b","c") hyp=("a","x","c") -> sub b->x
    # u2: ref=("a","b","c") hyp=("a","b","x") -> sub c->x
    # u3: ref=("a","d") hyp=("x","d") -> sub a->x (second occurrence)
    return [
        UtteranceScore("u0", ("a", "b", "c"), ("x", "b", "c")),
        UtteranceScore("u1", ("a", "b", "c"), ("a", "x", "c")),
        UtteranceScore("u2", ("a", "b", "c"), ("a", "b", "x")),
        UtteranceScore("u3", ("a", "d"), ("x", "d")),
    ]


def test_directional_keeps_order() -> None:
    pairs = substitution_pairs(_items(), directional=True)
    # a->x appears twice (u0 and u3)
    assert pairs[("a", "x")] == 2
    assert pairs[("b", "x")] == 1
    assert pairs[("c", "x")] == 1


def test_nondirectional_collapses_pairs() -> None:
    items = [
        UtteranceScore("u0", ("a",), ("b",)),  # sub a->b
        UtteranceScore("u1", ("b",), ("a",)),  # sub b->a
    ]
    pairs = substitution_pairs(items, directional=False)
    # (a,b) and (b,a) collapse to sorted pair ("a","b")
    assert pairs[("a", "b")] == 2
    assert ("b", "a") not in pairs


def test_min_count_cutoff() -> None:
    pairs = substitution_pairs(_items(), min_count=2)
    assert ("a", "x") in pairs
    assert ("b", "x") not in pairs
    assert ("c", "x") not in pairs


def test_top_n_with_stable_tiebreak() -> None:
    # b->x and c->x both have count=1; tie-break is lexicographic
    pairs = substitution_pairs(_items(), top_n=2)
    keys = list(pairs.keys())
    assert len(keys) == 2
    # a->x (count=2) first; then b->x (count=1, "b" < "c")
    assert keys[0] == ("a", "x")
    assert keys[1] == ("b", "x")


def test_defaults_match_original_api() -> None:
    items = [UtteranceScore("u0", ("a", "b"), ("x", "b"))]
    pairs = substitution_pairs(items)
    assert pairs[("a", "x")] == 1
