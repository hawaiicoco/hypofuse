"""UtteranceScore.error_breakdown and is_perfect property."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore


def test_perfect_alignment() -> None:
    us = UtteranceScore("u1", ("a", "b", "c"), ("a", "b", "c"))
    assert us.is_perfect
    subs, dels, ins = us.error_breakdown()
    assert subs == 0
    assert dels == 0
    assert ins == 0


def test_all_wrong_substitutions() -> None:
    us = UtteranceScore("u1", ("a", "b", "c"), ("x", "y", "z"))
    assert not us.is_perfect
    subs, dels, ins = us.error_breakdown()
    # three reference tokens, three hypothesis tokens, all different -> 3 subs
    assert subs == 3
    assert dels == 0
    assert ins == 0


def test_deletion_and_insertion() -> None:
    # ref=("a","b","c") hyp=("a","c") -> MATCH(a), DEL(b), MATCH(c)
    us = UtteranceScore("u1", ("a", "b", "c"), ("a", "c"))
    assert not us.is_perfect
    subs, dels, ins = us.error_breakdown()
    assert subs == 0
    assert dels == 1
    assert ins == 0


def test_insertion_only() -> None:
    # ref=("a",) hyp=("a","b") -> MATCH(a), INS(b)
    us = UtteranceScore("u1", ("a",), ("a", "b"))
    subs, dels, ins = us.error_breakdown()
    assert subs == 0
    assert dels == 0
    assert ins == 1


def test_empty_alignment() -> None:
    us = UtteranceScore("u1", (), ())
    assert us.is_perfect
    subs, dels, ins = us.error_breakdown()
    assert (subs, dels, ins) == (0, 0, 0)
