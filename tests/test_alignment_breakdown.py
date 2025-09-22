"""Error breakdown invariants over exhaustive short pairs."""

from __future__ import annotations

import itertools

from hypofuse.alignment import edit_alignment, error_breakdown


def _all_strings(alphabet: tuple[str, ...], max_len: int) -> list[list[str]]:
    """Synthetic: all token lists up to *max_len* from *alphabet*."""
    result: list[list[str]] = [[]]
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            result.append(list(combo))
    return result


def test_error_breakdown_invariants_exhaustive() -> None:
    """Exhaustive over all pairs from a 3-symbol alphabet, lengths 0-4.

    121 strings, 14641 pairs -- well under one second.
    """
    alphabet = ("a", "b", "c")
    strings = _all_strings(alphabet, 4)
    for ref in strings:
        for hyp in strings:
            align = edit_alignment(ref, hyp)
            eb = error_breakdown(align)
            # total == substitutions + insertions + deletions
            assert eb.total == eb.substitutions + eb.insertions + eb.deletions
            # total <= max(ref_length, hyp_length)
            assert eb.total <= max(eb.ref_length, eb.hyp_length)
            # lengths match the input
            assert eb.ref_length == len(ref)
            assert eb.hyp_length == len(hyp)


def test_error_breakdown_perfect_match() -> None:
    align = edit_alignment(["a", "b"], ["a", "b"])
    eb = error_breakdown(align)
    assert eb.substitutions == 0
    assert eb.insertions == 0
    assert eb.deletions == 0
    assert eb.total == 0


def test_error_breakdown_mixed_errors() -> None:
    # ref = ["a", "b", "c"], hyp = ["a", "x", "c", "d"]
    # Expected: MATCH(a,a), SUB(b,x), MATCH(c,c), INS(d)
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c", "d"])
    eb = error_breakdown(align)
    assert eb.substitutions == 1
    assert eb.insertions == 1
    assert eb.deletions == 0
    assert eb.total == 2
