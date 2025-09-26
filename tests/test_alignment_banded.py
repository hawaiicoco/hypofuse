"""Banded alignment: Sakoe-Chiba restriction tests."""

from __future__ import annotations

import random

import pytest

from hypofuse.alignment import edit_alignment
from hypofuse.exceptions import AlignmentError


def test_wide_band_equals_unbanded() -> None:
    ref = ["a", "b", "c", "d"]
    hyp = ["a", "x", "c", "d"]
    unbanded = edit_alignment(ref, hyp)
    banded = edit_alignment(ref, hyp, band=max(len(ref), len(hyp)))
    assert banded.ops == unbanded.ops
    assert banded.score == pytest.approx(unbanded.score)


def test_wide_band_property_seeded() -> None:
    """Synthetic seeded pairs: band >= max(n,m) equals unbanded."""
    rng = random.Random(42)
    alphabet = ["a", "b", "c", "d", "e"]
    for _ in range(50):
        n = rng.randint(0, 20)
        m = rng.randint(0, 20)
        ref = [rng.choice(alphabet) for _ in range(n)]
        hyp = [rng.choice(alphabet) for _ in range(m)]
        unbanded = edit_alignment(ref, hyp)
        banded = edit_alignment(ref, hyp, band=max(n, m))
        assert banded.ops == unbanded.ops


def test_narrow_band_still_returns_valid_path() -> None:
    ref = ["a"] * 10
    hyp = ["b"] * 10
    align = edit_alignment(ref, hyp, band=2)
    ref_pos = hyp_pos = 0
    for op in align.ops:
        if op.op in ("MATCH", "SUB"):
            ref_pos += 1
            hyp_pos += 1
        elif op.op == "INS":
            hyp_pos += 1
        elif op.op == "DEL":
            ref_pos += 1
    assert ref_pos == len(ref)
    assert hyp_pos == len(hyp)


def test_band_zero_equal_lengths() -> None:
    align = edit_alignment(["a", "b"], ["a", "b"], band=0)
    assert all(op.op == "MATCH" for op in align.ops)


def test_band_zero_unequal_lengths_raises() -> None:
    with pytest.raises(AlignmentError, match="too narrow"):
        edit_alignment(["a", "b"], ["a"], band=0)


def test_negative_band_raises() -> None:
    with pytest.raises(AlignmentError, match="non-negative"):
        edit_alignment(["a"], ["b"], band=-1)


@pytest.mark.slow
def test_large_banded_alignment() -> None:
    """Synthetic: 2000-token sequences with band=200."""
    rng = random.Random(123)
    tokens = [f"t{i}" for i in range(2000)]
    hyp = tokens[:]
    for _ in range(100):
        i = rng.randint(0, 1999)
        j = rng.randint(0, 1999)
        hyp[i], hyp[j] = hyp[j], hyp[i]
    align = edit_alignment(tokens, hyp, band=200)
    ref_pos = hyp_pos = 0
    for op in align.ops:
        if op.op in ("MATCH", "SUB"):
            ref_pos += 1
            hyp_pos += 1
        elif op.op == "INS":
            hyp_pos += 1
        elif op.op == "DEL":
            ref_pos += 1
    assert ref_pos == 2000
    assert hyp_pos == 2000
