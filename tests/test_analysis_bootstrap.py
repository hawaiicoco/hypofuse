"""Paired bootstrap CI for system comparisons."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, paired_bootstrap_ci


def _items(n: int) -> list[UtteranceScore]:
    out = []
    for i in range(n):
        ref = ("the", "cat")
        hyp_a = ("the", "cat") if i % 2 == 0 else ("the", "bat")
        hyp_b = ("the", "cat") if i % 3 == 0 else ("a", "cat")
        out.append((ref, hyp_a, hyp_b))
    refs, a, b = zip(*out, strict=True)
    return [UtteranceScore(f"u{i}", r, h) for i, (r, h) in enumerate(zip(refs, a, strict=True))], [
        UtteranceScore(f"u{i}", r, h) for i, (r, h) in enumerate(zip(refs, b, strict=True))
    ]


def test_bootstrap_ci_deterministic() -> None:
    a, b = _items(20)
    r1 = paired_bootstrap_ci(a, b, seed=123, n_bootstrap=100)
    r2 = paired_bootstrap_ci(a, b, seed=123, n_bootstrap=100)
    assert r1.cer_ci_low == r2.cer_ci_low
    assert r1.cer_ci_high == r2.cer_ci_high


def test_bootstrap_ci_changes_with_seed() -> None:
    a, b = _items(20)
    r1 = paired_bootstrap_ci(a, b, seed=1, n_bootstrap=50)
    r2 = paired_bootstrap_ci(a, b, seed=2, n_bootstrap=50)
    assert r1.cer_ci_low != r2.cer_ci_low


def test_bootstrap_ci_length_mismatch() -> None:
    import pytest

    from hypofuse.analysis import UtteranceScore as US

    a = [US("u1", ("a",), ("a",))]
    b = []
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b, n_bootstrap=10)
