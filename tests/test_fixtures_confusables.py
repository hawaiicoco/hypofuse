"""Confusable pair injection in synthetic fixtures."""

from __future__ import annotations

from collections import Counter

from hypofuse.alignment import SUB, edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_confusable_substitution_uses_pair() -> None:
    cfg = FixtureConfig(
        n_utterances=10,
        n_best=1,
        seed=42,
        substitution_rate=0.5,
        insertion_rate=0.0,
        deletion_rate=0.0,
        confusables=(("their", "there"),),
    )
    fx = generate_fixture(cfg)
    pairs: Counter[tuple[str, str]] = Counter()
    for u in fx:
        ops = edit_alignment(u.reference, u.hypotheses[0]).ops
        for op in ops:
            if op.op == SUB:
                pairs[(str(op.ref_token), str(op.hyp_token))] += 1
    assert pairs, "expected at least one substitution"
    top_pair, _ = pairs.most_common(1)[0]
    assert top_pair == ("their", "there")
    for pair in pairs:
        assert pair == ("their", "there")
