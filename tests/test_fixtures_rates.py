"""Known error rates produce expected substitution counts."""

from __future__ import annotations

from hypofuse.alignment import DEL, edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_zero_rates_produce_perfect_hypotheses() -> None:
    cfg = FixtureConfig(
        n_utterances=3,
        n_best=1,
        seed=0,
        substitution_rate=0.0,
        insertion_rate=0.0,
        deletion_rate=0.0,
    )
    fx = generate_fixture(cfg)
    for u in fx:
        assert u.hypotheses[0] == u.reference


def test_high_deletion_rate_produces_deletions() -> None:
    cfg = FixtureConfig(
        n_utterances=10,
        n_best=1,
        seed=1,
        substitution_rate=0.0,
        insertion_rate=0.0,
        deletion_rate=0.8,
    )
    fx = generate_fixture(cfg)
    deletions = 0
    total = 0
    for u in fx:
        ops = edit_alignment(u.reference, u.hypotheses[0]).ops
        deletions += sum(1 for op in ops if op.op == DEL)
        total += len(u.reference)
    assert deletions / total > 0.4
