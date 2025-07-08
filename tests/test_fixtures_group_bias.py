"""Speaker group bias on substitution rate."""

from __future__ import annotations

from hypofuse.alignment import SUB, edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def _count_substitutions(fx: list, group: str) -> tuple[int, int]:
    subs = 0
    total = 0
    for u in fx:
        if u.speaker_group != group:
            continue
        ops = edit_alignment(u.reference, u.hypotheses[0]).ops
        subs += sum(1 for op in ops if op.op == SUB)
        total += len(u.reference)
    return subs, total


def test_group_bias_a_higher_than_b() -> None:
    cfg = FixtureConfig(
        n_utterances=300,
        n_best=1,
        seed=42,
        substitution_rate=0.1,
        insertion_rate=0.0,
        deletion_rate=0.0,
        speaker_groups=("A", "B"),
        group_bias={"A": 2.0, "B": 0.0},
    )
    fx = generate_fixture(cfg)
    subs_a, total_a = _count_substitutions(fx, "A")
    subs_b, _total_b = _count_substitutions(fx, "B")
    rate_a = subs_a / total_a if total_a else 0.0
    assert rate_a > 0.0, "group A should have substitutions"
    assert subs_b == 0, "group B should have zero substitutions"
