"""Rank decay increases edit distance with rank."""

from __future__ import annotations

from hypofuse.alignment import edit_alignment
from hypofuse.fixtures import FixtureConfig, generate_fixture


def test_mean_edit_distance_non_decreasing_with_rank() -> None:
    cfg = FixtureConfig(
        n_utterances=200,
        n_best=4,
        seed=42,
        substitution_rate=0.05,
        insertion_rate=0.02,
        deletion_rate=0.02,
        rank_decay=0.5,
    )
    fx = generate_fixture(cfg)
    means: list[float] = []
    for rank in range(cfg.n_best):
        total_errors = 0
        total_ref = 0
        for u in fx:
            ops = edit_alignment(u.reference, u.hypotheses[rank]).ops
            total_errors += sum(1 for op in ops if op.op != "MATCH")
            total_ref += len(u.reference)
        means.append(total_errors / total_ref if total_ref else 0.0)
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1], (
            f"mean[{i}]={means[i]:.4f} > mean[{i + 1}]={means[i + 1]:.4f}"
        )
