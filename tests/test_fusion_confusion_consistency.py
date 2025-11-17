"""Property test: fuse(majority) matches confusion network one-best.

Under policy='majority' with no scores (uniform weights), the fused
tokens equal minimal_cut_one_best(build_confusion_network(grid)) for
every grid tested. Both use lexicographic tie-breaking and count-based
voting, so they are guaranteed to agree.
"""

from __future__ import annotations

import random

from hypofuse.confusion import build_confusion_network, minimal_cut_one_best
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_majority_matches_confusion_one_best_200_grids() -> None:
    """Property test over 200 seeded random grids."""
    rng = random.Random(2024)
    for _ in range(200):
        n_hyps = rng.randint(2, 5)
        hyps = []
        for _ in range(n_hyps):
            length = rng.randint(1, 5)
            hyps.append(tuple(rng.choice(_VOCAB) for _ in range(length)))
        grid = progressive_align(hyps)
        rover = fuse(grid, config=FusionConfig(policy="majority"))
        net = build_confusion_network(grid)
        one_best = minimal_cut_one_best(net)
        assert list(rover.tokens) == list(one_best), (
            f"Mismatch for hyps={hyps}: rover={rover.tokens}, cn={one_best}"
        )


def test_score_weighted_with_uniform_scores_matches_confusion() -> None:
    """score_weighted with uniform scores also matches confusion network."""
    rng = random.Random(2025)
    for _ in range(100):
        n_hyps = rng.randint(2, 4)
        hyps = []
        for _ in range(n_hyps):
            length = rng.randint(1, 4)
            hyps.append(tuple(rng.choice(_VOCAB) for _ in range(length)))
        grid = progressive_align(hyps)
        # Uniform scores: all 1.0
        scores = [[1.0] * grid.width for _ in range(n_hyps)]
        rover = fuse(
            grid,
            scores=scores,
            config=FusionConfig(policy="score_weighted"),
        )
        net = build_confusion_network(grid)
        one_best = minimal_cut_one_best(net)
        assert list(rover.tokens) == list(one_best)


def test_documented_tiebreak_caveat() -> None:
    """With non-uniform scores, fuse and confusion network may disagree.

    The confusion network normalizes posteriors to sum to 1 per slot,
    while score_weighted fusion uses raw scores. When scores differ
    across hypotheses, the ranking can change. This test documents
    the weaker invariant: they agree when using majority with no scores.
    """
    grid = progressive_align([("a",), ("b",)])
    rover_majority = fuse(grid, config=FusionConfig(policy="majority"))
    net = build_confusion_network(grid)
    # Under majority (no scores), they always agree
    assert list(rover_majority.tokens) == list(minimal_cut_one_best(net))
