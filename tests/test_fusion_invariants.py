"""Property test: fused tokens are always drawn from inputs."""

from __future__ import annotations

from hypofuse.fusion import (
    POLICY_LM_WEIGHTED,
    POLICY_MAJORITY,
    POLICY_SCORE_WEIGHTED,
    FusionConfig,
    fuse,
)
from hypofuse.multi_align import progressive_align


def test_no_invented_tokens_random_inputs() -> None:
    inputs = [("alpha", "beta"), ("gamma", "beta"), ("alpha", "delta")]
    grid = progressive_align(inputs)
    for policy in (POLICY_MAJORITY, POLICY_SCORE_WEIGHTED, POLICY_LM_WEIGHTED):
        result = fuse(grid, config=FusionConfig(policy=policy))
        universe = {tok for row in inputs for tok in row}
        for tok in result.tokens:
            assert tok in universe


def test_no_invented_tokens_with_scores() -> None:
    inputs = [("cat",), ("dog",), ("fish",)]
    grid = progressive_align(inputs)
    result = fuse(
        grid,
        scores=[[10.0], [0.1], [0.1]],
        lm_scores=[[0.0], [0.0], [0.0]],
        config=FusionConfig(policy="score_weighted"),
    )
    assert result.tokens[0] in {"cat", "dog", "fish"}


def test_fusion_result_lengths_match_grid() -> None:
    inputs = [("a", "b", "c"), ("a", "c"), ("x", "b", "c")]
    grid = progressive_align(inputs)
    result = fuse(grid)
    assert len(result.tokens) == grid.width
    assert len(result.confidences) == grid.width
    assert len(result.chosen) == grid.width
