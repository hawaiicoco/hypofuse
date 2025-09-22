"""Cost model behaviour: cheap, expensive and unknown substitution pairs."""

from __future__ import annotations

import pytest

from hypofuse.alignment import SUB, AlignmentCosts, edit_alignment


def test_cheap_substitution_preferred_over_ins_del() -> None:
    # ref = ["a"], hyp = ["b"]
    # Default substitution = 10.0 (expensive), so DEL+INS (2.0) would win.
    # But token_costs[("a","b")] = 0.1 makes SUB cheaper.
    # Arithmetic: SUB = 0.1 < DEL + INS = 1.0 + 1.0 = 2.0
    costs = AlignmentCosts(
        substitution=10.0,
        token_costs={("a", "b"): 0.1},
    )
    align = edit_alignment(["a"], ["b"], costs=costs)
    assert [op.op for op in align.ops] == [SUB]
    assert align.score == pytest.approx(0.1)


def test_expensive_substitution_prefers_ins_del() -> None:
    # ref = ["a"], hyp = ["b"]
    # substitution = 10.0 (expensive)
    # Arithmetic: SUB = 10.0 > DEL + INS = 1.0 + 1.0 = 2.0
    costs = AlignmentCosts(substitution=10.0)
    align = edit_alignment(["a"], ["b"], costs=costs)
    ops = [op.op for op in align.ops]
    assert SUB not in ops
    assert align.score == pytest.approx(2.0)


def test_unknown_pair_falls_back_to_substitution() -> None:
    # token_costs has (a, b) = 0.1, but we align (a, c)
    # Should use default substitution cost = 1.0
    costs = AlignmentCosts(
        substitution=1.0,
        token_costs={("a", "b"): 0.1},
    )
    align = edit_alignment(["a"], ["c"], costs=costs)
    assert [op.op for op in align.ops] == [SUB]
    assert align.score == pytest.approx(1.0)


def test_token_costs_exact_pair() -> None:
    costs = AlignmentCosts(
        substitution=5.0,
        token_costs={("a", "b"): 0.5},
    )
    align = edit_alignment(["a"], ["b"], costs=costs)
    assert align.score == pytest.approx(0.5)
