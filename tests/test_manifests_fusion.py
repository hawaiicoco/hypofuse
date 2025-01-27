"""Tests for FusionRun / FusionArc."""

from __future__ import annotations

import pytest

from hypofuse.manifests.fusion_run import FusionArc, FusionRun


def test_fusion_run_num_tokens() -> None:
    run = FusionRun(
        utterance_id="u1",
        systems=("a", "b"),
        tokens=("hello", "world"),
        confidences=(0.9, 0.7),
        policy="majority",
    )
    assert run.num_tokens() == 2


def test_fusion_arc_collects_candidates() -> None:
    arc = FusionArc(pivot="*", candidates=(("a", 0.3), ("b", 0.7)))
    assert arc.pivot == "*"
    assert arc.candidates == (("a", 0.3), ("b", 0.7))


def test_fusion_run_frozen() -> None:
    from dataclasses import FrozenInstanceError

    run = FusionRun(
        utterance_id="u1",
        systems=("a",),
        tokens=("hello",),
        confidences=(1.0,),
        policy="majority",
    )
    with pytest.raises(FrozenInstanceError):
        run.tokens = ()  # type: ignore[misc]
