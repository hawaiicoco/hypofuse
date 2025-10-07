"""Tests for fusion_run_from_row / fusion_run_to_row."""

from __future__ import annotations

import json

import pytest

from hypofuse.convert import fusion_run_from_row, fusion_run_to_row
from hypofuse.exceptions import SchemaError
from hypofuse.manifests.fusion_run import FusionArc, FusionRun


def _row(**overrides: object) -> dict:
    base: dict = {
        "schema": "hypofuse.fusion_run",
        "utterance_id": "u1",
        "systems": ["a"],
        "tokens": ["hello"],
        "confidences": [0.9],
        "policy": "majority",
    }
    base.update(overrides)
    return base


def test_fusion_roundtrip() -> None:
    original = FusionRun(
        utterance_id="u1",
        systems=("a", "b"),
        tokens=("hello", "world"),
        confidences=(0.9, 0.7),
        policy="majority",
        config_hash="abc123",
        arcs=(
            FusionArc(pivot="hello", candidates=(("hello", 0.9),)),
            FusionArc(pivot="world", candidates=(("world", 0.7), ("word", 0.3))),
        ),
    )
    row = fusion_run_to_row(original)
    restored = fusion_run_from_row(row)
    assert restored == original
    assert json.loads(json.dumps(row)) == row


def test_fusion_missing_required() -> None:
    row = _row()
    del row["policy"]
    with pytest.raises(SchemaError, match="missing required field: policy"):
        fusion_run_from_row(row)


def test_fusion_wrong_type() -> None:
    with pytest.raises(SchemaError, match="field utterance_id expected str"):
        fusion_run_from_row(_row(utterance_id=99))


def test_fusion_unknown_key() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        fusion_run_from_row(_row(extra="x"))


def test_fusion_wrong_type_confidences() -> None:
    with pytest.raises(SchemaError, match="confidences"):
        fusion_run_from_row(_row(confidences=["bad"]))


def test_fusion_arc_wrong_type() -> None:
    # Nested conversion errors name the inner field, not the container.
    with pytest.raises(SchemaError, match="pivot"):
        fusion_run_from_row(_row(arcs=[{"pivot": 123, "candidates": []}]))
