"""Manifest bridge: fusion_to_manifest_row and fusion_from_manifest_row."""

from __future__ import annotations

import pytest

from hypofuse.fusion import (
    FusionConfig,
    fuse,
    fusion_from_manifest_row,
    fusion_to_manifest_row,
)
from hypofuse.manifests.validate import validate_record
from hypofuse.multi_align import progressive_align


def test_to_manifest_row_passes_validation() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    row = fusion_to_manifest_row(result, utterance_id="utt001")
    schema = validate_record(row)
    assert schema == "hypofuse.fusion_run"


def test_to_manifest_row_field_names() -> None:
    grid = progressive_align([("a",), ("b",)])
    result = fuse(grid)
    row = fusion_to_manifest_row(
        result, utterance_id="utt002", system="rover", config=FusionConfig()
    )
    assert row["utterance_id"] == "utt002"
    assert row["systems"] == ["rover"]
    assert row["policy"] == "majority"
    assert row["schema"] == "hypofuse.fusion_run"
    assert isinstance(row["tokens"], list)
    assert isinstance(row["confidences"], list)
    assert isinstance(row["arcs"], list)
    assert "config_hash" in row


def test_roundtrip_preserves_tokens_and_confidences() -> None:
    grid = progressive_align([("x", "y"), ("x", "z"), ("w", "y")])
    result = fuse(grid)
    row = fusion_to_manifest_row(result, utterance_id="utt003")
    restored = fusion_from_manifest_row(row)
    assert restored.tokens == result.tokens
    assert restored.confidences == result.confidences


def test_roundtrip_preserves_chosen() -> None:
    grid = progressive_align([("a", "b"), ("c", "d")])
    result = fuse(grid)
    row = fusion_to_manifest_row(result, utterance_id="utt004")
    restored = fusion_from_manifest_row(row)
    assert len(restored.chosen) == len(result.chosen)
    for orig, rest in zip(result.chosen, restored.chosen, strict=True):
        assert str(orig[0]) == str(rest[0])
        assert float(orig[1]) == pytest.approx(float(rest[1]))


def test_default_system_is_rover() -> None:
    grid = progressive_align([("a",), ("b",)])
    result = fuse(grid)
    row = fusion_to_manifest_row(result, utterance_id="utt005")
    assert row["systems"] == ["rover"]


def test_config_hash_is_deterministic() -> None:
    grid = progressive_align([("a",), ("b",)])
    result = fuse(grid)
    cfg = FusionConfig(policy="majority")
    row1 = fusion_to_manifest_row(result, utterance_id="utt006", config=cfg)
    row2 = fusion_to_manifest_row(result, utterance_id="utt006", config=cfg)
    assert row1["config_hash"] == row2["config_hash"]
    assert row1["config_hash"] != ""
