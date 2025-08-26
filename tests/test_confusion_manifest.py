"""Manifest bridge roundtrip and validation tests."""

from __future__ import annotations

from hypofuse.confusion import (
    build_confusion_network,
    confusion_from_manifest_row,
    confusion_to_manifest_row,
)
from hypofuse.manifests.validate import validate_record
from hypofuse.multi_align import progressive_align


def test_to_manifest_row_passes_validation() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    row = confusion_to_manifest_row(net, utterance_id="utt001")
    schema = validate_record(row)
    assert schema == "hypofuse.fusion_run"


def test_to_manifest_row_field_names() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    row = confusion_to_manifest_row(net, utterance_id="utt002", system="sysA")
    assert row["utterance_id"] == "utt002"
    assert row["systems"] == ["sysA"]
    assert row["policy"] == "confusion"
    assert row["schema"] == "hypofuse.fusion_run"
    assert isinstance(row["tokens"], list)
    assert isinstance(row["confidences"], list)
    assert isinstance(row["arcs"], list)


def test_manifest_roundtrip() -> None:
    grid = progressive_align([("x", "y"), ("x", "z"), ("w", "y")])
    net = build_confusion_network(grid)
    row = confusion_to_manifest_row(net, utterance_id="utt003")
    restored = confusion_from_manifest_row(row)
    assert restored.to_dict() == net.to_dict()


def test_manifest_default_system() -> None:
    grid = progressive_align([("a",), ("b",)])
    net = build_confusion_network(grid)
    row = confusion_to_manifest_row(net, utterance_id="utt004")
    assert row["systems"] == ["fusion"]


def test_manifest_empty_network() -> None:
    from hypofuse.confusion import ConfusionNetwork

    net = ConfusionNetwork(slots=())
    row = confusion_to_manifest_row(net, utterance_id="utt005")
    assert row["tokens"] == []
    assert row["arcs"] == []
    restored = confusion_from_manifest_row(row)
    assert restored.slots == ()
