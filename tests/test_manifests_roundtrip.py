"""End-to-end JSONL roundtrip with mixed schemas."""

from __future__ import annotations

from pathlib import Path

from hypofuse.manifests.jsonl import read_manifest, write_manifest


def test_roundtrip_mixed_schemas(tmp_path: Path) -> None:
    rows = [
        {"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"},
        {"schema": "hypofuse.nbest", "utterance_id": "u2", "system": "b", "language": "en"},
        {"schema": "hypofuse.reference", "utterance_id": "u1", "text": "hi"},
        {"schema": "hypofuse.reference", "utterance_id": "u2", "text": "bye"},
        {"schema": "hypofuse.system", "system_id": "a", "language": "en"},
        {"schema": "hypofuse.system", "system_id": "b", "language": "en"},
    ]
    target = tmp_path / "mixed.jsonl"
    write_manifest(target, rows)
    parsed = read_manifest(target)
    assert parsed == rows


def test_roundtrip_preserves_order(tmp_path: Path) -> None:
    rows = [
        {"schema": "hypofuse.nbest", "utterance_id": f"u{i}", "system": "s", "language": "en"}
        for i in range(20)
    ]
    target = tmp_path / "ordered.jsonl"
    write_manifest(target, rows)
    parsed = read_manifest(target)
    assert [r["utterance_id"] for r in parsed] == [r["utterance_id"] for r in rows]
