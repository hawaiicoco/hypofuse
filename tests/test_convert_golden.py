"""Tests reading golden JSONL data files and drift detection."""

from __future__ import annotations

from pathlib import Path

from hypofuse.convert import migrate_row, nbest_from_row, reference_from_row, system_from_row
from hypofuse.manifests.jsonl import read_manifest

_DATA = Path(__file__).parent / "data"


def test_read_nbest_v1_golden() -> None:
    rows = read_manifest(_DATA / "nbest_v1.jsonl")
    assert len(rows) == 3
    assert rows[0]["utterance_id"] == "u001"


def test_read_nbest_v1_migrate_and_convert() -> None:
    rows = read_manifest(_DATA / "nbest_v1.jsonl")
    for row in rows:
        v2 = migrate_row(row, to_version=2)
        obj = nbest_from_row(v2)
        assert obj.utterance_id.startswith("u")
        assert obj.hypotheses[0].acoustic_log10 != 0.0 or obj.hypotheses[0].lm_log10 == 0.0


def test_read_nbest_v2_convert_directly() -> None:
    rows = read_manifest(_DATA / "nbest_v2.jsonl")
    assert len(rows) == 3
    obj = nbest_from_row(rows[0])
    assert obj.utterance_id == "u001"
    assert obj.hypotheses[0].text == "hello world"
    assert obj.hypotheses[0].acoustic_log10 == -1.5
    assert obj.hypotheses[0].lm_log10 == 0.0


def test_read_reference_v1_golden() -> None:
    rows = read_manifest(_DATA / "reference_v1.jsonl")
    assert len(rows) == 3
    # CJK reference (row 3)
    cjk = reference_from_row(rows[2])
    assert cjk.utterance_id == "u003"
    # \u4f60\u597d\u4e16\u754c == ni hao shi jie
    assert len(cjk.text) == 4


def test_read_system_v1_golden() -> None:
    rows = read_manifest(_DATA / "system_v1.jsonl")
    assert len(rows) == 3
    obj = system_from_row(rows[0])
    assert obj.system_id == "synth_a"
    assert obj.vocabulary_size == 500


def test_reference_golden_exact_content() -> None:
    """Byte-exact check: accidental edits to the golden file are caught."""
    content = (_DATA / "reference_v1.jsonl").read_text(encoding="utf-8")
    expected = (
        '{"schema": "hypofuse.reference", "utterance_id": "u001",'
        ' "text": "hello world"}\n'
        '{"schema": "hypofuse.reference", "utterance_id": "u002",'
        ' "text": "good morning", "speaker_group": "A",'
        ' "intent_domain": "greeting"}\n'
        '{"schema": "hypofuse.reference", "utterance_id": "u003",'
        ' "text": "\\u4f60\\u597d\\u4e16\\u754c",'
        ' "speaker_group": "B", "intent_domain": "greeting"}\n'
    )
    assert content == expected
