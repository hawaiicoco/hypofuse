"""Tests for nbest_from_row strict conversion."""

from __future__ import annotations

import pytest

from hypofuse.convert import nbest_from_row
from hypofuse.exceptions import SchemaError


def _row(**overrides: object) -> dict:
    base: dict = {
        "schema": "hypofuse.nbest",
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
        "hypotheses": [{"rank": 1, "text": "hi", "tokens": ["hi"]}],
    }
    base.update(overrides)
    return base


def test_nbest_from_row_basic() -> None:
    obj = nbest_from_row(_row())
    assert obj.utterance_id == "u1"
    assert obj.system == "a"
    assert obj.language == "en"
    assert len(obj.hypotheses) == 1
    assert obj.hypotheses[0].rank == 1
    assert obj.hypotheses[0].text == "hi"
    assert obj.hypotheses[0].tokens == ("hi",)


def test_nbest_from_row_missing_required() -> None:
    row = _row()
    del row["utterance_id"]
    with pytest.raises(SchemaError, match="missing required field: utterance_id"):
        nbest_from_row(row)


def test_nbest_from_row_wrong_type() -> None:
    with pytest.raises(SchemaError, match="field system expected str"):
        nbest_from_row(_row(system=42))


def test_nbest_from_row_unknown_key() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        nbest_from_row(_row(extra_field="bad"))


def test_nbest_from_row_allow_extra() -> None:
    obj = nbest_from_row(_row(extra_field="ok"), allow_extra=True)
    assert obj.utterance_id == "u1"


def test_nbest_from_row_hypothesis_unknown_key() -> None:
    row = _row(hypotheses=[{"rank": 1, "text": "hi", "bogus": True}])
    with pytest.raises(SchemaError, match="unknown keys"):
        nbest_from_row(row)


def test_nbest_from_row_empty_hypotheses() -> None:
    obj = nbest_from_row(_row(hypotheses=[]))
    assert obj.hypotheses == ()


def test_nbest_from_row_defaults() -> None:
    obj = nbest_from_row(_row())
    assert obj.audio_path == ""
    h = obj.hypotheses[0]
    assert h.posteriors == ()
    assert h.acoustic_log10 == 0.0
    assert h.lm_log10 == 0.0
    assert h.start_time is None
    assert h.end_time is None


def test_nbest_roundtrip() -> None:
    """nbest_from_row(nbest_to_row(x)) == x for a populated object."""
    import json

    from hypofuse.convert import nbest_to_row
    from hypofuse.manifests.nbest import NBestHypothesis, NBestList

    original = NBestList(
        utterance_id="u7",
        system="sys",
        language="en",
        hypotheses=(
            NBestHypothesis(
                rank=1,
                text="hello world",
                tokens=("hello", "world"),
                posteriors=(0.9, 0.8),
                acoustic_log10=-1.5,
                lm_log10=-0.3,
                start_time=0.0,
                end_time=1.2,
            ),
        ),
        audio_path="corpus/u7.wav",
    )
    row = nbest_to_row(original)
    restored = nbest_from_row(row)
    assert restored == original
    # dict must be JSON-serializable
    assert json.loads(json.dumps(row)) == row
