"""Smoke tests for ReferenceTranscript."""

from __future__ import annotations

from hypofuse.manifests.reference import ReferenceTranscript


def test_reference_defaults() -> None:
    r = ReferenceTranscript(utterance_id="u1", text="hi")
    assert r.speaker_id == ""
    assert r.speaker_group == ""
    assert r.duration_s == 0.0
    assert r.noise_db == 0.0
    assert r.intent_domain == ""
    assert r.tokens == ()


def test_reference_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    r = ReferenceTranscript(utterance_id="u1", text="hi")
    with pytest.raises(FrozenInstanceError):
        r.text = "bye"  # type: ignore[misc]
